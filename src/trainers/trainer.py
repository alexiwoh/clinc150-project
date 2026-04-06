"""Reusable training and validation loop logic shared across models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.constants import OOS_LABEL_ID
from src.metrics import compute_classification_metrics, compute_oos_metrics
from src.utils import save_checkpoint


class Trainer:
    """Model-agnostic training loop with early stopping and checkpointing.

    Works with any ``nn.Module`` that returns logits from ``forward()``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: LRScheduler | None = None,
    ) -> None:
        self.model: nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.criterion: nn.Module = criterion
        self.device: torch.device = device
        self.scheduler: LRScheduler | None = scheduler

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Run one training epoch. Returns ``{"loss": avg_loss}``."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for inputs, targets in dataloader:
            # Check if the inputs are finite then convert them to the device
            assert torch.isfinite(inputs).all(), "Non-finite values in training inputs"
            inputs: torch.Tensor = inputs.to(self.device)
            targets: torch.Tensor = targets.to(self.device)

            # Zero the gradients and compute the logits, loss, and backward pass
            self.optimizer.zero_grad()
            logits: torch.Tensor = self.model(inputs)
            loss: torch.Tensor = self.criterion(logits, targets)
            assert not torch.isnan(loss), "NaN loss detected during training"
            loss.backward()
            self.optimizer.step()

            # Update the total loss and the number of batches
            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / max(n_batches, 1)}

    def evaluate(self, dataloader: DataLoader) -> dict[str, Any]:
        """Evaluate the model on a dataloader. Returns metrics + predictions."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds: list[int] = []
        all_targets: list[int] = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                # Check if the inputs are finite then convert them to the device
                assert torch.isfinite(inputs).all(), "Non-finite values in evaluation inputs"
                inputs: torch.Tensor = inputs.to(self.device)
                targets: torch.Tensor = targets.to(self.device)

                # Compute the logits, loss, and update the total loss and the number of batches
                logits: torch.Tensor = self.model(inputs)
                loss: torch.Tensor = self.criterion(logits, targets)
                total_loss += loss.item()
                n_batches += 1

                # Compute the predictions and update the all_preds and all_targets lists
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().tolist())

        # Compute the classification metrics and the OOS metrics
        cls_metrics = compute_classification_metrics(all_targets, all_preds)
        oos_metrics = compute_oos_metrics(all_targets, all_preds, OOS_LABEL_ID)

        return {
            "loss": total_loss / max(n_batches, 1),
            "accuracy": cls_metrics["accuracy"],
            "macro_f1": cls_metrics["macro_f1"],
            "precision": cls_metrics["macro_precision"],
            "recall": cls_metrics["macro_recall"],
            "oos_precision": oos_metrics["oos_precision"],
            "oos_recall": oos_metrics["oos_recall"],
            "oos_f1": oos_metrics["oos_f1"],
            "predictions": all_preds,
            "targets": all_targets,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int,
        patience: int,
        checkpoint_dir: Path | str,
        run_name: str,
        config: dict[str, Any],
        artifact_refs: dict[str, str],
        log_dir: Path | str,
        monitor_metric: str = "val_macro_f1",
    ) -> dict[str, Any]:
        """Full training loop with early stopping, checkpointing, and logging.

        Returns a summary dict with epoch history, best metrics, and timing.
        """
        checkpoint_dir = Path(checkpoint_dir)
        log_dir = Path(log_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"mlp_best_{run_name}.pt"

        best_metric = -float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        epoch_history: list[dict[str, Any]] = []
        reason_for_stopping = "max_epochs_reached"

        start_time = time.time()

        for epoch in range(1, max_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_oos_f1": val_metrics["oos_f1"],
            }
            epoch_history.append(epoch_record)

            metric_key = monitor_metric.replace("val_", "")
            current_metric = val_metrics.get(metric_key, val_metrics.get("macro_f1"))
            assert current_metric is not None, f"Monitor metric {monitor_metric} not found in val metrics"

            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_path,
                    self.model,
                    self.optimizer,
                    epoch,
                    best_metric,
                    config,
                    artifact_refs,
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                reason_for_stopping = "patience_exceeded"
                break

        training_duration = time.time() - start_time

        assert checkpoint_path.exists(), f"Best checkpoint not saved: {checkpoint_path}"

        all_monitored = [h.get(monitor_metric, h.get(monitor_metric.replace("val_", ""))) for h in epoch_history]
        true_best_idx = int(max(range(len(all_monitored)), key=lambda i: all_monitored[i]))
        assert epoch_history[true_best_idx]["epoch"] == best_epoch, (
            f"best_epoch {best_epoch} does not match true max at epoch {epoch_history[true_best_idx]['epoch']}"
        )

        log_path = log_dir / f"mlp_training_log_{run_name}.json"
        log_path.write_text(json.dumps(epoch_history, indent=2))

        return {
            "run_name": run_name,
            "best_epoch": best_epoch,
            "best_val_metric": best_metric,
            "stopping_epoch": epoch_history[-1]["epoch"],
            "reason_for_stopping": reason_for_stopping,
            "training_duration": training_duration,
            "checkpoint_path": str(checkpoint_path),
            "log_path": str(log_path),
            "epoch_history": epoch_history,
            "config": config,
        }
