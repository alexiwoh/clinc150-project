"""Reusable training and validation loop logic shared across models."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.constants import OOS_LABEL_ID
from src.metrics import compute_classification_metrics, compute_oos_metrics
from src.training_contracts import MONITOR_METRIC_DIRECTIONS
from src.utils import save_checkpoint


class Trainer:
    """Model-agnostic training loop with early stopping and checkpointing.

    Works with any ``nn.Module`` that returns class logits from ``forward()``.
    The loss contract is mean-reduced ``nn.CrossEntropyLoss`` with integer class
    targets. Epoch losses use the full target-weight denominator, including when
    the last batch is short. Ignored targets are excluded from metrics as well.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.CrossEntropyLoss,
        device: torch.device,
        scheduler: LRScheduler | None = None,
        max_grad_norm: float | None = None,
    ) -> None:
        if criterion.reduction != "mean":
            raise ValueError("Trainer requires CrossEntropyLoss with reduction='mean'")
        if criterion.weight is not None and (
            not torch.isfinite(criterion.weight).all() or (criterion.weight < 0).any()
        ):
            raise ValueError("CrossEntropyLoss class weights must be finite and nonnegative")

        self.model: nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.criterion: nn.CrossEntropyLoss = criterion
        self.device: torch.device = device
        self.scheduler: LRScheduler | None = scheduler
        self.max_grad_norm: float | None = max_grad_norm

    def _loss_denominator(self, targets: torch.Tensor) -> float:
        """Return PyTorch's mean cross-entropy denominator for class targets."""
        if targets.ndim != 1 or targets.dtype != torch.long:
            raise ValueError("Trainer requires one-dimensional integer class targets")
        valid_targets = targets[targets != self.criterion.ignore_index]
        denominator = (
            float(valid_targets.numel())
            if self.criterion.weight is None
            else float(self.criterion.weight[valid_targets].sum().item())
        )
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("Each batch must have a positive finite cross-entropy denominator")
        return denominator

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Run one training epoch. Returns ``{"loss": avg_loss}``."""
        self.model.train()
        total_loss = 0.0
        total_weight = 0.0
        for inputs, targets in dataloader:
            if inputs.is_floating_point():
                assert torch.isfinite(inputs).all(), "Non-finite values in training inputs"
            inputs: torch.Tensor = inputs.to(self.device)
            targets: torch.Tensor = targets.to(self.device)
            batch_weight = self._loss_denominator(targets)

            # Zero the gradients and compute the logits, loss, and backward pass
            self.optimizer.zero_grad()
            logits: torch.Tensor = self.model(inputs)
            loss: torch.Tensor = self.criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss detected during training")
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Recover the numerator of each mean-reduced batch loss.
            total_loss += loss.item() * batch_weight
            total_weight += batch_weight

        if total_weight == 0:
            raise ValueError("Cannot train on an empty dataloader")
        return {"loss": total_loss / total_weight}

    def evaluate(self, dataloader: DataLoader) -> dict[str, Any]:
        """Evaluate the model on a dataloader. Returns metrics + predictions."""
        self.model.eval()
        total_loss = 0.0
        total_weight = 0.0
        all_preds: list[int] = []
        all_targets: list[int] = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                if inputs.is_floating_point():
                    assert torch.isfinite(inputs).all(), "Non-finite values in evaluation inputs"
                inputs: torch.Tensor = inputs.to(self.device)
                targets: torch.Tensor = targets.to(self.device)
                batch_weight = self._loss_denominator(targets)

                # Recover the loss numerator using the criterion's reduction denominator.
                logits: torch.Tensor = self.model(inputs)
                loss: torch.Tensor = self.criterion(logits, targets)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss detected during evaluation")
                total_loss += loss.item() * batch_weight
                total_weight += batch_weight

                # Compute the predictions and update the all_preds and all_targets lists
                valid_targets = targets != self.criterion.ignore_index
                preds = logits.argmax(dim=1)[valid_targets].cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(targets[valid_targets].cpu().tolist())

        if total_weight == 0:
            raise ValueError("Cannot evaluate an empty dataloader")
        # Compute the classification metrics and the OOS metrics
        cls_metrics = compute_classification_metrics(all_targets, all_preds)
        oos_metrics = compute_oos_metrics(all_targets, all_preds, OOS_LABEL_ID)

        return {
            "loss": total_loss / total_weight,
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
        model_prefix: str = "mlp",
    ) -> dict[str, Any]:
        """Full training loop with early stopping, checkpointing, and logging.

        Supported ``val_*`` metrics are declared in ``MONITOR_METRIC_DIRECTIONS``.
        Loss is minimized and scores are maximized. Ties prefer lower validation
        loss, then the earliest epoch. Patience counts consecutive epochs without
        an improvement under that complete selection rule.

        Returns a summary dict with epoch history, best metrics, and timing.
        """
        if monitor_metric not in MONITOR_METRIC_DIRECTIONS:
            raise ValueError(f"Unsupported monitor metric {monitor_metric!r}")
        if max_epochs < 1 or patience < 1:
            raise ValueError("max_epochs and patience must be positive")
        direction = MONITOR_METRIC_DIRECTIONS[monitor_metric]
        metric_key = monitor_metric.removeprefix("val_")

        checkpoint_dir = Path(checkpoint_dir)
        log_dir = Path(log_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{model_prefix}_best_{run_name}.pt"

        best_metric = float("inf") if direction == "min" else -float("inf")
        best_val_loss = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        epoch_history: list[dict[str, Any]] = []
        reason_for_stopping = "max_epochs_reached"

        start_time = time.perf_counter()

        for epoch in range(1, max_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            current_metric: float = val_metrics[metric_key]
            current_val_loss: float = val_metrics["loss"]
            if not math.isfinite(current_metric) or not math.isfinite(current_val_loss):
                raise ValueError("Validation monitor metric and loss must be finite")

            is_improvement = current_metric < best_metric if direction == "min" else current_metric > best_metric
            if not is_improvement and current_metric == best_metric:
                is_improvement = current_val_loss < best_val_loss

            epoch_record: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": current_val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_oos_precision": val_metrics["oos_precision"],
                "val_oos_recall": val_metrics["oos_recall"],
                "val_oos_f1": val_metrics["oos_f1"],
                "learning_rate": current_lr,
                "checkpoint_updated": is_improvement,
            }
            epoch_history.append(epoch_record)

            if is_improvement:
                best_metric = current_metric
                best_val_loss = current_val_loss
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

        training_duration = time.perf_counter() - start_time

        assert checkpoint_path.exists(), f"Best checkpoint not saved: {checkpoint_path}"

        all_monitored = [h[monitor_metric] for h in epoch_history]
        optimal_metric_value = min(all_monitored) if direction == "min" else max(all_monitored)
        best_epoch_metric = all_monitored[best_epoch - 1]
        assert best_epoch_metric == optimal_metric_value, (
            f"best_epoch {best_epoch} metric {best_epoch_metric} != optimal metric {optimal_metric_value}"
        )

        log_path = log_dir / f"{model_prefix}_training_log_{run_name}.json"
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
