"""Tests for Trainer: model-agnostic design, early stopping, checkpointing, finite checks."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.trainers.trainer import Trainer
from src.training_contracts import MONITOR_METRIC_DIRECTIONS
from src.utils import load_checkpoint


def _make_synthetic_data(
    n_samples: int = 100,
    input_dim: int = 20,
    num_classes: int = 5,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create tiny synthetic train/val loaders with a simple nn.Linear model."""
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, input_dim, generator=rng)
    y = torch.randint(0, num_classes, (n_samples,), generator=rng)
    split = n_samples // 2
    train_ds = TensorDataset(x[:split], y[:split])
    val_ds = TensorDataset(x[split:], y[split:])
    train_loader = DataLoader(train_ds, batch_size=16)
    val_loader = DataLoader(val_ds, batch_size=16)
    return train_loader, val_loader


class TestTrainerModelAgnostic:
    """Trainer must work with any nn.Module, not just MLPClassifier."""

    def test_trains_with_linear_model(self, tmp_path) -> None:
        train_loader, val_loader = _make_synthetic_data()
        model = nn.Linear(20, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        trainer = Trainer(model, optimizer, criterion, device)
        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=5,
            patience=3,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="linear_test",
            config={"lr": 0.01},
            artifact_refs={"test": "path"},
            log_dir=tmp_path / "logs",
        )

        assert "best_epoch" in result
        assert "epoch_history" in result
        assert len(result["epoch_history"]) > 0


class TestTrainerEarlyStopping:
    def test_patience_exceeded(self, tmp_path) -> None:
        train_loader, val_loader = _make_synthetic_data()
        model = nn.Linear(20, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))

        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=200,
            patience=3,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="es_test",
            config={},
            artifact_refs={},
            log_dir=tmp_path / "logs",
        )

        assert result["reason_for_stopping"] in ("patience_exceeded", "max_epochs_reached")

    def test_max_epochs_reached(self, tmp_path) -> None:
        train_loader, val_loader = _make_synthetic_data()
        model = nn.Linear(20, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))

        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=2,
            patience=999,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="max_ep",
            config={},
            artifact_refs={},
            log_dir=tmp_path / "logs",
        )

        assert result["stopping_epoch"] == 2
        assert result["reason_for_stopping"] == "max_epochs_reached"

    def test_best_epoch_matches_true_max(self, tmp_path) -> None:
        train_loader, val_loader = _make_synthetic_data()
        model = nn.Linear(20, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))

        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=10,
            patience=5,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="best_ep",
            config={},
            artifact_refs={},
            log_dir=tmp_path / "logs",
        )

        history = result["epoch_history"]
        f1_values = [h["val_macro_f1"] for h in history]
        best_idx = max(range(len(f1_values)), key=lambda i: (f1_values[i], -history[i]["val_loss"]))
        assert result["best_epoch"] == history[best_idx]["epoch"]


class TestTrainerEpochHistory:
    def test_history_persisted_to_disk(self, tmp_path) -> None:
        train_loader, val_loader = _make_synthetic_data()
        model = nn.Linear(20, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))

        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=3,
            patience=10,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="hist_test",
            config={},
            artifact_refs={},
            log_dir=tmp_path / "logs",
        )

        log_path = tmp_path / "logs" / "mlp_training_log_hist_test.json"
        assert log_path.exists()

        saved_history = json.loads(log_path.read_text())
        assert len(saved_history) == len(result["epoch_history"])
        for saved, mem in zip(saved_history, result["epoch_history"], strict=True):
            assert saved["epoch"] == mem["epoch"]
            assert abs(saved["val_macro_f1"] - mem["val_macro_f1"]) < 1e-9

    def test_history_has_required_fields(self, tmp_path) -> None:
        train_loader, val_loader = _make_synthetic_data()
        model = nn.Linear(20, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))

        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=2,
            patience=10,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="fields",
            config={},
            artifact_refs={},
            log_dir=tmp_path / "logs",
        )

        required = {
            "epoch",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "val_precision",
            "val_recall",
            "val_oos_f1",
        }
        for record in result["epoch_history"]:
            assert required.issubset(record.keys()), f"Missing fields: {required - record.keys()}"


class TestTrainerCheckpoint:
    def test_checkpoint_exists_after_fit(self, tmp_path) -> None:
        train_loader, val_loader = _make_synthetic_data()
        model = nn.Linear(20, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))

        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=3,
            patience=10,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="ckpt_test",
            config={"lr": 0.01},
            artifact_refs={"v": "path"},
            log_dir=tmp_path / "logs",
        )

        ckpt_path = Path(result["checkpoint_path"])
        assert ckpt_path.exists()

        loaded = torch.load(ckpt_path, weights_only=False)
        assert "model_state_dict" in loaded
        assert "config" in loaded

    def test_checkpoint_reloadable(self, tmp_path) -> None:
        train_loader, val_loader = _make_synthetic_data()
        model = nn.Linear(20, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))

        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=3,
            patience=10,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="reload",
            config={},
            artifact_refs={},
            log_dir=tmp_path / "logs",
        )

        new_model = nn.Linear(20, 5)
        meta = load_checkpoint(result["checkpoint_path"], new_model, torch.device("cpu"))
        assert meta["epoch"] == result["best_epoch"]


class TestTrainerIntegerInputs:
    """Trainer must handle integer-tensor inputs (e.g. token-id sequences for CNN/BiLSTM)."""

    def test_trains_with_int_inputs(self, tmp_path) -> None:
        """Verify trainer works with nn.Embedding + integer inputs (spec U2)."""
        n = 64
        seq_len = 10
        vocab_size = 50
        num_classes = 5
        rng = torch.Generator().manual_seed(42)

        x = torch.randint(0, vocab_size, (n, seq_len), generator=rng)
        y = torch.randint(0, num_classes, (n,), generator=rng)

        split = n // 2
        train_ds = TensorDataset(x[:split], y[:split])
        val_ds = TensorDataset(x[split:], y[split:])
        train_loader = DataLoader(train_ds, batch_size=16)
        val_loader = DataLoader(val_ds, batch_size=16)

        model = nn.Sequential(
            nn.Embedding(vocab_size, 16),
            nn.Flatten(),
            nn.Linear(seq_len * 16, num_classes),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))

        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=3,
            patience=5,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="int_input_test",
            config={"test": True},
            artifact_refs={"test": "path"},
            log_dir=tmp_path / "logs",
            model_prefix="dummy",
        )

        assert result["best_epoch"] >= 1
        assert len(result["epoch_history"]) == 3

    def test_int_inputs_skip_finite_check(self) -> None:
        """Verify integer inputs don't trigger the isfinite assertion (spec U2)."""
        x = torch.randint(0, 50, (4, 10))
        y = torch.zeros(4, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        model = nn.Sequential(
            nn.Embedding(50, 8),
            nn.Flatten(),
            nn.Linear(80, 3),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))

        metrics = trainer.train_epoch(loader)
        assert "loss" in metrics


class TestTrainerFiniteChecks:
    def test_nan_input_raises(self) -> None:
        x = torch.full((4, 10), float("nan"))
        y = torch.zeros(4, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        model = nn.Linear(10, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))

        with pytest.raises(AssertionError, match="Non-finite"):
            trainer.train_epoch(loader)

    def test_inf_input_raises(self) -> None:
        x = torch.full((4, 10), float("inf"))
        y = torch.zeros(4, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        model = nn.Linear(10, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))

        with pytest.raises(AssertionError, match="Non-finite"):
            trainer.train_epoch(loader)

    def test_inf_input_raises_on_evaluate(self) -> None:
        x = torch.full((4, 10), float("inf"))
        y = torch.zeros(4, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        model = nn.Linear(10, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, torch.device("cpu"))

        with pytest.raises(AssertionError, match="Non-finite"):
            trainer.evaluate(loader)


class TestTrainerLossReduction:
    """Reported losses must match full-dataset cross-entropy, independent of batching."""

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("weights", [None, [1.0, 4.0]])
    @pytest.mark.parametrize("label_smoothing", [0.0, 0.2])
    @pytest.mark.parametrize("training", [False, True])
    def test_matches_dataset_mean(
        self, batch_size: int, weights: list[float] | None, label_smoothing: float, training: bool
    ) -> None:
        logits = torch.tensor([[5.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
        targets = torch.tensor([0, 1, 0])
        loader = DataLoader(TensorDataset(logits, targets), batch_size=batch_size)
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        criterion = nn.CrossEntropyLoss(
            weight=None if weights is None else torch.tensor(weights), label_smoothing=label_smoothing
        )
        # Zero learning rate keeps the oracle logits fixed while exercising backward/optimizer steps.
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.0), criterion, torch.device("cpu"))

        expected = criterion(logits, targets).item()
        metrics = trainer.train_epoch(loader) if training else trainer.evaluate(loader)

        assert metrics["loss"] == pytest.approx(expected, rel=1e-6)

    def test_uneven_batch_audit_example(self) -> None:
        logits = torch.tensor([[5.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
        targets = torch.zeros(3, dtype=torch.long)
        loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        trainer = Trainer(
            model, torch.optim.SGD(model.parameters(), lr=0.0), nn.CrossEntropyLoss(), torch.device("cpu")
        )

        assert trainer.evaluate(loader)["loss"] == pytest.approx(1.6733819, rel=1e-6)

    def test_ignored_targets_excluded_from_loss_and_metrics(self) -> None:
        logits = torch.tensor([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]])
        targets = torch.tensor([-100, 1, 0])
        loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]))
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.0), criterion, torch.device("cpu"))

        metrics = trainer.evaluate(loader)

        assert metrics["loss"] == pytest.approx(criterion(logits, targets).item(), rel=1e-6)
        assert metrics["targets"] == [1, 0]
        assert metrics["predictions"] == [1, 0]
        assert metrics["accuracy"] == 1.0

    @pytest.mark.parametrize("reduction", ["sum", "none"])
    def test_unsupported_reduction_rejected(self, reduction: str) -> None:
        model = nn.Linear(2, 2)
        with pytest.raises(ValueError, match="reduction='mean'"):
            Trainer(
                model,
                torch.optim.SGD(model.parameters(), lr=0.0),
                nn.CrossEntropyLoss(reduction=reduction),
                torch.device("cpu"),
            )

    @pytest.mark.parametrize("weights", [[1.0, -1.0], [1.0, float("nan")], [1.0, float("inf")]])
    def test_invalid_weights_rejected(self, weights: list[float]) -> None:
        model = nn.Linear(2, 2)
        with pytest.raises(ValueError, match="weights must be finite and nonnegative"):
            Trainer(
                model,
                torch.optim.SGD(model.parameters(), lr=0.0),
                nn.CrossEntropyLoss(weight=torch.tensor(weights)),
                torch.device("cpu"),
            )

    @pytest.mark.parametrize("training", [False, True])
    @pytest.mark.parametrize("targets", [torch.tensor([-100, -100]), torch.tensor([0, 0])])
    def test_zero_denominator_rejected(self, training: bool, targets: torch.Tensor) -> None:
        model = nn.Linear(2, 2)
        trainer = Trainer(
            model,
            torch.optim.SGD(model.parameters(), lr=0.0),
            nn.CrossEntropyLoss(weight=torch.tensor([0.0, 1.0])),
            torch.device("cpu"),
        )
        loader = DataLoader(TensorDataset(torch.zeros(2, 2), targets), batch_size=2)
        with pytest.raises(ValueError, match="positive finite cross-entropy denominator"):
            if training:
                trainer.train_epoch(loader)
            else:
                trainer.evaluate(loader)

    @pytest.mark.parametrize("training", [False, True])
    def test_soft_targets_rejected(self, training: bool) -> None:
        model = nn.Linear(2, 2)
        trainer = Trainer(
            model, torch.optim.SGD(model.parameters(), lr=0.0), nn.CrossEntropyLoss(), torch.device("cpu")
        )
        loader = DataLoader(TensorDataset(torch.zeros(2, 2), torch.full((2, 2), 0.5)), batch_size=2)
        with pytest.raises(ValueError, match="integer class targets"):
            if training:
                trainer.train_epoch(loader)
            else:
                trainer.evaluate(loader)

    @pytest.mark.parametrize("training", [False, True])
    def test_empty_loader_rejected(self, training: bool) -> None:
        model = nn.Linear(2, 2)
        trainer = Trainer(
            model, torch.optim.SGD(model.parameters(), lr=0.0), nn.CrossEntropyLoss(), torch.device("cpu")
        )
        loader = DataLoader(TensorDataset(torch.zeros(0, 2), torch.zeros(0, dtype=torch.long)), batch_size=2)
        with pytest.raises(ValueError, match="empty dataloader"):
            if training:
                trainer.train_epoch(loader)
            else:
                trainer.evaluate(loader)

    @pytest.mark.parametrize("training", [False, True])
    def test_nonfinite_loss_rejected(self, training: bool) -> None:
        model = nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.fill_(float("inf"))
        trainer = Trainer(
            model, torch.optim.SGD(model.parameters(), lr=0.0), nn.CrossEntropyLoss(), torch.device("cpu")
        )
        loader = DataLoader(TensorDataset(torch.ones(2, 2), torch.zeros(2, dtype=torch.long)), batch_size=2)
        with pytest.raises(ValueError, match="Non-finite loss"):
            if training:
                trainer.train_epoch(loader)
            else:
                trainer.evaluate(loader)


class _ScriptedTrainer(Trainer):
    """Exercise real fit/checkpoint behavior with a deterministic validation sequence."""

    def __init__(self, losses: list[float], scores: list[float], monitor_metric: str) -> None:
        model = nn.Linear(2, 2)
        super().__init__(model, torch.optim.SGD(model.parameters(), lr=0.0), nn.CrossEntropyLoss(), torch.device("cpu"))
        self._validation_metrics: list[dict[str, float]] = []
        self.train_calls: int = 0
        for loss, score in zip(losses, scores, strict=True):
            metrics = {name.removeprefix("val_"): 0.5 for name in MONITOR_METRIC_DIRECTIONS}
            metrics["loss"] = loss
            if monitor_metric != "val_loss":
                metrics[monitor_metric.removeprefix("val_")] = score
            self._validation_metrics.append(metrics)

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.train_calls += 1
        return {"loss": 0.0}

    def evaluate(self, dataloader: DataLoader) -> dict[str, Any]:
        return self._validation_metrics[self.train_calls - 1]


def _fit_scripted(trainer: _ScriptedTrainer, tmp_path: Path, monitor_metric: str, patience: int = 10) -> dict[str, Any]:
    loader = DataLoader(TensorDataset(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long)))
    return trainer.fit(
        loader,
        loader,
        max_epochs=len(trainer._validation_metrics),
        patience=patience,
        checkpoint_dir=tmp_path / "checkpoints",
        run_name="scripted",
        config={},
        artifact_refs={},
        log_dir=tmp_path / "logs",
        monitor_metric=monitor_metric,
    )


class TestTrainerMonitorContract:
    def test_loss_is_minimized(self, tmp_path: Path) -> None:
        trainer = _ScriptedTrainer([0.4, 0.8, 0.2], [0.0] * 3, "val_loss")
        result = _fit_scripted(trainer, tmp_path, "val_loss")

        assert result["best_epoch"] == 3
        assert result["best_val_metric"] == 0.2
        assert [epoch["checkpoint_updated"] for epoch in result["epoch_history"]] == [True, False, True]
        checkpoint = torch.load(result["checkpoint_path"], weights_only=False)
        assert checkpoint["epoch"] == 3
        assert checkpoint["best_metric"] == 0.2

    @pytest.mark.parametrize("monitor_metric", [name for name in MONITOR_METRIC_DIRECTIONS if name != "val_loss"])
    def test_each_score_is_maximized_and_logged(self, tmp_path: Path, monitor_metric: str) -> None:
        trainer = _ScriptedTrainer([0.5] * 3, [0.4, 0.8, 0.6], monitor_metric)
        result = _fit_scripted(trainer, tmp_path, monitor_metric)

        assert result["best_epoch"] == 2
        assert result["best_val_metric"] == 0.8
        assert [epoch[monitor_metric] for epoch in result["epoch_history"]] == [0.4, 0.8, 0.6]

    def test_score_ties_prefer_lower_loss_then_earliest(self, tmp_path: Path) -> None:
        trainer = _ScriptedTrainer([0.5, 0.3, 0.3, 0.4], [0.8] * 4, "val_macro_f1")
        result = _fit_scripted(trainer, tmp_path, "val_macro_f1")

        assert result["best_epoch"] == 2
        assert [epoch["checkpoint_updated"] for epoch in result["epoch_history"]] == [True, True, False, False]

    def test_loss_ties_keep_earliest(self, tmp_path: Path) -> None:
        trainer = _ScriptedTrainer([0.3, 0.3, 0.4], [0.0] * 3, "val_loss")
        result = _fit_scripted(trainer, tmp_path, "val_loss")
        assert result["best_epoch"] == 1

    @pytest.mark.parametrize("monitor_metric", ["val_loss", "val_macro_f1"])
    def test_patience_counts_nonimproving_epochs(self, tmp_path: Path, monitor_metric: str) -> None:
        trainer = _ScriptedTrainer([0.5, 0.3, 0.3, 0.4, 0.1], [0.8] * 5, monitor_metric)
        result = _fit_scripted(trainer, tmp_path, monitor_metric, patience=2)

        assert result["best_epoch"] == 2
        assert result["stopping_epoch"] == 4
        assert result["reason_for_stopping"] == "patience_exceeded"
        assert trainer.train_calls == 4

    def test_tie_break_improvement_resets_patience(self, tmp_path: Path) -> None:
        trainer = _ScriptedTrainer([0.5, 0.6, 0.3, 0.3, 0.4], [0.8] * 5, "val_macro_f1")
        result = _fit_scripted(trainer, tmp_path, "val_macro_f1", patience=2)

        assert result["best_epoch"] == 3
        assert result["stopping_epoch"] == 5
        assert result["reason_for_stopping"] == "patience_exceeded"

    @pytest.mark.parametrize("monitor_metric", ["val_typo", "loss", "macro_f1", "val_predictions", "val_val_loss"])
    def test_invalid_name_fails_before_training_or_outputs(self, tmp_path: Path, monitor_metric: str) -> None:
        trainer = _ScriptedTrainer([0.4], [0.8], "val_macro_f1")

        with pytest.raises(ValueError, match="Unsupported monitor metric"):
            _fit_scripted(trainer, tmp_path, monitor_metric)

        assert trainer.train_calls == 0
        assert not (tmp_path / "checkpoints").exists()
        assert not (tmp_path / "logs").exists()

    @pytest.mark.parametrize("max_epochs, patience", [(0, 1), (-1, 1), (1, 0), (1, -1)])
    def test_invalid_bounds_fail_before_outputs(self, tmp_path: Path, max_epochs: int, patience: int) -> None:
        trainer = _ScriptedTrainer([0.4], [0.8], "val_macro_f1")
        loader = DataLoader(TensorDataset(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long)))
        with pytest.raises(ValueError, match="must be positive"):
            trainer.fit(
                loader, loader, max_epochs, patience, tmp_path / "checkpoints", "bounds", {}, {}, tmp_path / "logs"
            )
        assert trainer.train_calls == 0
        assert not (tmp_path / "checkpoints").exists()
        assert not (tmp_path / "logs").exists()

    @pytest.mark.parametrize("monitor_metric", ["val_loss", "val_macro_f1"])
    @pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), -float("inf")])
    def test_nonfinite_monitor_fails_before_checkpoint(
        self, tmp_path: Path, monitor_metric: str, invalid_value: float
    ) -> None:
        trainer = _ScriptedTrainer([invalid_value], [invalid_value], monitor_metric)
        with pytest.raises(ValueError, match="must be finite"):
            _fit_scripted(trainer, tmp_path, monitor_metric)
        assert not list((tmp_path / "checkpoints").glob("*.pt"))
