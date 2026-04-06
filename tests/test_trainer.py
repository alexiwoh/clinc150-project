"""Tests for Trainer: model-agnostic design, early stopping, checkpointing, finite checks."""

import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.trainers.trainer import Trainer


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
        best_idx = max(range(len(f1_values)), key=lambda i: f1_values[i])
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

        from pathlib import Path

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

        from src.utils import load_checkpoint

        new_model = nn.Linear(20, 5)
        meta = load_checkpoint(result["checkpoint_path"], new_model, torch.device("cpu"))
        assert meta["epoch"] == result["best_epoch"]


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
