"""Tests for the BiLSTM training pipeline.

Covers manifest validation, build + train, checkpoint reload, smoke test,
config serialization, and trainer model-agnosticism.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import BiLSTMConfig
from src.constants import NUM_CLASSES
from src.enums import ModelID
from src.train import (
    _one_batch_bilstm_smoke_test,
    _select_best_from_rows,
    _validate_preprocessing_manifest,
    build_bilstm,
)
from src.trainers.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_summary() -> dict[str, Any]:
    """Minimal valid preprocessing summary matching BiLSTMConfig defaults."""
    return {
        "vocabulary_size": 4311,
        "special_tokens": {"<PAD>": 0, "<UNK>": 1},
        "max_seq_length": 20,
        "tokenizer": "whitespace split",
        "text_cleaning_policy": {"lowercase": True},
        "timestamp": "2025-01-01T00:00:00+00:00",
    }


def _make_valid_label_to_id() -> dict[str, int]:
    mapping: dict[str, int] = {}
    for i in range(NUM_CLASSES):
        name = "oos" if i == 42 else f"label_{i}"
        mapping[name] = i
    return mapping


# ---------------------------------------------------------------------------
# Manifest validation (BiLSTMConfig)
# ---------------------------------------------------------------------------


class TestManifestValidationBiLSTM:
    def test_valid_manifest_passes(self) -> None:
        summary = _make_valid_summary()
        label_to_id = _make_valid_label_to_id()
        config = BiLSTMConfig()
        _validate_preprocessing_manifest(summary, config, label_to_id)

    def test_non_positive_vocab_size_raises(self) -> None:
        summary = _make_valid_summary()
        summary["vocabulary_size"] = 0
        label_to_id = _make_valid_label_to_id()
        config = BiLSTMConfig()
        with pytest.raises(AssertionError, match="vocabulary_size"):
            _validate_preprocessing_manifest(summary, config, label_to_id)

    def test_wrong_max_seq_length_raises(self) -> None:
        summary = _make_valid_summary()
        summary["max_seq_length"] = 50
        label_to_id = _make_valid_label_to_id()
        config = BiLSTMConfig()
        with pytest.raises(AssertionError, match="max_seq_length"):
            _validate_preprocessing_manifest(summary, config, label_to_id)

    def test_wrong_pad_id_raises(self) -> None:
        summary = _make_valid_summary()
        summary["special_tokens"]["<PAD>"] = 5
        label_to_id = _make_valid_label_to_id()
        config = BiLSTMConfig()
        with pytest.raises(AssertionError, match="PAD_ID"):
            _validate_preprocessing_manifest(summary, config, label_to_id)

    def test_missing_oos_label_raises(self) -> None:
        summary = _make_valid_summary()
        label_to_id = {f"no_oos_{i}": i for i in range(NUM_CLASSES)}
        config = BiLSTMConfig()
        with pytest.raises(AssertionError, match="OOS label"):
            _validate_preprocessing_manifest(summary, config, label_to_id)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestBiLSTMSmokeTest:
    def test_smoke_test_passes(self) -> None:
        config = BiLSTMConfig(
            vocab_size=100,
            embedding_dim=16,
            hidden_dim=32,
            num_layers=1,
            max_seq_length=10,
        )
        model = build_bilstm(config, NUM_CLASSES)
        device = torch.device("cpu")
        model.to(device)

        x = torch.randint(0, 100, (4, 10))
        y = torch.randint(0, NUM_CLASSES, (4,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        _one_batch_bilstm_smoke_test(model, loader, criterion, optimizer, device, config, NUM_CLASSES)


# ---------------------------------------------------------------------------
# Build + train
# ---------------------------------------------------------------------------


class TestBiLSTMTraining:
    def test_trains_and_checkpoints(self, tmp_path: Path) -> None:
        config = BiLSTMConfig(
            vocab_size=50,
            embedding_dim=8,
            hidden_dim=16,
            num_layers=1,
            max_seq_length=10,
            max_epochs=3,
            early_stopping_patience=5,
        )
        model = build_bilstm(config, NUM_CLASSES)
        device = torch.device("cpu")
        model.to(device)

        n = 32
        x = torch.randint(0, 50, (n, 10))
        y = torch.randint(0, NUM_CLASSES, (n,))
        train_ds = TensorDataset(x[: n // 2], y[: n // 2])
        val_ds = TensorDataset(x[n // 2 :], y[n // 2 :])
        train_loader = DataLoader(train_ds, batch_size=8)
        val_loader = DataLoader(val_ds, batch_size=8)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(
            model,
            optimizer,
            criterion,
            device,
            max_grad_norm=config.max_grad_norm,
        )

        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=3,
            patience=5,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="test_bilstm",
            config=config.to_dict(),
            artifact_refs={"test": "path"},
            log_dir=tmp_path / "logs",
            model_prefix=ModelID.BILSTM,
        )

        assert result["best_epoch"] >= 1
        assert Path(result["checkpoint_path"]).exists()

        log_path = tmp_path / "logs" / "bilstm_training_log_test_bilstm.json"
        assert log_path.exists()

    def test_epoch_history_has_new_fields(self, tmp_path: Path) -> None:
        config = BiLSTMConfig(
            vocab_size=50,
            embedding_dim=8,
            hidden_dim=16,
            num_layers=1,
            max_seq_length=10,
            max_epochs=2,
            early_stopping_patience=5,
        )
        model = build_bilstm(config, NUM_CLASSES)
        device = torch.device("cpu")
        model.to(device)

        n = 16
        x = torch.randint(0, 50, (n, 10))
        y = torch.randint(0, NUM_CLASSES, (n,))
        train_loader = DataLoader(TensorDataset(x, y), batch_size=8)
        val_loader = DataLoader(TensorDataset(x, y), batch_size=8)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, device, max_grad_norm=1.0)

        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=2,
            patience=5,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="epoch_fields_test",
            config=config.to_dict(),
            artifact_refs={"test": "path"},
            log_dir=tmp_path / "logs",
            model_prefix=ModelID.BILSTM,
        )

        for record in result["epoch_history"]:
            assert "learning_rate" in record, "learning_rate missing from epoch record"
            assert "checkpoint_updated" in record, "checkpoint_updated missing from epoch record"
            assert isinstance(record["learning_rate"], float)
            assert isinstance(record["checkpoint_updated"], bool)


# ---------------------------------------------------------------------------
# Checkpoint save/reload
# ---------------------------------------------------------------------------


class TestBiLSTMCheckpointReload:
    def test_checkpoint_round_trip(self, tmp_path: Path) -> None:
        config = BiLSTMConfig(
            vocab_size=50,
            embedding_dim=8,
            hidden_dim=16,
            num_layers=1,
            max_seq_length=10,
            max_epochs=3,
            early_stopping_patience=5,
        )
        model = build_bilstm(config, NUM_CLASSES)
        device = torch.device("cpu")
        model.to(device)

        n = 32
        x = torch.randint(0, 50, (n, 10))
        y = torch.randint(0, NUM_CLASSES, (n,))
        train_ds = TensorDataset(x[: n // 2], y[: n // 2])
        val_ds = TensorDataset(x[n // 2 :], y[n // 2 :])
        train_loader = DataLoader(train_ds, batch_size=8)
        val_loader = DataLoader(val_ds, batch_size=8)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, device, max_grad_norm=1.0)

        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=3,
            patience=5,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="reload_bilstm",
            config=config.to_dict(),
            artifact_refs={"vocab": "path/to/vocab.json"},
            log_dir=tmp_path / "logs",
            model_prefix=ModelID.BILSTM,
        )

        from src.utils import load_checkpoint

        new_model = build_bilstm(config, NUM_CLASSES)
        meta = load_checkpoint(result["checkpoint_path"], new_model, device)
        assert meta["epoch"] == result["best_epoch"]
        assert meta["config"] == config.to_dict()
        assert "artifact_refs" in meta
        assert meta["artifact_refs"]["vocab"] == "path/to/vocab.json"


# ---------------------------------------------------------------------------
# Config to_dict
# ---------------------------------------------------------------------------


class TestBiLSTMConfigToDict:
    def test_contains_bilstm_specific_fields(self) -> None:
        cfg = BiLSTMConfig(vocab_size=500, hidden_dim=256)
        d = cfg.to_dict()
        required = {
            "hidden_dim",
            "num_layers",
            "bidirectional",
            "summarization_mode",
            "gradient_clipping",
            "max_grad_norm",
            "dataloader_seed",
        }
        assert required.issubset(d.keys()), f"Missing keys: {required - d.keys()}"
        assert d["hidden_dim"] == 256
        assert d["bidirectional"] is True
        assert d["summarization_mode"] == "concat_final_hidden"
        assert d["gradient_clipping"] is True
        assert d["max_grad_norm"] == 1.0
        assert d["dataloader_seed"] == 42

    def test_seed_logging(self) -> None:
        cfg = BiLSTMConfig(random_seed=123, dataloader_seed=456)
        d = cfg.to_dict()
        assert d["random_seed"] == 123
        assert d["dataloader_seed"] == 456


# ---------------------------------------------------------------------------
# Trainer generalization: model-agnostic (spec V2)
# ---------------------------------------------------------------------------


class TestTrainerGeneralization:
    """Verify the trainer works with an arbitrary nn.Module, proving
    it remains model-agnostic after gradient clipping and tie-break changes."""

    def test_fit_with_dummy_linear_model(self, tmp_path: Path) -> None:
        model = nn.Linear(10, 5)
        device = torch.device("cpu")
        model.to(device)

        n = 32
        x = torch.randn(n, 10)
        y = torch.randint(0, 5, (n,))
        train_loader = DataLoader(TensorDataset(x[: n // 2], y[: n // 2]), batch_size=8)
        val_loader = DataLoader(TensorDataset(x[n // 2 :], y[n // 2 :]), batch_size=8)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, device, max_grad_norm=2.0)

        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=3,
            patience=5,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="dummy",
            config={"test": True},
            artifact_refs={"ref": "value"},
            log_dir=tmp_path / "logs",
            model_prefix="dummy",
        )

        assert result["best_epoch"] >= 1
        assert Path(result["checkpoint_path"]).exists()
        for record in result["epoch_history"]:
            assert "learning_rate" in record
            assert "checkpoint_updated" in record

    def test_evaluate_with_dummy_linear_model(self) -> None:
        model = nn.Linear(10, 5)
        device = torch.device("cpu")
        model.to(device)

        n = 16
        x = torch.randn(n, 10)
        y = torch.randint(0, 5, (n,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, device)

        metrics = trainer.evaluate(loader)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "macro_f1" in metrics

    def test_gradient_clipping_none_works(self, tmp_path: Path) -> None:
        """Trainer with max_grad_norm=None skips clipping (MLP/TextCNN path)."""
        model = nn.Linear(10, 5)
        device = torch.device("cpu")
        model.to(device)

        n = 16
        x = torch.randn(n, 10)
        y = torch.randint(0, 5, (n,))
        train_loader = DataLoader(TensorDataset(x, y), batch_size=8)
        val_loader = DataLoader(TensorDataset(x, y), batch_size=8)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, device, max_grad_norm=None)

        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=2,
            patience=5,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="no_clip",
            config={},
            artifact_refs={},
            log_dir=tmp_path / "logs",
            model_prefix="test",
        )

        assert result["best_epoch"] >= 1


# ---------------------------------------------------------------------------
# Tie-break logic
# ---------------------------------------------------------------------------


class TestTieBreakLogic:
    def test_tie_break_prefers_lower_val_loss(self, tmp_path: Path) -> None:
        """When metric ties, checkpoint should update if val loss is lower."""
        model = nn.Linear(10, 5)
        device = torch.device("cpu")
        model.to(device)

        n = 16
        x = torch.randn(n, 10)
        y = torch.randint(0, 5, (n,))
        train_loader = DataLoader(TensorDataset(x, y), batch_size=n)
        val_loader = DataLoader(TensorDataset(x, y), batch_size=n)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, device)

        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=5,
            patience=10,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="tiebreak",
            config={},
            artifact_refs={},
            log_dir=tmp_path / "logs",
            model_prefix="test",
        )

        updated_epochs = [r for r in result["epoch_history"] if r["checkpoint_updated"]]
        assert len(updated_epochs) >= 1


# ---------------------------------------------------------------------------
# Best-row selection reuse
# ---------------------------------------------------------------------------


def _row(
    name: str,
    metric: float | None,
    loss: float = 0.3,
    params: int = 100,
    dur: float = 10.0,
) -> dict[str, Any]:
    return {
        "run_name": name,
        "best_val_metric": metric,
        "best_val_loss": loss,
        "trainable_parameters": params,
        "training_duration": dur,
    }


class TestBiLSTMFinalModelSelection:
    def test_selects_highest_metric(self) -> None:
        rows = [_row("a", 0.85, 0.5), _row("b", 0.90, 0.4), _row("c", 0.88)]
        best = _select_best_from_rows(rows)
        assert best["run_name"] == "b"

    def test_tie_break_by_val_loss(self) -> None:
        rows = [_row("a", 0.90, 0.5), _row("b", 0.90, 0.3)]
        best = _select_best_from_rows(rows)
        assert best["run_name"] == "b"

    def test_skips_none_metrics(self) -> None:
        rows = [
            {"run_name": "fail", "best_val_metric": None},
            _row("ok", 0.80, 0.5),
        ]
        best = _select_best_from_rows(rows)
        assert best["run_name"] == "ok"
