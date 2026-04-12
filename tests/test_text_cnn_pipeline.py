"""Tests for the Text CNN training pipeline.

Covers manifest validation, frozen vocab, data loading, training, and checkpoint reload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import TextCNNConfig
from src.constants import NUM_CLASSES
from src.enums import ModelID
from src.train import (
    _one_batch_smoke_test,
    _select_best_from_rows,
    _validate_preprocessing_manifest,
    build_text_cnn,
)
from src.trainers.trainer import Trainer


def _make_valid_summary() -> dict[str, Any]:
    """Minimal valid preprocessing summary matching TextCNNConfig defaults."""
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


class TestManifestValidation:
    def test_valid_manifest_passes(self) -> None:
        summary = _make_valid_summary()
        label_to_id = _make_valid_label_to_id()
        config = TextCNNConfig()
        _validate_preprocessing_manifest(summary, config, label_to_id)

    def test_non_positive_vocab_size_raises(self) -> None:
        summary = _make_valid_summary()
        summary["vocabulary_size"] = 0
        label_to_id = _make_valid_label_to_id()
        config = TextCNNConfig()
        with pytest.raises(AssertionError, match="vocabulary_size"):
            _validate_preprocessing_manifest(summary, config, label_to_id)

    def test_negative_vocab_size_raises(self) -> None:
        summary = _make_valid_summary()
        summary["vocabulary_size"] = -1
        label_to_id = _make_valid_label_to_id()
        config = TextCNNConfig()
        with pytest.raises(AssertionError, match="vocabulary_size"):
            _validate_preprocessing_manifest(summary, config, label_to_id)

    def test_wrong_max_seq_length_raises(self) -> None:
        summary = _make_valid_summary()
        summary["max_seq_length"] = 50
        label_to_id = _make_valid_label_to_id()
        config = TextCNNConfig()
        with pytest.raises(AssertionError, match="max_seq_length"):
            _validate_preprocessing_manifest(summary, config, label_to_id)

    def test_wrong_pad_id_raises(self) -> None:
        summary = _make_valid_summary()
        summary["special_tokens"]["<PAD>"] = 5
        label_to_id = _make_valid_label_to_id()
        config = TextCNNConfig()
        with pytest.raises(AssertionError, match="PAD_ID"):
            _validate_preprocessing_manifest(summary, config, label_to_id)

    def test_missing_timestamp_raises(self) -> None:
        summary = _make_valid_summary()
        del summary["timestamp"]
        label_to_id = _make_valid_label_to_id()
        config = TextCNNConfig()
        with pytest.raises(AssertionError, match="timestamp"):
            _validate_preprocessing_manifest(summary, config, label_to_id)

    def test_missing_oos_label_raises(self) -> None:
        summary = _make_valid_summary()
        label_to_id = {f"no_oos_{i}": i for i in range(NUM_CLASSES)}
        config = TextCNNConfig()
        with pytest.raises(AssertionError, match="OOS label"):
            _validate_preprocessing_manifest(summary, config, label_to_id)


class TestSmokeTesting:
    def test_smoke_test_passes(self) -> None:
        config = TextCNNConfig(
            vocab_size=100,
            embedding_dim=16,
            num_filters=8,
            kernel_sizes=(3,),
            max_seq_length=10,
        )
        model = build_text_cnn(config, NUM_CLASSES)
        device = torch.device("cpu")
        model.to(device)

        x = torch.randint(0, 100, (4, 10))
        y = torch.randint(0, NUM_CLASSES, (4,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        _one_batch_smoke_test(model, loader, criterion, optimizer, device, config, NUM_CLASSES)


class TestSingleTrainingRun:
    def test_trains_and_checkpoints(self, tmp_path: Path) -> None:
        config = TextCNNConfig(
            vocab_size=50,
            embedding_dim=8,
            num_filters=4,
            kernel_sizes=(3,),
            max_seq_length=10,
            max_epochs=3,
            early_stopping_patience=5,
        )
        model = build_text_cnn(config, NUM_CLASSES)
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
        trainer = Trainer(model, optimizer, criterion, device)

        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=3,
            patience=5,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="test_cnn",
            config=config.to_dict(),
            artifact_refs={"test": "path"},
            log_dir=tmp_path / "logs",
            model_prefix=ModelID.TEXT_CNN,
        )

        assert result["best_epoch"] >= 1
        assert Path(result["checkpoint_path"]).exists()

        log_path = tmp_path / "logs" / "text_cnn_training_log_test_cnn.json"
        assert log_path.exists()

    def test_checkpoint_reload(self, tmp_path: Path) -> None:
        config = TextCNNConfig(
            vocab_size=50,
            embedding_dim=8,
            num_filters=4,
            kernel_sizes=(3,),
            max_seq_length=10,
            max_epochs=3,
            early_stopping_patience=5,
        )
        model = build_text_cnn(config, NUM_CLASSES)
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
        trainer = Trainer(model, optimizer, criterion, device)

        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=3,
            patience=5,
            checkpoint_dir=tmp_path / "ckpt",
            run_name="reload_cnn",
            config=config.to_dict(),
            artifact_refs={"test": "path"},
            log_dir=tmp_path / "logs",
            model_prefix=ModelID.TEXT_CNN,
        )

        from src.utils import load_checkpoint

        new_model = build_text_cnn(config, NUM_CLASSES)
        meta = load_checkpoint(result["checkpoint_path"], new_model, device)
        assert meta["epoch"] == result["best_epoch"]
        assert meta["config"] == config.to_dict()
        assert "artifact_refs" in meta


class TestSeedLogging:
    def test_seeds_appear_in_metadata(self) -> None:
        config = TextCNNConfig(random_seed=123, dataloader_seed=456)
        d = config.to_dict()
        assert d["random_seed"] == 123
        assert d["dataloader_seed"] == 456


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


class TestFinalModelSelection:
    def test_selects_highest_metric(self) -> None:
        rows = [_row("a", 0.85, 0.5), _row("b", 0.90, 0.4), _row("c", 0.88)]
        best = _select_best_from_rows(rows)
        assert best["run_name"] == "b"

    def test_tie_break_by_val_loss(self) -> None:
        rows = [_row("a", 0.90, 0.5), _row("b", 0.90, 0.3)]
        best = _select_best_from_rows(rows)
        assert best["run_name"] == "b"

    def test_tie_break_by_params(self) -> None:
        rows = [_row("a", 0.90, params=200), _row("b", 0.90, params=100)]
        best = _select_best_from_rows(rows)
        assert best["run_name"] == "b"

    def test_tie_break_by_duration(self) -> None:
        rows = [_row("a", 0.90, dur=20.0), _row("b", 0.90, dur=5.0)]
        best = _select_best_from_rows(rows)
        assert best["run_name"] == "b"

    def test_tie_break_by_run_name(self) -> None:
        rows = [_row("b_run", 0.90), _row("a_run", 0.90)]
        best = _select_best_from_rows(rows)
        assert best["run_name"] == "a_run"

    def test_skips_none_metrics(self) -> None:
        rows = [
            {"run_name": "fail", "best_val_metric": None},
            _row("ok", 0.80, 0.5),
        ]
        best = _select_best_from_rows(rows)
        assert best["run_name"] == "ok"
