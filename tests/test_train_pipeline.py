"""Integration tests for production input validation and training-only loader access."""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader

from src.config import BiLSTMConfig, MLPBaselineConfig, TextCNNConfig
from src.constants import NUM_CLASSES, OOS_LABEL_ID, OOS_LABEL_NAME
from src.dataset import CLINCDataset, IntentDataset, TFIDFDataset, create_dataloaders
from src.enums import ModelID
from src.preprocessing import build_vocab
from src.train import (
    NeuralMetadata,
    TFIDFMetadata,
    _validate_split_labels,
    build_mlp_baseline,
    compute_class_weights,
    load_neural_data,
    load_tfidf_data,
    train_bilstm,
    train_mlp_baseline,
    train_text_cnn,
)

_SPLITS = ("train", "validation", "test")
_SEQUENCE_LENGTH = 4
_INPUT_DIM = 3


@dataclass
class PreflightFixture:
    """Mutable malformed-input fixtures consumed by the real data-loading functions."""

    labels: dict[str, list[int | float | list[int]]]
    features: dict[str, csr_matrix]


@pytest.fixture
def preflight_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PreflightFixture:
    label_names = [OOS_LABEL_NAME if i == OOS_LABEL_ID else f"label_{i}" for i in range(NUM_CLASSES)]
    label_to_id = {name: i for i, name in enumerate(label_names)}
    vocab = build_vocab([["first", "second", "common"]])
    summary = {
        "vocabulary_size": vocab.size,
        "special_tokens": {"<PAD>": 0, "<UNK>": 1},
        "max_seq_length": _SEQUENCE_LENGTH,
        "tokenizer": "whitespace split",
        "text_cleaning_policy": {"lowercase": True},
        "timestamp": "2026-09-30T00:00:00+00:00",
        "tfidf_fitted_feature_dim": _INPUT_DIM,
    }
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    for filename, content in (
        ("label_to_id.json", label_to_id),
        ("id_to_label.json", dict(enumerate(label_names))),
        ("vocab.json", vocab.to_dict()),
        ("preprocessing_summary.json", summary),
        ("sequence_length_stats.json", {}),
    ):
        (artifacts_dir / filename).write_text(json.dumps(content))
    vectorizer = TfidfVectorizer().fit(["first common", "second common"])
    with (artifacts_dir / "tfidf_vectorizer.pkl").open("wb") as handle:
        pickle.dump(vectorizer, handle)

    fixture = PreflightFixture(
        labels={split: [0, NUM_CLASSES - 1] for split in _SPLITS},
        features={split: csr_matrix(np.ones((2, _INPUT_DIM), dtype=np.float32)) for split in _SPLITS},
    )
    dataset = MagicMock(spec=CLINCDataset)
    dataset.label_names = label_names
    dataset.__getitem__.side_effect = lambda split: {
        "text": [f"{split} first", f"{split} second"],
        "intent": fixture.labels[split],
    }
    monkeypatch.setattr("src.train.CLINCDataset.load", lambda _config: dataset)
    monkeypatch.setattr("src.train.ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr("src.train.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.train.CHECKPOINTS_DIR", tmp_path / "checkpoints")

    def transform(_vectorizer: TfidfVectorizer, texts: list[str]) -> csr_matrix:
        return fixture.features[texts[0].split()[0]]

    monkeypatch.setattr(TfidfVectorizer, "transform", transform)
    return fixture


def _load_pipeline(model_id: ModelID) -> dict[str, DataLoader]:
    """Exercise the same production loader used by each architecture."""
    if model_id is ModelID.MLP:
        loaders, _, _, _ = load_tfidf_data(MLPBaselineConfig(batch_size=2))
    else:
        config = (
            TextCNNConfig(batch_size=2, max_seq_length=_SEQUENCE_LENGTH)
            if model_id is ModelID.TEXT_CNN
            else BiLSTMConfig(batch_size=2, max_seq_length=_SEQUENCE_LENGTH)
        )
        loaders, _ = load_neural_data(config)
    return loaders


class TestBuildMLPBaseline:
    def test_correct_architecture_one_layer(self) -> None:
        config = MLPBaselineConfig(hidden_dim=128)
        model = build_mlp_baseline(config, input_dim=100, num_classes=NUM_CLASSES)
        x = torch.randn(2, 100)
        logits = model(x)
        assert logits.shape == (2, NUM_CLASSES)

    def test_correct_architecture_two_layer(self) -> None:
        config = MLPBaselineConfig(hidden_dim=128, second_hidden_dim=64)
        model = build_mlp_baseline(config, input_dim=100, num_classes=NUM_CLASSES)
        x = torch.randn(2, 100)
        logits = model(x)
        assert logits.shape == (2, NUM_CLASSES)

    def test_zero_input_dim_raises(self) -> None:
        config = MLPBaselineConfig()
        with pytest.raises(AssertionError, match="input_dim must be positive"):
            build_mlp_baseline(config, input_dim=0, num_classes=NUM_CLASSES)

    def test_wrong_num_classes_raises(self) -> None:
        config = MLPBaselineConfig()
        with pytest.raises(AssertionError, match="num_classes"):
            build_mlp_baseline(config, input_dim=100, num_classes=10)


class TestComputeClassWeights:
    def test_shape(self) -> None:
        labels = [0, 0, 0, 1, 1, 2]
        weights = compute_class_weights(labels, num_classes=3, device=torch.device("cpu"))
        assert weights.shape == (3,)

    def test_from_tensor(self) -> None:
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        weights = compute_class_weights(labels, num_classes=3, device=torch.device("cpu"))
        assert weights.shape == (3,)
        assert torch.allclose(weights, torch.ones(3), atol=1e-5)

    def test_imbalanced(self) -> None:
        labels = [0] * 100 + [1] * 10
        weights = compute_class_weights(labels, num_classes=2, device=torch.device("cpu"))
        assert weights[1] > weights[0]


class TestPreflightAssertions:
    """Malformed fixtures must fail in production, without downloading the dataset."""

    def test_empty_labels_rejected(self) -> None:
        with pytest.raises(ValueError, match="Empty labels in train"):
            _validate_split_labels(torch.empty(0, dtype=torch.long), NUM_CLASSES, "train", expected_rows=0)

    @pytest.mark.parametrize("model_id", list(ModelID))
    def test_valid_inputs_and_boundary_labels_pass(self, preflight_data: PreflightFixture, model_id: ModelID) -> None:
        loaders = _load_pipeline(model_id)
        for split in _SPLITS:
            _, labels = next(iter(loaders[split]))
            assert labels.dtype == torch.long
            assert sorted(labels.tolist()) == [0, NUM_CLASSES - 1]

    @pytest.mark.parametrize("split", _SPLITS)
    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_features_rejected(self, preflight_data: PreflightFixture, split: str, bad_value: float) -> None:
        preflight_data.features[split].data[0] = bad_value
        with pytest.raises(ValueError, match=f"Non-finite TF-IDF features in {split}"):
            _load_pipeline(ModelID.MLP)

    @pytest.mark.parametrize("model_id", list(ModelID))
    @pytest.mark.parametrize("split", _SPLITS)
    @pytest.mark.parametrize(
        ("labels", "error", "message"),
        [
            ([0, -1], ValueError, "Label ids out of range"),
            ([0, NUM_CLASSES], ValueError, "Label ids out of range"),
            ([0, 0.5], TypeError, "expected torch.long"),
            ([0, 1.0], TypeError, "expected torch.long"),
            ([False, True], TypeError, "expected torch.long"),
            ([0, float("nan")], TypeError, "expected torch.long"),
            ([0, float("inf")], TypeError, "expected torch.long"),
            ([0], ValueError, "one-dimensional labels"),
            ([[0], [1]], ValueError, "one-dimensional labels"),
        ],
        ids=["negative", "upper-bound", "fractional", "float-dtype", "bool-dtype", "nan", "inf", "count", "rank"],
    )
    def test_malformed_labels_rejected(
        self,
        preflight_data: PreflightFixture,
        model_id: ModelID,
        split: str,
        labels: list[int | float | list[int]],
        error: type[Exception],
        message: str,
    ) -> None:
        preflight_data.labels[split] = labels
        with pytest.raises(error, match=message) as exception:
            _load_pipeline(model_id)
        assert split in str(exception.value)


class ExplodingTestLoader(DataLoader[tuple[torch.Tensor, torch.Tensor]]):
    """A class-level iterator guard that Python's special-method lookup cannot bypass."""

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        raise AssertionError("Test loader was iterated during training")


class TestNoTestBeforeSelection:
    """Run each real training path on CPU with an unusable test iterator."""

    def test_guard_intercepts_python_iterator_lookup(self) -> None:
        dataset = IntentDataset(torch.zeros((1, _SEQUENCE_LENGTH), dtype=torch.long), torch.zeros(1, dtype=torch.long))
        with pytest.raises(AssertionError, match="Test loader was iterated during training"):
            next(iter(ExplodingTestLoader(dataset, batch_size=1)))

    @pytest.mark.parametrize("model_id", list(ModelID))
    def test_test_loader_not_iterated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_id: ModelID) -> None:
        monkeypatch.setattr("src.train.get_device", lambda: torch.device("cpu"))
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        if model_id is ModelID.MLP:
            datasets = {
                split: TFIDFDataset(csr_matrix(np.ones((4, _INPUT_DIM), dtype=np.float32)), labels) for split in _SPLITS
            }
        else:
            datasets = {
                split: IntentDataset(torch.full((4, _SEQUENCE_LENGTH), 2, dtype=torch.long), labels)
                for split in _SPLITS
            }
        loaders = create_dataloaders(datasets, batch_size=2, num_workers=0, pin_memory=False)
        loaders["test"] = ExplodingTestLoader(datasets["test"], batch_size=2)
        checkpoint_dir = tmp_path / "ckpt"
        log_dir = tmp_path / "logs"
        label_names = [f"label_{i}" for i in range(NUM_CLASSES)]
        if model_id is ModelID.MLP:
            config = MLPBaselineConfig(hidden_dim=4, max_epochs=1, early_stopping_patience=2)
            metadata: TFIDFMetadata = {
                "artifact_refs": {"test": "path"},
                "label_names": label_names,
                "label_to_id": {},
                "id_to_label": {},
                "texts_by_split": {},
            }
            result = train_mlp_baseline(config, loaders, _INPUT_DIM, NUM_CLASSES, metadata, checkpoint_dir, log_dir)
        else:
            neural_metadata: NeuralMetadata = {
                "artifact_refs": {"test": "path"},
                "label_names": label_names,
                "label_to_id": {},
                "id_to_label": {},
                "num_classes": NUM_CLASSES,
                "vocab_size": 4,
                "oov_stats": {},
                "truncation_stats": {},
                "preprocessing_policy": {},
                "manifest_timestamp": None,
            }
            if model_id is ModelID.TEXT_CNN:
                cnn_config = TextCNNConfig(
                    vocab_size=4,
                    embedding_dim=4,
                    num_filters=2,
                    kernel_sizes=(2,),
                    max_seq_length=_SEQUENCE_LENGTH,
                    max_epochs=1,
                    early_stopping_patience=2,
                )
                result = train_text_cnn(cnn_config, loaders, NUM_CLASSES, neural_metadata, checkpoint_dir, log_dir)
            else:
                lstm_config = BiLSTMConfig(
                    vocab_size=4,
                    embedding_dim=4,
                    hidden_dim=2,
                    max_seq_length=_SEQUENCE_LENGTH,
                    max_epochs=1,
                    early_stopping_patience=2,
                )
                result = train_bilstm(lstm_config, loaders, NUM_CLASSES, neural_metadata, checkpoint_dir, log_dir)
        assert "failure_reason" not in result, result
        assert result["best_epoch"] == 1
        assert Path(result["checkpoint_path"]).is_file()
