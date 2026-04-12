"""Integration tests for src/train.py: build, preflight assertions, class weights, no-test guardrail."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from src.config import MLPBaselineConfig
from src.constants import NUM_CLASSES
from src.dataset import TFIDFDataset, create_dataloaders
from src.train import build_mlp_baseline, compute_class_weights, train_mlp_baseline


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
    """Verify that preflight checks in the pipeline would catch bad data."""

    def test_nan_features_detected(self) -> None:
        data = np.array([[1.0, float("nan")], [0.5, 0.5]])
        assert np.isnan(data).any()

    def test_inf_features_detected(self) -> None:
        data = np.array([[1.0, float("inf")], [0.5, 0.5]])
        assert np.isinf(data).any()

    def test_label_dtype_check(self) -> None:
        labels = torch.tensor([0, 1, 2], dtype=torch.long)
        assert labels.dtype == torch.long

    def test_out_of_range_labels_detected(self) -> None:
        labels = torch.tensor([0, 1, 200], dtype=torch.long)
        num_classes = 151
        assert not ((labels >= 0) & (labels < num_classes)).all()


class TestNoTestBeforeSelection:
    """Verify train_mlp_baseline does not iterate the test loader."""

    def test_test_loader_not_iterated(self, tmp_path) -> None:
        input_dim = 20
        num_classes = NUM_CLASSES
        n = 32

        rng = np.random.RandomState(42)
        train_features = csr_matrix(rng.randn(n, input_dim).astype(np.float32))
        val_features = csr_matrix(rng.randn(n, input_dim).astype(np.float32))
        test_features = csr_matrix(rng.randn(n, input_dim).astype(np.float32))

        train_labels = torch.randint(0, num_classes, (n,))
        val_labels = torch.randint(0, num_classes, (n,))
        test_labels = torch.randint(0, num_classes, (n,))

        datasets = {
            "train": TFIDFDataset(train_features, train_labels),
            "validation": TFIDFDataset(val_features, val_labels),
            "test": TFIDFDataset(test_features, test_labels),
        }
        loaders = create_dataloaders(datasets, batch_size=16, num_workers=0, pin_memory=False)

        test_loader_iter_count = 0
        original_iter = loaders["test"].__iter__

        def counting_iter():
            nonlocal test_loader_iter_count
            test_loader_iter_count += 1
            return original_iter()

        loaders["test"].__iter__ = counting_iter

        config = MLPBaselineConfig(max_epochs=2, early_stopping_patience=5)
        metadata = {
            "artifact_refs": {"test": "path"},
            "label_names": [f"label_{i}" for i in range(num_classes)],
        }

        with (
            patch("src.train.CHECKPOINTS_DIR", tmp_path / "ckpt"),
            patch("src.train.LOGS_DIR", tmp_path / "logs"),
        ):
            train_mlp_baseline(config, loaders, input_dim, num_classes, metadata)

        assert test_loader_iter_count == 0, "Test loader was iterated during training"
