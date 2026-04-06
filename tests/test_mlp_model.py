"""Tests for MLPClassifier: forward shapes, parameter count, dropout, activations."""

import pytest
import torch

from src.models.mlp import MLPClassifier
from src.utils import count_parameters

INPUT_DIM = 100
NUM_CLASSES = 10
BATCH_SIZE = 4


class TestMLPClassifierForwardShape:
    """Verify output shapes for 1-layer and 2-layer configurations."""

    def test_one_layer_output_shape(self) -> None:
        model = MLPClassifier(INPUT_DIM, NUM_CLASSES, hidden_dim=64)
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        logits = model(x)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_two_layer_output_shape(self) -> None:
        model = MLPClassifier(INPUT_DIM, NUM_CLASSES, hidden_dim=64, second_hidden_dim=32)
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        logits = model(x)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_single_example(self) -> None:
        model = MLPClassifier(INPUT_DIM, NUM_CLASSES, hidden_dim=64)
        x = torch.randn(1, INPUT_DIM)
        logits = model(x)
        assert logits.shape == (1, NUM_CLASSES)


class TestMLPClassifierParameterCount:
    def test_one_layer_param_count(self) -> None:
        model = MLPClassifier(INPUT_DIM, NUM_CLASSES, hidden_dim=64, dropout_rate=0.0)
        expected = (INPUT_DIM * 64 + 64) + (64 * NUM_CLASSES + NUM_CLASSES)
        assert count_parameters(model) == expected

    def test_two_layer_param_count(self) -> None:
        model = MLPClassifier(INPUT_DIM, NUM_CLASSES, hidden_dim=64, second_hidden_dim=32, dropout_rate=0.0)
        expected = (INPUT_DIM * 64 + 64) + (64 * 32 + 32) + (32 * NUM_CLASSES + NUM_CLASSES)
        assert count_parameters(model) == expected


class TestMLPClassifierDropout:
    def test_dropout_differs_train_vs_eval(self) -> None:
        model = MLPClassifier(INPUT_DIM, NUM_CLASSES, hidden_dim=256, dropout_rate=0.9)
        x = torch.randn(32, INPUT_DIM)

        model.train()
        torch.manual_seed(0)
        out_train = model(x)

        model.eval()
        torch.manual_seed(0)
        out_eval = model(x)

        assert not torch.allclose(out_train, out_eval, atol=1e-6)


class TestMLPClassifierActivations:
    @pytest.mark.parametrize("activation", ["relu", "gelu", "tanh"])
    def test_supported_activations(self, activation: str) -> None:
        model = MLPClassifier(INPUT_DIM, NUM_CLASSES, hidden_dim=64, activation=activation)
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        logits = model(x)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_unsupported_activation_raises(self) -> None:
        with pytest.raises(AssertionError, match="Unsupported activation"):
            MLPClassifier(INPUT_DIM, NUM_CLASSES, hidden_dim=64, activation="sigmoid")


class TestMLPClassifierLogits:
    def test_output_is_raw_logits(self) -> None:
        """Logits should not be softmax-normalized (values can exceed 1 or be negative)."""
        model = MLPClassifier(INPUT_DIM, NUM_CLASSES, hidden_dim=64)
        model.eval()
        x = torch.randn(BATCH_SIZE, INPUT_DIM) * 10
        logits = model(x)
        assert logits.min().item() < 0 or logits.max().item() > 1
