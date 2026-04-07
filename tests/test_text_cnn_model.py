"""Tests for the TextCNN model: shapes, padding_idx, UNK init, kernel validation, activation, param counts."""

import pytest
import torch
from torch import nn

from src.models.text_cnn import TextCNN
from src.preprocessing import PAD_ID


VOCAB_SIZE = 100
EMBEDDING_DIM = 32
NUM_FILTERS = 16
KERNEL_SIZES = (3, 4, 5)
NUM_CLASSES = 10
DROPOUT = 0.1
MAX_SEQ_LEN = 20
BATCH_SIZE = 8


def _make_model(**overrides) -> TextCNN:
    defaults = {
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "num_filters": NUM_FILTERS,
        "kernel_sizes": KERNEL_SIZES,
        "num_classes": NUM_CLASSES,
        "dropout_rate": DROPOUT,
        "max_seq_length": MAX_SEQ_LEN,
    }
    defaults.update(overrides)
    return TextCNN(**defaults)


class TestTextCNNInit:
    def test_basic_init(self) -> None:
        model = _make_model()
        assert isinstance(model, nn.Module)

    def test_classifier_input_dim(self) -> None:
        model = _make_model()
        expected = len(KERNEL_SIZES) * NUM_FILTERS
        assert model._classifier_input_dim == expected
        assert model.classifier.in_features == expected

    def test_output_dim_matches_num_classes(self) -> None:
        model = _make_model()
        assert model.classifier.out_features == NUM_CLASSES

    def test_kernel_size_exceeds_seq_length_raises(self) -> None:
        with pytest.raises(AssertionError, match="exceeds max_seq_length"):
            _make_model(kernel_sizes=(25,), max_seq_length=20)

    def test_unsupported_activation_raises(self) -> None:
        with pytest.raises(AssertionError, match="Unsupported activation"):
            _make_model(activation="leaky_relu")


class TestTextCNNForward:
    def test_output_shape(self) -> None:
        model = _make_model()
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
        logits = model(x)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_dtype(self) -> None:
        model = _make_model()
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
        logits = model(x)
        assert logits.dtype == torch.float32

    def test_single_kernel_size(self) -> None:
        model = _make_model(kernel_sizes=(3,))
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
        logits = model(x)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_batch_size_one(self) -> None:
        model = _make_model()
        x = torch.randint(0, VOCAB_SIZE, (1, MAX_SEQ_LEN))
        logits = model(x)
        assert logits.shape == (1, NUM_CLASSES)


class TestTextCNNPaddingIdx:
    def test_pad_embedding_is_zero(self) -> None:
        model = _make_model()
        pad_emb = model.embedding.weight.data[PAD_ID]
        assert torch.all(pad_emb == 0), "PAD embedding should be zero"

    def test_pad_embedding_stays_zero_after_forward(self) -> None:
        model = _make_model()
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
        x[:, -1] = PAD_ID
        model.train()
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        pad_grad = model.embedding.weight.grad[PAD_ID]
        assert torch.all(pad_grad == 0), "PAD embedding gradient should be zero"

    def test_unk_embedding_is_not_zero(self) -> None:
        from src.preprocessing import UNK_ID

        model = _make_model()
        unk_emb = model.embedding.weight.data[UNK_ID]
        assert not torch.all(unk_emb == 0), "UNK embedding should NOT be zero (random init)"


class TestTextCNNActivation:
    @pytest.mark.parametrize("act", ["relu", "gelu", "tanh"])
    def test_activation_variants(self, act: str) -> None:
        model = _make_model(activation=act)
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
        logits = model(x)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)


class TestTextCNNParameters:
    def test_total_vs_trainable_with_frozen_embeddings(self) -> None:
        model_trainable = _make_model(trainable_embeddings=True)
        model_frozen = _make_model(trainable_embeddings=False)

        total_trainable = sum(p.numel() for p in model_trainable.parameters())
        total_frozen = sum(p.numel() for p in model_frozen.parameters())
        assert total_trainable == total_frozen, "Total params should be the same"

        grad_trainable = sum(p.numel() for p in model_trainable.parameters() if p.requires_grad)
        grad_frozen = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
        assert grad_trainable > grad_frozen, "Trainable model should have more grad-requiring params"

        embedding_params = VOCAB_SIZE * EMBEDDING_DIM
        assert grad_trainable - grad_frozen == embedding_params

    def test_embedding_frozen_flag(self) -> None:
        model = _make_model(trainable_embeddings=False)
        assert not model.embedding.weight.requires_grad
