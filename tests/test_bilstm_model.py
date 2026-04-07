"""Tests for the BiLSTMClassifier model: shapes, padding_idx, hidden state extraction,
trainable embeddings, parameter counts, bidirectional vs unidirectional, NaN guard."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from src.config import BiLSTMConfig
from src.models.bilstm import BiLSTMClassifier
from src.preprocessing import PAD_ID, UNK_ID

VOCAB_SIZE = 100
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
NUM_CLASSES = 10
NUM_LAYERS = 1
DROPOUT = 0.1
BATCH_SIZE = 8
SEQ_LEN = 20


def _make_model(**overrides: Any) -> BiLSTMClassifier:
    defaults: dict[str, Any] = {
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_classes": NUM_CLASSES,
        "num_layers": NUM_LAYERS,
        "dropout_rate": DROPOUT,
        "bidirectional": True,
    }
    defaults.update(overrides)
    return BiLSTMClassifier(**defaults)


def _random_input(batch: int = BATCH_SIZE, seq_len: int = SEQ_LEN) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (batch, seq_len))


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
class TestBiLSTMInit:
    def test_basic_init(self) -> None:
        model = _make_model()
        assert isinstance(model, nn.Module)

    def test_classifier_input_dim_bidirectional(self) -> None:
        model = _make_model(bidirectional=True)
        expected = HIDDEN_DIM * 2
        assert model._classifier_input_dim == expected
        assert model.classifier.in_features == expected

    def test_classifier_input_dim_unidirectional(self) -> None:
        model = _make_model(bidirectional=False)
        expected = HIDDEN_DIM
        assert model._classifier_input_dim == expected
        assert model.classifier.in_features == expected

    def test_output_dim_matches_num_classes(self) -> None:
        model = _make_model()
        assert model.classifier.out_features == NUM_CLASSES

    def test_bidirectional_flag_stored(self) -> None:
        model_bi = _make_model(bidirectional=True)
        model_uni = _make_model(bidirectional=False)
        assert model_bi._bidirectional is True
        assert model_uni._bidirectional is False

    def test_num_directions(self) -> None:
        assert _make_model(bidirectional=True)._num_directions == 2
        assert _make_model(bidirectional=False)._num_directions == 1

    def test_lstm_batch_first(self) -> None:
        model = _make_model()
        assert model.lstm.batch_first is True

    def test_lstm_dropout_noop_single_layer(self) -> None:
        model = _make_model(num_layers=1, dropout_rate=0.5)
        assert model.lstm.dropout == 0.0

    def test_lstm_dropout_active_multi_layer(self) -> None:
        model = _make_model(num_layers=2, dropout_rate=0.5)
        assert model.lstm.dropout == 0.5


# ---------------------------------------------------------------------------
# Forward pass shapes and dtypes
# ---------------------------------------------------------------------------
class TestBiLSTMForward:
    def test_output_shape_bidirectional(self) -> None:
        model = _make_model(bidirectional=True)
        logits = model(_random_input())
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_shape_unidirectional(self) -> None:
        model = _make_model(bidirectional=False)
        logits = model(_random_input())
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_dtype(self) -> None:
        model = _make_model()
        logits = model(_random_input())
        assert logits.dtype == torch.float32

    def test_batch_size_one(self) -> None:
        model = _make_model()
        logits = model(_random_input(batch=1))
        assert logits.shape == (1, NUM_CLASSES)

    def test_variable_seq_length(self) -> None:
        model = _make_model()
        for seq_len in [5, 10, 30]:
            logits = model(_random_input(seq_len=seq_len))
            assert logits.shape == (BATCH_SIZE, NUM_CLASSES)


# ---------------------------------------------------------------------------
# Hidden state extraction shapes
# ---------------------------------------------------------------------------
class TestBiLSTMHiddenStates:
    def test_h_n_shape_single_layer_bidirectional(self) -> None:
        model = _make_model(num_layers=1, bidirectional=True)
        embedded = model.embedding(_random_input())
        _output, (h_n, _c_n) = model.lstm(embedded)
        assert h_n.shape == (1 * 2, BATCH_SIZE, HIDDEN_DIM)

    def test_h_n_shape_two_layer_bidirectional(self) -> None:
        model = _make_model(num_layers=2, bidirectional=True)
        embedded = model.embedding(_random_input())
        _output, (h_n, _c_n) = model.lstm(embedded)
        assert h_n.shape == (2 * 2, BATCH_SIZE, HIDDEN_DIM)

    def test_h_n_shape_single_layer_unidirectional(self) -> None:
        model = _make_model(num_layers=1, bidirectional=False)
        embedded = model.embedding(_random_input())
        _output, (h_n, _c_n) = model.lstm(embedded)
        assert h_n.shape == (1, BATCH_SIZE, HIDDEN_DIM)

    def test_h_n_shape_two_layer_unidirectional(self) -> None:
        model = _make_model(num_layers=2, bidirectional=False)
        embedded = model.embedding(_random_input())
        _output, (h_n, _c_n) = model.lstm(embedded)
        assert h_n.shape == (2, BATCH_SIZE, HIDDEN_DIM)

    def test_summarized_dim_bidirectional(self) -> None:
        model = _make_model(bidirectional=True)
        model.eval()
        embedded = model.embedding(_random_input())
        _output, (h_n, _c_n) = model.lstm(embedded)
        forward_h = h_n[-2, :, :]
        backward_h = h_n[-1, :, :]
        summarized = torch.cat([forward_h, backward_h], dim=-1)
        assert summarized.shape == (BATCH_SIZE, HIDDEN_DIM * 2)

    def test_summarized_dim_unidirectional(self) -> None:
        model = _make_model(bidirectional=False)
        model.eval()
        embedded = model.embedding(_random_input())
        _output, (h_n, _c_n) = model.lstm(embedded)
        summarized = h_n[-1, :, :]
        assert summarized.shape == (BATCH_SIZE, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# Padding index
# ---------------------------------------------------------------------------
class TestBiLSTMPaddingIdx:
    def test_pad_embedding_is_zero(self) -> None:
        model = _make_model()
        pad_emb = model.embedding.weight.data[PAD_ID]
        assert torch.all(pad_emb == 0), "PAD embedding should be zero"

    def test_pad_embedding_stays_zero_after_backward(self) -> None:
        model = _make_model()
        x = _random_input()
        x[:, -1] = PAD_ID
        model.train()
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        pad_grad = model.embedding.weight.grad[PAD_ID]
        assert torch.all(pad_grad == 0), "PAD embedding gradient should be zero"

    def test_unk_embedding_is_not_zero(self) -> None:
        model = _make_model()
        unk_emb = model.embedding.weight.data[UNK_ID]
        assert not torch.all(unk_emb == 0), "UNK embedding should NOT be zero (random init)"


# ---------------------------------------------------------------------------
# Trainable vs frozen embeddings
# ---------------------------------------------------------------------------
class TestBiLSTMEmbeddingTrainability:
    def test_trainable_embeddings_require_grad(self) -> None:
        model = _make_model(trainable_embeddings=True)
        assert model.embedding.weight.requires_grad is True

    def test_frozen_embeddings_no_grad(self) -> None:
        model = _make_model(trainable_embeddings=False)
        assert model.embedding.weight.requires_grad is False

    def test_total_vs_trainable_with_frozen_embeddings(self) -> None:
        model_trainable = _make_model(trainable_embeddings=True)
        model_frozen = _make_model(trainable_embeddings=False)

        total_trainable = sum(p.numel() for p in model_trainable.parameters())
        total_frozen = sum(p.numel() for p in model_frozen.parameters())
        assert total_trainable == total_frozen, "Total params should be the same"

        grad_trainable = sum(p.numel() for p in model_trainable.parameters() if p.requires_grad)
        grad_frozen = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
        assert grad_trainable > grad_frozen

        embedding_params = VOCAB_SIZE * EMBEDDING_DIM
        assert grad_trainable - grad_frozen == embedding_params


# ---------------------------------------------------------------------------
# Parameter count sanity
# ---------------------------------------------------------------------------
class TestBiLSTMParameterCount:
    def test_bidirectional_has_more_params_than_unidirectional(self) -> None:
        model_bi = _make_model(bidirectional=True)
        model_uni = _make_model(bidirectional=False)
        params_bi = sum(p.numel() for p in model_bi.parameters())
        params_uni = sum(p.numel() for p in model_uni.parameters())
        assert params_bi > params_uni

    def test_two_layer_has_more_params_than_one_layer(self) -> None:
        model_1l = _make_model(num_layers=1)
        model_2l = _make_model(num_layers=2)
        params_1l = sum(p.numel() for p in model_1l.parameters())
        params_2l = sum(p.numel() for p in model_2l.parameters())
        assert params_2l > params_1l

    def test_expected_embedding_params(self) -> None:
        model = _make_model()
        emb_params = model.embedding.weight.numel()
        assert emb_params == VOCAB_SIZE * EMBEDDING_DIM

    def test_expected_classifier_params(self) -> None:
        model = _make_model(bidirectional=True)
        weight_params = model.classifier.weight.numel()
        bias_params = model.classifier.bias.numel()
        assert weight_params == (HIDDEN_DIM * 2) * NUM_CLASSES
        assert bias_params == NUM_CLASSES


# ---------------------------------------------------------------------------
# NaN / Inf guard
# ---------------------------------------------------------------------------
class TestBiLSTMNaNGuard:
    def test_normal_input_no_nan(self) -> None:
        model = _make_model()
        model.eval()
        logits = model(_random_input())
        assert torch.isfinite(logits).all()

    def test_all_pad_input_no_nan(self) -> None:
        model = _make_model()
        model.eval()
        x = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
        logits = model(x)
        assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# BiLSTMConfig
# ---------------------------------------------------------------------------
class TestBiLSTMConfig:
    def test_default_values(self) -> None:
        cfg = BiLSTMConfig()
        assert cfg.vocab_size == 0
        assert cfg.embedding_dim == 128
        assert cfg.hidden_dim == 128
        assert cfg.num_layers == 1
        assert cfg.bidirectional is True
        assert cfg.dropout_rate == 0.3
        assert cfg.learning_rate == 5e-4
        assert cfg.weight_decay == 0.0
        assert cfg.batch_size == 64
        assert cfg.max_epochs == 100
        assert cfg.early_stopping_patience == 10
        assert cfg.optimizer == "adam"
        assert cfg.trainable_embeddings is True
        assert cfg.use_class_weights is False
        assert cfg.use_lr_scheduler is False
        assert cfg.random_seed == 42
        assert cfg.dataloader_seed == 42
        assert cfg.monitor_metric == "val_macro_f1"
        assert cfg.oos_strategy == "explicit_class"
        assert cfg.max_seq_length == 20
        assert cfg.summarization_mode == "concat_final_hidden"
        assert cfg.gradient_clipping is True
        assert cfg.max_grad_norm == 1.0

    def test_frozen(self) -> None:
        cfg = BiLSTMConfig()
        with pytest.raises(AttributeError):
            cfg.hidden_dim = 256  # type: ignore[misc]

    def test_to_dict_contains_all_bilstm_fields(self) -> None:
        cfg = BiLSTMConfig(vocab_size=500, hidden_dim=256)
        d = cfg.to_dict()

        bilstm_specific_keys = {
            "vocab_size",
            "embedding_dim",
            "hidden_dim",
            "num_layers",
            "bidirectional",
            "dropout_rate",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "max_epochs",
            "early_stopping_patience",
            "optimizer",
            "trainable_embeddings",
            "use_class_weights",
            "use_lr_scheduler",
            "random_seed",
            "dataloader_seed",
            "monitor_metric",
            "oos_strategy",
            "max_seq_length",
            "summarization_mode",
            "gradient_clipping",
            "max_grad_norm",
        }
        assert bilstm_specific_keys.issubset(d.keys())
        assert d["vocab_size"] == 500
        assert d["hidden_dim"] == 256

    def test_to_dict_values_match_fields(self) -> None:
        cfg = BiLSTMConfig(
            vocab_size=200,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2,
            bidirectional=False,
            dropout_rate=0.5,
            gradient_clipping=False,
            max_grad_norm=2.0,
            summarization_mode="concat_final_hidden",
        )
        d = cfg.to_dict()
        assert d["vocab_size"] == 200
        assert d["embedding_dim"] == 64
        assert d["hidden_dim"] == 128
        assert d["num_layers"] == 2
        assert d["bidirectional"] is False
        assert d["dropout_rate"] == 0.5
        assert d["gradient_clipping"] is False
        assert d["max_grad_norm"] == 2.0
        assert d["summarization_mode"] == "concat_final_hidden"


# ---------------------------------------------------------------------------
# Multi-layer forward pass
# ---------------------------------------------------------------------------
class TestBiLSTMMultiLayer:
    def test_two_layer_bidirectional_forward(self) -> None:
        model = _make_model(num_layers=2, bidirectional=True)
        logits = model(_random_input())
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_three_layer_unidirectional_forward(self) -> None:
        model = _make_model(num_layers=3, bidirectional=False)
        logits = model(_random_input())
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
