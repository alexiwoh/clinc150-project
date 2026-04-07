"""BiLSTM sentence classifier for intent classification.

Architecture:
    Embedding -> BiLSTM -> concat final forward/backward h_n -> Dropout -> Linear classifier

Embedding policy:
    - Random initialization (PyTorch default) for all tokens.
    - PAD embedding is zeroed via ``padding_idx=0``.
    - UNK embedding uses PyTorch default random init (NOT zeroed).
    - Embeddings are trainable by default; can be frozen via ``trainable_embeddings=False``.

Sequence handling:
    - Fixed-length padded batches (batch_first=True).
    - No packed sequences.
    - padding_idx=PAD_ID in nn.Embedding ensures PAD embeddings stay zero.

Summarization:
    - Concatenated final forward and backward hidden states from the last BiLSTM layer.
    - Forward: h_n[-2, :, :], Backward: h_n[-1, :, :] (when bidirectional=True).
    - Unidirectional fallback: h_n[-1, :, :] only.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from src.preprocessing import PAD_ID

logger = logging.getLogger(__name__)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM sentence classifier over token-id sequences."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout_rate: float = 0.3,
        bidirectional: bool = True,
        padding_idx: int = PAD_ID,
        trainable_embeddings: bool = True,
    ) -> None:
        super().__init__()

        num_directions: int = 2 if bidirectional else 1
        classifier_input_dim: int = hidden_dim * num_directions
        assert classifier_input_dim > 0, "classifier_input_dim must be positive"

        lstm_inter_layer_dropout: float = 0.0 if num_layers == 1 else dropout_rate
        if num_layers == 1:
            logger.info(
                "LSTM internal dropout is a no-op with num_layers=1; classifier-path dropout (%.2f) remains active.",
                dropout_rate,
            )

        self.embedding: nn.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if not trainable_embeddings:
            self.embedding.weight.requires_grad_(False)

        self.lstm: nn.LSTM = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_inter_layer_dropout,
        )

        self.dropout: nn.Dropout = nn.Dropout(dropout_rate)
        self.classifier: nn.Linear = nn.Linear(classifier_input_dim, num_classes)

        self._hidden_dim: int = hidden_dim
        self._num_layers: int = num_layers
        self._num_directions: int = num_directions
        self._bidirectional: bool = bidirectional
        self._classifier_input_dim: int = classifier_input_dim

    def forward(self, x: Tensor) -> Tensor:
        """Return logits of shape ``(batch_size, num_classes)``.

        Args:
            x: integer token-id tensor of shape ``(batch_size, seq_len)``.
        """
        embedded: Tensor = self.embedding(x)  # (B, L, E) -> (batch_size, sequence_length, embedding_dim)
        _output, (h_n, _c_n) = self.lstm(embedded)
        # h_n: (num_layers * num_directions, B, hidden_dim)
        # _c_n: (num_layers * num_directions, B, hidden_dim)

        if self._bidirectional:
            forward_h: Tensor = h_n[-2, :, :]  # (B, hidden_dim)
            backward_h: Tensor = h_n[-1, :, :]  # (B, hidden_dim)
            summarized: Tensor = torch.cat([forward_h, backward_h], dim=-1)
        else:
            summarized = h_n[-1, :, :]  # (B, hidden_dim)

        assert summarized.shape[-1] == self._classifier_input_dim, (
            f"Summarized feature dim {summarized.shape[-1]} != "
            f"expected classifier input dim {self._classifier_input_dim}"
        )

        logits: Tensor = self.classifier(self.dropout(summarized))

        assert torch.isfinite(logits).all(), "NaN or Inf detected in logits"

        return logits
