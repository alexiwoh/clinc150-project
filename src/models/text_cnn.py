"""Kim-style Text CNN for sentence classification.

Architecture:
    Embedding -> parallel Conv1d (multiple kernel sizes) -> ReLU -> max-over-time pooling
    -> concatenation -> Dropout -> Linear classifier

Embedding policy:
    - Random initialization (PyTorch default) for all tokens.
    - PAD embedding is zeroed via ``padding_idx=0``.
    - UNK embedding uses PyTorch default random init (NOT zeroed).
    - Embeddings are trainable by default; can be frozen via ``trainable_embeddings=False``.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from src.preprocessing import PAD_ID

logger = logging.getLogger(__name__)

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class TextCNN(nn.Module):
    """Kim-style sentence-level CNN over token-id sequences."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: tuple[int, ...],
        num_classes: int,
        dropout_rate: float = 0.3,
        activation: str = "relu",
        max_seq_length: int = 20,
        trainable_embeddings: bool = True,
    ) -> None:
        super().__init__()
        assert activation in _ACTIVATIONS, f"Unsupported activation: {activation!r}. Choose from {list(_ACTIVATIONS)}"
        assert max(kernel_sizes) <= max_seq_length, (
            f"max(kernel_sizes)={max(kernel_sizes)} exceeds max_seq_length={max_seq_length}"
        )

        self.embedding: nn.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        if not trainable_embeddings:
            self.embedding.weight.requires_grad_(False)

        act_cls: type[nn.Module] = _ACTIVATIONS[activation]

        self.convs: nn.ModuleList = nn.ModuleList(
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k) for k in kernel_sizes
        )
        self.activations: nn.ModuleList = nn.ModuleList(act_cls() for _ in kernel_sizes)

        classifier_input_dim: int = len(kernel_sizes) * num_filters
        assert classifier_input_dim > 0, "classifier_input_dim must be positive"

        self.dropout: nn.Dropout = nn.Dropout(dropout_rate)
        self.classifier: nn.Linear = nn.Linear(classifier_input_dim, num_classes)

        self._kernel_sizes: tuple[int, ...] = kernel_sizes
        self._num_filters: int = num_filters
        self._classifier_input_dim: int = classifier_input_dim

    def forward(self, x: Tensor) -> Tensor:
        """Return logits of shape ``(batch_size, num_classes)``.

        Args:
            x: integer token-id tensor of shape ``(batch_size, seq_len)``.
        """
        embedded: Tensor = self.embedding(x)  # (B, L, E)
        embedded = embedded.transpose(1, 2)  # (B, E, L) for Conv1d

        pooled_outputs: list[Tensor] = []
        for conv, act in zip(self.convs, self.activations, strict=True):
            h: Tensor = act(conv(embedded))  # (B, num_filters, L')
            h = h.max(dim=2).values  # (B, num_filters)
            pooled_outputs.append(h)

        cat: Tensor = torch.cat(pooled_outputs, dim=1)  # (B, len(kernel_sizes) * num_filters)
        return self.classifier(self.dropout(cat))
