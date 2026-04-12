"""TF-IDF + MLP baseline model definition."""

from __future__ import annotations

from torch import Tensor, nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class MLPClassifier(nn.Module):
    """Multi-layer perceptron for intent classification on TF-IDF features.

    Architecture:
        Linear -> Activation -> Dropout
        [-> Linear -> Activation -> Dropout]  (optional second hidden layer)
        -> Linear (output logits)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        second_hidden_dim: int | None = None,
        dropout_rate: float = 0.3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        assert activation in _ACTIVATIONS, f"Unsupported activation: {activation!r}. Choose from {list(_ACTIVATIONS)}"
        act_cls: type[nn.Module] = _ACTIVATIONS[activation]

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            act_cls(),
            nn.Dropout(dropout_rate),
        ]
        last_dim: int = hidden_dim

        if second_hidden_dim is not None:
            layers.extend(
                [
                    nn.Linear(hidden_dim, second_hidden_dim),
                    act_cls(),
                    nn.Dropout(dropout_rate),
                ]
            )
            last_dim = second_hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Return raw logits (no softmax)."""
        return self.head(self.backbone(x))
