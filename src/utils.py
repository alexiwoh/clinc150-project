"""Shared utilities such as seeding, logging helpers, checkpoint helpers, and device selection."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    """Seed all RNG sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
    artifact_refs: dict[str, str],
) -> None:
    """Save a training checkpoint to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
            "artifact_refs": artifact_refs,
        },
        path,
    )


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Load a checkpoint, restoring model weights and returning metadata."""
    path = Path(path)
    assert path.exists(), f"Checkpoint not found: {path}"
    checkpoint: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return {
        "epoch": checkpoint["epoch"],
        "best_metric": checkpoint["best_metric"],
        "config": checkpoint["config"],
        "artifact_refs": checkpoint["artifact_refs"],
        "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
    }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: Path | str) -> Path:
    """Create directory (and parents) if it doesn't exist; return the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
