"""Shared matplotlib figure utilities for error analysis plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

matplotlib.use("Agg")

FIGURE_DPI: int = 150
_STYLE: str = "seaborn-v0_8-whitegrid"
COLOR_PALETTE: list[str] = [
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#CCB974",
    "#64B5CD",
]


def apply_style() -> None:
    """Activate the shared matplotlib style."""
    plt.style.use(_STYLE)


def save_figure(fig: Figure, path: Path) -> None:
    """Apply tight layout, save figure, and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reusable grouped bar chart
# ---------------------------------------------------------------------------


def plot_grouped_bar(
    group_labels: list[str],
    series: dict[str, list[float]],
    ylabel: str,
    title: str,
    out_path: Path,
    *,
    ylim: tuple[float, float] | None = None,
    annotate_values: bool = False,
) -> None:
    """Draw a grouped bar chart with one bar per series per group."""
    apply_style()
    n_groups = len(group_labels)
    n_series = len(series)
    x = np.arange(n_groups)
    width = 0.8 / max(n_series, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.5), 5))
    for i, (label, values) in enumerate(series.items()):
        offset = (i - n_series / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=COLOR_PALETTE[i % len(COLOR_PALETTE)])
        if annotate_values:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Overlaid line/curve comparison
# ---------------------------------------------------------------------------


def plot_curve_comparison(
    curves: list[dict[str, Any]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    *,
    diagonal: bool = False,
) -> None:
    """Overlay multiple curves on a single axes.

    Each element of *curves* must have ``x``, ``y``, ``label``,
    and optionally ``auc_value`` (displayed in the legend).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, curve in enumerate(curves):
        label = curve["label"]
        if "auc_value" in curve:
            label += f" (AUC={curve['auc_value']:.4f})"
        ax.plot(curve["x"], curve["y"], label=label, color=COLOR_PALETTE[i % len(COLOR_PALETTE)], linewidth=1.5)

    if diagonal:
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    save_figure(fig, out_path)
