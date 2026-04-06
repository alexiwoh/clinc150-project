"""Base visualizer with shared style constants and figure-saving utilities.

All project visualizer subclasses inherit from ``Visualizer`` to get a
consistent look across dataset exploration, training monitoring, and results
reporting figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

matplotlib.use("Agg")

_STYLE = "seaborn-v0_8-whitegrid"


class Visualizer:
    """Shared infrastructure for all project visualizers.

    Provides a consistent matplotlib style, color palette, DPI setting, and
    a helper that applies tight layout, saves, and closes a figure.
    Subclasses add domain-specific plotting methods.
    """

    FIGURE_DPI: int = 150
    COLOR_PALETTE: list[str] = [
        "#4C72B0",
        "#55A868",
        "#C44E52",
        "#8172B2",
        "#CCB974",
        "#64B5CD",
    ]

    @classmethod
    def _apply_style(cls) -> None:
        """Activate the shared matplotlib style for all subsequent plots."""
        plt.style.use(_STYLE)

    @staticmethod
    def _save_figure(fig: Figure, path: Path) -> None:
        """Apply tight layout, save *fig* to *path*, and close it."""
        fig.tight_layout()
        fig.savefig(path, dpi=Visualizer.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
