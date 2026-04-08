"""Training-level visualizations for monitoring experiments (Steps 4-6 and 9).

All methods accept pre-extracted plain Python data (lists of epoch-record
dicts) so they have no dependency on model or trainer objects and can be
reused across MLP, CNN, and BiLSTM runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.visualizers.base import Visualizer


class TrainingVisualizer(Visualizer):
    """Plotting methods for training monitoring."""

    @staticmethod
    def plot_loss_curves(
        epoch_history: list[dict[str, Any]],
        output_path: Path,
        *,
        best_epoch: int | None = None,
    ) -> None:
        """Train + validation loss curves overlaid on one figure.

        Args:
            epoch_history: list of epoch records, each containing
                ``epoch``, ``train_loss``, and ``val_loss``.
            output_path: where to save the figure.
            best_epoch: if provided, draws a vertical dashed line at this
                epoch to mark the best checkpoint.
        """
        Visualizer._apply_style()
        epochs = [r["epoch"] for r in epoch_history]
        train_loss = [r["train_loss"] for r in epoch_history]
        val_loss = [r["val_loss"] for r in epoch_history]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_loss, label="Train Loss", color=Visualizer.COLOR_PALETTE[0], linewidth=1.5)
        ax.plot(epochs, val_loss, label="Val Loss", color=Visualizer.COLOR_PALETTE[2], linewidth=1.5)

        if best_epoch is not None:
            ax.axvline(
                best_epoch,
                color=Visualizer.COLOR_PALETTE[3],
                linestyle="--",
                alpha=0.7,
                label=f"Best Epoch ({best_epoch})",
            )

        stop_epoch = epoch_history[-1]["epoch"]
        if best_epoch is not None and stop_epoch != best_epoch:
            ax.axvline(
                stop_epoch,
                color=Visualizer.COLOR_PALETTE[4],
                linestyle=":",
                alpha=0.7,
                label=f"Early Stop ({stop_epoch})",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        Visualizer._save_figure(fig, output_path)

    @staticmethod
    def plot_val_metric_curve(
        epoch_history: list[dict[str, Any]],
        output_path: Path,
        *,
        metric_key: str = "val_macro_f1",
        best_epoch: int | None = None,
    ) -> None:
        """Validation metric curve with best-epoch marker.

        Args:
            epoch_history: list of epoch records containing ``epoch``
                and the column named *metric_key*.
            output_path: where to save the figure.
            metric_key: key in the epoch record to plot.
            best_epoch: if provided, marks this epoch with a star and
                annotation showing the metric value.
        """
        Visualizer._apply_style()
        epochs = [r["epoch"] for r in epoch_history]
        values = [r[metric_key] for r in epoch_history]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, values, color=Visualizer.COLOR_PALETTE[0], linewidth=1.5, label=metric_key)

        if best_epoch is not None:
            best_idx = next(i for i, r in enumerate(epoch_history) if r["epoch"] == best_epoch)
            best_val = values[best_idx]
            ax.plot(best_epoch, best_val, marker="*", markersize=14, color=Visualizer.COLOR_PALETTE[2], zorder=5)
            ax.annotate(
                f"Best: {best_val:.4f}",
                xy=(best_epoch, best_val),
                xytext=(best_epoch + 0.5, best_val - 0.01),
                fontsize=9,
                arrowprops={"arrowstyle": "->", "color": Visualizer.COLOR_PALETTE[2]},
                color=Visualizer.COLOR_PALETTE[2],
            )

        stop_epoch = epoch_history[-1]["epoch"]
        if best_epoch is not None and stop_epoch != best_epoch:
            ax.axvline(
                stop_epoch,
                color=Visualizer.COLOR_PALETTE[4],
                linestyle=":",
                alpha=0.7,
                label=f"Early Stop ({stop_epoch})",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_key.replace("_", " ").title())
        ax.set_title(f"Validation {metric_key.replace('_', ' ').title()}")
        ax.legend()
        Visualizer._save_figure(fig, output_path)
