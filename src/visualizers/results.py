"""Results-level visualizations for model evaluation (Steps 4-9).

All methods accept pre-extracted plain Python data (DataFrames, lists,
dicts) so they have no dependency on model objects and can be reused
across MLP, CNN, and BiLSTM evaluations.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from sklearn.metrics import f1_score

from src.visualizers.base import Visualizer


class ResultsVisualizer(Visualizer):
    """Plotting methods for model evaluation results."""

    @staticmethod
    def plot_tuning_summary(
        tuning_df: pd.DataFrame,
        output_path: Path,
        *,
        metric_col: str = "best_val_metric",
    ) -> None:
        """Horizontal bar chart of best validation metric by run, sorted ascending.

        Args:
            tuning_df: DataFrame with at least ``run_name`` and *metric_col*.
            output_path: where to save the figure.
            metric_col: column name containing the metric to plot.
        """
        Visualizer._apply_style()
        df = tuning_df.sort_values(metric_col, ascending=True).reset_index(drop=True)
        names = df["run_name"].tolist()
        values = df[metric_col].tolist()

        height = max(6, len(names) * 0.3)
        fig, ax = plt.subplots(figsize=(10, height))
        ax.barh(range(len(names)), values, color=Visualizer.COLOR_PALETTE[0])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Best Validation Macro F1")
        ax.set_title("Hyperparameter Tuning Summary")

        if values:
            min_val = min(values)
            ax.set_xlim(left=max(0, min_val - 0.01))

        Visualizer._save_figure(fig, output_path)

    @staticmethod
    def plot_confusion_matrix(
        confusion_df: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Heatmap of the full confusion matrix (log-scale colour).

        Args:
            confusion_df: square DataFrame indexed and columned by label names.
            output_path: where to save the figure.
        """
        Visualizer._apply_style()
        matrix = confusion_df.values.astype(float)
        matrix_log = matrix + 1  # shift for log scale

        fig, ax = plt.subplots(figsize=(18, 16))
        im = ax.imshow(matrix_log, cmap="Blues", norm=LogNorm(vmin=1, vmax=matrix_log.max()), aspect="auto")
        fig.colorbar(im, ax=ax, label="Count (log scale, +1 offset)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        ax.set_xticks([])
        ax.set_yticks([])
        Visualizer._save_figure(fig, output_path)

    @staticmethod
    def plot_top_confused_pairs(
        confusions: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Horizontal bar chart of the most confused label pairs.

        Args:
            confusions: list of dicts with ``true_label``, ``predicted_label``,
                and ``count``.
            output_path: where to save the figure.
        """
        Visualizer._apply_style()
        labels = [f"{c['true_label']} -> {c['predicted_label']}" for c in confusions]
        counts = [c["count"] for c in confusions]

        labels.reverse()
        counts.reverse()

        fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.4)))
        ax.barh(range(len(labels)), counts, color=Visualizer.COLOR_PALETTE[2])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Count")
        ax.set_title("Top Confused Intent Pairs")
        Visualizer._save_figure(fig, output_path)

    @staticmethod
    def plot_bottom_classes_f1(
        targets: list[int],
        predictions: list[int],
        label_names: list[str],
        output_path: Path,
        *,
        bottom_n: int = 15,
    ) -> None:
        """Horizontal bar chart of the worst-performing classes by F1.

        Args:
            targets: true label ids.
            predictions: predicted label ids.
            label_names: ordered list mapping id -> name.
            output_path: where to save the figure.
            bottom_n: how many worst classes to show.
        """
        Visualizer._apply_style()
        per_class_f1: np.ndarray = f1_score(
            targets,
            predictions,
            labels=list(range(len(label_names))),
            average=None,
            zero_division=0.0,
        )
        paired = sorted(zip(label_names, per_class_f1), key=lambda x: x[1])
        bottom = paired[:bottom_n]
        names, f1s = zip(*bottom)

        fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
        ax.barh(range(len(names)), f1s, color=Visualizer.COLOR_PALETTE[2])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("F1 Score")
        ax.set_title(f"Bottom {bottom_n} Classes by F1")
        ax.set_xlim(0, 1)
        Visualizer._save_figure(fig, output_path)

    @staticmethod
    def plot_oos_metrics(
        oos_metrics: dict[str, float],
        output_path: Path,
    ) -> None:
        """Bar chart of OOS precision, recall, and F1.

        Args:
            oos_metrics: dict with keys ``oos_precision``, ``oos_recall``,
                ``oos_f1``.
            output_path: where to save the figure.
        """
        Visualizer._apply_style()
        metric_names = ["Precision", "Recall", "F1"]
        values = [
            oos_metrics["oos_precision"],
            oos_metrics["oos_recall"],
            oos_metrics["oos_f1"],
        ]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(metric_names, values, color=Visualizer.COLOR_PALETTE[:3])
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_ylabel("Score")
        ax.set_title("OOS Detection Metrics")
        ax.set_ylim(0, 1)
        Visualizer._save_figure(fig, output_path)

    @staticmethod
    def plot_error_summary(
        top_errors: list[dict[str, Any]],
        output_path: Path,
        *,
        top_n: int = 15,
    ) -> None:
        """Bar chart of the most frequent misclassification pairs.

        Args:
            top_errors: list of error dicts with ``true_label`` and
                ``predicted_label`` keys.
            output_path: where to save the figure.
            top_n: how many top error pairs to show.
        """
        Visualizer._apply_style()
        pair_counts: Counter[tuple[str, str]] = Counter()
        for err in top_errors:
            pair_counts[(err["true_label"], err["predicted_label"])] += 1

        most_common = pair_counts.most_common(top_n)
        labels = [f"{true} -> {pred}" for (true, pred), _ in reversed(most_common)]
        counts = [c for _, c in reversed(most_common)]

        fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.35)))
        ax.barh(range(len(labels)), counts, color=Visualizer.COLOR_PALETTE[4])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Error Count")
        ax.set_title("Most Frequent Misclassification Pairs")
        Visualizer._save_figure(fig, output_path)

    @staticmethod
    def plot_model_comparison_bar(
        model_stats: list[tuple[str, float, float]],
        metric_label: str,
        output_path: Path,
    ) -> None:
        """Grouped bar chart comparing one aggregate metric across models.

        Args:
            model_stats: ``(display_name, mean, std)`` per model, in
                canonical order.
            metric_label: human-readable metric name for axis/title.
            output_path: where to save the figure.
        """
        Visualizer._apply_style()
        names = [s[0] for s in model_stats]
        means = [s[1] for s in model_stats]
        stds = [s[2] for s in model_stats]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(names))
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            color=Visualizer.COLOR_PALETTE[: len(names)],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel(metric_label)
        ax.set_title(f"Model Comparison: {metric_label}")
        ax.set_ylim(0, 1)
        Visualizer._save_figure(fig, output_path)

    @staticmethod
    def plot_oos_metrics_comparison(
        rows: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Grouped bar chart of OOS precision, recall, and F1 across models.

        Args:
            rows: list of dicts from ``oos_summary_table.json``, each with
                ``display_name`` and ``oos_{precision,recall,f1}_{mean,std}``.
            output_path: where to save the figure.
        """
        Visualizer._apply_style()
        metric_keys = ("oos_precision", "oos_recall", "oos_f1")
        metric_labels = ("Precision", "Recall", "F1")
        n_metrics = len(metric_keys)
        n_models = len(rows)
        bar_width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(n_models)

        for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
            means = [r[f"{key}_mean"] for r in rows]
            stds = [r[f"{key}_std"] for r in rows]
            offset = (i - (n_metrics - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset,
                means,
                bar_width,
                yerr=stds,
                capsize=4,
                label=label,
                color=Visualizer.COLOR_PALETTE[i],
                edgecolor="black",
                linewidth=0.5,
            )
            for bar, val in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        names = [r["display_name"] for r in rows]
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel("Score")
        ax.set_title("OOS Detection Metrics Comparison")
        ax.set_ylim(0, 1)
        ax.legend()
        Visualizer._save_figure(fig, output_path)

    @staticmethod
    def plot_efficiency_comparison(
        rows: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Multi-panel efficiency comparison across models.

        Three subplots: training time, inference throughput, and parameter
        count.

        Args:
            rows: list of dicts from ``efficiency_summary_table.json``.
            output_path: where to save the figure.
        """
        Visualizer._apply_style()
        names = [r["display_name"] for r in rows]
        x = np.arange(len(names))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Training time
        means = [r["training_time_seconds_mean"] for r in rows]
        stds = [r["training_time_seconds_std"] for r in rows]
        axes[0].bar(x, means, yerr=stds, capsize=5, color=Visualizer.COLOR_PALETTE[0], edgecolor="black", linewidth=0.5)
        for xi, val in zip(x, means):
            axes[0].text(xi, val, f"{val:.1f}s", ha="center", va="bottom", fontsize=9)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, fontsize=9)
        axes[0].set_ylabel("Seconds")
        axes[0].set_title("Training Time")

        # Inference throughput
        means = [r["inference_examples_per_sec_mean"] for r in rows]
        stds = [r["inference_examples_per_sec_std"] for r in rows]
        axes[1].bar(x, means, yerr=stds, capsize=5, color=Visualizer.COLOR_PALETTE[1], edgecolor="black", linewidth=0.5)
        for xi, val in zip(x, means):
            axes[1].text(xi, val, f"{val:.0f}", ha="center", va="bottom", fontsize=9)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, fontsize=9)
        axes[1].set_ylabel("Examples / sec")
        axes[1].set_title("Inference Throughput")

        # Parameter count
        params = [r["trainable_parameter_count"] for r in rows]
        axes[2].bar(x, params, color=Visualizer.COLOR_PALETTE[2], edgecolor="black", linewidth=0.5)
        for xi, val in zip(x, params):
            axes[2].text(xi, val, f"{val / 1e6:.2f}M", ha="center", va="bottom", fontsize=9)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names, fontsize=9)
        axes[2].set_ylabel("Parameters")
        axes[2].set_title("Trainable Parameters")

        fig.suptitle("Model Efficiency Comparison", fontsize=13, y=1.02)
        Visualizer._save_figure(fig, output_path)

    @staticmethod
    def plot_bottom_classes_f1_from_metrics(
        class_metrics: list[dict[str, Any]],
        output_path: Path,
        *,
        bottom_n: int = 15,
    ) -> None:
        """Horizontal bar chart of worst classes by F1, from pre-computed metrics.

        Unlike :meth:`plot_bottom_classes_f1` which recomputes F1 from raw
        predictions, this method accepts the ``per_class_metrics.json``
        structure directly.

        Args:
            class_metrics: list of dicts each with ``label_name`` and ``f1``.
            output_path: where to save the figure.
            bottom_n: how many worst classes to show.
        """
        Visualizer._apply_style()
        sorted_classes = sorted(class_metrics, key=lambda c: c["f1"])
        bottom = sorted_classes[:bottom_n]
        names = [c["label_name"] for c in bottom]
        f1s = [c["f1"] for c in bottom]

        fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
        ax.barh(range(len(names)), f1s, color=Visualizer.COLOR_PALETTE[2])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("F1 Score")
        ax.set_title(f"Bottom {bottom_n} Classes by F1")
        ax.set_xlim(0, 1)
        Visualizer._save_figure(fig, output_path)
