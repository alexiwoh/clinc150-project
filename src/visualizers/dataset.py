"""Dataset-level visualizations for CLINC150 exploration (Step 2).

All methods accept pre-extracted plain Python data (dicts, lists) so they
have no dependency on ``CLINCDataset`` and can be reused or tested with
synthetic data.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import numpy as np

from src.constants import OOS_LABEL_NAME
from src.visualizers.base import Visualizer

_SPLIT_ORDER = ("train", "validation", "test")


class DatasetVisualizer(Visualizer):
    """Plotting and export methods for dataset exploration.

    Each static method produces one output file (PNG or CSV).  The methods
    are grouped here so they inherit the shared style / save helpers from
    ``Visualizer`` and keep dataset-specific constants (e.g. the OOS
    highlight color) co-located.
    """

    OOS_HIGHLIGHT_COLOR: str = "#C44E52"
    IN_SCOPE_COLOR: str = "#4C72B0"

    # ------------------------------------------------------------------
    # 1. Split size bar chart
    # ------------------------------------------------------------------

    @staticmethod
    def plot_split_sizes(split_sizes: dict[str, int], output_path: Path) -> None:
        """Bar chart showing the number of examples in each dataset split.

        **Dataset field:** split-level row counts (train / validation / test).

        **Usefulness:** confirms the dataset was loaded correctly, verifies
        split sizes match expectations, and documents dataset scale for the
        report.
        """
        Visualizer._apply_style()
        splits = [s for s in _SPLIT_ORDER if s in split_sizes]
        counts = [split_sizes[s] for s in splits]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(splits, counts, color=Visualizer.COLOR_PALETTE[: len(splits)])
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_xlabel("Split")
        ax.set_ylabel("Number of Examples")
        ax.set_title("Dataset Split Sizes")
        Visualizer._save_figure(fig, output_path)

    # ------------------------------------------------------------------
    # 2. Intent class distribution bar chart
    # ------------------------------------------------------------------

    @staticmethod
    def plot_class_distribution(
        label_names: list[str],
        counts: list[int],
        output_path: Path,
        *,
        top_n: int | None = None,
    ) -> None:
        """Horizontal bar chart of per-class example counts, sorted descending.

        **Dataset fields:** ``intent`` label counts from the training split.

        **Usefulness:** verifies whether the dataset is balanced, surfaces
        unexpected skew or missing labels, and provides context for later
        per-class performance analysis.

        Args:
            label_names: label name for every class, aligned with *counts*.
            counts: example count for every class, aligned with *label_names*.
            output_path: where to save the figure.
            top_n: if set, only the *top_n* most frequent classes are plotted
                   (a readable companion to the full chart).
        """
        Visualizer._apply_style()
        paired = sorted(zip(label_names, counts), key=lambda x: x[1])
        if top_n is not None:
            paired = paired[-top_n:]
        names, vals = zip(*paired)

        colors = [
            DatasetVisualizer.OOS_HIGHLIGHT_COLOR if n == OOS_LABEL_NAME else DatasetVisualizer.IN_SCOPE_COLOR
            for n in names
        ]

        height = max(4, len(names) * 0.22)
        fig, ax = plt.subplots(figsize=(8, height))
        ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Count")

        title = "Intent Class Distribution (Training Set)"
        if top_n is not None:
            title += f" — Top {top_n}"
        ax.set_title(title)

        Visualizer._save_figure(fig, output_path)

    # ------------------------------------------------------------------
    # 3. OOS vs in-scope distribution bar chart
    # ------------------------------------------------------------------

    @staticmethod
    def plot_oos_vs_inscope(
        oos_counts: dict[str, int],
        in_scope_counts: dict[str, int],
        output_path: Path,
    ) -> None:
        """Grouped bar chart of in-scope vs OOS counts, broken down by split.

        **Dataset fields:** ``intent`` label id compared against the OOS label
        id for each split.

        **Usefulness:** directly supports the OOS-detection project goal,
        confirms how OOS examples are represented across splits, and highlights
        the train-to-test distribution shift.
        """
        Visualizer._apply_style()
        splits = [s for s in _SPLIT_ORDER if s in oos_counts]
        oos_vals = [oos_counts[s] for s in splits]
        ins_vals = [in_scope_counts[s] for s in splits]

        x = np.arange(len(splits))
        width = 0.35

        fig, ax = plt.subplots(figsize=(7, 4))
        bars_in = ax.bar(x - width / 2, ins_vals, width, label="In-scope", color=DatasetVisualizer.IN_SCOPE_COLOR)
        bars_oos = ax.bar(x + width / 2, oos_vals, width, label="OOS", color=DatasetVisualizer.OOS_HIGHLIGHT_COLOR)

        for bar_group in (bars_in, bars_oos):
            for bar in bar_group:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{int(bar.get_height()):,}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_xlabel("Split")
        ax.set_ylabel("Number of Examples")
        ax.set_title("OOS vs In-Scope Distribution by Split")
        ax.legend()
        Visualizer._save_figure(fig, output_path)

    # ------------------------------------------------------------------
    # 4. Query length histogram
    # ------------------------------------------------------------------

    @staticmethod
    def plot_query_length_histogram(lengths: list[int], output_path: Path) -> None:
        """Histogram of utterance lengths (whitespace word count).

        **Dataset field:** ``text`` — each value is split on whitespace to
        obtain a word count.

        **Usefulness:** reveals how short or long CLINC150 utterances are,
        informs max-sequence-length choices for neural models, and helps
        anticipate truncation risk.
        """
        Visualizer._apply_style()
        mean_len = mean(lengths)
        median_len = median(lengths)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(lengths, bins=30, color=Visualizer.COLOR_PALETTE[0], edgecolor="white")
        ax.axvline(mean_len, color=Visualizer.COLOR_PALETTE[2], linestyle="--", label=f"Mean ({mean_len:.1f})")
        ax.axvline(median_len, color=Visualizer.COLOR_PALETTE[3], linestyle=":", label=f"Median ({median_len:.1f})")
        ax.set_xlabel("Query Length (words)")
        ax.set_ylabel("Frequency")
        ax.set_title("Query Length Distribution (All Splits)")
        ax.legend()
        Visualizer._save_figure(fig, output_path)

    # ------------------------------------------------------------------
    # 5. Query length boxplot by split
    # ------------------------------------------------------------------

    @staticmethod
    def plot_query_length_boxplot(
        lengths_by_split: dict[str, list[int]],
        output_path: Path,
    ) -> None:
        """Side-by-side boxplots of query length distributions per split.

        **Dataset field:** ``text`` — whitespace word count, computed
        separately for each split.

        **Usefulness:** checks whether splits have similar length distributions,
        detects accidental inconsistencies, and documents that the splits are
        linguistically comparable.
        """
        Visualizer._apply_style()
        splits = [s for s in _SPLIT_ORDER if s in lengths_by_split]
        data = [lengths_by_split[s] for s in splits]

        fig, ax = plt.subplots(figsize=(6, 4))
        bp = ax.boxplot(data, tick_labels=splits, patch_artist=True)
        for patch, color in zip(bp["boxes"], Visualizer.COLOR_PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xlabel("Split")
        ax.set_ylabel("Query Length (words)")
        ax.set_title("Query Length by Split")
        Visualizer._save_figure(fig, output_path)

    # ------------------------------------------------------------------
    # 6. Representative sample queries table
    # ------------------------------------------------------------------

    @staticmethod
    def save_representative_queries(
        rows: list[dict[str, str]],
        output_path: Path,
    ) -> None:
        """Save a CSV table of representative example queries with labels.

        **Dataset fields:** ``text`` and ``intent`` for a small curated
        subset of labels, including both in-scope and OOS examples.

        **Usefulness:** makes the dataset concrete and human-interpretable,
        helps verify that labels and OOS examples were parsed correctly, and
        is directly reusable in the project report and later error analysis.

        Each dict in *rows* must have keys ``split``, ``label``, ``scope``,
        and ``query``.
        """
        fieldnames = ["split", "label", "scope", "query"]
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
