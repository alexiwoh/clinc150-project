"""Section C: Calibration metrics and figures (ECE, MCE, Brier, NLL, confidence stats)."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.constants import (
    CALIBRATION_COMPARISON_FILENAME,
    CALIBRATION_METRICS_FILENAME,
    CALIBRATION_SUMMARY_FILENAME,
    CONFIDENCE_HISTOGRAM_FILENAME,
    ECE_DEFAULT_BINS,
    RELIABILITY_DIAGRAM_FILENAME,
)
from src.analysis.figures import COLOR_PALETTE, apply_style, save_figure
from src.analysis.metrics import (
    compute_brier_score,
    compute_confidence_statistics,
    compute_ece,
    compute_mce,
    compute_nll,
)
from src.analysis.utils import (
    HandoffContext,
    analysis_output_dir,
    artifact_envelope,
    load_confidences,
    repo_relative,
    shared_analysis_dir,
    write_json,
)
from src.enums import ModelID
from src.report_figure_generation import FigureRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-model calibration
# ---------------------------------------------------------------------------


def compute_and_save_calibration(ctx: HandoffContext) -> tuple[dict[str, Any], list[FigureRecord]]:
    """Compute calibration metrics + figures for one model.

    Returns ``(calibration_artifact, figure_records)``.
    """
    conf = load_confidences(ctx.confidences_path)
    records: list[FigureRecord] = []
    out_dir = analysis_output_dir(ctx.model_id)

    ece, bin_data = compute_ece(conf.probabilities, conf.targets)
    mce = compute_mce(conf.probabilities, conf.targets)
    brier = compute_brier_score(conf.probabilities, conf.targets)
    nll = compute_nll(conf.probabilities, conf.targets)
    conf_stats = compute_confidence_statistics(conf.probabilities, conf.predictions, conf.targets)

    artifact = artifact_envelope(
        model_id=ctx.model_id,
        representative_run_id=ctx.representative_run_id,
        ece=ece,
        mce=mce,
        brier_score=brier,
        nll=nll,
        n_bins=ECE_DEFAULT_BINS,
        bin_data=bin_data,
        confidence_statistics=conf_stats,
        source_artifact=repo_relative(ctx.confidences_path),
    )
    write_json(out_dir / CALIBRATION_METRICS_FILENAME, artifact)
    logger.info("Saved: %s", out_dir / CALIBRATION_METRICS_FILENAME)

    # Reliability diagram
    rel_path = out_dir / RELIABILITY_DIAGRAM_FILENAME
    _plot_reliability_diagram(bin_data, ctx.model_id.display_name, rel_path)
    records.append(
        FigureRecord(
            figure_path=repo_relative(rel_path),
            figure_type="reliability_diagram",
            scope="analysis",
            source_artifact_paths=[repo_relative(ctx.confidences_path)],
            model_name=str(ctx.model_id),
            representative_run_id=ctx.representative_run_id,
            caption_context={
                "analysis_basis": "representative-run calibration analysis",
                "dataset_name": "CLINC150",
                "dataset_split": "test",
                "metric_names": ["calibration accuracy by confidence bin"],
            },
        )
    )

    # Confidence histogram
    hist_path = out_dir / CONFIDENCE_HISTOGRAM_FILENAME
    max_probs = np.max(conf.probabilities, axis=1)
    correct_mask = conf.predictions == conf.targets
    _plot_confidence_histogram(max_probs, correct_mask, ctx.model_id.display_name, hist_path)
    records.append(
        FigureRecord(
            figure_path=repo_relative(hist_path),
            figure_type="confidence_histogram",
            scope="analysis",
            source_artifact_paths=[repo_relative(ctx.confidences_path)],
            model_name=str(ctx.model_id),
            representative_run_id=ctx.representative_run_id,
            caption_context={
                "analysis_basis": "representative-run calibration analysis",
                "dataset_name": "CLINC150",
                "dataset_split": "test",
                "metric_names": ["prediction confidence distribution"],
            },
        )
    )

    return artifact, records


# ---------------------------------------------------------------------------
# Shared calibration comparison
# ---------------------------------------------------------------------------


def generate_calibration_comparison(
    model_ids: list[ModelID],
    per_model_calibration: dict[str, dict[str, Any]],
) -> list[FigureRecord]:
    """Generate shared calibration summary table and comparison figure."""
    records: list[FigureRecord] = []
    shared_dir = shared_analysis_dir()
    source_paths: list[str] = []

    rows: list[dict[str, Any]] = []
    all_bin_data: list[tuple[str, list[dict[str, Any]]]] = []

    for mid in model_ids:
        cal = per_model_calibration[str(mid)]
        rows.append(
            {
                "model_id": str(mid),
                "model_name": mid.display_name,
                "ece": cal["ece"],
                "mce": cal["mce"],
                "brier_score": cal["brier_score"],
                "nll": cal["nll"],
            }
        )
        all_bin_data.append((mid.display_name, cal["bin_data"]))
        src = analysis_output_dir(mid) / CALIBRATION_METRICS_FILENAME
        source_paths.append(repo_relative(src))

    summary = artifact_envelope(rows=rows)
    write_json(shared_dir / CALIBRATION_SUMMARY_FILENAME, summary)
    logger.info("Saved: %s", shared_dir / CALIBRATION_SUMMARY_FILENAME)

    # Overlaid reliability curves
    comp_path = shared_dir / CALIBRATION_COMPARISON_FILENAME
    _plot_calibration_comparison(all_bin_data, comp_path)
    records.append(
        FigureRecord(
            figure_path=repo_relative(comp_path),
            figure_type="calibration_comparison",
            scope="analysis",
            source_artifact_paths=source_paths,
            caption_context={
                "analysis_basis": "cross-model calibration comparison using one representative run per model",
                "dataset_name": "CLINC150",
                "dataset_split": "test",
                "metric_names": ["ECE", "MCE", "Brier score", "NLL"],
            },
        )
    )

    return records


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _plot_reliability_diagram(
    bin_data: list[dict[str, Any]],
    model_name: str,
    out_path: Any,
) -> None:
    apply_style()
    bins_with_data = [b for b in bin_data if b["count"] > 0]
    midpoints = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bins_with_data]
    accuracies = [b["accuracy"] for b in bins_with_data]
    confidences = [b["confidence"] for b in bins_with_data]
    counts = [b["count"] for b in bins_with_data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect calibration")
    ax1.bar(midpoints, accuracies, width=1.0 / len(bin_data), alpha=0.6, color=COLOR_PALETTE[0], label="Accuracy")
    ax1.scatter(confidences, accuracies, color=COLOR_PALETTE[2], s=20, zorder=3)
    ax1.set_xlabel("Mean Predicted Confidence")
    ax1.set_ylabel("Fraction of Positives (Accuracy)")
    ax1.set_title(f"Reliability Diagram - {model_name}")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.legend(fontsize=8)

    ax2.bar(midpoints, counts, width=1.0 / len(bin_data), color=COLOR_PALETTE[1], alpha=0.7)
    ax2.set_xlabel("Mean Predicted Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Bin Counts")

    save_figure(fig, out_path)


def _plot_confidence_histogram(
    max_confs: Any,
    correct_mask: Any,
    model_name: str,
    out_path: Any,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    bins = np.linspace(0, 1, 31)
    ax.hist(max_confs[correct_mask], bins=bins, alpha=0.6, label="Correct", color=COLOR_PALETTE[0])
    ax.hist(max_confs[~correct_mask], bins=bins, alpha=0.6, label="Incorrect", color=COLOR_PALETTE[2])
    ax.set_xlabel("Max Predicted Confidence")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence Distribution - {model_name}")
    ax.legend()

    save_figure(fig, out_path)


def _plot_calibration_comparison(
    all_bin_data: list[tuple[str, list[dict[str, Any]]]],
    out_path: Any,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect calibration")

    for i, (name, bin_data) in enumerate(all_bin_data):
        bins_with_data = [b for b in bin_data if b["count"] > 0]
        confidences = [b["confidence"] for b in bins_with_data]
        accuracies = [b["accuracy"] for b in bins_with_data]
        ax.plot(confidences, accuracies, "o-", label=name, color=COLOR_PALETTE[i % len(COLOR_PALETTE)], markersize=4)

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Positives (Accuracy)")
    ax.set_title("Calibration Comparison")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=8)

    save_figure(fig, out_path)
