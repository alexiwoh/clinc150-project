"""Section G: Confidence-stratified error analysis and high-confidence error identification."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.analysis.constants import (
    CONFIDENCE_ACCURACY_COMPARISON_FILENAME,
    CONFIDENCE_STRATIFICATION_FILENAME,
    CONFIDENCE_VS_ACCURACY_FILENAME,
    HIGH_CONFIDENCE_ERRORS_TOP_K,
    HIGH_CONFIDENCE_THRESHOLD,
)
from src.analysis.figures import COLOR_PALETTE, apply_style, plot_grouped_bar, save_figure
from src.analysis.metrics import stratify_by_confidence
from src.analysis.utils import (
    HandoffContext,
    analysis_output_dir,
    artifact_envelope,
    load_confidences,
    load_predictions,
    repo_relative,
    shared_analysis_dir,
    write_json,
)
import matplotlib.pyplot as plt

from src.enums import ModelID
from src.report_figure_generation import FigureRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-model confidence stratification
# ---------------------------------------------------------------------------


def compute_and_save_confidence_stratification(ctx: HandoffContext) -> tuple[dict[str, Any], list[FigureRecord]]:
    """Stratify predictions by confidence and identify high-confidence errors.

    Returns ``(stratification_artifact, figure_records)``.
    """
    conf = load_confidences(ctx.confidences_path)
    preds_df = load_predictions(ctx.final_predictions_path)
    records: list[FigureRecord] = []
    out_dir = analysis_output_dir(ctx.model_id)

    max_probs = np.max(conf.probabilities, axis=1)
    correct_mask = conf.predictions == conf.targets

    strata = stratify_by_confidence(max_probs, correct_mask)

    # High-confidence errors
    high_conf_error_mask = (max_probs >= HIGH_CONFIDENCE_THRESHOLD) & ~correct_mask
    high_conf_indices = np.where(high_conf_error_mask)[0]
    sorted_indices = high_conf_indices[np.argsort(-max_probs[high_conf_indices])][:HIGH_CONFIDENCE_ERRORS_TOP_K]

    eps = 1e-12
    log_probs = np.log(np.clip(conf.probabilities, eps, None))
    entropies = -np.sum(conf.probabilities * log_probs, axis=1)

    sorted_probs = np.sort(conf.probabilities, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]

    high_confidence_errors: list[dict[str, Any]] = []
    for idx in sorted_indices:
        row = preds_df.iloc[idx]
        high_confidence_errors.append(
            {
                "text": row["text"],
                "true_label_name": row["true_label_name"],
                "predicted_label_name": row["predicted_label_name"],
                "max_confidence": float(max_probs[idx]),
                "prediction_entropy": float(entropies[idx]),
                "confidence_margin": float(margins[idx]),
            }
        )

    artifact = artifact_envelope(
        model_id=ctx.model_id,
        representative_run_id=ctx.representative_run_id,
        strata=strata,
        high_confidence_errors=high_confidence_errors,
        high_confidence_threshold=HIGH_CONFIDENCE_THRESHOLD,
        source_artifact=repo_relative(ctx.confidences_path),
    )

    write_json(out_dir / CONFIDENCE_STRATIFICATION_FILENAME, artifact)
    logger.info("Saved: %s", out_dir / CONFIDENCE_STRATIFICATION_FILENAME)

    # Per-model confidence vs accuracy figure
    fig_path = out_dir / CONFIDENCE_VS_ACCURACY_FILENAME
    _plot_confidence_vs_accuracy(strata, ctx.model_id.display_name, fig_path)
    records.append(
        FigureRecord(
            figure_path=repo_relative(fig_path),
            figure_type="confidence_vs_accuracy",
            scope="analysis",
            source_artifact_paths=[repo_relative(ctx.confidences_path)],
            model_name=str(ctx.model_id),
            representative_run_id=ctx.representative_run_id,
            caption_context={
                "analysis_basis": "representative-run confidence stratification",
                "dataset_name": "CLINC150",
                "dataset_split": "test",
                "metric_names": ["accuracy by confidence bin"],
            },
        )
    )

    return artifact, records


# ---------------------------------------------------------------------------
# Shared confidence comparison
# ---------------------------------------------------------------------------


def generate_confidence_comparison(
    model_ids: list[ModelID],
    per_model_stratification: dict[str, dict[str, Any]],
) -> list[FigureRecord]:
    """Generate shared confidence-accuracy comparison figure."""
    records: list[FigureRecord] = []
    shared_dir = shared_analysis_dir()

    group_labels = [s["bin_label"] for s in per_model_stratification[str(model_ids[0])]["strata"]]
    series: dict[str, list[float]] = {}
    for mid in model_ids:
        strat = per_model_stratification[str(mid)]
        series[mid.display_name] = [s["accuracy"] for s in strat["strata"]]

    fig_path = shared_dir / CONFIDENCE_ACCURACY_COMPARISON_FILENAME
    plot_grouped_bar(
        group_labels,
        series,
        ylabel="Accuracy",
        title="Confidence-Stratified Accuracy Comparison",
        out_path=fig_path,
        ylim=(0.0, 1.05),
    )
    source_paths = [repo_relative(analysis_output_dir(m) / CONFIDENCE_STRATIFICATION_FILENAME) for m in model_ids]
    records.append(
        FigureRecord(
            figure_path=repo_relative(fig_path),
            figure_type="confidence_vs_accuracy",
            scope="analysis",
            source_artifact_paths=source_paths,
            caption_context={
                "analysis_basis": "cross-model confidence stratification comparison using representative runs",
                "dataset_name": "CLINC150",
                "dataset_split": "test",
                "metric_names": ["accuracy by confidence bin"],
            },
        )
    )

    return records


# ---------------------------------------------------------------------------
# Figure helper
# ---------------------------------------------------------------------------


def _plot_confidence_vs_accuracy(strata: list[dict[str, Any]], model_name: str, out_path: Any) -> None:
    apply_style()
    labels = [s["bin_label"] for s in strata]
    accuracies = [s["accuracy"] for s in strata]
    counts = [s["count"] for s in strata]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(labels)), accuracies, color=COLOR_PALETTE[0], alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Confidence vs Accuracy - {model_name}")
    ax.set_ylim(0, 1.05)

    for bar, count in zip(bars, counts):
        h = bar.get_height()
        ax.annotate(f"n={count}", xy=(bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", fontsize=8)

    save_figure(fig, out_path)
