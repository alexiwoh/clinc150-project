"""Section H: Slice-based analysis by utterance length and supervised class scope."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.analysis.constants import (
    FREQUENCY_SLICE_COMPARISON_FILENAME,
    FREQUENCY_SLICE_FILENAME,
    LENGTH_SLICE_COMPARISON_FILENAME,
    LENGTH_SLICE_FILENAME,
)
from src.analysis.enums import LengthBucket, ScopeSlice
from src.analysis.figures import plot_grouped_bar
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
from src.enums import ModelID
from src.report_figure_generation import FigureRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Length bucketing
# ---------------------------------------------------------------------------

_LENGTH_THRESHOLDS: dict[LengthBucket, tuple[int, float]] = {
    LengthBucket.SHORT: (1, 4),
    LengthBucket.MEDIUM: (5, 8),
    LengthBucket.LONG: (9, float("inf")),
}


def bucket_by_length(texts: pd.Series) -> pd.Series:
    """Assign a :class:`LengthBucket` to each text based on whitespace token count."""
    token_counts: pd.Series = texts.str.split().str.len()
    buckets = pd.Series(LengthBucket.LONG, index=texts.index, dtype=object)
    for bucket, (lo, hi) in _LENGTH_THRESHOLDS.items():
        buckets[(token_counts >= lo) & (token_counts <= hi)] = bucket
    return buckets


def _compute_slice_metrics(
    preds_df: pd.DataFrame,
    max_probs: npt.NDArray[np.floating[Any]],
    correct_mask: npt.NDArray[np.bool_],
    slice_col: str,
) -> list[dict[str, Any]]:
    """Compute accuracy, error count/rate, and mean confidence per slice."""
    results: list[dict[str, Any]] = []
    for label in preds_df[slice_col].unique():
        mask = (preds_df[slice_col] == label).values
        count = int(mask.sum())
        if count == 0:
            continue
        n_correct = int(correct_mask[mask].sum())
        n_errors = count - n_correct
        results.append(
            {
                "slice": str(label),
                "count": count,
                "accuracy": float(n_correct / count),
                "error_count": n_errors,
                "error_rate": float(n_errors / count),
                "mean_confidence_correct": float(max_probs[mask & correct_mask].mean())
                if (mask & correct_mask).any()
                else 0.0,
                "mean_confidence_incorrect": float(max_probs[mask & ~correct_mask].mean())
                if (mask & ~correct_mask).any()
                else 0.0,
            }
        )
    bucket_order = {str(b): i for i, b in enumerate(LengthBucket)}
    scope_order = {scope: i for i, scope in enumerate(ScopeSlice)}
    order = {**bucket_order, **scope_order}
    results.sort(key=lambda r: order.get(r["slice"], 999))
    return results


# ---------------------------------------------------------------------------
# Per-model length analysis
# ---------------------------------------------------------------------------


def compute_and_save_length_analysis(ctx: HandoffContext) -> dict[str, Any]:
    """Compute length-slice metrics for one model and save."""
    conf = load_confidences(ctx.confidences_path)
    preds_df = load_predictions(ctx.final_predictions_path)
    out_dir = analysis_output_dir(ctx.model_id)

    preds_df = preds_df.copy()
    preds_df["length_bucket"] = bucket_by_length(preds_df["text"])
    max_probs = np.max(conf.probabilities, axis=1)
    correct_mask = conf.predictions == conf.targets

    slices = _compute_slice_metrics(preds_df, max_probs, correct_mask, "length_bucket")

    artifact = artifact_envelope(
        model_id=ctx.model_id,
        representative_run_id=ctx.representative_run_id,
        slices=slices,
        source_artifact=repo_relative(ctx.final_predictions_path),
    )
    write_json(out_dir / LENGTH_SLICE_FILENAME, artifact)
    logger.info("Saved: %s", out_dir / LENGTH_SLICE_FILENAME)
    return artifact


# ---------------------------------------------------------------------------
# Per-model in-scope/OOS analysis
# ---------------------------------------------------------------------------


def compute_and_save_frequency_analysis(ctx: HandoffContext) -> dict[str, Any]:
    """Compute in-scope/OOS metrics; retain the historical filename for compatibility."""
    conf = load_confidences(ctx.confidences_path)
    preds_df = load_predictions(ctx.final_predictions_path)
    out_dir = analysis_output_dir(ctx.model_id)

    preds_df = preds_df.copy()
    preds_df["class_scope"] = np.where(
        preds_df["true_label_id"] == conf.oos_class_idx, ScopeSlice.OOS, ScopeSlice.IN_SCOPE
    )
    max_probs = np.max(conf.probabilities, axis=1)
    correct_mask = conf.predictions == conf.targets

    slices = _compute_slice_metrics(preds_df, max_probs, correct_mask, "class_scope")

    artifact = artifact_envelope(
        model_id=ctx.model_id,
        representative_run_id=ctx.representative_run_id,
        analysis_basis="in_scope_vs_oos",
        note="Supervised in-scope versus OOS class accuracy, not a training-frequency analysis. "
        "The frequency_slice filename is retained for compatibility with historical artifacts.",
        slices=slices,
        source_artifact=repo_relative(ctx.final_predictions_path),
    )
    write_json(out_dir / FREQUENCY_SLICE_FILENAME, artifact)
    logger.info("Saved: %s", out_dir / FREQUENCY_SLICE_FILENAME)
    return artifact


# ---------------------------------------------------------------------------
# Shared comparison figures
# ---------------------------------------------------------------------------


def generate_slice_comparisons(
    model_ids: list[ModelID],
    per_model_length: dict[str, dict[str, Any]],
    per_model_frequency: dict[str, dict[str, Any]],
) -> list[FigureRecord]:
    """Generate shared length-slice and in-scope/OOS comparison figures."""
    records: list[FigureRecord] = []
    shared_dir = shared_analysis_dir()

    # Length slice comparison
    _generate_grouped_comparison(
        model_ids,
        per_model_length,
        shared_dir / LENGTH_SLICE_COMPARISON_FILENAME,
        "Accuracy by Utterance Length",
        LENGTH_SLICE_FILENAME,
        records,
        figure_type="length_slice",
    )

    # In-scope/OOS comparison (historical frequency_slice identifier)
    _generate_grouped_comparison(
        model_ids,
        per_model_frequency,
        shared_dir / FREQUENCY_SLICE_COMPARISON_FILENAME,
        "Accuracy by In-Scope / OOS Class",
        FREQUENCY_SLICE_FILENAME,
        records,
        figure_type="frequency_slice",
    )

    return records


def _generate_grouped_comparison(
    model_ids: list[ModelID],
    per_model_data: dict[str, dict[str, Any]],
    fig_path: Any,
    title: str,
    source_filename: str,
    records: list[FigureRecord],
    *,
    figure_type: str,
) -> None:
    first = per_model_data[str(model_ids[0])]
    group_labels = [s["slice"] for s in first["slices"]]

    series: dict[str, list[float]] = {}
    for mid in model_ids:
        data = per_model_data[str(mid)]
        series[mid.display_name] = [s["accuracy"] for s in data["slices"]]

    plot_grouped_bar(group_labels, series, ylabel="Accuracy", title=title, out_path=fig_path, ylim=(0.0, 1.05))
    source_paths = [repo_relative(analysis_output_dir(m) / source_filename) for m in model_ids]
    records.append(
        FigureRecord(
            figure_path=repo_relative(fig_path),
            figure_type=figure_type,
            scope="analysis",
            source_artifact_paths=source_paths,
            caption_context={
                "analysis_basis": "cross-model slice comparison using representative runs",
                "dataset_name": "CLINC150",
                "dataset_split": "test",
                "metric_names": ["accuracy by slice"],
                "top_k": len(group_labels),
            },
        )
    )
