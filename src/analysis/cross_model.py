"""Section I: Cross-model comparative error analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.constants import (
    CROSS_MODEL_ERROR_COMPARISON_FILENAME,
    CROSS_MODEL_ERROR_OVERLAP_FILENAME,
    UNIVERSALLY_MISCLASSIFIED_FILENAME,
)
from src.analysis.figures import COLOR_PALETTE, apply_style, save_figure
from src.analysis.utils import (
    artifact_envelope,
    load_predictions,
    repo_relative,
    resolve_handoff,
    shared_analysis_dir,
    write_json,
)
from src.enums import ModelID
from src.report_figure_generation import FigureRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction alignment
# ---------------------------------------------------------------------------


def align_predictions_across_models(
    model_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Align predictions from multiple models by ``text`` column.

    Returns a DataFrame with columns ``text``, ``true_label_name``, and per-model
    ``{model_id}_pred``, ``{model_id}_correct`` columns.
    """
    model_ids = list(model_predictions.keys())
    base_id = model_ids[0]
    base_df = model_predictions[base_id][["text", "true_label_name"]].copy()
    base_df[f"{base_id}_pred"] = model_predictions[base_id]["predicted_label_name"]
    base_df[f"{base_id}_correct"] = (
        model_predictions[base_id]["true_label_name"] == model_predictions[base_id]["predicted_label_name"]
    )

    aligned = base_df
    for mid in model_ids[1:]:
        other = model_predictions[mid][["text", "predicted_label_name", "true_label_name"]].copy()
        other = other.rename(columns={"predicted_label_name": f"{mid}_pred"})
        other[f"{mid}_correct"] = other["true_label_name"] == other[f"{mid}_pred"]
        other = other.drop(columns=["true_label_name"])
        aligned = aligned.merge(other, on="text", how="inner")

    return aligned


def _categorize_row(row: pd.Series, model_ids: list[str]) -> str:
    correct_flags = [row[f"{mid}_correct"] for mid in model_ids]
    n_correct = sum(correct_flags)
    if n_correct == len(model_ids):
        return "all_correct"
    if n_correct == 0:
        return "all_wrong"
    if n_correct == len(model_ids) - 1:
        return "model_specific_error"
    return "partial_error"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_and_save_cross_model_comparison(model_ids: list[ModelID]) -> tuple[dict[str, Any], list[FigureRecord]]:
    """Run cross-model error comparison and save artifacts.

    Returns ``(comparison_artifact, figure_records)``.
    """
    records: list[FigureRecord] = []
    shared_dir = shared_analysis_dir()

    model_preds: dict[str, pd.DataFrame] = {}
    run_ids: dict[str, str] = {}
    source_paths: list[str] = []
    for mid in model_ids:
        ctx = resolve_handoff(mid)
        preds = load_predictions(ctx.final_predictions_path)
        model_preds[str(mid)] = preds
        run_ids[str(mid)] = ctx.representative_run_id
        source_paths.append(repo_relative(ctx.final_predictions_path))

    mid_strs = [str(m) for m in model_ids]
    aligned = align_predictions_across_models(model_preds)

    total = len(aligned)
    aligned["category"] = aligned.apply(lambda row: _categorize_row(row, mid_strs), axis=1)

    category_counts: dict[str, int] = {}
    for cat in ("all_correct", "all_wrong", "model_specific_error", "partial_error"):
        category_counts[cat] = int((aligned["category"] == cat).sum())

    # Model-specific errors
    model_specific_counts: dict[str, int] = {}
    specific_mask = aligned["category"] == "model_specific_error"
    for mid in mid_strs:
        model_specific_counts[mid] = int((specific_mask & ~aligned[f"{mid}_correct"]).sum())

    # All-wrong agreement
    all_wrong_df = aligned[aligned["category"] == "all_wrong"]
    if len(all_wrong_df) > 0:
        pred_cols = [f"{m}_pred" for m in mid_strs]
        all_same = all_wrong_df.apply(lambda r: len(set(r[col] for col in pred_cols)) == 1, axis=1)
        agreement_fraction = float(all_same.mean())
    else:
        agreement_fraction = 0.0

    artifact = artifact_envelope(
        models_compared=mid_strs,
        representative_run_ids=run_ids,
        total_test_examples=total,
        categories={
            cat: {"count": category_counts[cat], "fraction": float(category_counts[cat] / total) if total > 0 else 0.0}
            for cat in category_counts
        },
        model_specific_errors=model_specific_counts,
        all_wrong_agreement=agreement_fraction,
    )
    write_json(shared_dir / CROSS_MODEL_ERROR_COMPARISON_FILENAME, artifact)
    logger.info("Saved: %s", shared_dir / CROSS_MODEL_ERROR_COMPARISON_FILENAME)

    # Universally misclassified examples CSV
    univ_path = shared_dir / UNIVERSALLY_MISCLASSIFIED_FILENAME
    if len(all_wrong_df) > 0:
        export_cols = ["text", "true_label_name"] + [f"{m}_pred" for m in mid_strs]
        all_wrong_df[export_cols].to_csv(univ_path, index=False)
    else:
        pd.DataFrame(columns=["text", "true_label_name"]).to_csv(univ_path, index=False)
    logger.info("Saved: %s (%d examples)", univ_path, len(all_wrong_df))

    # Overlap figure
    fig_path = shared_dir / CROSS_MODEL_ERROR_OVERLAP_FILENAME
    _plot_error_overlap(model_ids, category_counts, model_specific_counts, total, fig_path)
    records.append(
        FigureRecord(
            figure_path=repo_relative(fig_path),
            figure_type="cross_model_error_overlap",
            scope="analysis",
            source_artifact_paths=source_paths,
            caption_context={
                "analysis_basis": "cross-model error overlap using one representative run per model",
                "dataset_name": "CLINC150",
                "dataset_split": "test",
                "metric_names": ["all-correct count", "all-wrong count", "model-specific errors"],
            },
        )
    )

    return artifact, records


def _plot_error_overlap(
    model_ids: list[ModelID],
    category_counts: dict[str, int],
    model_specific_counts: dict[str, int],
    total: int,
    out_path: Path,
) -> None:
    apply_style()
    labels = ["All Correct", "All Wrong", "Partial\nError"] + [f"Only\n{m.display_name}" for m in model_ids]
    values = [
        category_counts["all_correct"],
        category_counts["all_wrong"],
        category_counts["partial_error"],
    ] + [model_specific_counts.get(str(m), 0) for m in model_ids]

    colors = [COLOR_PALETTE[1], COLOR_PALETTE[2], COLOR_PALETTE[3]] + [
        COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(4, 4 + len(model_ids))
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.75)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Example Count")
    ax.set_title("Cross-Model Error Overlap")

    for bar, val in zip(bars, values):
        frac = val / total * 100 if total > 0 else 0
        ax.annotate(
            f"{val}\n({frac:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=7,
        )

    save_figure(fig, out_path)
