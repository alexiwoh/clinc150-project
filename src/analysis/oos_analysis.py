"""Sections D + F: OOS threshold analysis and OOS error deep dive."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.constants import (
    OOS_ERROR_BREAKDOWN_FILENAME,
    OOS_ERROR_COMPARISON_FILENAME,
    OOS_FALSE_ACCEPT_COMPARISON_FILENAME,
    OOS_PR_COMPARISON_FILENAME,
    OOS_PR_CURVE_FILENAME,
    OOS_ROC_COMPARISON_FILENAME,
    OOS_ROC_CURVE_FILENAME,
    OOS_THRESHOLD_COMPARISON_FILENAME,
    OOS_THRESHOLD_METRICS_FILENAME,
    OOS_ERROR_DEEP_DIVE_FILENAME,
    TOP_OOS_EXAMPLES_K,
    TOP_OOS_INTENTS_K,
)
from src.analysis.figures import COLOR_PALETTE, apply_style, plot_curve_comparison, save_figure
from src.analysis.metrics import compute_msp_oos_detection_metrics, compute_oos_detection_metrics
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
# Section D: OOS threshold analysis
# ---------------------------------------------------------------------------


def compute_and_save_oos_threshold(ctx: HandoffContext) -> tuple[dict[str, Any], list[FigureRecord]]:
    """Compute OOS detection metrics and save curves for one model.

    Returns ``(oos_metrics_artifact, figure_records)``.
    """
    conf = load_confidences(ctx.confidences_path)
    records: list[FigureRecord] = []
    out_dir = analysis_output_dir(ctx.model_id)

    oos_metrics = compute_oos_detection_metrics(conf.probabilities, conf.targets, conf.oos_class_idx)
    msp_metrics = compute_msp_oos_detection_metrics(conf.probabilities, conf.targets, conf.oos_class_idx)

    artifact = artifact_envelope(
        model_id=ctx.model_id,
        representative_run_id=ctx.representative_run_id,
        method="explicit_oos_class_probability",
        auroc=oos_metrics["auroc"],
        aupr=oos_metrics["aupr"],
        fpr_at_95tpr=oos_metrics["fpr_at_95tpr"],
        fpr_at_90tpr=oos_metrics["fpr_at_90tpr"],
        msp_baseline=msp_metrics,
        source_artifact=repo_relative(ctx.confidences_path),
    )

    write_json(out_dir / OOS_THRESHOLD_METRICS_FILENAME, artifact)
    logger.info("Saved: %s", out_dir / OOS_THRESHOLD_METRICS_FILENAME)

    artifact["roc_curve"] = oos_metrics["roc_curve"]
    artifact["pr_curve"] = oos_metrics["pr_curve"]

    src_paths = [repo_relative(ctx.confidences_path)]

    # ROC curve
    roc_path = out_dir / OOS_ROC_CURVE_FILENAME
    _plot_single_roc(oos_metrics, ctx.model_id.display_name, roc_path)
    records.append(
        FigureRecord(
            figure_path=repo_relative(roc_path),
            figure_type="oos_roc_curve",
            scope="representative",
            source_artifact_paths=src_paths,
            model_name=str(ctx.model_id),
            representative_run_id=ctx.representative_run_id,
        )
    )

    # PR curve
    pr_path = out_dir / OOS_PR_CURVE_FILENAME
    _plot_single_pr(oos_metrics, ctx.model_id.display_name, pr_path)
    records.append(
        FigureRecord(
            figure_path=repo_relative(pr_path),
            figure_type="oos_pr_curve",
            scope="representative",
            source_artifact_paths=src_paths,
            model_name=str(ctx.model_id),
            representative_run_id=ctx.representative_run_id,
        )
    )

    return artifact, records


def generate_oos_threshold_comparison(
    model_ids: list[ModelID],
    per_model_oos: dict[str, dict[str, Any]],
) -> list[FigureRecord]:
    """Generate shared OOS threshold comparison table and overlay figures."""
    records: list[FigureRecord] = []
    shared_dir = shared_analysis_dir()
    source_paths: list[str] = []

    rows: list[dict[str, Any]] = []
    roc_curves: list[dict[str, Any]] = []
    pr_curves: list[dict[str, Any]] = []

    for mid in model_ids:
        oos = per_model_oos[str(mid)]
        rows.append(
            {
                "model_id": str(mid),
                "model_name": mid.display_name,
                "auroc": oos["auroc"],
                "aupr": oos["aupr"],
                "fpr_at_95tpr": oos["fpr_at_95tpr"],
                "fpr_at_90tpr": oos["fpr_at_90tpr"],
                "msp_auroc": oos.get("msp_baseline", {}).get("msp_auroc"),
                "msp_aupr": oos.get("msp_baseline", {}).get("msp_aupr"),
            }
        )

        src = analysis_output_dir(mid) / OOS_THRESHOLD_METRICS_FILENAME
        source_paths.append(repo_relative(src))

        if "roc_curve" in oos:
            roc_curves.append(
                {
                    "x": oos["roc_curve"]["fpr"],
                    "y": oos["roc_curve"]["tpr"],
                    "label": mid.display_name,
                    "auc_value": oos["auroc"],
                }
            )
        if "pr_curve" in oos:
            pr_curves.append(
                {
                    "x": oos["pr_curve"]["recall"],
                    "y": oos["pr_curve"]["precision"],
                    "label": mid.display_name,
                    "auc_value": oos["aupr"],
                }
            )

    summary = artifact_envelope(rows=rows)
    write_json(shared_dir / OOS_THRESHOLD_COMPARISON_FILENAME, summary)
    logger.info("Saved: %s", shared_dir / OOS_THRESHOLD_COMPARISON_FILENAME)

    if roc_curves:
        roc_comp = shared_dir / OOS_ROC_COMPARISON_FILENAME
        plot_curve_comparison(
            roc_curves,
            "False Positive Rate",
            "True Positive Rate",
            "OOS ROC Comparison",
            roc_comp,
            diagonal=True,
        )
        records.append(
            FigureRecord(
                figure_path=repo_relative(roc_comp),
                figure_type="oos_roc_comparison",
                scope="aggregate",
                source_artifact_paths=source_paths,
            )
        )

    if pr_curves:
        pr_comp = shared_dir / OOS_PR_COMPARISON_FILENAME
        plot_curve_comparison(pr_curves, "Recall", "Precision", "OOS Precision-Recall Comparison", pr_comp)
        records.append(
            FigureRecord(
                figure_path=repo_relative(pr_comp),
                figure_type="oos_pr_comparison",
                scope="aggregate",
                source_artifact_paths=source_paths,
            )
        )

    return records


# ---------------------------------------------------------------------------
# Section F: OOS error deep dive
# ---------------------------------------------------------------------------


def compute_and_save_oos_deep_dive(ctx: HandoffContext) -> tuple[dict[str, Any], list[FigureRecord]]:
    """Analyze OOS false accepts and false rejects for one model.

    Returns ``(deep_dive_artifact, figure_records)``.
    """
    conf = load_confidences(ctx.confidences_path)
    preds_df = load_predictions(ctx.final_predictions_path)
    records: list[FigureRecord] = []
    out_dir = analysis_output_dir(ctx.model_id)

    is_oos = conf.targets == conf.oos_class_idx
    pred_oos = conf.predictions == conf.oos_class_idx
    max_probs = np.max(conf.probabilities, axis=1)

    # False accepts: true OOS predicted as in-scope
    false_accept_mask = is_oos & ~pred_oos
    fa_df = preds_df[false_accept_mask].copy()
    fa_df["max_confidence"] = max_probs[false_accept_mask]

    fa_by_intent = (
        fa_df.groupby("predicted_label_name")
        .agg(count=("text", "size"), mean_confidence=("max_confidence", "mean"))
        .sort_values("count", ascending=False)
        .head(TOP_OOS_INTENTS_K)
    )

    fa_top_examples = fa_df.nlargest(TOP_OOS_EXAMPLES_K, "max_confidence")[
        ["text", "true_label_name", "predicted_label_name", "max_confidence"]
    ].to_dict("records")

    # False rejects: true in-scope predicted as OOS
    false_reject_mask = ~is_oos & pred_oos
    fr_df = preds_df[false_reject_mask].copy()
    fr_df["max_confidence"] = max_probs[false_reject_mask]

    fr_by_intent = (
        fr_df.groupby("true_label_name")
        .agg(count=("text", "size"), mean_confidence=("max_confidence", "mean"))
        .sort_values("count", ascending=False)
        .head(TOP_OOS_INTENTS_K)
    )

    fr_top_examples = fr_df.nsmallest(TOP_OOS_EXAMPLES_K, "max_confidence")[
        ["text", "true_label_name", "predicted_label_name", "max_confidence"]
    ].to_dict("records")

    artifact = artifact_envelope(
        model_id=ctx.model_id,
        representative_run_id=ctx.representative_run_id,
        false_accepts={
            "count": int(false_accept_mask.sum()),
            "mean_confidence": float(max_probs[false_accept_mask].mean()) if false_accept_mask.any() else 0.0,
            "std_confidence": float(max_probs[false_accept_mask].std()) if false_accept_mask.any() else 0.0,
            "top_capturing_intents": (
                fa_by_intent.reset_index().rename(columns={"predicted_label_name": "intent"}).to_dict("records")
            ),
            "top_examples": fa_top_examples,
        },
        false_rejects={
            "count": int(false_reject_mask.sum()),
            "mean_confidence": float(max_probs[false_reject_mask].mean()) if false_reject_mask.any() else 0.0,
            "std_confidence": float(max_probs[false_reject_mask].std()) if false_reject_mask.any() else 0.0,
            "top_rejected_intents": (
                fr_by_intent.reset_index().rename(columns={"true_label_name": "intent"}).to_dict("records")
            ),
            "top_examples": fr_top_examples,
        },
        source_artifact=repo_relative(ctx.final_predictions_path),
    )

    write_json(out_dir / OOS_ERROR_DEEP_DIVE_FILENAME, artifact)
    logger.info("Saved: %s", out_dir / OOS_ERROR_DEEP_DIVE_FILENAME)

    # Per-model breakdown figure
    fig_path = out_dir / OOS_ERROR_BREAKDOWN_FILENAME
    _plot_oos_error_breakdown(artifact, ctx.model_id.display_name, fig_path)
    records.append(
        FigureRecord(
            figure_path=repo_relative(fig_path),
            figure_type="oos_error_breakdown",
            scope="representative",
            source_artifact_paths=[repo_relative(ctx.final_predictions_path)],
            model_name=str(ctx.model_id),
            representative_run_id=ctx.representative_run_id,
        )
    )

    return artifact, records


def generate_oos_error_comparison(
    model_ids: list[ModelID],
    per_model_deep_dive: dict[str, dict[str, Any]],
) -> list[FigureRecord]:
    """Generate shared OOS false-accept comparison and OOS error comparison figure."""
    records: list[FigureRecord] = []
    shared_dir = shared_analysis_dir()

    # False-accept comparison: which intents capture OOS across models
    fa_comparison: dict[str, Any] = {}

    for mid in model_ids:
        dd = per_model_deep_dive[str(mid)]
        fa_comparison[str(mid)] = {
            "model_name": mid.display_name,
            "false_accept_count": dd["false_accepts"]["count"],
            "top_capturing_intents": dd["false_accepts"]["top_capturing_intents"],
        }
    fa_artifact = artifact_envelope(models=fa_comparison)
    write_json(shared_dir / OOS_FALSE_ACCEPT_COMPARISON_FILENAME, fa_artifact)
    logger.info("Saved: %s", shared_dir / OOS_FALSE_ACCEPT_COMPARISON_FILENAME)

    # OOS error comparison figure
    comp_fig_path = shared_dir / OOS_ERROR_COMPARISON_FILENAME
    _plot_oos_error_comparison_bar(model_ids, per_model_deep_dive, comp_fig_path)
    source_paths = [repo_relative(analysis_output_dir(m) / OOS_ERROR_DEEP_DIVE_FILENAME) for m in model_ids]
    records.append(
        FigureRecord(
            figure_path=repo_relative(comp_fig_path),
            figure_type="oos_error_comparison",
            scope="aggregate",
            source_artifact_paths=source_paths,
        )
    )

    return records


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _plot_single_roc(oos_metrics: dict[str, Any], model_name: str, out_path: Any) -> None:
    apply_style()
    fpr = oos_metrics["roc_curve"]["fpr"]
    tpr = oos_metrics["roc_curve"]["tpr"]
    auroc = oos_metrics["auroc"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=COLOR_PALETTE[0], linewidth=1.5, label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"OOS ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    save_figure(fig, out_path)


def _plot_single_pr(oos_metrics: dict[str, Any], model_name: str, out_path: Any) -> None:
    apply_style()
    precision = oos_metrics["pr_curve"]["precision"]
    recall = oos_metrics["pr_curve"]["recall"]
    aupr = oos_metrics["aupr"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color=COLOR_PALETTE[0], linewidth=1.5, label=f"AUPR = {aupr:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"OOS Precision-Recall Curve - {model_name}")
    ax.legend(loc="best")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    save_figure(fig, out_path)


def _plot_oos_error_breakdown(artifact: dict[str, Any], model_name: str, out_path: Any) -> None:
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # False accepts
    fa_intents = artifact["false_accepts"]["top_capturing_intents"]
    if fa_intents:
        labels = [e["intent"] for e in fa_intents]
        counts = [e["count"] for e in fa_intents]
        ax1.barh(range(len(labels)), counts, color=COLOR_PALETTE[2], alpha=0.7)
        ax1.set_yticks(range(len(labels)))
        ax1.set_yticklabels(labels, fontsize=7)
        ax1.invert_yaxis()
    ax1.set_xlabel("Count")
    ax1.set_title(f"Top Intents Capturing OOS\n(False Accepts) - {model_name}")

    # False rejects
    fr_intents = artifact["false_rejects"]["top_rejected_intents"]
    if fr_intents:
        labels = [e["intent"] for e in fr_intents]
        counts = [e["count"] for e in fr_intents]
        ax2.barh(range(len(labels)), counts, color=COLOR_PALETTE[0], alpha=0.7)
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels, fontsize=7)
        ax2.invert_yaxis()
    ax2.set_xlabel("Count")
    ax2.set_title(f"Top Intents Rejected as OOS\n(False Rejects) - {model_name}")

    save_figure(fig, out_path)


def _plot_oos_error_comparison_bar(
    model_ids: list[ModelID],
    per_model_deep_dive: dict[str, dict[str, Any]],
    out_path: Any,
) -> None:
    apply_style()
    names = [m.display_name for m in model_ids]
    fa_counts = [per_model_deep_dive[str(m)]["false_accepts"]["count"] for m in model_ids]
    fr_counts = [per_model_deep_dive[str(m)]["false_rejects"]["count"] for m in model_ids]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, fa_counts, width, label="False Accepts", color=COLOR_PALETTE[2], alpha=0.7)
    ax.bar(x + width / 2, fr_counts, width, label="False Rejects", color=COLOR_PALETTE[0], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Count")
    ax.set_title("OOS Error Comparison")
    ax.legend()
    save_figure(fig, out_path)
