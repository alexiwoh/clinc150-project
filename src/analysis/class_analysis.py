"""Sections J + K: Per-class deep dive and cross-run confusion stability."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.constants import (
    CONFUSION_STABILITY_FIGURE_FILENAME,
    CONFUSION_STABILITY_FILENAME,
    CONFUSION_STABILITY_TOP_K,
    WORST_CLASSES_COMPARISON_FILENAME,
    WORST_CLASSES_FILENAME,
    WORST_CLASSES_HEATMAP_FILENAME,
    WORST_CLASSES_K,
    CLINC150_INTENT_DOMAINS,
)
from src.analysis.figures import COLOR_PALETTE, apply_style, save_figure
from src.analysis.utils import (
    HandoffContext,
    analysis_output_dir,
    artifact_envelope,
    discover_all_run_dirs,
    load_confidences,
    load_predictions,
    read_json,
    repo_relative,
    shared_analysis_dir,
    write_json,
)
from src.enums import ModelID
from src.report_figure_generation import FigureRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section J: Per-class deep dive (worst classes)
# ---------------------------------------------------------------------------


def compute_and_save_worst_classes(ctx: HandoffContext) -> tuple[dict[str, Any], list[FigureRecord]]:
    """Identify bottom-K classes by F1 and analyze confusion patterns.

    Returns ``(worst_classes_artifact, figure_records)``.
    """
    records: list[FigureRecord] = []
    out_dir = analysis_output_dir(ctx.model_id)

    per_class = read_json(ctx.per_class_metrics_path)
    classes: list[dict[str, Any]] = per_class["classes"]
    sorted_by_f1 = sorted(classes, key=lambda c: c.get("f1", 0.0))
    bottom_k = sorted_by_f1[:WORST_CLASSES_K]
    bottom_k_names = {c["label_name"] for c in bottom_k}

    conf_matrix_df = pd.read_csv(ctx.confusion_matrix_path, index_col=0)
    conf = load_confidences(ctx.confidences_path)
    preds_df = load_predictions(ctx.final_predictions_path)
    max_probs = np.max(conf.probabilities, axis=1)

    deep_dive_entries: list[dict[str, Any]] = []
    for cls in bottom_k:
        name = cls["label_name"]
        if name in conf_matrix_df.index:
            row = conf_matrix_df.loc[name].drop(name, errors="ignore")
            top_confused = row.nlargest(3)
            confusion_targets = [
                {
                    "target_class": str(t),
                    "count": int(c),
                    "shares_domain": CLINC150_INTENT_DOMAINS.get(name) == CLINC150_INTENT_DOMAINS.get(str(t)),
                }
                for t, c in top_confused.items()
                if c > 0
            ]
        else:
            confusion_targets = []

        mis_mask = (preds_df["true_label_name"] == name) & (
            preds_df["true_label_name"] != preds_df["predicted_label_name"]
        )
        mean_conf_misclassified = float(max_probs[mis_mask.values].mean()) if mis_mask.any() else 0.0

        deep_dive_entries.append(
            {
                "label_name": name,
                "precision": cls.get("precision", 0.0),
                "recall": cls.get("recall", 0.0),
                "f1": cls.get("f1", 0.0),
                "support": cls.get("support", 0),
                "domain": CLINC150_INTENT_DOMAINS.get(name, "unknown"),
                "top_confusion_targets": confusion_targets,
                "mean_confidence_misclassified": mean_conf_misclassified,
            }
        )

    artifact = artifact_envelope(
        model_id=ctx.model_id,
        representative_run_id=ctx.representative_run_id,
        k=WORST_CLASSES_K,
        worst_classes=deep_dive_entries,
        source_artifacts={
            "per_class_metrics": repo_relative(ctx.per_class_metrics_path),
            "confusion_matrix": repo_relative(ctx.confusion_matrix_path),
        },
    )
    write_json(out_dir / WORST_CLASSES_FILENAME, artifact)
    logger.info("Saved: %s", out_dir / WORST_CLASSES_FILENAME)

    # Heatmap figure
    fig_path = out_dir / WORST_CLASSES_HEATMAP_FILENAME
    _plot_worst_classes_heatmap(bottom_k_names, conf_matrix_df, ctx.model_id.display_name, fig_path)
    records.append(
        FigureRecord(
            figure_path=repo_relative(fig_path),
            figure_type="worst_classes_heatmap",
            scope="analysis",
            source_artifact_paths=[repo_relative(ctx.confusion_matrix_path)],
            model_name=str(ctx.model_id),
            representative_run_id=ctx.representative_run_id,
            caption_context={
                "analysis_basis": "representative-run worst-class analysis",
                "class_count": len(bottom_k_names),
                "dataset_name": "CLINC150",
                "dataset_split": "test",
            },
        )
    )

    return artifact, records


def _plot_worst_classes_heatmap(
    bottom_k_names: set[str],
    conf_matrix_df: pd.DataFrame,
    model_name: str,
    out_path: Path,
) -> None:
    apply_style()
    ordered = sorted(bottom_k_names)
    all_targets: set[str] = set()
    for cls in ordered:
        if cls in conf_matrix_df.index:
            row = conf_matrix_df.loc[cls].drop(cls, errors="ignore")
            top3 = row.nlargest(3)
            all_targets.update(str(t) for t, c in top3.items() if c > 0)
    target_cols = sorted(all_targets | bottom_k_names)

    matrix = np.zeros((len(ordered), len(target_cols)))
    for i, cls in enumerate(ordered):
        if cls in conf_matrix_df.index:
            for j, tgt in enumerate(target_cols):
                if tgt in conf_matrix_df.columns:
                    matrix[i, j] = conf_matrix_df.loc[cls, tgt]

    fig, ax = plt.subplots(figsize=(max(10, len(target_cols) * 0.6), max(6, len(ordered) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(target_cols)))
    ax.set_xticklabels(target_cols, rotation=90, fontsize=6)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=7)
    ax.set_title(f"Worst {len(ordered)} Classes Confusion Heatmap - {model_name}")
    fig.colorbar(im, ax=ax, shrink=0.6)
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# J2: Cross-model worst-class comparison
# ---------------------------------------------------------------------------


def generate_worst_classes_comparison(
    model_ids: list[ModelID],
    per_model_worst: dict[str, dict[str, Any]],
) -> None:
    """Compare bottom-K classes across models."""
    shared_dir = shared_analysis_dir()

    per_model_sets: dict[str, set[str]] = {}
    for mid in model_ids:
        classes = per_model_worst[str(mid)]["worst_classes"]
        per_model_sets[str(mid)] = {c["label_name"] for c in classes}

    all_sets = list(per_model_sets.values())
    shared_worst = set.intersection(*all_sets) if all_sets else set()
    unique_per_model: dict[str, list[str]] = {}
    for mid_str, s in per_model_sets.items():
        other_sets = [v for k, v in per_model_sets.items() if k != mid_str]
        others = set.union(*other_sets) if other_sets else set()
        unique_per_model[mid_str] = sorted(s - others)

    artifact = artifact_envelope(
        k=WORST_CLASSES_K,
        shared_worst_classes=sorted(shared_worst),
        shared_count=len(shared_worst),
        unique_per_model=unique_per_model,
        per_model_worst_classes={mid_str: sorted(s) for mid_str, s in per_model_sets.items()},
    )
    write_json(shared_dir / WORST_CLASSES_COMPARISON_FILENAME, artifact)
    logger.info("Saved: %s", shared_dir / WORST_CLASSES_COMPARISON_FILENAME)


# ---------------------------------------------------------------------------
# Section K: Cross-run confusion stability
# ---------------------------------------------------------------------------


def compute_and_save_confusion_stability(
    ctx: HandoffContext,
) -> tuple[dict[str, Any], list[FigureRecord]]:
    """Analyze confusion stability across all runs for one model.

    Returns ``(stability_artifact, figure_records)``.
    """
    records: list[FigureRecord] = []
    out_dir = analysis_output_dir(ctx.model_id)
    run_dirs = discover_all_run_dirs(ctx.model_id)

    if len(run_dirs) < 2:
        logger.warning("Only %d run(s) for %s — skipping confusion stability.", len(run_dirs), ctx.model_id)
        artifact = artifact_envelope(
            model_id=ctx.model_id,
            note="Fewer than 2 runs available; confusion stability analysis skipped.",
            stable_pairs=[],
            unstable_pairs=[],
        )
        write_json(out_dir / CONFUSION_STABILITY_FILENAME, artifact)
        return artifact, records

    all_pair_counts: dict[tuple[str, str], list[int]] = {}
    for rd in run_dirs:
        cm_path = rd / "confusion_matrix.csv"
        if not cm_path.exists():
            continue
        cm_df = pd.read_csv(cm_path, index_col=0)
        for true_cls in cm_df.index:
            for pred_cls in cm_df.columns:
                if true_cls == pred_cls:
                    continue
                count = int(cm_df.loc[true_cls, pred_cls])
                if count > 0:
                    key = (str(true_cls), str(pred_cls))
                    all_pair_counts.setdefault(key, []).append(count)

    n_runs = len(run_dirs)
    pair_stats: list[dict[str, Any]] = []
    for (true_cls, pred_cls), counts in all_pair_counts.items():
        padded = counts + [0] * (n_runs - len(counts))
        pair_stats.append(
            {
                "true_class": true_cls,
                "predicted_class": pred_cls,
                "mean_count": float(np.mean(padded)),
                "std_count": float(np.std(padded)),
                "runs_present": len(counts),
                "total_runs": n_runs,
            }
        )
    pair_stats.sort(key=lambda p: p["mean_count"], reverse=True)
    top_pairs = pair_stats[:CONFUSION_STABILITY_TOP_K]

    top_pair_keys: set[tuple[str, str]] = set()
    per_run_top_pairs: list[set[tuple[str, str]]] = []
    for rd in run_dirs:
        cm_path = rd / "confusion_matrix.csv"
        if not cm_path.exists():
            continue
        cm_df = pd.read_csv(cm_path, index_col=0)
        run_pairs: list[tuple[str, str, int]] = []
        for true_cls in cm_df.index:
            for pred_cls in cm_df.columns:
                if true_cls == pred_cls:
                    continue
                c = int(cm_df.loc[true_cls, pred_cls])
                if c > 0:
                    run_pairs.append((str(true_cls), str(pred_cls), c))
        run_pairs.sort(key=lambda x: x[2], reverse=True)
        run_top = {(p[0], p[1]) for p in run_pairs[:CONFUSION_STABILITY_TOP_K]}
        per_run_top_pairs.append(run_top)
        top_pair_keys.update(run_top)

    stable_pairs: list[dict[str, str]] = []
    unstable_pairs: list[dict[str, str]] = []
    for pair in top_pair_keys:
        present_in = sum(1 for rtp in per_run_top_pairs if pair in rtp)
        entry = {"true_class": pair[0], "predicted_class": pair[1]}
        if present_in == n_runs:
            stable_pairs.append(entry)
        elif present_in == 1:
            unstable_pairs.append(entry)

    artifact = artifact_envelope(
        model_id=ctx.model_id,
        n_runs=n_runs,
        top_confusion_pairs=top_pairs,
        stable_pairs=stable_pairs,
        unstable_pairs=unstable_pairs,
        stable_count=len(stable_pairs),
        unstable_count=len(unstable_pairs),
    )
    write_json(out_dir / CONFUSION_STABILITY_FILENAME, artifact)
    logger.info("Saved: %s", out_dir / CONFUSION_STABILITY_FILENAME)

    # Stability figure
    fig_path = out_dir / CONFUSION_STABILITY_FIGURE_FILENAME
    _plot_confusion_stability(top_pairs, ctx.model_id.display_name, fig_path)
    records.append(
        FigureRecord(
            figure_path=repo_relative(fig_path),
            figure_type="confusion_stability",
            scope="analysis",
            source_artifact_paths=[
                repo_relative(rd / "confusion_matrix.csv") for rd in run_dirs if (rd / "confusion_matrix.csv").exists()
            ],
            model_name=str(ctx.model_id),
            representative_run_id=ctx.representative_run_id,
            caption_context={
                "analysis_basis": "cross-run confusion stability analysis",
                "dataset_name": "CLINC150",
                "dataset_split": "test",
                "metric_names": ["mean confusion count", "std confusion count"],
                "run_count": len(run_dirs),
                "top_k": len(top_pairs),
            },
        )
    )

    return artifact, records


def _plot_confusion_stability(
    top_pairs: list[dict[str, Any]],
    model_name: str,
    out_path: Path,
) -> None:
    if not top_pairs:
        return
    apply_style()
    labels = [f"{p['true_class']}\n→{p['predicted_class']}" for p in top_pairs[:15]]
    means = [p["mean_count"] for p in top_pairs[:15]]
    stds = [p["std_count"] for p in top_pairs[:15]]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(labels)), means, yerr=stds, capsize=3, color=COLOR_PALETTE[0], alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Mean Confusion Count")
    ax.set_title(f"Top Confusion Pairs (mean ± std across runs) - {model_name}")
    save_figure(fig, out_path)
