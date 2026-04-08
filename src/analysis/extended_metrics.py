"""Section B: Extended classification metrics (micro/weighted F1, precision, recall)."""

from __future__ import annotations

import logging
from typing import Any

from src.analysis.metrics import compute_extended_classification_metrics
from src.analysis.utils import (
    HandoffContext,
    analysis_output_dir,
    artifact_envelope,
    load_predictions,
    read_json,
    repo_relative,
    shared_analysis_dir,
    write_json,
)
from src.enums import ModelID

from src.analysis.constants import (
    EXTENDED_METRICS_COMPARISON_FILENAME,
    EXTENDED_TEST_METRICS_FILENAME,
)

logger = logging.getLogger(__name__)


def compute_and_save_extended_metrics(ctx: HandoffContext) -> dict[str, Any]:
    """Compute extended metrics for one model and save to analysis dir.

    Returns the saved artifact dict for downstream aggregation.
    """
    preds_df = load_predictions(ctx.final_predictions_path)
    y_true = preds_df["true_label_id"].values
    y_pred = preds_df["predicted_label_id"].values

    extended = compute_extended_classification_metrics(y_true, y_pred)

    test_metrics = read_json(ctx.test_metrics_path)

    artifact = artifact_envelope(
        model_id=ctx.model_id,
        representative_run_id=ctx.representative_run_id,
        macro_f1=test_metrics.get("macro_f1"),
        micro_f1=extended["micro_f1"],
        weighted_f1=extended["weighted_f1"],
        macro_precision=test_metrics.get("precision", test_metrics.get("macro_precision")),
        macro_recall=test_metrics.get("recall", test_metrics.get("macro_recall")),
        micro_precision=extended["micro_precision"],
        micro_recall=extended["micro_recall"],
        weighted_precision=extended["weighted_precision"],
        weighted_recall=extended["weighted_recall"],
        accuracy=test_metrics.get("accuracy"),
        source_artifact=repo_relative(ctx.final_predictions_path),
    )

    out_path = analysis_output_dir(ctx.model_id) / EXTENDED_TEST_METRICS_FILENAME
    write_json(out_path, artifact)
    logger.info("Saved: %s", out_path)
    return artifact


def generate_extended_metrics_comparison(
    model_ids: list[ModelID],
    per_model_metrics: dict[str, dict[str, Any]],
) -> None:
    """Combine per-model extended metrics into a shared comparison table."""
    rows: list[dict[str, Any]] = []
    for mid in model_ids:
        m = per_model_metrics[str(mid)]
        rows.append(
            {
                "model_id": str(mid),
                "model_name": mid.display_name,
                "accuracy": m.get("accuracy"),
                "macro_f1": m.get("macro_f1"),
                "micro_f1": m.get("micro_f1"),
                "weighted_f1": m.get("weighted_f1"),
                "macro_precision": m.get("macro_precision"),
                "macro_recall": m.get("macro_recall"),
                "micro_precision": m.get("micro_precision"),
                "micro_recall": m.get("micro_recall"),
                "weighted_precision": m.get("weighted_precision"),
                "weighted_recall": m.get("weighted_recall"),
            }
        )

    artifact = artifact_envelope(
        note="Single representative-run values, not aggregate means.",
        rows=rows,
    )
    out_path = shared_analysis_dir() / EXTENDED_METRICS_COMPARISON_FILENAME
    write_json(out_path, artifact)
    logger.info("Saved: %s", out_path)
