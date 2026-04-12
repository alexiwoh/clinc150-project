"""Section E: Structured error taxonomy classification and domain mapping."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.analysis.constants import (
    CLINC150_INTENT_DOMAINS,
    ERROR_TAXONOMY_COMPARISON_FILENAME,
    ERROR_TAXONOMY_FILENAME,
    ERROR_TAXONOMY_SUMMARY_FILENAME,
    INTENT_DOMAIN_MAPPING_FILENAME,
    SHORT_QUERY_TOKEN_THRESHOLD,
)
from src.analysis.enums import ErrorCategory
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
from src.constants import OOS_LABEL_NAME
from src.enums import ModelID
from src.report_figure_generation import FigureRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-example classification
# ---------------------------------------------------------------------------


def classify_single_error(
    text: str,
    true_label: str,
    pred_label: str,
    oos_label: str = OOS_LABEL_NAME,
    domain_map: dict[str, str] | None = None,
) -> tuple[ErrorCategory, list[ErrorCategory]]:
    """Classify one misclassified example into primary + secondary categories.

    Priority order: oos_as_inscope > inscope_as_oos > near_semantic_confusion
    > cross_domain_confusion > short_query_ambiguity.
    """
    if domain_map is None:
        domain_map = CLINC150_INTENT_DOMAINS

    categories: list[ErrorCategory] = []

    if true_label == oos_label and pred_label != oos_label:
        categories.append(ErrorCategory.OOS_AS_INSCOPE)
    if true_label != oos_label and pred_label == oos_label:
        categories.append(ErrorCategory.INSCOPE_AS_OOS)

    if true_label != oos_label and pred_label != oos_label:
        true_domain = domain_map.get(true_label)
        pred_domain = domain_map.get(pred_label)
        if true_domain is not None and pred_domain is not None:
            if true_domain == pred_domain:
                categories.append(ErrorCategory.NEAR_SEMANTIC_CONFUSION)
            else:
                categories.append(ErrorCategory.CROSS_DOMAIN_CONFUSION)
        else:
            categories.append(ErrorCategory.CROSS_DOMAIN_CONFUSION)

    token_count = len(text.split())
    if token_count <= SHORT_QUERY_TOKEN_THRESHOLD:
        categories.append(ErrorCategory.SHORT_QUERY_AMBIGUITY)

    if not categories:
        categories.append(ErrorCategory.CROSS_DOMAIN_CONFUSION)

    primary = categories[0]
    secondary = categories[1:]
    return primary, secondary


# ---------------------------------------------------------------------------
# Per-model taxonomy
# ---------------------------------------------------------------------------


def compute_and_save_error_taxonomy(ctx: HandoffContext) -> dict[str, Any]:
    """Classify all misclassified examples for one model and save taxonomy artifact."""
    conf = load_confidences(ctx.confidences_path)
    preds_df = load_predictions(ctx.final_predictions_path)
    out_dir = analysis_output_dir(ctx.model_id)

    max_probs = np.max(conf.probabilities, axis=1)
    misclassified_mask = preds_df["true_label_id"].values != preds_df["predicted_label_id"].values
    total_test = len(preds_df)
    total_misclassified = int(misclassified_mask.sum())

    category_counts: dict[str, int] = {cat: 0 for cat in ErrorCategory}
    examples: list[dict[str, Any]] = []

    for idx in np.where(misclassified_mask)[0]:
        row = preds_df.iloc[idx]
        primary, secondary = classify_single_error(
            text=row["text"],
            true_label=row["true_label_name"],
            pred_label=row["predicted_label_name"],
        )
        category_counts[primary] += 1
        examples.append(
            {
                "text": row["text"],
                "true_label_name": row["true_label_name"],
                "predicted_label_name": row["predicted_label_name"],
                "max_confidence": float(max_probs[idx]),
                "primary_category": str(primary),
                "secondary_categories": [str(s) for s in secondary],
            }
        )

    assert sum(category_counts.values()) == total_misclassified, (
        f"Category counts {sum(category_counts.values())} != total misclassified {total_misclassified}"
    )

    categories_detail: dict[str, dict[str, Any]] = {}
    for cat in ErrorCategory:
        c = category_counts[cat]
        categories_detail[cat] = {
            "count": c,
            "fraction_of_errors": float(c / total_misclassified) if total_misclassified > 0 else 0.0,
            "fraction_of_test_set": float(c / total_test) if total_test > 0 else 0.0,
        }

    artifact = artifact_envelope(
        model_id=ctx.model_id,
        representative_run_id=ctx.representative_run_id,
        total_misclassified=total_misclassified,
        total_test_examples=total_test,
        error_rate=float(total_misclassified / total_test) if total_test > 0 else 0.0,
        categories=categories_detail,
        examples=examples,
        source_artifact=repo_relative(ctx.final_predictions_path),
    )

    write_json(out_dir / ERROR_TAXONOMY_FILENAME, artifact)
    logger.info("Saved: %s (%d misclassified examples)", out_dir / ERROR_TAXONOMY_FILENAME, total_misclassified)
    return artifact


# ---------------------------------------------------------------------------
# Shared taxonomy summary + figure
# ---------------------------------------------------------------------------


def generate_taxonomy_summary(
    model_ids: list[ModelID],
    per_model_taxonomy: dict[str, dict[str, Any]],
) -> list[FigureRecord]:
    """Generate cross-model taxonomy summary table and comparison figure."""
    records: list[FigureRecord] = []
    shared_dir = shared_analysis_dir()

    rows: list[dict[str, Any]] = []
    for mid in model_ids:
        tax = per_model_taxonomy[str(mid)]
        row: dict[str, Any] = {"model_id": str(mid), "model_name": mid.display_name}
        for cat in ErrorCategory:
            cat_data = tax["categories"][cat]
            row[f"{cat}_count"] = cat_data["count"]
            row[f"{cat}_fraction"] = cat_data["fraction_of_errors"]
        rows.append(row)

    summary = artifact_envelope(rows=rows)
    write_json(shared_dir / ERROR_TAXONOMY_SUMMARY_FILENAME, summary)
    logger.info("Saved: %s", shared_dir / ERROR_TAXONOMY_SUMMARY_FILENAME)

    # Grouped bar chart
    group_labels = [cat.value for cat in ErrorCategory]
    series: dict[str, list[float]] = {}
    for mid in model_ids:
        tax = per_model_taxonomy[str(mid)]
        series[mid.display_name] = [tax["categories"][cat]["fraction_of_errors"] for cat in ErrorCategory]

    fig_path = shared_dir / ERROR_TAXONOMY_COMPARISON_FILENAME
    plot_grouped_bar(
        group_labels,
        series,
        ylabel="Fraction of Errors",
        title="Error Taxonomy Comparison",
        out_path=fig_path,
    )
    source_paths = [repo_relative(analysis_output_dir(m) / ERROR_TAXONOMY_FILENAME) for m in model_ids]
    records.append(
        FigureRecord(
            figure_path=repo_relative(fig_path),
            figure_type="error_taxonomy",
            scope="analysis",
            source_artifact_paths=source_paths,
            caption_context={
                "analysis_basis": "cross-model error taxonomy comparison using representative runs",
                "dataset_name": "CLINC150",
                "dataset_split": "test",
                "metric_names": ["fraction of errors by taxonomy category"],
            },
        )
    )

    return records


# ---------------------------------------------------------------------------
# Intent domain mapping artifact
# ---------------------------------------------------------------------------


def save_intent_domain_mapping() -> None:
    """Save the CLINC150 intent-to-domain mapping as a shared analysis artifact."""
    from src.analysis.constants import CLINC150_DOMAIN_NAMES

    domains_to_intents: dict[str, list[str]] = {d: [] for d in CLINC150_DOMAIN_NAMES}
    for intent, domain in CLINC150_INTENT_DOMAINS.items():
        domains_to_intents[domain].append(intent)

    artifact = artifact_envelope(
        source="https://github.com/clinc/oos-eval/blob/master/data/domains.json",
        n_domains=len(CLINC150_DOMAIN_NAMES),
        n_intents=len(CLINC150_INTENT_DOMAINS),
        intent_to_domain=CLINC150_INTENT_DOMAINS,
        domain_to_intents=domains_to_intents,
    )

    out_path = shared_analysis_dir() / INTENT_DOMAIN_MAPPING_FILENAME
    write_json(out_path, artifact)
    logger.info("Saved: %s", out_path)
