"""Section L: Curate representative misclassification examples for the report."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.analysis.constants import (
    CONFIDENCE_STRATIFICATION_FILENAME,
    CURATED_EXAMPLES_CSV_FILENAME,
    CURATED_EXAMPLES_JSON_FILENAME,
    ERROR_TAXONOMY_FILENAME,
    HIGH_CONFIDENCE_THRESHOLD,
    OOS_ERROR_DEEP_DIVE_FILENAME,
)
from src.analysis.enums import ErrorCategory
from src.analysis.utils import (
    analysis_output_dir,
    artifact_envelope,
    load_predictions,
    read_json,
    resolve_handoff,
    shared_analysis_dir,
    write_json,
)
from src.enums import ModelID

logger = logging.getLogger(__name__)

_EXAMPLES_PER_CATEGORY: int = 5
_MIN_HIGH_CONF_PER_MODEL: int = 2
_MIN_OOS_FA_PER_MODEL: int = 2
_MIN_SEMANTIC_PER_MODEL: int = 2
_MINIMUM_TOTAL_CURATED: int = 50


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def curate_report_examples(model_ids: list[ModelID]) -> dict[str, Any]:
    """Select curated examples across models and categories.

    Returns the curated examples artifact dict.
    """
    shared_dir = shared_analysis_dir()
    all_curated: list[dict[str, Any]] = []

    for mid in model_ids:
        ctx = resolve_handoff(mid)
        preds_df = load_predictions(ctx.final_predictions_path)
        out_dir = analysis_output_dir(mid)

        taxonomy = read_json(out_dir / ERROR_TAXONOMY_FILENAME)
        oos_dd = read_json(out_dir / OOS_ERROR_DEEP_DIVE_FILENAME)
        conf_strat = read_json(out_dir / CONFIDENCE_STRATIFICATION_FILENAME)

        taxonomy_examples: list[dict[str, Any]] = taxonomy.get("examples", [])
        model_curated: list[dict[str, Any]] = []
        seen_texts: set[str] = set()

        for cat in ErrorCategory:
            cat_examples = [e for e in taxonomy_examples if e["primary_category"] == str(cat)]
            cat_examples.sort(key=lambda e: e.get("max_confidence", 0), reverse=True)
            added = 0
            for ex in cat_examples:
                if ex["text"] in seen_texts:
                    continue
                if added >= _EXAMPLES_PER_CATEGORY:
                    break
                model_curated.append(
                    {
                        "text": ex["text"],
                        "true_label_name": ex["true_label_name"],
                        "predicted_label_name": ex["predicted_label_name"],
                        "max_confidence": ex["max_confidence"],
                        "primary_category": ex["primary_category"],
                        "model_id": str(mid),
                        "annotation_tag": _annotation_tag(ErrorCategory(ex["primary_category"])),
                    }
                )
                seen_texts.add(ex["text"])
                added += 1

        _ensure_minimums(model_curated, seen_texts, str(mid), oos_dd, conf_strat, taxonomy_examples)

        _validate_traceability(model_curated, preds_df, str(mid))
        all_curated.extend(model_curated)

    if len(all_curated) < _MINIMUM_TOTAL_CURATED:
        logger.warning(
            "Only %d curated examples (minimum %d). Padding with additional examples.",
            len(all_curated),
            _MINIMUM_TOTAL_CURATED,
        )
        all_curated = _pad_to_minimum(all_curated, model_ids)

    curated_df = pd.DataFrame(all_curated)
    curated_df.to_csv(shared_dir / CURATED_EXAMPLES_CSV_FILENAME, index=False)
    logger.info("Saved: %s (%d examples)", shared_dir / CURATED_EXAMPLES_CSV_FILENAME, len(all_curated))

    artifact = artifact_envelope(
        total_curated=len(all_curated),
        models=sorted({e["model_id"] for e in all_curated}),
        examples=all_curated,
    )
    write_json(shared_dir / CURATED_EXAMPLES_JSON_FILENAME, artifact)
    logger.info("Saved: %s", shared_dir / CURATED_EXAMPLES_JSON_FILENAME)

    assert len(all_curated) >= _MINIMUM_TOTAL_CURATED, (
        f"Curated examples ({len(all_curated)}) below minimum ({_MINIMUM_TOTAL_CURATED})"
    )
    return artifact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _annotation_tag(cat: ErrorCategory) -> str:
    return {
        ErrorCategory.OOS_AS_INSCOPE: "confident false accept",
        ErrorCategory.INSCOPE_AS_OOS: "false rejection",
        ErrorCategory.NEAR_SEMANTIC_CONFUSION: "same-domain confusion",
        ErrorCategory.CROSS_DOMAIN_CONFUSION: "cross-domain mix-up",
        ErrorCategory.SHORT_QUERY_AMBIGUITY: "short query",
    }[cat]


def _ensure_minimums(
    curated: list[dict[str, Any]],
    seen_texts: set[str],
    model_id: str,
    oos_dd: dict[str, Any],
    conf_strat: dict[str, Any],
    taxonomy_examples: list[dict[str, Any]],
) -> None:
    """Pad curated list to satisfy per-model minimums."""
    # High-confidence errors
    hc_count = sum(
        1 for e in curated if e.get("max_confidence", 0) >= HIGH_CONFIDENCE_THRESHOLD and e["model_id"] == model_id
    )
    if hc_count < _MIN_HIGH_CONF_PER_MODEL:
        for he in conf_strat.get("high_confidence_errors", []):
            if he["text"] in seen_texts:
                continue
            curated.append(
                {
                    "text": he["text"],
                    "true_label_name": he["true_label_name"],
                    "predicted_label_name": he["predicted_label_name"],
                    "max_confidence": he["max_confidence"],
                    "primary_category": "high_confidence_error",
                    "model_id": model_id,
                    "annotation_tag": "confident wrong",
                }
            )
            seen_texts.add(he["text"])
            hc_count += 1
            if hc_count >= _MIN_HIGH_CONF_PER_MODEL:
                break

    # OOS false accepts
    fa_count = sum(
        1
        for e in curated
        if e.get("primary_category") == str(ErrorCategory.OOS_AS_INSCOPE) and e["model_id"] == model_id
    )
    if fa_count < _MIN_OOS_FA_PER_MODEL:
        for fa in oos_dd.get("false_accepts", {}).get("top_examples", []):
            if fa["text"] in seen_texts:
                continue
            curated.append(
                {
                    "text": fa["text"],
                    "true_label_name": fa["true_label_name"],
                    "predicted_label_name": fa["predicted_label_name"],
                    "max_confidence": fa["max_confidence"],
                    "primary_category": str(ErrorCategory.OOS_AS_INSCOPE),
                    "model_id": model_id,
                    "annotation_tag": "confident false accept",
                }
            )
            seen_texts.add(fa["text"])
            fa_count += 1
            if fa_count >= _MIN_OOS_FA_PER_MODEL:
                break

    # Same-domain confusions
    sem_count = sum(
        1
        for e in curated
        if e.get("primary_category") == str(ErrorCategory.NEAR_SEMANTIC_CONFUSION) and e["model_id"] == model_id
    )
    if sem_count < _MIN_SEMANTIC_PER_MODEL:
        for ex in taxonomy_examples:
            if ex["primary_category"] != str(ErrorCategory.NEAR_SEMANTIC_CONFUSION):
                continue
            if ex["text"] in seen_texts:
                continue
            curated.append(
                {
                    "text": ex["text"],
                    "true_label_name": ex["true_label_name"],
                    "predicted_label_name": ex["predicted_label_name"],
                    "max_confidence": ex["max_confidence"],
                    "primary_category": str(ErrorCategory.NEAR_SEMANTIC_CONFUSION),
                    "model_id": model_id,
                    "annotation_tag": _annotation_tag(ErrorCategory.NEAR_SEMANTIC_CONFUSION),
                }
            )
            seen_texts.add(ex["text"])
            sem_count += 1
            if sem_count >= _MIN_SEMANTIC_PER_MODEL:
                break


def _validate_traceability(
    curated: list[dict[str, Any]],
    preds_df: pd.DataFrame,
    model_id: str,
) -> None:
    """Verify every curated example traces to a real predictions row."""
    pred_texts = set(preds_df["text"].values)
    for ex in curated:
        if ex["model_id"] != model_id:
            continue
        if ex["text"] not in pred_texts:
            logger.warning("Curated example text not found in predictions: %s", ex["text"][:80])


def _pad_to_minimum(
    curated: list[dict[str, Any]],
    model_ids: list[ModelID],
) -> list[dict[str, Any]]:
    """Add more examples from taxonomy if below the minimum threshold."""
    seen_texts = {e["text"] for e in curated}
    for mid in model_ids:
        out_dir = analysis_output_dir(mid)
        taxonomy = read_json(out_dir / ERROR_TAXONOMY_FILENAME)
        for ex in taxonomy.get("examples", []):
            if len(curated) >= _MINIMUM_TOTAL_CURATED:
                return curated
            if ex["text"] in seen_texts:
                continue
            curated.append(
                {
                    "text": ex["text"],
                    "true_label_name": ex["true_label_name"],
                    "predicted_label_name": ex["predicted_label_name"],
                    "max_confidence": ex["max_confidence"],
                    "primary_category": ex["primary_category"],
                    "model_id": str(mid),
                    "annotation_tag": _annotation_tag(ErrorCategory(ex["primary_category"])),
                }
            )
            seen_texts.add(ex["text"])
    return curated
