"""Preflight validation and top-level orchestration for Step 10 error analysis."""

from __future__ import annotations

import logging
from typing import Any

from src.analysis.utils import (
    HandoffContext,
    read_json,
    resolve_handoff,
    validate_consistency,
    load_confidences,
    load_predictions,
)
from src.constants import (
    AGGREGATE_SUBDIR,
    SHARED_DIR,
    model_output_dir,
    PROJECT_ROOT,
)
from src.enums import ModelID
from src.report_figure_generation import FigureRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preflight validation (Section A)
# ---------------------------------------------------------------------------

_SHARED_REQUIRED = (
    "model_comparison_aggregate.json",
    "oos_summary_table.json",
    "most_confused_pairs_table.json",
    "representative_examples_index.json",
)

_PER_RUN_REQUIRED = (
    "final_predictions.csv",
    "confidences.npz",
    "per_class_metrics.json",
    "confusion_matrix.csv",
    "top_confusions.json",
    "top_errors.json",
    "label_order.json",
)


def run_preflight_validation(model_ids: list[ModelID]) -> None:
    """Verify all required input artifacts exist before running any analysis.

    Raises ``FileNotFoundError`` with a summary of all missing paths.
    """
    missing: list[str] = []

    for shared_name in _SHARED_REQUIRED:
        p = SHARED_DIR / shared_name
        if not p.exists():
            missing.append(str(p))

    for mid in model_ids:
        handoff_path = model_output_dir(mid) / AGGREGATE_SUBDIR / "error_analysis_handoff.json"
        if not handoff_path.exists():
            missing.append(str(handoff_path))
            continue

        handoff = read_json(handoff_path)
        artifacts: dict[str, str] = handoff["artifacts"]
        run_dir = (PROJECT_ROOT / artifacts["final_predictions"]).parent

        for rel in _PER_RUN_REQUIRED:
            p = run_dir / rel
            if not p.exists():
                missing.append(str(p))

    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(
            f"Step 10 preflight failed — {len(missing)} required artifact(s) missing:\n  {formatted}"
        )

    logger.info("Step 10 preflight validation passed for %d model(s).", len(model_ids))


# ---------------------------------------------------------------------------
# Per-model analysis orchestration (Sections B-G)
# ---------------------------------------------------------------------------


def _run_per_model_analysis(ctx: HandoffContext) -> tuple[dict[str, dict[str, Any]], list[FigureRecord]]:
    """Run all per-model analyses for one model.

    Returns ``(section_artifacts, figure_records)`` where *section_artifacts*
    maps section keys to their saved artifact dicts.
    """
    from src.analysis.calibration import compute_and_save_calibration
    from src.analysis.confidence import compute_and_save_confidence_stratification
    from src.analysis.extended_metrics import compute_and_save_extended_metrics
    from src.analysis.oos_analysis import compute_and_save_oos_deep_dive, compute_and_save_oos_threshold
    from src.analysis.taxonomy import compute_and_save_error_taxonomy

    conf = load_confidences(ctx.confidences_path)
    preds_df = load_predictions(ctx.final_predictions_path)
    validate_consistency(conf, preds_df)

    all_records: list[FigureRecord] = []
    section_artifacts: dict[str, dict[str, Any]] = {}

    # Section B: Extended metrics
    extended = compute_and_save_extended_metrics(ctx)
    section_artifacts["extended_metrics"] = extended

    # Section C: Calibration
    calibration, cal_records = compute_and_save_calibration(ctx)
    section_artifacts["calibration"] = calibration
    all_records.extend(cal_records)

    # Section D: OOS threshold
    oos_threshold, oos_records = compute_and_save_oos_threshold(ctx)
    section_artifacts["oos_threshold"] = oos_threshold
    all_records.extend(oos_records)

    # Section E: Error taxonomy
    taxonomy = compute_and_save_error_taxonomy(ctx)
    section_artifacts["taxonomy"] = taxonomy

    # Section F: OOS deep dive
    oos_deep_dive, oos_dd_records = compute_and_save_oos_deep_dive(ctx)
    section_artifacts["oos_deep_dive"] = oos_deep_dive
    all_records.extend(oos_dd_records)

    # Section G: Confidence stratification
    conf_strat, conf_records = compute_and_save_confidence_stratification(ctx)
    section_artifacts["confidence_stratification"] = conf_strat
    all_records.extend(conf_records)

    return section_artifacts, all_records


# ---------------------------------------------------------------------------
# Shared comparison generation (Sections B-G cross-model)
# ---------------------------------------------------------------------------


def _run_shared_comparisons(
    model_ids: list[ModelID],
    all_model_artifacts: dict[str, dict[str, dict[str, Any]]],
) -> list[FigureRecord]:
    """Generate all shared cross-model comparison artifacts and figures for Phase 1."""
    from src.analysis.calibration import generate_calibration_comparison
    from src.analysis.confidence import generate_confidence_comparison
    from src.analysis.extended_metrics import generate_extended_metrics_comparison
    from src.analysis.oos_analysis import generate_oos_error_comparison, generate_oos_threshold_comparison
    from src.analysis.taxonomy import generate_taxonomy_summary, save_intent_domain_mapping

    all_records: list[FigureRecord] = []

    # Intent domain mapping
    save_intent_domain_mapping()

    # Section B: Extended metrics comparison
    ext_metrics = {mid: all_model_artifacts[mid]["extended_metrics"] for mid in all_model_artifacts}
    generate_extended_metrics_comparison(model_ids, ext_metrics)

    # Section C: Calibration comparison
    cal_metrics = {mid: all_model_artifacts[mid]["calibration"] for mid in all_model_artifacts}
    cal_records = generate_calibration_comparison(model_ids, cal_metrics)
    all_records.extend(cal_records)

    # Section D: OOS threshold comparison
    oos_metrics = {mid: all_model_artifacts[mid]["oos_threshold"] for mid in all_model_artifacts}
    oos_records = generate_oos_threshold_comparison(model_ids, oos_metrics)
    all_records.extend(oos_records)

    # Section E: Taxonomy summary
    tax_metrics = {mid: all_model_artifacts[mid]["taxonomy"] for mid in all_model_artifacts}
    tax_records = generate_taxonomy_summary(model_ids, tax_metrics)
    all_records.extend(tax_records)

    # Section F: OOS error comparison
    oos_dd = {mid: all_model_artifacts[mid]["oos_deep_dive"] for mid in all_model_artifacts}
    oos_err_records = generate_oos_error_comparison(model_ids, oos_dd)
    all_records.extend(oos_err_records)

    # Section G: Confidence comparison
    conf_strat = {mid: all_model_artifacts[mid]["confidence_stratification"] for mid in all_model_artifacts}
    conf_records = generate_confidence_comparison(model_ids, conf_strat)
    all_records.extend(conf_records)

    return all_records


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_error_analysis(model_ids: list[ModelID]) -> bool:
    """Run the full Step 10 error analysis pipeline.

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        run_preflight_validation(model_ids)

        all_records: list[FigureRecord] = []
        all_model_artifacts: dict[str, dict[str, dict[str, Any]]] = {}

        # Phase 1: Per-model analyses (Sections B-G)
        for mid in model_ids:
            logger.info("Running per-model analysis for %s ...", mid.display_name)
            ctx = resolve_handoff(mid)
            section_artifacts, records = _run_per_model_analysis(ctx)
            all_model_artifacts[str(mid)] = section_artifacts
            all_records.extend(records)

        # Phase 1: Shared comparisons
        logger.info("Generating shared comparison artifacts ...")
        shared_records = _run_shared_comparisons(model_ids, all_model_artifacts)
        all_records.extend(shared_records)

        # Phase 2 hooks (slicing, cross-model, class analysis, curation, summary, handoff)
        # will be wired here in Phase 2 implementation

        logger.info(
            "Step 10 error analysis complete — %d figures generated, %d models analyzed.",
            len(all_records),
            len(model_ids),
        )
        return True

    except Exception:
        logger.exception("Step 10 error analysis failed.")
        return False
