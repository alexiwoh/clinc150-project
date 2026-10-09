"""Preflight validation and top-level orchestration for Step 10 error analysis."""

from __future__ import annotations

import logging
from typing import Any

from src.analysis.utils import (
    HandoffContext,
    load_confidences,
    load_predictions,
    read_json,
    resolve_handoff,
    validate_consistency,
)
from src.constants import (
    AGGREGATE_SUBDIR,
    PROJECT_ROOT,
    SHARED_DIR,
    model_output_dir,
)
from src.enums import ModelID
from src.report_figure_generation import FigureRecord
from src.run_ledger import load_current_generation

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

    # The analysis headline reads the all-model shared comparison table.
    load_current_generation(
        list(ModelID), model_dirs={mid: model_output_dir(mid) for mid in ModelID}, project_root=PROJECT_ROOT
    )
    for mid in model_ids:
        resolve_handoff(mid)
    logger.info("Step 10 preflight validation passed for %d model(s).", len(model_ids))


# ---------------------------------------------------------------------------
# Per-model analysis orchestration (Sections B-K)
# ---------------------------------------------------------------------------


def _run_per_model_analysis(ctx: HandoffContext) -> tuple[dict[str, dict[str, Any]], list[FigureRecord]]:
    """Run all per-model analyses for one model.

    Returns ``(section_artifacts, figure_records)`` where *section_artifacts*
    maps section keys to their saved artifact dicts.
    """
    from src.analysis.calibration import compute_and_save_calibration
    from src.analysis.class_analysis import compute_and_save_confusion_stability, compute_and_save_worst_classes
    from src.analysis.confidence import compute_and_save_confidence_stratification
    from src.analysis.extended_metrics import compute_and_save_extended_metrics
    from src.analysis.oos_analysis import compute_and_save_oos_deep_dive, compute_and_save_oos_threshold
    from src.analysis.slicing import compute_and_save_frequency_analysis, compute_and_save_length_analysis
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

    # Section H: Length and frequency slicing
    length = compute_and_save_length_analysis(ctx)
    section_artifacts["length_slice"] = length

    frequency = compute_and_save_frequency_analysis(ctx)
    section_artifacts["frequency_slice"] = frequency

    # Section J: Worst classes deep dive
    worst_classes, wc_records = compute_and_save_worst_classes(ctx)
    section_artifacts["worst_classes"] = worst_classes
    all_records.extend(wc_records)

    # Section K: Confusion stability
    stability, stab_records = compute_and_save_confusion_stability(ctx)
    section_artifacts["confusion_stability"] = stability
    all_records.extend(stab_records)

    return section_artifacts, all_records


# ---------------------------------------------------------------------------
# Shared comparison generation
# ---------------------------------------------------------------------------


def _run_shared_comparisons(
    model_ids: list[ModelID],
    all_model_artifacts: dict[str, dict[str, dict[str, Any]]],
) -> list[FigureRecord]:
    """Generate all shared cross-model comparison artifacts and figures."""
    from src.analysis.calibration import generate_calibration_comparison
    from src.analysis.class_analysis import generate_worst_classes_comparison
    from src.analysis.confidence import generate_confidence_comparison
    from src.analysis.extended_metrics import generate_extended_metrics_comparison
    from src.analysis.oos_analysis import generate_oos_error_comparison, generate_oos_threshold_comparison
    from src.analysis.slicing import generate_slice_comparisons
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

    # Section H: Slice comparisons
    length_data = {mid: all_model_artifacts[mid]["length_slice"] for mid in all_model_artifacts}
    freq_data = {mid: all_model_artifacts[mid]["frequency_slice"] for mid in all_model_artifacts}
    slice_records = generate_slice_comparisons(model_ids, length_data, freq_data)
    all_records.extend(slice_records)

    # Section J2: Worst classes comparison
    worst_data = {mid: all_model_artifacts[mid]["worst_classes"] for mid in all_model_artifacts}
    generate_worst_classes_comparison(model_ids, worst_data)

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

        # Per-model analyses (Sections B-K)
        for mid in model_ids:
            logger.info("Running per-model analysis for %s ...", mid.display_name)
            ctx = resolve_handoff(mid)
            section_artifacts, records = _run_per_model_analysis(ctx)
            all_model_artifacts[str(mid)] = section_artifacts
            all_records.extend(records)

        # Shared comparisons (Sections B-K cross-model)
        logger.info("Generating shared comparison artifacts ...")
        shared_records = _run_shared_comparisons(model_ids, all_model_artifacts)
        all_records.extend(shared_records)

        # Section I: Cross-model error comparison
        from src.analysis.cross_model import compute_and_save_cross_model_comparison

        logger.info("Running cross-model error comparison ...")
        _, cross_records = compute_and_save_cross_model_comparison(model_ids)
        all_records.extend(cross_records)

        # Section L: Curated report examples
        from src.analysis.curation import curate_report_examples

        logger.info("Curating report examples ...")
        curate_report_examples(model_ids)

        # Section M: Error analysis summary and notes
        from src.analysis.summary import generate_error_analysis_notes, generate_error_analysis_summary

        logger.info("Generating error analysis summary ...")
        generate_error_analysis_summary(model_ids)
        generate_error_analysis_notes(model_ids)

        # Section N: Step 11 handoff and figure manifest
        from src.analysis.handoff import (
            generate_step11_handoff,
            update_figure_manifest,
            validate_handoff_paths,
            validate_manifest_paths,
        )

        logger.info("Building Step 11 handoff ...")
        generate_step11_handoff(model_ids, all_records)
        update_figure_manifest(all_records)

        from src.analysis.utils import shared_analysis_dir
        from src.analysis.constants import STEP11_HANDOFF_FILENAME

        handoff_path = shared_analysis_dir() / STEP11_HANDOFF_FILENAME
        validate_handoff_paths(handoff_path)
        validate_manifest_paths(SHARED_DIR / "figure_manifest.json")

        logger.info(
            "Step 10 error analysis complete — %d figures generated, %d models analyzed.",
            len(all_records),
            len(model_ids),
        )
        return True

    except Exception:
        logger.exception("Step 10 error analysis failed.")
        return False
