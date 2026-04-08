"""Section M: Error analysis summary and report discussion notes."""

from __future__ import annotations

import logging
from typing import Any

from src.analysis.constants import (
    CALIBRATION_SUMMARY_FILENAME,
    CROSS_MODEL_ERROR_COMPARISON_FILENAME,
    ERROR_ANALYSIS_NOTES_FILENAME,
    ERROR_ANALYSIS_SUMMARY_FILENAME,
    ERROR_TAXONOMY_SUMMARY_FILENAME,
    LENGTH_SLICE_COMPARISON_FILENAME,
    OOS_THRESHOLD_COMPARISON_FILENAME,
)
from src.analysis.utils import (
    artifact_envelope,
    read_json,
    shared_analysis_dir,
    write_json,
)
from src.constants import SHARED_DIR
from src.enums import ModelID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------


def generate_error_analysis_summary(model_ids: list[ModelID]) -> dict[str, Any]:
    """Produce structured summary tying together all analyses."""
    shared_dir = shared_analysis_dir()

    cal_summary = read_json(shared_dir / CALIBRATION_SUMMARY_FILENAME)
    oos_summary = read_json(shared_dir / OOS_THRESHOLD_COMPARISON_FILENAME)
    taxonomy_summary = read_json(shared_dir / ERROR_TAXONOMY_SUMMARY_FILENAME)
    cross_model = read_json(shared_dir / CROSS_MODEL_ERROR_COMPARISON_FILENAME)
    agg_table = read_json(SHARED_DIR / "model_comparison_aggregate.json")

    # Headline: best overall model by macro F1
    agg_rows = agg_table.get("rows", [])
    best_agg = max(agg_rows, key=lambda r: r.get("test_macro_f1_mean", 0)) if agg_rows else {}
    headline = {
        "best_model": best_agg.get("model_id", "unknown"),
        "test_macro_f1_mean": best_agg.get("test_macro_f1_mean"),
        "test_macro_f1_std": best_agg.get("test_macro_f1_std"),
        "rationale": "Highest aggregate test macro F1 across repeated runs.",
        "source_artifact": "outputs/shared/model_comparison_aggregate.json",
    }

    # Calibration
    cal_rows = cal_summary.get("rows", [])
    best_cal = min(cal_rows, key=lambda r: r.get("ece", float("inf"))) if cal_rows else {}
    worst_cal = max(cal_rows, key=lambda r: r.get("ece", 0)) if cal_rows else {}
    calibration_finding = {
        "best_calibrated": best_cal.get("model_id"),
        "best_ece": best_cal.get("ece"),
        "worst_calibrated": worst_cal.get("model_id"),
        "worst_ece": worst_cal.get("ece"),
        "source_artifact": f"outputs/shared/analysis/{CALIBRATION_SUMMARY_FILENAME}",
    }

    # OOS
    oos_rows = oos_summary.get("rows", [])
    best_oos = max(oos_rows, key=lambda r: r.get("auroc", 0)) if oos_rows else {}
    oos_finding = {
        "best_model": best_oos.get("model_id"),
        "best_auroc": best_oos.get("auroc"),
        "best_aupr": best_oos.get("aupr"),
        "source_artifact": f"outputs/shared/analysis/{OOS_THRESHOLD_COMPARISON_FILENAME}",
    }

    # Confusion
    taxonomy_rows = taxonomy_summary.get("rows", [])
    dominant_cats: dict[str, str] = {}
    for row in taxonomy_rows:
        from src.analysis.enums import ErrorCategory

        cat_fracs = {cat: row.get(f"{cat}_fraction", 0) for cat in ErrorCategory}
        dominant_cats[row["model_id"]] = max(cat_fracs, key=lambda c: cat_fracs[c])

    confusion_finding = {
        "dominant_error_categories": dominant_cats,
        "source_artifact": f"outputs/shared/analysis/{ERROR_TAXONOMY_SUMMARY_FILENAME}",
    }

    # Architecture comparison
    cats = cross_model.get("categories", {})
    arch_finding = {
        "all_wrong_count": cats.get("all_wrong", {}).get("count"),
        "all_wrong_fraction": cats.get("all_wrong", {}).get("fraction"),
        "model_specific_count": cats.get("model_specific_error", {}).get("count"),
        "model_specific_fraction": cats.get("model_specific_error", {}).get("fraction"),
        "all_wrong_agreement": cross_model.get("all_wrong_agreement"),
        "source_artifact": f"outputs/shared/analysis/{CROSS_MODEL_ERROR_COMPARISON_FILENAME}",
    }

    # Length finding
    length_finding: dict[str, Any] = {"source_artifact": f"outputs/shared/analysis/{LENGTH_SLICE_COMPARISON_FILENAME}"}
    for mid in model_ids:
        from src.analysis.constants import LENGTH_SLICE_FILENAME
        from src.analysis.utils import analysis_output_dir

        length_path = analysis_output_dir(mid) / LENGTH_SLICE_FILENAME
        if length_path.exists():
            length_data = read_json(length_path)
            slices = length_data.get("slices", [])
            short_slice = next((s for s in slices if s["slice"] == "short"), None)
            if short_slice:
                length_finding[str(mid)] = {
                    "short_accuracy": short_slice["accuracy"],
                    "short_error_rate": short_slice["error_rate"],
                }

    recommendation = (
        "Focus on improving OOS detection and addressing semantically ambiguous intent pairs "
        "within the same domain. Consider intent merging for persistently confused pairs and "
        "confidence thresholding for high-confidence errors."
    )

    artifact = artifact_envelope(
        headline=headline,
        calibration=calibration_finding,
        oos_detection=oos_finding,
        confusion=confusion_finding,
        architecture_comparison=arch_finding,
        length=length_finding,
        top_recommendation=recommendation,
    )
    write_json(shared_dir / ERROR_ANALYSIS_SUMMARY_FILENAME, artifact)
    logger.info("Saved: %s", shared_dir / ERROR_ANALYSIS_SUMMARY_FILENAME)
    return artifact


# ---------------------------------------------------------------------------
# Discussion notes (Markdown)
# ---------------------------------------------------------------------------


def generate_error_analysis_notes(model_ids: list[ModelID]) -> None:
    """Generate human-readable discussion notes for report section."""
    shared_dir = shared_analysis_dir()
    summary = read_json(shared_dir / ERROR_ANALYSIS_SUMMARY_FILENAME)

    headline = summary.get("headline", {})
    cal = summary.get("calibration", {})
    oos = summary.get("oos_detection", {})
    confusion = summary.get("confusion", {})
    arch = summary.get("architecture_comparison", {})
    recommendation = summary.get("top_recommendation", "N/A")

    lines: list[str] = [
        "# Error Analysis Discussion Notes",
        "",
        "## Overview",
        "",
        f"The best-performing model overall is **{headline.get('best_model', 'N/A')}** "
        f"with aggregate test macro F1 of {_fmt(headline.get('test_macro_f1_mean'))} "
        f"(±{_fmt(headline.get('test_macro_f1_std'))}) across repeated runs. "
        "This analysis examines calibration, OOS detection, systematic confusions, "
        "and architecture-specific failure modes to understand not just how well each model "
        "performs, but why and where it fails.",
        "",
        "## Calibration and Confidence",
        "",
        f"The best-calibrated model is **{cal.get('best_calibrated', 'N/A')}** "
        f"(ECE = {_fmt(cal.get('best_ece'))}), while the worst-calibrated is "
        f"**{cal.get('worst_calibrated', 'N/A')}** (ECE = {_fmt(cal.get('worst_ece'))}). "
        f"See `{cal.get('source_artifact', '')}` for full calibration comparison.",
        "",
        "## OOS Detection Quality",
        "",
        f"The best OOS detector is **{oos.get('best_model', 'N/A')}** "
        f"with AUROC = {_fmt(oos.get('best_auroc'))} and AUPR = {_fmt(oos.get('best_aupr'))}. "
        f"See `{oos.get('source_artifact', '')}` for threshold comparison.",
        "",
        "## Systematic Confusions",
        "",
    ]

    dom_cats = confusion.get("dominant_error_categories", {})
    for mid_str, cat in dom_cats.items():
        lines.append(f"- **{mid_str}**: dominant error category is `{cat}`")
    lines.append("")
    lines.append(f"See `{confusion.get('source_artifact', '')}` for per-model taxonomy.")
    lines.append("")

    lines.extend(
        [
            "## Architecture-Specific vs Shared Errors",
            "",
            f"Across all models, **{arch.get('all_wrong_count', 'N/A')}** examples "
            f"({_fmt_pct(arch.get('all_wrong_fraction'))}) are misclassified by all three models, "
            f"while **{arch.get('model_specific_count', 'N/A')}** examples "
            f"({_fmt_pct(arch.get('model_specific_fraction'))}) are unique to a single model. "
            f"Of universally wrong examples, {_fmt_pct(arch.get('all_wrong_agreement'))} "
            "predict the same incorrect class. "
            f"See `{arch.get('source_artifact', '')}` for overlap analysis.",
            "",
            "## Limitations",
            "",
            "- Qualitative analysis uses a single representative run per model, not all seeds.",
            "- CLINC150 is balanced; real-world class distributions may differ significantly.",
            "- No interpretability analysis (attention, saliency) was performed.",
            "- Post-hoc calibration (e.g., temperature scaling) was not applied.",
            "- Length and frequency slicing uses simple whitespace tokenization.",
            "",
            "## Recommendations",
            "",
            f"- **Top recommendation**: {recommendation}",
            "- Collect more OOS training examples to reduce false accepts.",
            "- Consider merging or relabeling persistently confused intent pairs within the same domain.",
            "- Apply confidence thresholding in deployment to flag uncertain predictions for human review.",
            "- Evaluate temperature scaling to improve calibration without retraining.",
            "",
        ]
    )

    notes_path = shared_dir / ERROR_ANALYSIS_NOTES_FILENAME
    notes_path.write_text("\n".join(lines))
    logger.info("Saved: %s", notes_path)


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value * 100:.1f}%"
    return str(value)
