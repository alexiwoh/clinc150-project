"""Figure catalogue generator (Spec Section I / I2).

Reads ``figure_manifest.json`` and produces an organized catalogue with
descriptive captions, report-section assignments, and scope tags for every
figure in the project.
"""

from __future__ import annotations

from typing import Any

from src.constants import PROTOCOL_VERSION, SCHEMA_VERSION
from src.enums import ModelID
from src.report.artifact_loader import load_json, resolve_repo_path

SectionOutput = tuple[str, dict[str, Any]]

_MANIFEST_PATH = "outputs/shared/figure_manifest.json"

# ---------------------------------------------------------------------------
# Figure-type -> report section mapping
# ---------------------------------------------------------------------------

_SECTION_MAP: dict[str, str] = {
    "train_val_loss_curve": "Training Diagnostics",
    "val_macro_f1_curve": "Training Diagnostics",
    "val_accuracy_curve": "Training Diagnostics",
    "confusion_matrix": "Confusion Analysis",
    "top_confused_pairs": "Confusion Analysis",
    "bottom_classes_f1": "Confusion Analysis",
    "oos_metrics": "OOS Detection",
    "error_summary": "Error Analysis",
    "model_comparison_test_accuracy": "Model Comparison",
    "model_comparison_test_macro_f1": "Model Comparison",
    "model_comparison_oos_f1": "Model Comparison",
    "oos_metrics_comparison": "Model Comparison",
    "model_efficiency_comparison": "Efficiency",
    "reliability_diagram": "Error Analysis",
    "confidence_histogram": "Error Analysis",
    "oos_roc_curve": "OOS Detection",
    "oos_pr_curve": "OOS Detection",
    "oos_error_breakdown": "OOS Detection",
    "confidence_vs_accuracy": "Error Analysis",
    "worst_classes_heatmap": "Error Analysis",
    "confusion_stability": "Error Analysis",
    "calibration_comparison": "Error Analysis",
    "oos_roc_comparison": "OOS Detection",
    "oos_pr_comparison": "OOS Detection",
    "error_taxonomy": "Error Analysis",
    "oos_error_comparison": "OOS Detection",
    "length_slice": "Error Analysis",
    "frequency_slice": "Error Analysis",
    "cross_model_error_overlap": "Error Analysis",
}

_SECTION_ORDER: list[str] = [
    "Training Diagnostics",
    "Model Comparison",
    "OOS Detection",
    "Confusion Analysis",
    "Error Analysis",
    "Efficiency",
]

# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------

_TYPE_DESCRIPTIONS: dict[str, str] = {
    "train_val_loss_curve": "Training and validation loss curves",
    "val_macro_f1_curve": "Validation macro F1 progression during training",
    "val_accuracy_curve": "Validation accuracy progression during training",
    "confusion_matrix": "Test-set confusion matrix across 151 intent classes including OOS",
    "top_confused_pairs": "Top confused intent pairs on the test set",
    "bottom_classes_f1": "Bottom classes by test F1 score",
    "oos_metrics": "OOS detection metrics summary (precision, recall, F1)",
    "error_summary": "Most frequent misclassification categories on the test set",
    "model_comparison_test_accuracy": "Aggregate test accuracy comparison with error bars",
    "model_comparison_test_macro_f1": "Aggregate test macro F1 comparison with error bars",
    "model_comparison_oos_f1": "Aggregate OOS F1 comparison with error bars",
    "oos_metrics_comparison": "Aggregate OOS precision, recall, and F1 comparison",
    "model_efficiency_comparison": "Model efficiency comparison (parameters, training time, throughput)",
    "reliability_diagram": "Reliability diagram showing calibration quality (predicted confidence vs actual accuracy)",
    "confidence_histogram": "Prediction confidence distribution for correct and incorrect predictions",
    "oos_roc_curve": "OOS detection ROC curve (explicit OOS class probability method)",
    "oos_pr_curve": "OOS detection precision-recall curve",
    "oos_error_breakdown": "OOS error breakdown: false accepts (OOS as in-scope) vs false rejects",
    "confidence_vs_accuracy": "Confidence vs accuracy analysis across confidence bins",
    "worst_classes_heatmap": "Confusion heatmap for worst-performing intent classes",
    "confusion_stability": "Confusion stability across repeated runs (3 seeds)",
    "calibration_comparison": "Calibration comparison (ECE, MCE, Brier, NLL) across models",
    "oos_roc_comparison": "OOS ROC curve comparison overlay across models",
    "oos_pr_comparison": "OOS precision-recall curve comparison overlay across models",
    "error_taxonomy": "Error taxonomy comparison (OOS-as-inscope, inscope-as-OOS, semantic, cross-domain) per model",
    "oos_error_comparison": "OOS false-accept and false-reject pattern comparison across models",
    "length_slice": "Accuracy by query length (short/medium/long) across models",
    "frequency_slice": "Accuracy by class frequency across models",
    "cross_model_error_overlap": "Cross-model error overlap: examples wrong by all/some/one model",
}


def _build_caption(fig: dict[str, Any]) -> str:
    """Generate a self-contained caption for a figure entry."""
    ft: str = fig["figure_type"]
    scope: str = fig.get("scope", "representative")
    model_id: str | None = fig.get("model_name")

    model_str = ModelID(model_id).display_name if model_id else "all models"

    if scope == "representative":
        run_id = fig.get("representative_run_id", "")
        scope_str = f"representative run ({run_id})" if run_id else "representative run"
    elif scope == "aggregate":
        scope_str = "aggregate over 3 repeated runs"
    else:
        scope_str = scope

    desc = _TYPE_DESCRIPTIONS.get(ft, ft.replace("_", " ").title())
    return f"{desc} for {model_str}. Data: {scope_str}, CLINC150 test set."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_figure_catalogue() -> SectionOutput:
    """Spec section I / I2."""
    manifest = load_json(resolve_repo_path(_MANIFEST_PATH))
    figures: list[dict[str, Any]] = manifest["figures"]

    # Group by report section
    by_section: dict[str, list[dict[str, Any]]] = {s: [] for s in _SECTION_ORDER}

    for fig in figures:
        section = _SECTION_MAP.get(fig["figure_type"], "Error Analysis")
        entry = {
            "figure_path": fig["figure_path"],
            "caption": _build_caption(fig),
            "scope": fig.get("scope", "representative"),
            "model_name": fig.get("model_name"),
            "figure_type": fig["figure_type"],
        }
        by_section.setdefault(section, []).append(entry)

    # Build markdown
    md_parts: list[str] = ["## Figure Catalogue", ""]

    for section_name in _SECTION_ORDER:
        section_figs = by_section.get(section_name, [])
        if not section_figs:
            continue
        md_parts.append(f"### {section_name}")
        md_parts.append("")
        for entry in section_figs:
            md_parts.append(f"- **{entry['figure_type']}** ({entry['scope']}): {entry['caption']}")
            md_parts.append(f"  Path: `{entry['figure_path']}`")
        md_parts.append("")

    md_parts.append(f"Total figures: {len(figures)}")
    md_parts.append("")

    # Build metadata
    sections_json = [
        {
            "section_name": section_name,
            "figures": by_section.get(section_name, []),
        }
        for section_name in _SECTION_ORDER
        if by_section.get(section_name)
    ]

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "total_figure_count": len(figures),
        "sections": sections_json,
        "source_artifact": _MANIFEST_PATH,
    }

    return "\n".join(md_parts), metadata
