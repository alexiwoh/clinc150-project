"""Figure catalogue generator (Spec Section I / I2).

Reads ``figure_manifest.json`` and produces an organized catalogue with
descriptive captions, report-section assignments, and scope tags for every
figure in the project.
"""

from __future__ import annotations

from typing import Any

from src.constants import PROTOCOL_VERSION, SCHEMA_VERSION
from src.enums import ModelID
from src.report.figure_metadata import FigureCaptionContext, FigureManifestEntry, load_figure_manifest_entries

SectionOutput = tuple[str, dict[str, Any]]
CatalogueEntry = dict[str, Any]

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
    "train_val_loss_curve": "Training and validation loss by epoch",
    "val_macro_f1_curve": "Validation macro F1 by training epoch",
    "val_accuracy_curve": "Validation accuracy by training epoch",
    "confusion_matrix": "Confusion matrix for intent predictions",
    "top_confused_pairs": "Most frequent true-label and predicted-label confusion pairs",
    "bottom_classes_f1": "Lowest-F1 intent classes in the representative evaluation",
    "oos_metrics": "Representative-run OOS precision, recall, and F1 summary",
    "error_summary": "Representative-run summary of the most frequent error patterns",
    "model_comparison_test_accuracy": "Cross-model test accuracy comparison with error bars",
    "model_comparison_test_macro_f1": "Cross-model test macro F1 comparison with error bars",
    "model_comparison_oos_f1": "Cross-model OOS F1 comparison with error bars",
    "oos_metrics_comparison": "Cross-model OOS precision, recall, and F1 comparison",
    "model_efficiency_comparison": "Cross-model efficiency comparison",
    "reliability_diagram": "Reliability diagram comparing confidence with empirical accuracy",
    "confidence_histogram": "Confidence distribution for correct versus incorrect predictions",
    "oos_roc_curve": "ROC curve for explicit-OOS detection",
    "oos_pr_curve": "Precision-recall curve for explicit-OOS detection",
    "oos_error_breakdown": "False-accept and false-reject intent breakdown for OOS analysis",
    "confidence_vs_accuracy": "Accuracy across confidence buckets",
    "worst_classes_heatmap": "Confusion heatmap for the weakest intent classes",
    "confusion_stability": "Run-to-run stability of the highest-count confusion pairs",
    "calibration_comparison": "Cross-model calibration comparison",
    "oos_roc_comparison": "Cross-model ROC comparison for explicit-OOS detection",
    "oos_pr_comparison": "Cross-model precision-recall comparison for explicit-OOS detection",
    "error_taxonomy": "Cross-model comparison of error-taxonomy fractions",
    "oos_error_comparison": "Cross-model false-accept versus false-reject comparison",
    "length_slice": "Cross-model accuracy comparison by utterance length",
    "frequency_slice": "Cross-model accuracy comparison by class-frequency tier",
    "cross_model_error_overlap": "Overlap between shared and model-specific prediction failures",
}


def _build_caption(fig: FigureManifestEntry) -> str:
    """Generate a self-contained caption for a figure entry."""
    figure_type = fig["figure_type"]
    context = fig.get("caption_context", {})
    intro = _caption_intro(fig, context)
    context_parts = [
        _basis_sentence(fig, context),
        _dataset_sentence(context),
        _detail_sentence(figure_type, context),
    ]
    context_parts = [part for part in context_parts if part]
    if not context_parts:
        return intro
    return " ".join([intro, *context_parts])


def _caption_intro(fig: FigureManifestEntry, context: FigureCaptionContext) -> str:
    figure_type = fig["figure_type"]
    description = _TYPE_DESCRIPTIONS.get(figure_type, figure_type.replace("_", " ").replace("-", " ").title())
    model_id = fig.get("model_name")
    if model_id:
        return f"{description} for {ModelID(model_id).display_name}."
    return f"{description} across all evaluated models."


def _basis_sentence(fig: FigureManifestEntry, context: FigureCaptionContext) -> str | None:
    basis = context.get("analysis_basis")
    scope = fig.get("scope", "representative")
    run_id = fig.get("representative_run_id")
    selection_rule = context.get("selection_rule")
    if scope == "representative":
        if run_id is None:
            return None
        sentence = f"Computed from representative run `{run_id}`"
        if selection_rule:
            sentence += f", selected by {selection_rule.replace('_', ' ')}"
        return sentence + "."
    if scope == "aggregate":
        run_count = context.get("run_count")
        if run_count is not None:
            return f"Uses aggregate statistics over {run_count} repeated runs per model."
        return "Uses aggregate statistics over repeated runs."
    if basis is None:
        return None
    if run_id is not None and "representative-run" in basis:
        return f"Computed from {basis} for representative run `{run_id}`."
    return f"Computed from {basis}."


def _dataset_sentence(context: FigureCaptionContext) -> str | None:
    dataset_name = context.get("dataset_name")
    dataset_split = context.get("dataset_split")
    if dataset_name and dataset_split:
        return f"Uses the {dataset_name} {dataset_split} split."
    if dataset_name:
        return f"Uses {dataset_name} artifacts."
    return None


def _detail_sentence(figure_type: str, context: FigureCaptionContext) -> str | None:
    metric_names = context.get("metric_names", [])
    class_count = context.get("class_count")
    run_count = context.get("run_count")
    top_k = context.get("top_k")

    if figure_type == "confusion_matrix" and class_count is not None:
        return f"Covers {class_count} intent classes including OOS."
    if figure_type == "top_confused_pairs" and top_k is not None:
        return f"Highlights the top {top_k} confusion pairs."
    if figure_type == "error_summary" and top_k is not None:
        return f"Summarizes {top_k} ranked error examples or categories."
    if figure_type == "confusion_stability" and run_count is not None:
        if top_k is not None:
            return f"Tracks the top {top_k} confusion pairs across {run_count} completed runs."
        return f"Tracks run-to-run confusion stability across {run_count} completed runs."
    if figure_type == "model_efficiency_comparison" and metric_names:
        return f"Metrics shown: {', '.join(metric_names)}."
    if metric_names and figure_type in {
        "oos_metrics",
        "oos_metrics_comparison",
        "calibration_comparison",
        "oos_error_comparison",
    }:
        return f"Metrics shown: {', '.join(metric_names)}."
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_figure_catalogue() -> SectionOutput:
    """Spec section I / I2."""
    figures = load_figure_manifest_entries()

    # Group by report section
    by_section: dict[str, list[CatalogueEntry]] = {s: [] for s in _SECTION_ORDER}

    for fig in figures:
        section = _SECTION_MAP.get(fig["figure_type"], "Error Analysis")
        entry = {
            "figure_path": fig["figure_path"],
            "caption": _build_caption(fig),
            "scope": fig.get("scope", "representative"),
            "model_name": fig.get("model_name"),
            "figure_type": fig["figure_type"],
            "source_artifact_paths": list(fig.get("source_artifact_paths", [])),
            "caption_context": dict(fig.get("caption_context", {})),
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
            md_parts.append(f"  Figure path: `{entry['figure_path']}`")
            sources = entry.get("source_artifact_paths", [])
            if sources:
                joined = ", ".join(f"`{path}`" for path in sources)
                md_parts.append(f"  Source artifacts: {joined}")
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
