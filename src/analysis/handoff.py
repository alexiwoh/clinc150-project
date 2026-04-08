"""Section N: Step 11 handoff generation and figure manifest update."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.analysis.constants import (
    CALIBRATION_COMPARISON_FILENAME,
    CALIBRATION_METRICS_FILENAME,
    CALIBRATION_SUMMARY_FILENAME,
    CONFIDENCE_ACCURACY_COMPARISON_FILENAME,
    CONFIDENCE_HISTOGRAM_FILENAME,
    CONFIDENCE_STRATIFICATION_FILENAME,
    CONFIDENCE_VS_ACCURACY_FILENAME,
    CROSS_MODEL_ERROR_COMPARISON_FILENAME,
    CROSS_MODEL_ERROR_OVERLAP_FILENAME,
    CURATED_EXAMPLES_CSV_FILENAME,
    CURATED_EXAMPLES_JSON_FILENAME,
    ERROR_ANALYSIS_NOTES_FILENAME,
    ERROR_ANALYSIS_SUMMARY_FILENAME,
    ERROR_TAXONOMY_COMPARISON_FILENAME,
    ERROR_TAXONOMY_FILENAME,
    ERROR_TAXONOMY_SUMMARY_FILENAME,
    EXTENDED_METRICS_COMPARISON_FILENAME,
    EXTENDED_TEST_METRICS_FILENAME,
    FREQUENCY_SLICE_COMPARISON_FILENAME,
    FREQUENCY_SLICE_FILENAME,
    INTENT_DOMAIN_MAPPING_FILENAME,
    LENGTH_SLICE_COMPARISON_FILENAME,
    LENGTH_SLICE_FILENAME,
    OOS_ERROR_BREAKDOWN_FILENAME,
    OOS_ERROR_COMPARISON_FILENAME,
    OOS_ERROR_DEEP_DIVE_FILENAME,
    OOS_FALSE_ACCEPT_COMPARISON_FILENAME,
    OOS_PR_COMPARISON_FILENAME,
    OOS_PR_CURVE_FILENAME,
    OOS_ROC_COMPARISON_FILENAME,
    OOS_ROC_CURVE_FILENAME,
    OOS_THRESHOLD_COMPARISON_FILENAME,
    OOS_THRESHOLD_METRICS_FILENAME,
    RELIABILITY_DIAGRAM_FILENAME,
    STEP11_HANDOFF_FILENAME,
    UNIVERSALLY_MISCLASSIFIED_FILENAME,
    WORST_CLASSES_COMPARISON_FILENAME,
    WORST_CLASSES_FILENAME,
    WORST_CLASSES_HEATMAP_FILENAME,
    CONFUSION_STABILITY_FILENAME,
    CONFUSION_STABILITY_FIGURE_FILENAME,
)
from src.analysis.utils import (
    analysis_output_dir,
    artifact_envelope,
    read_json,
    repo_relative,
    shared_analysis_dir,
    write_json,
)
from src.constants import PROJECT_ROOT, SHARED_DIR
from src.enums import ModelID
from src.report_figure_generation import FigureRecord

logger = logging.getLogger(__name__)

_PER_MODEL_JSON_FILES: tuple[str, ...] = (
    EXTENDED_TEST_METRICS_FILENAME,
    CALIBRATION_METRICS_FILENAME,
    OOS_THRESHOLD_METRICS_FILENAME,
    ERROR_TAXONOMY_FILENAME,
    OOS_ERROR_DEEP_DIVE_FILENAME,
    CONFIDENCE_STRATIFICATION_FILENAME,
    LENGTH_SLICE_FILENAME,
    FREQUENCY_SLICE_FILENAME,
    WORST_CLASSES_FILENAME,
    CONFUSION_STABILITY_FILENAME,
)

_PER_MODEL_FIGURE_FILES: tuple[str, ...] = (
    RELIABILITY_DIAGRAM_FILENAME,
    CONFIDENCE_HISTOGRAM_FILENAME,
    OOS_ROC_CURVE_FILENAME,
    OOS_PR_CURVE_FILENAME,
    OOS_ERROR_BREAKDOWN_FILENAME,
    CONFIDENCE_VS_ACCURACY_FILENAME,
    WORST_CLASSES_HEATMAP_FILENAME,
    CONFUSION_STABILITY_FIGURE_FILENAME,
)

_SHARED_JSON_FILES: tuple[str, ...] = (
    EXTENDED_METRICS_COMPARISON_FILENAME,
    CALIBRATION_SUMMARY_FILENAME,
    OOS_THRESHOLD_COMPARISON_FILENAME,
    INTENT_DOMAIN_MAPPING_FILENAME,
    ERROR_TAXONOMY_SUMMARY_FILENAME,
    OOS_FALSE_ACCEPT_COMPARISON_FILENAME,
    CROSS_MODEL_ERROR_COMPARISON_FILENAME,
    WORST_CLASSES_COMPARISON_FILENAME,
    CURATED_EXAMPLES_JSON_FILENAME,
    ERROR_ANALYSIS_SUMMARY_FILENAME,
)

_SHARED_FIGURE_FILES: tuple[str, ...] = (
    CALIBRATION_COMPARISON_FILENAME,
    OOS_ROC_COMPARISON_FILENAME,
    OOS_PR_COMPARISON_FILENAME,
    ERROR_TAXONOMY_COMPARISON_FILENAME,
    OOS_ERROR_COMPARISON_FILENAME,
    CONFIDENCE_ACCURACY_COMPARISON_FILENAME,
    LENGTH_SLICE_COMPARISON_FILENAME,
    FREQUENCY_SLICE_COMPARISON_FILENAME,
    CROSS_MODEL_ERROR_OVERLAP_FILENAME,
)

_SHARED_OTHER_FILES: tuple[str, ...] = (
    CURATED_EXAMPLES_CSV_FILENAME,
    UNIVERSALLY_MISCLASSIFIED_FILENAME,
    ERROR_ANALYSIS_NOTES_FILENAME,
)


# ---------------------------------------------------------------------------
# Handoff generation
# ---------------------------------------------------------------------------


def generate_step11_handoff(
    model_ids: list[ModelID],
    figure_records: list[FigureRecord],
) -> dict[str, Any]:
    """Create step11_handoff.json referencing all Step 10 outputs."""
    shared_dir = shared_analysis_dir()

    per_model_artifacts: dict[str, dict[str, str]] = {}
    for mid in model_ids:
        out_dir = analysis_output_dir(mid)
        model_paths: dict[str, str] = {}
        for fname in _PER_MODEL_JSON_FILES + _PER_MODEL_FIGURE_FILES:
            p = out_dir / fname
            model_paths[fname] = repo_relative(p) if p.exists() else f"MISSING:{fname}"
        per_model_artifacts[str(mid)] = model_paths

    shared_artifacts: dict[str, str] = {}
    for fname in _SHARED_JSON_FILES + _SHARED_FIGURE_FILES + _SHARED_OTHER_FILES:
        p = shared_dir / fname
        shared_artifacts[fname] = repo_relative(p) if p.exists() else f"MISSING:{fname}"

    artifact = artifact_envelope(
        per_model_artifacts=per_model_artifacts,
        shared_artifacts=shared_artifacts,
        figure_count=len(figure_records),
    )
    handoff_path = shared_dir / STEP11_HANDOFF_FILENAME
    write_json(handoff_path, artifact)
    logger.info("Saved: %s", handoff_path)
    return artifact


# ---------------------------------------------------------------------------
# Figure manifest update
# ---------------------------------------------------------------------------


def update_figure_manifest(figure_records: list[FigureRecord]) -> None:
    """Append Step 10 figure records to the existing figure manifest."""
    from src.constants import PROTOCOL_VERSION, SCHEMA_VERSION

    manifest_path = SHARED_DIR / "figure_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        manifest = {"schema_version": SCHEMA_VERSION, "protocol_version": PROTOCOL_VERSION, "figures": []}

    existing_paths = {entry["figure_path"] for entry in manifest.get("figures", [])}

    for rec in figure_records:
        if rec.figure_path in existing_paths:
            continue
        entry: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "figure_path": rec.figure_path,
            "figure_type": rec.figure_type,
            "scope": rec.scope,
            "source_artifact_paths": rec.source_artifact_paths,
        }
        if rec.model_name is not None:
            entry["model_name"] = rec.model_name
        if rec.representative_run_id is not None:
            entry["representative_run_id"] = rec.representative_run_id
        if rec.seed is not None:
            entry["seed"] = rec.seed
        if rec.best_epoch is not None:
            entry["best_epoch"] = rec.best_epoch
        if rec.stopping_epoch is not None:
            entry["stopping_epoch"] = rec.stopping_epoch
        manifest["figures"].append(entry)

    write_json(manifest_path, manifest)
    logger.info("Updated figure manifest: %d total entries.", len(manifest["figures"]))


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def validate_handoff_paths(handoff_path: Path) -> None:
    """Assert that all artifact paths in the handoff resolve on disk."""
    handoff = read_json(handoff_path)
    missing: list[str] = []

    for model_id, paths in handoff.get("per_model_artifacts", {}).items():
        for fname, rel_path in paths.items():
            if rel_path.startswith("MISSING:"):
                missing.append(f"{model_id}/{fname}")
            elif not (PROJECT_ROOT / rel_path).exists():
                missing.append(rel_path)

    for fname, rel_path in handoff.get("shared_artifacts", {}).items():
        if rel_path.startswith("MISSING:"):
            missing.append(f"shared/{fname}")
        elif not (PROJECT_ROOT / rel_path).exists():
            missing.append(rel_path)

    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(f"Step 11 handoff validation failed — {len(missing)} path(s) missing:\n  {formatted}")

    logger.info("Step 11 handoff path validation passed.")


def validate_manifest_paths(manifest_path: Path) -> None:
    """Assert all figure paths in the manifest exist on disk."""
    manifest = read_json(manifest_path)
    missing: list[str] = []
    for entry in manifest.get("figures", []):
        fig_path = entry.get("figure_path", "")
        if not (PROJECT_ROOT / fig_path).exists():
            missing.append(fig_path)

    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(
            f"Figure manifest validation failed — {len(missing)} figure path(s) missing:\n  {formatted}"
        )

    logger.info("Figure manifest path validation passed (%d figures).", len(manifest.get("figures", [])))
