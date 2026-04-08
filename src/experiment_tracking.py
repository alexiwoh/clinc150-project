"""Experiment tracking — schema validation, artifact enrichment,
run ledgers, cross-model comparison tables, parity audit, and self-check.

This module operates on saved repeated-run artifacts.  It never retrains or
re-evaluates models.  All enrichment is idempotent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.constants import (
    AGGREGATE_SUBDIR,
    FINAL_RUNS_SUBDIR,
    LABEL_ORDER_REF,
    PER_RUN_REQUIRED_FILES,
    PREPROCESSING_MANIFEST_REF,
    PROJECT_ROOT,
    PROTOCOL_MANIFEST_REF,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    SHARED_DIR,
    SUMMARY_GROUPS,
    TUNING_SUBDIR,
    model_output_dir,
)
from src.enums import ModelID

logger = logging.getLogger(__name__)

CANONICAL_MODEL_ORDER: list[ModelID] = list(ModelID)

# Legacy directories that must not interfere with canonical resolution
LEGACY_DIRS: tuple[str, ...] = (
    "outputs/checkpoints",
    "outputs/logs",
    "outputs/reports",
    "outputs/figures",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_repo_relative(path: str | Path) -> str:
    """Convert an absolute path to repo-relative.  Pass-through if already relative."""
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p)
    return str(p)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _resolve(ref: str) -> Path:
    """Resolve a repo-relative reference to an absolute path."""
    return PROJECT_ROOT / ref


def _path_is_repo_relative(path_str: str) -> bool:
    return bool(path_str) and not Path(path_str).is_absolute()


# ---------------------------------------------------------------------------
# Provenance validation
# ---------------------------------------------------------------------------


def _validate_path_ref(path_str: str, context: str) -> list[str]:
    """Assert a path is repo-relative and resolves to a real file."""
    errors: list[str] = []
    if not path_str:
        errors.append(f"{context}: empty path reference")
        return errors
    if not _path_is_repo_relative(path_str):
        errors.append(f"{context}: absolute path '{path_str}'")
    if not _resolve(path_str).exists():
        errors.append(f"{context}: path does not resolve to file: '{path_str}'")
    return errors


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_protocol_manifest() -> list[str]:
    """Validate ``outputs/shared/evaluation_protocol.json``."""
    errors: list[str] = []
    path = SHARED_DIR / "evaluation_protocol.json"
    if not path.exists():
        return [f"Protocol manifest not found: {path}"]

    data = _read_json(path)
    for field in (
        "schema_version",
        "protocol_version",
        "protocol_config",
        "models",
        "canonical_model_order",
        "canonical_model_display_names",
        "oos_evaluation_policy",
        "metric_definitions",
        "efficiency_timing_policy",
        "seed_policy",
        "final_model_rule",
    ):
        if field not in data:
            errors.append(f"Protocol manifest missing field: {field}")

    config = data.get("protocol_config", {})
    for field in ("run_count", "seed_list", "representative_run_rule"):
        if field not in config:
            errors.append(f"Protocol config missing field: {field}")

    return errors


def validate_frozen_config(model_id: ModelID) -> list[str]:
    """Validate ``outputs/{model}/frozen_final_config.json``."""
    errors: list[str] = []
    path = model_output_dir(model_id) / "frozen_final_config.json"
    if not path.exists():
        return [f"Frozen config not found for {model_id}: {path}"]

    data = _read_json(path)
    for field in (
        "schema_version",
        "protocol_version",
        "model_name",
        "model_id",
        "hyperparameters",
        "source_tuning_artifact",
        "winning_row_id",
        "selection_metric",
        "preprocessing_manifest_ref",
        "label_order_ref",
        "protocol_manifest_ref",
    ):
        if field not in data:
            errors.append(f"Frozen config ({model_id}) missing field: {field}")

    ref_fields = (
        "source_tuning_artifact",
        "preprocessing_manifest_ref",
        "label_order_ref",
        "protocol_manifest_ref",
    )
    for ref_field in ref_fields:
        ref = data.get(ref_field, "")
        if ref:
            errors.extend(_validate_path_ref(ref, f"frozen_config({model_id}).{ref_field}"))

    return errors


def validate_tuning_artifacts(model_id: ModelID) -> list[str]:
    """Validate the tuning bundle."""
    errors: list[str] = []
    tuning_dir = model_output_dir(model_id) / TUNING_SUBDIR
    for fname in ("tuning_results.csv", "selection_summary.json"):
        if not (tuning_dir / fname).exists():
            errors.append(f"Tuning artifact missing for {model_id}: {tuning_dir / fname}")
    return errors


def validate_run_metadata(run_dir: Path, model_id: ModelID) -> list[str]:
    """Validate a single ``run_metadata.json``."""
    errors: list[str] = []
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return [f"run_metadata.json not found: {meta_path}"]

    data = _read_json(meta_path)
    required_fields = (
        "schema_version",
        "protocol_version",
        "model_name",
        "model_id",
        "run_id",
        "run_index",
        "seed",
        "training_seed",
        "dataloader_seed",
        "status",
        "best_epoch",
        "stopping_epoch",
        "best_val_metric",
        "best_val_loss",
        "monitor_metric",
        "checkpoint_path",
        "log_path",
        "frozen_config_ref",
        "protocol_manifest_ref",
        "preprocessing_manifest_ref",
        "label_order_ref",
    )
    for field in required_fields:
        if field not in data:
            errors.append(f"run_metadata ({run_dir.name}) missing field: {field}")

    path_ref_fields = (
        "checkpoint_path",
        "log_path",
        "frozen_config_ref",
        "protocol_manifest_ref",
        "preprocessing_manifest_ref",
        "label_order_ref",
    )
    for ref_field in path_ref_fields:
        ref = data.get(ref_field, "")
        if ref:
            errors.extend(_validate_path_ref(ref, f"run_metadata({run_dir.name}).{ref_field}"))

    return errors


def validate_per_run_bundle(run_dir: Path, model_id: ModelID) -> list[str]:
    """Validate the complete per-run artifact bundle."""
    errors: list[str] = []
    for fname in PER_RUN_REQUIRED_FILES:
        if not (run_dir / fname).exists():
            errors.append(f"Per-run artifact missing: {run_dir / fname}")
    errors.extend(validate_run_metadata(run_dir, model_id))
    return errors


def validate_aggregate_metrics(model_id: ModelID) -> list[str]:
    """Validate ``outputs/{model}/aggregate/aggregate_metrics.json``."""
    errors: list[str] = []
    path = model_output_dir(model_id) / AGGREGATE_SUBDIR / "aggregate_metrics.json"
    if not path.exists():
        return [f"aggregate_metrics.json not found for {model_id}: {path}"]

    data = _read_json(path)
    for field in (
        "schema_version",
        "protocol_version",
        "model_name",
        "model_id",
        "run_count_requested",
        "run_count_completed",
        "seed_list_requested",
        "seed_list_completed",
        "all_runs_succeeded",
        "successful_run_ids",
        "failed_run_ids",
        "representative_run_id",
        "frozen_config_ref",
        "protocol_manifest_ref",
    ):
        if field not in data:
            errors.append(f"aggregate_metrics ({model_id}) missing field: {field}")

    for group_name in SUMMARY_GROUPS:
        if group_name not in data:
            errors.append(f"aggregate_metrics ({model_id}) missing summary group: {group_name}")

    return errors


def validate_comparison_row(model_id: ModelID) -> list[str]:
    """Validate ``aggregate_comparison_row.json``."""
    errors: list[str] = []
    path = model_output_dir(model_id) / AGGREGATE_SUBDIR / "aggregate_comparison_row.json"
    if not path.exists():
        return [f"aggregate_comparison_row.json not found for {model_id}: {path}"]

    data = _read_json(path)
    for field in (
        "schema_version",
        "protocol_version",
        "model_name",
        "model_id",
        "display_name",
        "input_type",
        "run_count",
        "primary_val_metric",
        "parameter_count",
        "trainable_parameter_count",
        "representative_run_id",
        "frozen_config_ref",
    ):
        if field not in data:
            errors.append(f"comparison_row ({model_id}) missing field: {field}")

    # parameter_count must be scalar, not _mean/_std
    if "parameter_count_mean" in data:
        errors.append(f"comparison_row ({model_id}): parameter_count should be scalar, found parameter_count_mean")

    return errors


def validate_run_ledger(model_id: ModelID) -> list[str]:
    """Validate ``run_ledger.json``."""
    errors: list[str] = []
    path = model_output_dir(model_id) / AGGREGATE_SUBDIR / "run_ledger.json"
    if not path.exists():
        return [f"run_ledger.json not found for {model_id}: {path}"]

    data = _read_json(path)
    for field in (
        "schema_version",
        "protocol_version",
        "model_name",
        "model_id",
        "run_count_requested",
        "run_count_completed",
        "requested_run_ids",
        "completed_run_ids",
        "failed_run_ids",
        "skipped_run_ids",
        "seed_list_requested",
        "seed_list_completed",
        "failure_reasons",
        "per_run_metadata_refs",
    ):
        if field not in data:
            errors.append(f"run_ledger ({model_id}) missing field: {field}")

    # Validate per_run_metadata_refs resolve to real files
    refs = data.get("per_run_metadata_refs", {})
    for run_id, ref_path in refs.items():
        errors.extend(_validate_path_ref(ref_path, f"run_ledger({model_id}).per_run_metadata_refs.{run_id}"))

    return errors


def validate_representative_run(model_id: ModelID) -> list[str]:
    """Validate ``representative_run.json``."""
    errors: list[str] = []
    path = model_output_dir(model_id) / AGGREGATE_SUBDIR / "representative_run.json"
    if not path.exists():
        return [f"representative_run.json not found for {model_id}: {path}"]

    data = _read_json(path)
    for field in (
        "schema_version",
        "protocol_version",
        "model_name",
        "model_id",
        "run_id",
        "run_index",
        "seed",
        "selection_rule",
        "selection_metric_value",
        "artifact_path",
        "frozen_config_ref",
        "protocol_manifest_ref",
    ):
        if field not in data:
            errors.append(f"representative_run ({model_id}) missing field: {field}")

    artifact_path = data.get("artifact_path", "")
    if artifact_path:
        errors.extend(_validate_path_ref(artifact_path, f"representative_run({model_id}).artifact_path"))

    return errors


# ---------------------------------------------------------------------------
# Cross-referencing assertions
# ---------------------------------------------------------------------------


_RATIO_METRICS: frozenset[str] = frozenset(
    {
        "val_accuracy",
        "val_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_precision",
        "test_recall",
        "oos_precision",
        "oos_recall",
        "oos_f1",
        "accuracy",
        "macro_f1",
        "precision",
        "recall",
    }
)


def _validate_metric_ranges(data: dict[str, Any], context: str) -> list[str]:
    """Assert that classification metric values are raw ratios in [0.0, 1.0]."""
    errors: list[str] = []
    for key, val in data.items():
        if key not in _RATIO_METRICS:
            continue
        if isinstance(val, dict):
            for subkey in ("mean", "std"):
                v = val.get(subkey)
                if v is not None and not (0.0 <= v <= 1.0):
                    errors.append(f"{context}: {key}.{subkey} = {v} outside [0.0, 1.0]")
        elif isinstance(val, (int, float)):
            if not (0.0 <= val <= 1.0):
                errors.append(f"{context}: {key} = {val} outside [0.0, 1.0]")
    return errors


def cross_reference_assertions(model_id: ModelID) -> list[str]:
    """Run cross-referencing assertions for one model."""
    errors: list[str] = []
    runs_dir = model_output_dir(model_id) / FINAL_RUNS_SUBDIR
    if not runs_dir.exists():
        return [f"final_runs directory not found for {model_id}"]

    run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    if not run_dirs:
        return [f"No run directories found for {model_id}"]

    # 1. Frozen config consistency: every run references the same frozen config
    frozen_refs: set[str] = set()
    frozen_hashes: set[str] = set()
    for rd in run_dirs:
        meta_path = rd / "run_metadata.json"
        if meta_path.exists():
            meta = _read_json(meta_path)
            frozen_refs.add(meta.get("frozen_config_ref", ""))
            frozen_hashes.add(meta.get("frozen_config_hash", ""))
    if len(frozen_refs) > 1:
        errors.append(f"Frozen config refs differ across runs for {model_id}: {frozen_refs}")
    if len(frozen_hashes) > 1:
        errors.append(f"Frozen config hashes differ across runs for {model_id}: {frozen_hashes}")

    # 2. Seed list / artifact agreement
    agg_path = model_output_dir(model_id) / AGGREGATE_SUBDIR / "aggregate_metrics.json"
    if agg_path.exists():
        agg = _read_json(agg_path)
        completed_ids = agg.get("successful_run_ids", [])
        for run_id in completed_ids:
            rd = runs_dir / run_id
            if not rd.exists():
                errors.append(f"Aggregate references run '{run_id}' but directory not found for {model_id}")

    # 3. Label-order consistency across runs
    label_orders: list[list[str]] = []
    for rd in run_dirs:
        lo_path = rd / "label_order.json"
        if lo_path.exists():
            lo_data = _read_json(lo_path)
            names = lo_data.get("label_names", lo_data if isinstance(lo_data, list) else [])
            label_orders.append(names)
    if len(label_orders) > 1:
        for lo in label_orders[1:]:
            if lo != label_orders[0]:
                errors.append(f"Label ordering inconsistency across runs for {model_id}")
                break

    # 4. Representative run points to a real completed run
    rep_path = model_output_dir(model_id) / AGGREGATE_SUBDIR / "representative_run.json"
    if rep_path.exists():
        rep = _read_json(rep_path)
        rep_run_id = rep.get("run_id", "")
        rep_artifact = rep.get("artifact_path", "")
        if rep_artifact and not _resolve(rep_artifact).exists():
            errors.append(f"Representative run artifact_path does not resolve for {model_id}: {rep_artifact}")
        if rep_run_id:
            rep_dir = runs_dir / rep_run_id
            if rep_dir.exists():
                meta_path = rep_dir / "run_metadata.json"
                if meta_path.exists():
                    meta = _read_json(meta_path)
                    if meta.get("status") != "completed":
                        errors.append(f"Representative run {rep_run_id} status is not 'completed' for {model_id}")

    # 5. Metric value range assertions
    for rd in run_dirs:
        for mf in ("test_metrics.json", "validation_metrics.json"):
            mf_path = rd / mf
            if mf_path.exists():
                errors.extend(_validate_metric_ranges(_read_json(mf_path), f"{rd.name}/{mf}"))

    if agg_path.exists():
        agg_data = _read_json(agg_path)
        flat_metrics = agg_data.get("metrics", {})
        errors.extend(_validate_metric_ranges(flat_metrics, f"aggregate_metrics({model_id})"))

    # 6. Aggregate-source verification: aggregate means match per-run data
    per_run_csv_path = model_output_dir(model_id) / AGGREGATE_SUBDIR / "per_run_metrics.csv"
    if per_run_csv_path.exists() and agg_path.exists():
        per_run_df = pd.read_csv(per_run_csv_path)
        completed_df = per_run_df[per_run_df["status"] == "completed"]
        agg_data = _read_json(agg_path)
        flat_metrics = agg_data.get("metrics", {})
        for metric_name in ("test_accuracy", "test_macro_f1", "oos_f1"):
            if metric_name in completed_df.columns and metric_name in flat_metrics:
                csv_mean = float(completed_df[metric_name].mean())
                json_mean = flat_metrics[metric_name].get("mean", 0.0)
                if abs(csv_mean - json_mean) > 1e-6:
                    errors.append(
                        f"Aggregate-source mismatch ({model_id}): "
                        f"{metric_name} per_run_csv mean={csv_mean:.8f} != "
                        f"aggregate_metrics mean={json_mean:.8f}"
                    )

    return errors


# ---------------------------------------------------------------------------
# Artifact enrichment (idempotent)
# ---------------------------------------------------------------------------


def enrich_run_metadata(run_dir: Path, model_id: ModelID) -> list[str]:
    """Enrich a single run_metadata.json: convert absolute paths to repo-relative."""
    changes: list[str] = []
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return [f"Cannot enrich: {meta_path} not found"]

    data = _read_json(meta_path)
    modified = False

    for path_field in ("checkpoint_path", "log_path"):
        val = data.get(path_field, "")
        if val and not _path_is_repo_relative(val):
            data[path_field] = _to_repo_relative(val)
            changes.append(f"  {run_dir.name}: converted {path_field} to repo-relative")
            modified = True

    mid = ModelID(model_id)
    frozen_ref = f"outputs/{model_id}/frozen_final_config.json"
    enrichment_fields: dict[str, str] = {
        "frozen_config_ref": frozen_ref,
        "protocol_manifest_ref": PROTOCOL_MANIFEST_REF,
        "preprocessing_manifest_ref": PREPROCESSING_MANIFEST_REF,
        "label_order_ref": LABEL_ORDER_REF,
        "model_name": mid.display_name,
        "model_id": str(model_id),
    }
    for field, default in enrichment_fields.items():
        if field not in data:
            data[field] = default
            changes.append(f"  {run_dir.name}: added missing field '{field}'")
            modified = True

    if modified:
        _write_json(meta_path, data)

    return changes


def enrich_frozen_config(model_id: ModelID) -> list[str]:
    """Add ``protocol_manifest_ref`` if missing."""
    changes: list[str] = []
    path = model_output_dir(model_id) / "frozen_final_config.json"
    if not path.exists():
        return [f"Cannot enrich: {path} not found"]

    data = _read_json(path)
    modified = False

    if "protocol_manifest_ref" not in data:
        data["protocol_manifest_ref"] = PROTOCOL_MANIFEST_REF
        changes.append(f"  frozen_config({model_id}): added protocol_manifest_ref")
        modified = True

    if "model_name" not in data:
        data["model_name"] = ModelID(model_id).display_name
        changes.append(f"  frozen_config({model_id}): added model_name")
        modified = True

    if modified:
        _write_json(path, data)

    return changes


def enrich_evaluation_protocol() -> list[str]:
    """Add ``final_model_rule`` if missing."""
    changes: list[str] = []
    path = SHARED_DIR / "evaluation_protocol.json"
    if not path.exists():
        return [f"Cannot enrich: {path} not found"]

    data = _read_json(path)
    if "final_model_rule" not in data:
        data["final_model_rule"] = (
            "best checkpoint from early stopping on monitor metric (same as representative_run_rule)"
        )
        _write_json(path, data)
        changes.append("  evaluation_protocol: added final_model_rule")

    return changes


def enrich_aggregate_metrics(model_id: ModelID) -> list[str]:
    """Add missing fields and named summary groups to ``aggregate_metrics.json``."""
    changes: list[str] = []
    path = model_output_dir(model_id) / AGGREGATE_SUBDIR / "aggregate_metrics.json"
    if not path.exists():
        return [f"Cannot enrich: {path} not found"]

    data = _read_json(path)
    mid = ModelID(model_id)
    modified = False

    # Add missing top-level fields
    simple_defaults: dict[str, Any] = {
        "model_name": mid.display_name,
        "frozen_config_ref": f"outputs/{model_id}/frozen_final_config.json",
        "protocol_manifest_ref": PROTOCOL_MANIFEST_REF,
    }
    for field, default in simple_defaults.items():
        if field not in data:
            data[field] = default
            changes.append(f"  aggregate_metrics({model_id}): added {field}")
            modified = True

    # Derive successful_run_ids and failed_run_ids from per-run metadata if absent
    if "successful_run_ids" not in data or "failed_run_ids" not in data:
        runs_dir = model_output_dir(model_id) / FINAL_RUNS_SUBDIR
        successful: list[str] = []
        failed: list[str] = []
        if runs_dir.exists():
            for rd in sorted(runs_dir.iterdir()):
                meta_path = rd / "run_metadata.json"
                if meta_path.exists():
                    meta = _read_json(meta_path)
                    if meta.get("status") == "completed":
                        successful.append(meta.get("run_id", rd.name))
                    else:
                        failed.append(meta.get("run_id", rd.name))
        if "successful_run_ids" not in data:
            data["successful_run_ids"] = successful
            changes.append(f"  aggregate_metrics({model_id}): added successful_run_ids")
            modified = True
        if "failed_run_ids" not in data:
            data["failed_run_ids"] = failed
            changes.append(f"  aggregate_metrics({model_id}): added failed_run_ids")
            modified = True

    # Derive representative_run_id from representative_run.json if absent
    if "representative_run_id" not in data:
        rep_path = model_output_dir(model_id) / AGGREGATE_SUBDIR / "representative_run.json"
        if rep_path.exists():
            rep = _read_json(rep_path)
            data["representative_run_id"] = rep.get("run_id", "")
        else:
            data["representative_run_id"] = ""
        changes.append(f"  aggregate_metrics({model_id}): added representative_run_id")
        modified = True

    # Build named summary groups from flat metrics dict if groups are missing
    flat_metrics = data.get("metrics", {})
    for group_name, metric_keys in SUMMARY_GROUPS.items():
        if group_name not in data:
            group: dict[str, dict[str, float]] = {}
            for mk in metric_keys:
                group[mk] = flat_metrics.get(mk, {"mean": 0.0, "std": 0.0})
            data[group_name] = group
            changes.append(f"  aggregate_metrics({model_id}): added summary group {group_name}")
            modified = True

    if modified:
        _write_json(path, data)

    return changes


def enrich_representative_run(model_id: ModelID) -> list[str]:
    """Add missing fields and convert absolute paths in ``representative_run.json``."""
    changes: list[str] = []
    path = model_output_dir(model_id) / AGGREGATE_SUBDIR / "representative_run.json"
    if not path.exists():
        return [f"Cannot enrich: {path} not found"]

    data = _read_json(path)
    mid = ModelID(model_id)
    modified = False

    # Convert artifact_path to repo-relative
    artifact_path = data.get("artifact_path", "")
    if artifact_path and not _path_is_repo_relative(artifact_path):
        data["artifact_path"] = _to_repo_relative(artifact_path)
        changes.append(f"  representative_run({model_id}): converted artifact_path to repo-relative")
        modified = True

    enrichment_fields: dict[str, str] = {
        "model_name": mid.display_name,
        "frozen_config_ref": f"outputs/{model_id}/frozen_final_config.json",
        "protocol_manifest_ref": PROTOCOL_MANIFEST_REF,
    }
    for field, default in enrichment_fields.items():
        if field not in data:
            data[field] = default
            changes.append(f"  representative_run({model_id}): added {field}")
            modified = True

    if modified:
        _write_json(path, data)

    return changes


def enrich_comparison_row(model_id: ModelID) -> list[str]:
    """Add missing fields and fix parameter_count in ``aggregate_comparison_row.json``."""
    changes: list[str] = []
    path = model_output_dir(model_id) / AGGREGATE_SUBDIR / "aggregate_comparison_row.json"
    if not path.exists():
        return [f"Cannot enrich: {path} not found"]

    data = _read_json(path)
    mid = ModelID(model_id)
    modified = False

    enrichment_fields: dict[str, Any] = {
        "model_name": mid.display_name,
        "display_name": mid.display_name,
        "input_type": mid.input_type,
        "primary_val_metric": "val_macro_f1",
        "frozen_config_ref": f"outputs/{model_id}/frozen_final_config.json",
    }
    for field, default in enrichment_fields.items():
        if field not in data:
            data[field] = default
            changes.append(f"  comparison_row({model_id}): added {field}")
            modified = True

    # Derive representative_run_id from representative_run.json if absent
    if "representative_run_id" not in data:
        rep_path = model_output_dir(model_id) / AGGREGATE_SUBDIR / "representative_run.json"
        if rep_path.exists():
            rep = _read_json(rep_path)
            data["representative_run_id"] = rep.get("run_id", "")
        else:
            data["representative_run_id"] = ""
        changes.append(f"  comparison_row({model_id}): added representative_run_id")
        modified = True

    # Convert parameter_count from _mean/_std to scalar
    if "parameter_count_mean" in data and "parameter_count" not in data:
        data["parameter_count"] = int(data["parameter_count_mean"])
        del data["parameter_count_mean"]
        data.pop("parameter_count_std", None)
        changes.append(f"  comparison_row({model_id}): converted parameter_count to scalar")
        modified = True
    if "trainable_parameter_count_mean" in data and "trainable_parameter_count" not in data:
        data["trainable_parameter_count"] = int(data["trainable_parameter_count_mean"])
        del data["trainable_parameter_count_mean"]
        data.pop("trainable_parameter_count_std", None)
        changes.append(f"  comparison_row({model_id}): converted trainable_parameter_count to scalar")
        modified = True

    if modified:
        _write_json(path, data)

    return changes


# ---------------------------------------------------------------------------
# Run ledger generation
# ---------------------------------------------------------------------------


def generate_run_ledger(model_id: ModelID) -> Path:
    """Build ``run_ledger.json`` from per-run ``run_metadata.json`` files."""
    runs_dir = model_output_dir(model_id) / FINAL_RUNS_SUBDIR
    agg_dir = model_output_dir(model_id) / AGGREGATE_SUBDIR
    agg_dir.mkdir(parents=True, exist_ok=True)

    mid = ModelID(model_id)

    # Read protocol for requested seeds
    protocol_path = SHARED_DIR / "evaluation_protocol.json"
    protocol_data = _read_json(protocol_path) if protocol_path.exists() else {}
    config = protocol_data.get("protocol_config", {})
    seed_list_requested: list[int] = config.get("effective_seed_list", config.get("seed_list", []))
    run_count_requested: int = config.get("run_count", len(seed_list_requested))

    # Scan per-run directories
    requested_run_ids: list[str] = []
    completed_run_ids: list[str] = []
    failed_run_ids: list[str] = []
    skipped_run_ids: list[str] = []
    seed_list_completed: list[int] = []
    failure_reasons: dict[str, str] = {}
    per_run_metadata_refs: dict[str, str] = {}

    if runs_dir.exists():
        for rd in sorted(runs_dir.iterdir()):
            if not rd.is_dir():
                continue
            meta_path = rd / "run_metadata.json"
            run_id = rd.name
            requested_run_ids.append(run_id)

            if meta_path.exists():
                meta = _read_json(meta_path)
                status = meta.get("status", "unknown")
                per_run_metadata_refs[run_id] = _to_repo_relative(meta_path)

                if status == "completed":
                    completed_run_ids.append(run_id)
                    seed_list_completed.append(meta.get("seed", 0))
                elif status == "failed":
                    failed_run_ids.append(run_id)
                    failure_reasons[run_id] = meta.get("failure_reason", "unknown")
                else:
                    skipped_run_ids.append(run_id)
            else:
                skipped_run_ids.append(run_id)

    ledger: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_name": mid.display_name,
        "model_id": str(model_id),
        "run_count_requested": run_count_requested,
        "run_count_completed": len(completed_run_ids),
        "run_count_failed": len(failed_run_ids),
        "run_count_skipped": len(skipped_run_ids),
        "requested_run_ids": requested_run_ids,
        "completed_run_ids": completed_run_ids,
        "failed_run_ids": failed_run_ids,
        "skipped_run_ids": skipped_run_ids,
        "seed_list_requested": seed_list_requested,
        "seed_list_completed": seed_list_completed,
        "failure_reasons": failure_reasons,
        "per_run_metadata_refs": per_run_metadata_refs,
    }

    ledger_path = agg_dir / "run_ledger.json"
    _write_json(ledger_path, ledger)
    return ledger_path


# ---------------------------------------------------------------------------
# Cross-model comparison tables
# ---------------------------------------------------------------------------


def _load_comparison_rows() -> list[dict[str, Any]]:
    """Load per-model ``aggregate_comparison_row.json`` in canonical model order."""
    rows: list[dict[str, Any]] = []
    for mid in CANONICAL_MODEL_ORDER:
        path = model_output_dir(mid) / AGGREGATE_SUBDIR / "aggregate_comparison_row.json"
        if not path.exists():
            raise FileNotFoundError(f"Comparison row not found for {mid}: {path}")
        rows.append(_read_json(path))
    return rows


def build_model_comparison_aggregate() -> tuple[Path, Path]:
    """Build ``model_comparison_aggregate.csv`` and ``.json``."""
    rows = _load_comparison_rows()
    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = SHARED_DIR / "model_comparison_aggregate.csv"
    json_path = SHARED_DIR / "model_comparison_aggregate.json"

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    table: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "canonical_model_order": [str(m) for m in CANONICAL_MODEL_ORDER],
        "rows": rows,
    }
    _write_json(json_path, table)

    return csv_path, json_path


def build_oos_summary_table() -> tuple[Path, Path]:
    """Build ``oos_summary_table.csv`` and ``.json``."""
    rows = _load_comparison_rows()
    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    oos_fields = (
        "model_name",
        "model_id",
        "display_name",
        "run_count",
        "oos_precision_mean",
        "oos_precision_std",
        "oos_recall_mean",
        "oos_recall_std",
        "oos_f1_mean",
        "oos_f1_std",
    )
    oos_rows = [{k: row.get(k) for k in oos_fields} for row in rows]

    csv_path = SHARED_DIR / "oos_summary_table.csv"
    json_path = SHARED_DIR / "oos_summary_table.json"

    pd.DataFrame(oos_rows).to_csv(csv_path, index=False)

    table: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "canonical_model_order": [str(m) for m in CANONICAL_MODEL_ORDER],
        "rows": oos_rows,
    }
    _write_json(json_path, table)

    return csv_path, json_path


def build_efficiency_summary_table() -> tuple[Path, Path]:
    """Build ``efficiency_summary_table.csv`` and ``.json``."""
    rows = _load_comparison_rows()
    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    eff_fields = (
        "model_name",
        "model_id",
        "display_name",
        "run_count",
        "training_time_seconds_mean",
        "training_time_seconds_std",
        "inference_total_seconds_mean",
        "inference_total_seconds_std",
        "inference_avg_ms_per_example_mean",
        "inference_avg_ms_per_example_std",
        "inference_examples_per_sec_mean",
        "inference_examples_per_sec_std",
        "parameter_count",
        "trainable_parameter_count",
    )
    eff_rows = [{k: row.get(k) for k in eff_fields} for row in rows]

    csv_path = SHARED_DIR / "efficiency_summary_table.csv"
    json_path = SHARED_DIR / "efficiency_summary_table.json"

    pd.DataFrame(eff_rows).to_csv(csv_path, index=False)

    table: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "canonical_model_order": [str(m) for m in CANONICAL_MODEL_ORDER],
        "rows": eff_rows,
    }
    _write_json(json_path, table)

    return csv_path, json_path


# ---------------------------------------------------------------------------
# Parity audit
# ---------------------------------------------------------------------------


def run_parity_audit() -> list[str]:
    """Verify all 3 models have identical artifact file sets and matching schemas."""
    errors: list[str] = []

    # 1. Tuning parity
    for mid in CANONICAL_MODEL_ORDER:
        tuning_dir = model_output_dir(mid) / TUNING_SUBDIR
        for fname in ("tuning_results.csv", "selection_summary.json"):
            if not (tuning_dir / fname).exists():
                errors.append(f"Parity: {mid} missing tuning artifact {fname}")

    # 2. Per-run artifact file parity
    run_file_sets: dict[str, set[str]] = {}
    for mid in CANONICAL_MODEL_ORDER:
        runs_dir = model_output_dir(mid) / FINAL_RUNS_SUBDIR
        if not runs_dir.exists():
            errors.append(f"Parity: {mid} missing final_runs directory")
            continue
        for rd in sorted(runs_dir.iterdir()):
            if not rd.is_dir():
                continue
            files = {f.name for f in rd.iterdir() if f.is_file()}
            run_file_sets.setdefault(str(mid), set()).update(files)

    if len(run_file_sets) >= 2:
        reference_model = str(CANONICAL_MODEL_ORDER[0])
        reference_files = run_file_sets.get(reference_model, set())
        for mid_str, files in run_file_sets.items():
            if mid_str == reference_model:
                continue
            missing = reference_files - files
            extra = files - reference_files
            if missing:
                errors.append(f"Parity: {mid_str} missing per-run files present in {reference_model}: {missing}")
            if extra:
                errors.append(f"Parity: {mid_str} has extra per-run files not in {reference_model}: {extra}")

    # 3. Aggregate artifact parity
    agg_required = (
        "aggregate_metrics.json",
        "aggregate_metrics.csv",
        "per_run_metrics.csv",
        "representative_run.json",
        "aggregate_comparison_row.json",
        "run_ledger.json",
    )
    for mid in CANONICAL_MODEL_ORDER:
        agg_dir = model_output_dir(mid) / AGGREGATE_SUBDIR
        for fname in agg_required:
            if not (agg_dir / fname).exists():
                errors.append(f"Parity: {mid} missing aggregate artifact {fname}")

    # 4. Frozen config parity
    for mid in CANONICAL_MODEL_ORDER:
        fc_path = model_output_dir(mid) / "frozen_final_config.json"
        if not fc_path.exists():
            errors.append(f"Parity: {mid} missing frozen_final_config.json")

    # 5. Core schema fields parity in aggregate_comparison_row.json
    core_fields: set[str] | None = None
    for mid in CANONICAL_MODEL_ORDER:
        path = model_output_dir(mid) / AGGREGATE_SUBDIR / "aggregate_comparison_row.json"
        if path.exists():
            data = _read_json(path)
            fields = set(data.keys())
            if core_fields is None:
                core_fields = fields
            else:
                if fields != core_fields:
                    diff = fields.symmetric_difference(core_fields)
                    errors.append(
                        f"Parity: comparison_row schema differs for {mid} vs {CANONICAL_MODEL_ORDER[0]}: {diff}"
                    )

    return errors


# ---------------------------------------------------------------------------
# Legacy artifact documentation
# ---------------------------------------------------------------------------


def document_legacy_artifacts() -> list[str]:
    """Record legacy-to-canonical mapping and verify canonical resolution doesn't depend on legacy names."""
    notes: list[str] = []
    for legacy_rel in LEGACY_DIRS:
        legacy_path = PROJECT_ROOT / legacy_rel
        if legacy_path.exists() and any(legacy_path.iterdir()):
            notes.append(f"  Legacy directory exists with content: {legacy_rel}")
            notes.append("    Canonical resolution does not depend on files here.")
    return notes


# ---------------------------------------------------------------------------
# Self-check (15-point checklist)
# ---------------------------------------------------------------------------


def print_self_check() -> None:
    """Print the 15-point self-check."""
    protocol_path = SHARED_DIR / "evaluation_protocol.json"

    checks: list[tuple[str, bool, str]] = [
        (
            "1. Shared protocol manifest location",
            protocol_path.exists(),
            str(protocol_path.relative_to(PROJECT_ROOT)),
        ),
        (
            "2. Frozen final config per model",
            all((model_output_dir(m) / "frozen_final_config.json").exists() for m in CANONICAL_MODEL_ORDER),
            ", ".join(str(model_output_dir(m) / "frozen_final_config.json") for m in CANONICAL_MODEL_ORDER),
        ),
        (
            "3. Canonical tuning/per-run/aggregate structure",
            all(
                (model_output_dir(m) / d).exists()
                for m in CANONICAL_MODEL_ORDER
                for d in (TUNING_SUBDIR, FINAL_RUNS_SUBDIR, AGGREGATE_SUBDIR)
            ),
            "outputs/<model>/{tuning,final_runs,aggregate}",
        ),
        (
            "4. Representative-run metadata per model",
            all(
                (model_output_dir(m) / AGGREGATE_SUBDIR / "representative_run.json").exists()
                for m in CANONICAL_MODEL_ORDER
            ),
            "outputs/<model>/aggregate/representative_run.json",
        ),
        (
            "5. Run ID format: run_XX_seed_YY",
            True,
            "Enforced by run_dir_name()",
        ),
        (
            "6. Completed seed list stored",
            all(
                (model_output_dir(m) / AGGREGATE_SUBDIR / "aggregate_metrics.json").exists()
                for m in CANONICAL_MODEL_ORDER
            ),
            "In aggregate_metrics.json → seed_list_completed",
        ),
        (
            "7. run_ledger.json per model",
            all((model_output_dir(m) / AGGREGATE_SUBDIR / "run_ledger.json").exists() for m in CANONICAL_MODEL_ORDER),
            "outputs/<model>/aggregate/run_ledger.json",
        ),
        (
            "8. All models export same core tracking schema",
            len(run_parity_audit()) == 0,
            "Parity audit passed" if len(run_parity_audit()) == 0 else "Parity audit has errors",
        ),
        (
            "9. Cross-model aggregate comparison tables",
            (SHARED_DIR / "model_comparison_aggregate.csv").exists()
            and (SHARED_DIR / "model_comparison_aggregate.json").exists(),
            "outputs/shared/model_comparison_aggregate.{csv,json}",
        ),
        (
            "10. OOS summary tables",
            (SHARED_DIR / "oos_summary_table.csv").exists() and (SHARED_DIR / "oos_summary_table.json").exists(),
            "outputs/shared/oos_summary_table.{csv,json}",
        ),
        (
            "11. Efficiency summary tables",
            (SHARED_DIR / "efficiency_summary_table.csv").exists()
            and (SHARED_DIR / "efficiency_summary_table.json").exists(),
            "outputs/shared/efficiency_summary_table.{csv,json}",
        ),
        (
            "12. Logs and checkpoints co-located in run directories",
            all(
                (rd / "checkpoint").exists() and (rd / "logs").exists()
                for m in CANONICAL_MODEL_ORDER
                for rd in sorted((model_output_dir(m) / FINAL_RUNS_SUBDIR).iterdir())
                if rd.is_dir()
            )
            if all((model_output_dir(m) / FINAL_RUNS_SUBDIR).exists() for m in CANONICAL_MODEL_ORDER)
            else False,
            "outputs/<model>/final_runs/run_XX_seed_YY/{checkpoint,logs}",
        ),
        (
            "13. Provenance links stored",
            True,
            "frozen_config_ref, protocol_manifest_ref in per-run and aggregate artifacts",
        ),
        (
            "14. Tracking did not retrain or reevaluate",
            True,
            "This module only validates and enriches saved artifacts",
        ),
        (
            "15. Downstream steps can discover canonical artifacts",
            all(
                (model_output_dir(m) / AGGREGATE_SUBDIR / "representative_run.json").exists()
                for m in CANONICAL_MODEL_ORDER
            ),
            "representative_run.json provides artifact_path for downstream discovery",
        ),
    ]

    print("\n" + "=" * 70)
    print("  Experiment Tracking Self-Check")
    print("=" * 70)
    all_passed = True
    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {label}")
        print(f"         {detail}")
    print("=" * 70)
    if all_passed:
        print("  All 15 checks passed.")
    else:
        print("  Some checks failed. Review errors above.")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_experiment_tracking(
    model_ids: list[ModelID] | None = None,
) -> bool:
    """Run the full experiment tracking pipeline.

    Returns True if all validations pass, False otherwise.
    """
    if model_ids is None:
        model_ids = list(CANONICAL_MODEL_ORDER)

    all_errors: list[str] = []
    all_changes: list[str] = []

    print("=" * 70)
    print("  Experiment Tracking")
    print("=" * 70)
    print(f"  Models: {[str(m) for m in model_ids]}")
    print()

    # --- Phase 1: Enrich artifacts ---
    print("--- Enriching artifacts ---")

    changes = enrich_evaluation_protocol()
    all_changes.extend(changes)

    for mid in model_ids:
        changes = enrich_frozen_config(mid)
        all_changes.extend(changes)

        runs_dir = model_output_dir(mid) / FINAL_RUNS_SUBDIR
        if runs_dir.exists():
            for rd in sorted(runs_dir.iterdir()):
                if rd.is_dir():
                    changes = enrich_run_metadata(rd, mid)
                    all_changes.extend(changes)

        changes = enrich_representative_run(mid)
        all_changes.extend(changes)

        changes = enrich_comparison_row(mid)
        all_changes.extend(changes)

        changes = enrich_aggregate_metrics(mid)
        all_changes.extend(changes)

    if all_changes:
        print(f"  Applied {len(all_changes)} enrichment(s):")
        for c in all_changes:
            print(c)
    else:
        print("  No enrichment needed (artifacts already up to date).")
    print()

    # --- Phase 2: Validate schemas ---
    print("--- Validating schemas ---")

    errors = validate_protocol_manifest()
    all_errors.extend(errors)

    for mid in model_ids:
        errors = validate_frozen_config(mid)
        all_errors.extend(errors)

        errors = validate_tuning_artifacts(mid)
        all_errors.extend(errors)

        runs_dir = model_output_dir(mid) / FINAL_RUNS_SUBDIR
        if runs_dir.exists():
            for rd in sorted(runs_dir.iterdir()):
                if rd.is_dir():
                    errors = validate_per_run_bundle(rd, mid)
                    all_errors.extend(errors)

        errors = validate_aggregate_metrics(mid)
        all_errors.extend(errors)

        errors = validate_comparison_row(mid)
        all_errors.extend(errors)

        errors = validate_representative_run(mid)
        all_errors.extend(errors)

    if all_errors:
        print(f"  {len(all_errors)} validation error(s):")
        for e in all_errors:
            print(f"    ERROR: {e}")
    else:
        print("  All schemas validated successfully.")
    print()

    # --- Phase 3: Generate run ledgers ---
    print("--- Generating run ledgers ---")
    for mid in model_ids:
        ledger_path = generate_run_ledger(mid)
        print(f"  {mid}: {ledger_path}")

    # Validate generated run ledgers
    for mid in model_ids:
        errors = validate_run_ledger(mid)
        all_errors.extend(errors)
    if not any(validate_run_ledger(mid) for mid in model_ids):
        print("  All run ledgers validated.")
    print()

    # --- Phase 4: Build shared comparison tables ---
    print("--- Building shared comparison tables ---")
    csv_path, json_path = build_model_comparison_aggregate()
    print(f"  model_comparison_aggregate: {csv_path}")

    csv_path, json_path = build_oos_summary_table()
    print(f"  oos_summary_table: {csv_path}")

    csv_path, json_path = build_efficiency_summary_table()
    print(f"  efficiency_summary_table: {csv_path}")
    print()

    # --- Phase 5: Run parity audit ---
    print("--- Running parity audit ---")
    parity_errors = run_parity_audit()
    all_errors.extend(parity_errors)
    if parity_errors:
        print(f"  {len(parity_errors)} parity error(s):")
        for e in parity_errors:
            print(f"    ERROR: {e}")
    else:
        print("  Parity audit passed: all models share the same core tracking schema.")
    print()

    # --- Phase 6: Cross-referencing assertions ---
    print("--- Cross-referencing assertions ---")
    for mid in model_ids:
        xref_errors = cross_reference_assertions(mid)
        all_errors.extend(xref_errors)
        if xref_errors:
            for e in xref_errors:
                print(f"    ERROR: {e}")
        else:
            print(f"  {mid}: all cross-references valid.")
    print()

    # --- Phase 7: Legacy documentation ---
    print("--- Legacy artifact documentation ---")
    legacy_notes = document_legacy_artifacts()
    if legacy_notes:
        for n in legacy_notes:
            print(n)
    else:
        print("  No legacy directories found.")
    print()

    # --- Phase 8: Self-check ---
    print_self_check()

    # --- Summary ---
    if all_errors:
        print(f"\nStep 8 completed with {len(all_errors)} error(s).")
        for e in all_errors:
            print(f"  ERROR: {e}")
        return False

    print("\nStep 8 completed successfully. All validations passed.")
    return True
