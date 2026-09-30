"""Tests for Step 8 experiment tracking (spec section K).

Covers:
- Protocol-manifest schema validation (including ``final_model_rule`` presence)
- Per-run artifact schema validation (all C2 minimum fields)
- Aggregate artifact schema validation (including named summary groups from D2)
- ``run_ledger.json`` correctness (requested/completed/failed/skipped runs,
  seed lists, failure reasons, per-run metadata refs)
- Representative-run metadata correctness (points to real completed run)
- Cross-model comparison-table schema consistency (all 3 rows, canonical order,
  all E2 required columns)
- Parity across all 3 models (same file sets, same core fields across F parity items)
- Provenance references resolving to real files
- Cross-referencing assertions: frozen-config consistency, seed/artifact agreement,
  label-order match
- No-reconstruction rule enforcement: missing artifact causes clear failure
- Legacy isolation: canonical resolution does not depend on legacy filenames
- Content and linkage validation, not just filename presence
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

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
from src.run_ledger import load_current_run_ledger
from src.experiment_tracking import (
    CANONICAL_MODEL_ORDER,
    LEGACY_DIRS,
    _read_json,
    _to_repo_relative,
    _validate_metric_ranges,
    _validate_path_ref,
    _write_json,
    build_efficiency_summary_table,
    build_model_comparison_aggregate,
    build_oos_summary_table,
    cross_reference_assertions,
    document_legacy_artifacts,
    enrich_comparison_row,
    enrich_evaluation_protocol,
    enrich_frozen_config,
    enrich_representative_run,
    generate_run_ledger,
    run_experiment_tracking,
    run_parity_audit,
    validate_aggregate_metrics,
    validate_comparison_row,
    validate_frozen_config,
    validate_per_run_bundle,
    validate_protocol_manifest,
    validate_representative_run,
    validate_run_ledger,
    validate_run_metadata,
    validate_tuning_artifacts,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _minimal_protocol(shared_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_config": {
            "run_count": 3,
            "seed_list": [42, 1337, 2024],
            "effective_seed_list": [42, 1337, 2024],
            "representative_run_rule": "highest_validation_macro_f1",
        },
        "models": ["mlp", "text_cnn", "bilstm"],
        "canonical_model_order": ["mlp", "text_cnn", "bilstm"],
        "canonical_model_display_names": {
            "mlp": "TF-IDF + MLP",
            "text_cnn": "Text CNN",
            "bilstm": "BiLSTM",
        },
        "oos_evaluation_policy": {
            "oos_class_name": "oos",
            "oos_class_id": 42,
            "evaluation_method": "explicit_class",
            "one_vs_rest_rule": "oos is positive; all in-scope are negative",
        },
        "metric_definitions": {"accuracy": "overall accuracy"},
        "efficiency_timing_policy": {"timing_includes_dataloader_overhead": True},
        "seed_policy": {"seed_list": [42, 1337, 2024]},
        "final_model_rule": "best checkpoint from early stopping",
    }


def _minimal_frozen_config(model_id: str) -> dict[str, Any]:
    mid = ModelID(model_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_name": mid.display_name,
        "model_id": model_id,
        "hyperparameters": {"hidden_dim": 512},
        "source_tuning_artifact": f"outputs/{model_id}/tuning/tuning_results.csv",
        "winning_row_id": "run_01",
        "selection_metric": "best_val_metric",
        "preprocessing_manifest_ref": PREPROCESSING_MANIFEST_REF,
        "label_order_ref": LABEL_ORDER_REF,
        "protocol_manifest_ref": PROTOCOL_MANIFEST_REF,
    }


def _minimal_run_metadata(
    model_id: str,
    run_id: str,
    *,
    run_index: int = 1,
    seed: int = 42,
    checkpoint_path: str = "",
    log_path: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_name": ModelID(model_id).display_name,
        "model_id": model_id,
        "run_id": run_id,
        "run_index": run_index,
        "seed": seed,
        "training_seed": seed,
        "dataloader_seed": seed + 1,
        "status": "completed",
        "best_epoch": 10,
        "stopping_epoch": 15,
        "best_val_metric": 0.91,
        "best_val_loss": 0.3,
        "monitor_metric": "val_macro_f1",
        "checkpoint_path": checkpoint_path,
        "log_path": log_path,
        "frozen_config_ref": f"outputs/{model_id}/frozen_final_config.json",
        "protocol_manifest_ref": PROTOCOL_MANIFEST_REF,
        "preprocessing_manifest_ref": PREPROCESSING_MANIFEST_REF,
        "label_order_ref": LABEL_ORDER_REF,
        "frozen_config_hash": "abcd1234",
    }


def _minimal_aggregate_metrics(model_id: str) -> dict[str, Any]:
    mid = ModelID(model_id)
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_name": mid.display_name,
        "model_id": model_id,
        "run_count_requested": 3,
        "run_count_completed": 3,
        "seed_list_requested": [42, 1337, 2024],
        "seed_list_completed": [42, 1337, 2024],
        "all_runs_succeeded": True,
        "successful_run_ids": [
            "run_01_seed_42",
            "run_02_seed_1337",
            "run_03_seed_2024",
        ],
        "failed_run_ids": [],
        "representative_run_id": "run_01_seed_42",
        "frozen_config_ref": f"outputs/{model_id}/frozen_final_config.json",
        "protocol_manifest_ref": PROTOCOL_MANIFEST_REF,
        "metrics": {
            "test_accuracy": {"mean": 0.85, "std": 0.01},
            "test_macro_f1": {"mean": 0.83, "std": 0.02},
            "oos_f1": {"mean": 0.65, "std": 0.03},
        },
    }
    for group_name, metric_keys in SUMMARY_GROUPS.items():
        group: dict[str, dict[str, float]] = {}
        for mk in metric_keys:
            group[mk] = {"mean": 0.5, "std": 0.01}
        data[group_name] = group
    return data


def _minimal_comparison_row(model_id: str) -> dict[str, Any]:
    mid = ModelID(model_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_name": mid.display_name,
        "model_id": model_id,
        "display_name": mid.display_name,
        "input_type": mid.input_type,
        "run_count": 3,
        "primary_val_metric": "val_macro_f1",
        "val_accuracy_mean": 0.90,
        "val_accuracy_std": 0.01,
        "val_macro_f1_mean": 0.88,
        "val_macro_f1_std": 0.01,
        "test_accuracy_mean": 0.85,
        "test_accuracy_std": 0.01,
        "test_macro_f1_mean": 0.83,
        "test_macro_f1_std": 0.01,
        "test_precision_mean": 0.82,
        "test_precision_std": 0.01,
        "test_recall_mean": 0.81,
        "test_recall_std": 0.01,
        "oos_precision_mean": 0.70,
        "oos_precision_std": 0.01,
        "oos_recall_mean": 0.65,
        "oos_recall_std": 0.01,
        "oos_f1_mean": 0.67,
        "oos_f1_std": 0.01,
        "training_time_seconds_mean": 100.0,
        "training_time_seconds_std": 10.0,
        "inference_total_seconds_mean": 1.0,
        "inference_total_seconds_std": 0.1,
        "inference_avg_ms_per_example_mean": 0.5,
        "inference_avg_ms_per_example_std": 0.05,
        "inference_examples_per_sec_mean": 2000.0,
        "inference_examples_per_sec_std": 200.0,
        "parameter_count": 50000,
        "trainable_parameter_count": 50000,
        "representative_run_id": "run_01_seed_42",
        "frozen_config_ref": f"outputs/{model_id}/frozen_final_config.json",
    }


def _minimal_representative_run(model_id: str) -> dict[str, Any]:
    mid = ModelID(model_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_name": mid.display_name,
        "model_id": model_id,
        "run_id": "run_01_seed_42",
        "run_index": 1,
        "seed": 42,
        "selection_rule": "highest_validation_macro_f1",
        "selection_metric_value": 0.91,
        "artifact_path": f"outputs/{model_id}/final_runs/run_01_seed_42",
        "frozen_config_ref": f"outputs/{model_id}/frozen_final_config.json",
        "protocol_manifest_ref": PROTOCOL_MANIFEST_REF,
    }


def _minimal_run_ledger(model_id: str, seeds: tuple[int, ...] = (42, 1337, 2024)) -> dict[str, Any]:
    run_ids = [f"run_{index:02d}_seed_{seed}" for index, seed in enumerate(seeds, 1)]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_name": ModelID(model_id).display_name,
        "model_id": model_id,
        "run_count_requested": len(run_ids),
        "run_count_completed": len(run_ids),
        "run_count_failed": 0,
        "run_count_skipped": 0,
        "requested_run_ids": run_ids,
        "completed_run_ids": run_ids,
        "failed_run_ids": [],
        "skipped_run_ids": [],
        "seed_list_requested": list(seeds),
        "seed_list_completed": list(seeds),
        "failure_reasons": {},
        "per_run_metadata_refs": {
            run_id: f"outputs/{model_id}/{FINAL_RUNS_SUBDIR}/{run_id}/run_metadata.json" for run_id in run_ids
        },
    }


def _build_model_tree(
    root: Path,
    model_id: str,
    *,
    seeds: tuple[int, ...] = (42, 1337, 2024),
    include_per_run_files: bool = True,
) -> None:
    """Build a minimal model output tree under *root* (standing in for outputs/)."""
    model_dir = root / model_id

    # Tuning
    tuning_dir = model_dir / TUNING_SUBDIR
    tuning_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"run_name": ["run_01"], "best_val_metric": [0.91], "best_val_loss": [0.3]}).to_csv(
        tuning_dir / "tuning_results.csv", index=False
    )
    _write(
        tuning_dir / "selection_summary.json",
        {"schema_version": SCHEMA_VERSION, "model_id": model_id},
    )

    # Frozen config
    _write(model_dir / "frozen_final_config.json", _minimal_frozen_config(model_id))

    # Final runs
    for idx, seed in enumerate(seeds, 1):
        run_id = f"run_{idx:02d}_seed_{seed}"
        run_dir = model_dir / FINAL_RUNS_SUBDIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        ckpt_dir = run_dir / "checkpoint"
        log_dir = run_dir / "logs"
        ckpt_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)

        ckpt_file = ckpt_dir / f"best_{run_id}.pt"
        ckpt_file.write_bytes(b"dummy")
        log_file = log_dir / "log.json"
        log_file.write_text("{}")

        meta = _minimal_run_metadata(
            model_id,
            run_id,
            run_index=idx,
            seed=seed,
            checkpoint_path=str(ckpt_file.relative_to(root.parent)),
            log_path=str(log_file.relative_to(root.parent)),
        )
        _write(run_dir / "run_metadata.json", meta)

        if include_per_run_files:
            for fname in PER_RUN_REQUIRED_FILES:
                fpath = run_dir / fname
                if not fpath.exists():
                    if fname.endswith(".json"):
                        _write(fpath, {"schema_version": SCHEMA_VERSION})
                    elif fname.endswith(".csv"):
                        fpath.write_text("col1,col2\n1,2\n")

            lo = run_dir / "label_order.json"
            _write(
                lo,
                {
                    "schema_version": SCHEMA_VERSION,
                    "label_names": [f"label_{i}" for i in range(151)],
                },
            )

    # Aggregate
    agg_dir = model_dir / AGGREGATE_SUBDIR
    agg_dir.mkdir(parents=True, exist_ok=True)
    _write(agg_dir / "aggregate_metrics.json", _minimal_aggregate_metrics(model_id))
    _write(
        agg_dir / "aggregate_comparison_row.json",
        _minimal_comparison_row(model_id),
    )
    _write(
        agg_dir / "representative_run.json",
        _minimal_representative_run(model_id),
    )
    _write(agg_dir / "run_ledger.json", _minimal_run_ledger(model_id, seeds))

    per_run_csv = agg_dir / "per_run_metrics.csv"
    rows = []
    for idx, seed in enumerate(seeds, 1):
        rows.append(
            {
                "run_id": f"run_{idx:02d}_seed_{seed}",
                "status": "completed",
                "test_accuracy": 0.85,
                "test_macro_f1": 0.83,
                "oos_f1": 0.65,
            }
        )
    pd.DataFrame(rows).to_csv(per_run_csv, index=False)
    pd.DataFrame([{"metric": "test_accuracy", "mean": 0.85, "std": 0.01}]).to_csv(
        agg_dir / "aggregate_metrics.csv", index=False
    )


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a full mock output tree for all 3 models under *tmp_path*.

    Returns the project root (parent of ``outputs/``).
    """
    project_root = tmp_path
    outputs = project_root / "outputs"
    shared = outputs / "shared"
    shared.mkdir(parents=True, exist_ok=True)

    _write(shared / "evaluation_protocol.json", _minimal_protocol(shared))

    for mid in ("mlp", "text_cnn", "bilstm"):
        _build_model_tree(outputs, mid)

    # Create the data artifacts that provenance refs point to
    artifacts = project_root / "data" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    _write(artifacts / "preprocessing_summary.json", {"max_seq_length": 20})
    _write(artifacts / "id_to_label.json", {str(i): f"label_{i}" for i in range(151)})

    # Every helper must resolve this fixture's references inside the same root.
    monkeypatch.setattr("src.experiment_tracking.PROJECT_ROOT", project_root)
    monkeypatch.setattr("src.experiment_tracking.model_output_dir", lambda mid: outputs / str(mid))
    monkeypatch.setattr("src.experiment_tracking.SHARED_DIR", shared)

    return project_root


# ── Protocol-manifest schema validation ──────────────────────────────────


class TestProtocolManifestValidation:
    def test_valid_manifest_passes(self) -> None:
        errors = validate_protocol_manifest()
        assert errors == []

    def test_missing_manifest_reports_error(self, tmp_path: Path) -> None:
        with patch("src.experiment_tracking.SHARED_DIR", tmp_path):
            errors = validate_protocol_manifest()
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_missing_final_model_rule_reports_error(self, tmp_path: Path) -> None:
        proto = _minimal_protocol(tmp_path)
        del proto["final_model_rule"]
        _write(tmp_path / "evaluation_protocol.json", proto)
        with patch("src.experiment_tracking.SHARED_DIR", tmp_path):
            errors = validate_protocol_manifest()
        assert any("final_model_rule" in e for e in errors)

    def test_missing_protocol_config_fields(self, tmp_path: Path) -> None:
        proto = _minimal_protocol(tmp_path)
        del proto["protocol_config"]["representative_run_rule"]
        _write(tmp_path / "evaluation_protocol.json", proto)
        with patch("src.experiment_tracking.SHARED_DIR", tmp_path):
            errors = validate_protocol_manifest()
        assert any("representative_run_rule" in e for e in errors)


# ── Per-run artifact schema validation ───────────────────────────────────


class TestPerRunArtifactValidation:
    def test_valid_run_metadata_passes(self, mock_outputs: Path) -> None:
        run_dir = mock_outputs / "outputs" / "mlp" / FINAL_RUNS_SUBDIR / "run_01_seed_42"
        with patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs):
            errors = validate_run_metadata(run_dir, ModelID.MLP)
        assert errors == []

    def test_missing_run_metadata_reports_error(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_01_seed_42"
        run_dir.mkdir()
        errors = validate_run_metadata(run_dir, ModelID.MLP)
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_missing_c2_fields_reported(self, mock_outputs: Path) -> None:
        run_dir = mock_outputs / "outputs" / "mlp" / FINAL_RUNS_SUBDIR / "run_01_seed_42"
        meta = _read_json(run_dir / "run_metadata.json")
        del meta["seed"]
        del meta["training_seed"]
        _write_json(run_dir / "run_metadata.json", meta)
        with patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs):
            errors = validate_run_metadata(run_dir, ModelID.MLP)
        assert any("seed" in e for e in errors)
        assert any("training_seed" in e for e in errors)

    def test_per_run_bundle_validates_all_required_files(self, mock_outputs: Path) -> None:
        run_dir = mock_outputs / "outputs" / "mlp" / FINAL_RUNS_SUBDIR / "run_01_seed_42"
        with patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs):
            errors = validate_per_run_bundle(run_dir, ModelID.MLP)
        assert errors == []

    def test_missing_per_run_file_reported(self, mock_outputs: Path) -> None:
        run_dir = mock_outputs / "outputs" / "mlp" / FINAL_RUNS_SUBDIR / "run_01_seed_42"
        (run_dir / "confusion_matrix.csv").unlink()
        with patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs):
            errors = validate_per_run_bundle(run_dir, ModelID.MLP)
        assert any("confusion_matrix.csv" in e for e in errors)


# ── Aggregate artifact schema validation ─────────────────────────────────


class TestAggregateArtifactValidation:
    def test_valid_aggregate_passes(self, mock_outputs: Path) -> None:
        with patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs):
            with patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ):
                errors = validate_aggregate_metrics(ModelID.MLP)
        assert errors == []

    def test_missing_summary_group_reported(self, mock_outputs: Path) -> None:
        agg_path = mock_outputs / "outputs" / "mlp" / AGGREGATE_SUBDIR / "aggregate_metrics.json"
        data = _read_json(agg_path)
        del data["validation_summary"]
        _write_json(agg_path, data)
        with patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs):
            with patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ):
                errors = validate_aggregate_metrics(ModelID.MLP)
        assert any("validation_summary" in e for e in errors)

    def test_missing_aggregate_file_reported(self, tmp_path: Path) -> None:
        with patch(
            "src.experiment_tracking.model_output_dir",
            return_value=tmp_path / "mlp",
        ):
            errors = validate_aggregate_metrics(ModelID.MLP)
        assert any("not found" in e for e in errors)


# ── Run-ledger correctness ───────────────────────────────────────────────


class TestRunLedgerCorrectness:
    def test_generate_run_ledger_produces_valid_json(self, mock_outputs: Path) -> None:
        with (
            patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs),
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
            patch(
                "src.experiment_tracking.SHARED_DIR",
                mock_outputs / "outputs" / "shared",
            ),
        ):
            ledger_path = generate_run_ledger(ModelID.MLP)

        data = _read_json(ledger_path)
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["model_id"] == "mlp"
        assert data["run_count_requested"] == 3
        assert data["run_count_completed"] == 3
        assert len(data["completed_run_ids"]) == 3
        assert data["failed_run_ids"] == []
        assert data["skipped_run_ids"] == []
        assert len(data["per_run_metadata_refs"]) == 3

    def test_ledger_records_failed_runs(self, mock_outputs: Path) -> None:
        run_dir = mock_outputs / "outputs" / "mlp" / FINAL_RUNS_SUBDIR / "run_03_seed_2024"
        meta = _read_json(run_dir / "run_metadata.json")
        meta["status"] = "failed"
        meta["failure_reason"] = "OOM"
        _write_json(run_dir / "run_metadata.json", meta)
        ledger_path = run_dir.parent.parent / AGGREGATE_SUBDIR / "run_ledger.json"
        ledger = _read_json(ledger_path)
        ledger["completed_run_ids"] = ledger["completed_run_ids"][:2]
        ledger["failed_run_ids"] = ["run_03_seed_2024"]
        ledger["seed_list_completed"] = [42, 1337]
        ledger["run_count_completed"] = 2
        ledger["run_count_failed"] = 1
        ledger["failure_reasons"] = {"run_03_seed_2024": "OOM"}
        _write_json(ledger_path, ledger)

        with (
            patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs),
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
            patch(
                "src.experiment_tracking.SHARED_DIR",
                mock_outputs / "outputs" / "shared",
            ),
        ):
            ledger_path = generate_run_ledger(ModelID.MLP)

        data = _read_json(ledger_path)
        assert "run_03_seed_2024" in data["failed_run_ids"]
        assert data["failure_reasons"]["run_03_seed_2024"] == "OOM"
        assert data["run_count_completed"] == 2

    def test_ledger_seed_lists(self, mock_outputs: Path) -> None:
        with (
            patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs),
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
            patch(
                "src.experiment_tracking.SHARED_DIR",
                mock_outputs / "outputs" / "shared",
            ),
        ):
            ledger_path = generate_run_ledger(ModelID.MLP)

        data = _read_json(ledger_path)
        assert data["seed_list_requested"] == [42, 1337, 2024]
        assert set(data["seed_list_completed"]) == {42, 1337, 2024}

    def test_validate_run_ledger_passes(self, mock_outputs: Path) -> None:
        with (
            patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs),
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
            patch(
                "src.experiment_tracking.SHARED_DIR",
                mock_outputs / "outputs" / "shared",
            ),
        ):
            generate_run_ledger(ModelID.MLP)
            errors = validate_run_ledger(ModelID.MLP)
        assert errors == []

    def test_tracking_retains_current_one_run_ledger_and_ignores_stale_artifacts(self, mock_outputs: Path) -> None:
        model_dir = mock_outputs / "outputs" / "mlp"
        ledger_path = model_dir / AGGREGATE_SUBDIR / "run_ledger.json"
        _write(ledger_path, _minimal_run_ledger("mlp", (42,)))
        original_ledger = ledger_path.read_bytes()
        aggregate_path = model_dir / AGGREGATE_SUBDIR / "aggregate_metrics.json"
        aggregate = _read_json(aggregate_path)
        aggregate["successful_run_ids"] = ["run_01_seed_42"]
        aggregate["run_count_requested"] = aggregate["run_count_completed"] = 1
        aggregate["seed_list_requested"] = aggregate["seed_list_completed"] = [42]
        _write(aggregate_path, aggregate)
        stale_path = model_dir / FINAL_RUNS_SUBDIR / "run_02_seed_1337" / "run_metadata.json"
        stale = _read_json(stale_path)
        stale["frozen_config_ref"] = "historical/different.json"
        stale["model_id"] = "historical"
        _write(stale_path, stale)
        stale_before = stale_path.read_bytes()

        assert generate_run_ledger(ModelID.MLP) == ledger_path
        assert cross_reference_assertions(ModelID.MLP) == []
        assert ledger_path.read_bytes() == original_ledger
        assert stale_path.read_bytes() == stale_before
        assert (model_dir / FINAL_RUNS_SUBDIR / "run_03_seed_2024").is_dir()

    def test_tracking_does_not_create_missing_ledger_from_directories(self, mock_outputs: Path) -> None:
        ledger_path = mock_outputs / "outputs" / "mlp" / AGGREGATE_SUBDIR / "run_ledger.json"
        ledger_path.unlink()
        with pytest.raises(FileNotFoundError, match="Current run ledger not found"):
            generate_run_ledger(ModelID.MLP)
        assert not ledger_path.exists()

    def test_tracking_preflight_leaves_invalid_membership_unchanged(self, mock_outputs: Path) -> None:
        model_dir = mock_outputs / "outputs" / "mlp"
        ledger_path = model_dir / AGGREGATE_SUBDIR / "run_ledger.json"
        ledger = _read_json(ledger_path)
        ledger["completed_run_ids"].append(ledger["completed_run_ids"][0])
        _write(ledger_path, ledger)
        before = {path: path.read_bytes() for path in model_dir.rglob("*") if path.is_file()}

        assert run_experiment_tracking([ModelID.MLP]) is False

        assert all(path.read_bytes() == contents for path, contents in before.items())

    @pytest.mark.parametrize(
        "mutation, message",
        [
            ("duplicate_requested", "duplicate run IDs"),
            ("duplicate_completed", "duplicate run IDs"),
            ("missing_ref", "missing per_run_metadata_refs"),
            ("extra_ref", "unrequested run IDs"),
            ("other_run_ref", "points to a different run"),
            ("absolute_ref", "must be repository-relative"),
            ("missing_metadata", "run metadata not found"),
            ("other_model", "model_id/run_id"),
            ("other_run_id", "model_id/run_id"),
            ("wrong_status", "completed ledger run has status"),
            ("failed_status", "failed ledger run has status"),
            ("skipped_status", "skipped ledger run has status"),
            ("wrong_seed", "run_index/seed"),
            ("wrong_index", "run_index/seed"),
            ("wrong_count", "run_count_completed"),
            ("status_overlap", "statuses must partition"),
            ("wrong_completed_seeds", "seed_list_completed"),
            ("wrong_ledger_model", "model_id does not match"),
            ("malformed_refs", "per_run_metadata_refs"),
            ("metadata_not_object", "expected a JSON object"),
        ],
    )
    def test_identity_mutations_are_rejected(self, mock_outputs: Path, mutation: str, message: str) -> None:
        model_dir = mock_outputs / "outputs" / "mlp"
        ledger_path = model_dir / AGGREGATE_SUBDIR / "run_ledger.json"
        ledger = _read_json(ledger_path)
        first_run = ledger["completed_run_ids"][0]
        metadata_path = model_dir / FINAL_RUNS_SUBDIR / first_run / "run_metadata.json"
        metadata = _read_json(metadata_path)

        if mutation == "duplicate_requested":
            ledger["requested_run_ids"].append(first_run)
        elif mutation == "duplicate_completed":
            ledger["completed_run_ids"].append(first_run)
        elif mutation == "missing_ref":
            del ledger["per_run_metadata_refs"][first_run]
        elif mutation == "extra_ref":
            ledger["per_run_metadata_refs"]["historical_run"] = ledger["per_run_metadata_refs"][first_run]
        elif mutation == "other_run_ref":
            ledger["per_run_metadata_refs"][first_run] = ledger["per_run_metadata_refs"]["run_02_seed_1337"]
        elif mutation == "absolute_ref":
            ledger["per_run_metadata_refs"][first_run] = str(metadata_path)
        elif mutation == "missing_metadata":
            metadata_path.unlink()
        elif mutation == "other_model":
            metadata["model_id"] = "bilstm"
        elif mutation == "other_run_id":
            metadata["run_id"] = "run_02_seed_1337"
        elif mutation == "wrong_status":
            metadata["status"] = "failed"
        elif mutation in {"failed_status", "skipped_status"}:
            group_name = "failed" if mutation == "failed_status" else "skipped"
            ledger["completed_run_ids"] = ledger["completed_run_ids"][1:]
            ledger["seed_list_completed"] = ledger["seed_list_completed"][1:]
            ledger["run_count_completed"] = 2
            ledger[f"{group_name}_run_ids"] = [first_run]
            ledger[f"run_count_{group_name}"] = 1
        elif mutation == "wrong_seed":
            metadata["seed"] = 1337
        elif mutation == "wrong_index":
            metadata["run_index"] = 2
        elif mutation == "wrong_count":
            ledger["run_count_completed"] = 1
        elif mutation == "status_overlap":
            ledger["failed_run_ids"] = [first_run]
        elif mutation == "wrong_completed_seeds":
            ledger["seed_list_completed"] = [42]
        elif mutation == "wrong_ledger_model":
            ledger["model_id"] = "bilstm"
        elif mutation == "malformed_refs":
            ledger["per_run_metadata_refs"] = []
        elif mutation == "metadata_not_object":
            metadata_path.write_text("[]")
        else:
            pytest.fail(f"Unhandled mutation: {mutation}")

        if mutation not in {"missing_metadata", "metadata_not_object"}:
            _write(metadata_path, metadata)
        _write(ledger_path, ledger)
        with pytest.raises((FileNotFoundError, ValueError), match=message):
            load_current_run_ledger(ModelID.MLP, model_dir=model_dir, project_root=mock_outputs)


# ── Representative-run metadata correctness ──────────────────────────────


class TestRepresentativeRunValidation:
    def test_valid_representative_passes(self, mock_outputs: Path) -> None:
        with (
            patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs),
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
        ):
            errors = validate_representative_run(ModelID.MLP)
        assert errors == []

    def test_missing_field_reported(self, mock_outputs: Path) -> None:
        rep_path = mock_outputs / "outputs" / "mlp" / AGGREGATE_SUBDIR / "representative_run.json"
        data = _read_json(rep_path)
        del data["selection_rule"]
        _write_json(rep_path, data)
        with patch(
            "src.experiment_tracking.model_output_dir",
            side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
        ):
            errors = validate_representative_run(ModelID.MLP)
        assert any("selection_rule" in e for e in errors)

    def test_representative_artifact_path_resolves(self, mock_outputs: Path) -> None:
        rep_path = mock_outputs / "outputs" / "mlp" / AGGREGATE_SUBDIR / "representative_run.json"
        data = _read_json(rep_path)
        assert data["artifact_path"].startswith("outputs/")

        resolved = mock_outputs / data["artifact_path"]
        assert resolved.exists()

    def test_preserved_completed_run_outside_current_ledger_is_rejected(self, mock_outputs: Path) -> None:
        model_dir = mock_outputs / "outputs" / "mlp"
        _write(model_dir / AGGREGATE_SUBDIR / "run_ledger.json", _minimal_run_ledger("mlp", (42,)))
        rep_path = model_dir / AGGREGATE_SUBDIR / "representative_run.json"
        rep = _read_json(rep_path)
        rep.update(
            run_id="run_02_seed_1337",
            run_index=2,
            seed=1337,
            artifact_path="outputs/mlp/final_runs/run_02_seed_1337",
        )
        _write(rep_path, rep)
        assert any("not a completed member" in error for error in validate_representative_run(ModelID.MLP))

    def test_artifact_path_cannot_select_another_active_run(self, mock_outputs: Path) -> None:
        rep_path = mock_outputs / "outputs" / "mlp" / AGGREGATE_SUBDIR / "representative_run.json"
        rep = _read_json(rep_path)
        rep["artifact_path"] = "outputs/mlp/final_runs/run_02_seed_1337"
        _write(rep_path, rep)
        assert any("artifact_path does not match" in error for error in validate_representative_run(ModelID.MLP))


# ── Comparison-row validation ────────────────────────────────────────────


class TestComparisonRowValidation:
    def test_valid_comparison_row_passes(self, mock_outputs: Path) -> None:
        with patch(
            "src.experiment_tracking.model_output_dir",
            side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
        ):
            errors = validate_comparison_row(ModelID.MLP)
        assert errors == []

    def test_parameter_count_mean_rejected(self, mock_outputs: Path) -> None:
        row_path = mock_outputs / "outputs" / "mlp" / AGGREGATE_SUBDIR / "aggregate_comparison_row.json"
        data = _read_json(row_path)
        data["parameter_count_mean"] = 50000.0
        _write_json(row_path, data)
        with patch(
            "src.experiment_tracking.model_output_dir",
            side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
        ):
            errors = validate_comparison_row(ModelID.MLP)
        assert any("parameter_count should be scalar" in e for e in errors)

    def test_e2_required_columns_present(self, mock_outputs: Path) -> None:
        """Verify every E2-required column is present in comparison rows."""
        row_path = mock_outputs / "outputs" / "mlp" / AGGREGATE_SUBDIR / "aggregate_comparison_row.json"
        data = _read_json(row_path)

        e2_required = {
            "schema_version",
            "protocol_version",
            "model_name",
            "model_id",
            "display_name",
            "input_type",
            "run_count",
            "primary_val_metric",
            "val_accuracy_mean",
            "val_accuracy_std",
            "val_macro_f1_mean",
            "val_macro_f1_std",
            "test_accuracy_mean",
            "test_accuracy_std",
            "test_macro_f1_mean",
            "test_macro_f1_std",
            "test_precision_mean",
            "test_precision_std",
            "test_recall_mean",
            "test_recall_std",
            "oos_precision_mean",
            "oos_precision_std",
            "oos_recall_mean",
            "oos_recall_std",
            "oos_f1_mean",
            "oos_f1_std",
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
            "representative_run_id",
            "frozen_config_ref",
        }
        missing = e2_required - set(data.keys())
        assert missing == set(), f"Missing E2 columns: {missing}"


# ── Cross-model comparison-table consistency ─────────────────────────────


class TestCrossModelComparisonTables:
    def test_model_comparison_aggregate_has_3_rows(self, mock_outputs: Path) -> None:
        with (
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
            patch("src.experiment_tracking.SHARED_DIR", mock_outputs / "outputs" / "shared"),
        ):
            csv_path, json_path = build_model_comparison_aggregate()

        data = _read_json(json_path)
        assert len(data["rows"]) == 3
        assert data["canonical_model_order"] == ["mlp", "text_cnn", "bilstm"]

        df = pd.read_csv(csv_path)
        assert len(df) == 3

    def test_oos_summary_table_correct_fields(self, mock_outputs: Path) -> None:
        with (
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
            patch("src.experiment_tracking.SHARED_DIR", mock_outputs / "outputs" / "shared"),
        ):
            _, json_path = build_oos_summary_table()

        data = _read_json(json_path)
        assert len(data["rows"]) == 3
        for row in data["rows"]:
            assert "oos_f1_mean" in row
            assert "oos_precision_mean" in row
            assert "oos_recall_mean" in row
            assert "model_id" in row

    def test_efficiency_summary_table_correct_fields(self, mock_outputs: Path) -> None:
        with (
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
            patch("src.experiment_tracking.SHARED_DIR", mock_outputs / "outputs" / "shared"),
        ):
            _, json_path = build_efficiency_summary_table()

        data = _read_json(json_path)
        assert len(data["rows"]) == 3
        for row in data["rows"]:
            assert "parameter_count" in row
            assert "trainable_parameter_count" in row
            assert "training_time_seconds_mean" in row
            assert "inference_examples_per_sec_mean" in row

    def test_canonical_model_order_in_tables(self, mock_outputs: Path) -> None:
        with (
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
            patch("src.experiment_tracking.SHARED_DIR", mock_outputs / "outputs" / "shared"),
        ):
            _, json_path = build_model_comparison_aggregate()

        data = _read_json(json_path)
        model_ids = [r["model_id"] for r in data["rows"]]
        assert model_ids == ["mlp", "text_cnn", "bilstm"]


# ── Parity audit ─────────────────────────────────────────────────────────


class TestParityAudit:
    def test_identical_trees_pass(self, mock_outputs: Path) -> None:
        with (
            patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs),
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
            patch(
                "src.experiment_tracking.SHARED_DIR",
                mock_outputs / "outputs" / "shared",
            ),
        ):
            for mid in CANONICAL_MODEL_ORDER:
                generate_run_ledger(mid)
            errors = run_parity_audit()
        assert errors == []

    def test_missing_aggregate_artifact_detected(self, mock_outputs: Path) -> None:
        (mock_outputs / "outputs" / "bilstm" / AGGREGATE_SUBDIR / "aggregate_metrics.csv").unlink(missing_ok=True)
        with patch(
            "src.experiment_tracking.model_output_dir",
            side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
        ):
            errors = run_parity_audit()
        assert any("bilstm" in e and "aggregate_metrics.csv" in e for e in errors)

    def test_missing_tuning_artifact_detected(self, mock_outputs: Path) -> None:
        (mock_outputs / "outputs" / "text_cnn" / TUNING_SUBDIR / "tuning_results.csv").unlink(missing_ok=True)
        with patch(
            "src.experiment_tracking.model_output_dir",
            side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
        ):
            errors = run_parity_audit()
        assert any("text_cnn" in e and "tuning" in e for e in errors)

    def test_extra_per_run_files_detected(self, mock_outputs: Path) -> None:
        extra = mock_outputs / "outputs" / "bilstm" / FINAL_RUNS_SUBDIR / "run_01_seed_42" / "extra_file.json"
        extra.write_text("{}")
        with patch(
            "src.experiment_tracking.model_output_dir",
            side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
        ):
            errors = run_parity_audit()
        assert any("extra" in e.lower() for e in errors)

    def test_comparison_row_schema_drift_detected(self, mock_outputs: Path) -> None:
        row_path = mock_outputs / "outputs" / "bilstm" / AGGREGATE_SUBDIR / "aggregate_comparison_row.json"
        data = _read_json(row_path)
        data["extra_field"] = 42
        _write_json(row_path, data)
        with patch(
            "src.experiment_tracking.model_output_dir",
            side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
        ):
            errors = run_parity_audit()
        assert any("schema differs" in e.lower() for e in errors)


# ── Provenance references ────────────────────────────────────────────────


class TestProvenanceReferences:
    def test_repo_relative_path_passes(self) -> None:
        errors = _validate_path_ref(
            "outputs/shared/evaluation_protocol.json",
            "test",
        )
        assert errors == []

    def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        errors = _validate_path_ref(str(tmp_path / "something"), "test")
        assert any("absolute path" in e for e in errors)

    def test_empty_path_rejected(self) -> None:
        errors = _validate_path_ref("", "test")
        assert any("empty" in e for e in errors)

    def test_nonexistent_path_rejected(self) -> None:
        errors = _validate_path_ref(
            "outputs/shared/does_not_exist.json",
            "test",
        )
        assert any("does not resolve" in e for e in errors)

    def test_to_repo_relative_converts_absolute(self) -> None:
        abs_path = PROJECT_ROOT / "outputs" / "mlp" / "test.json"
        result = _to_repo_relative(abs_path)
        assert not Path(result).is_absolute()
        assert result == "outputs/mlp/test.json"

    def test_to_repo_relative_passthrough_relative(self) -> None:
        assert _to_repo_relative("outputs/mlp/test.json") == "outputs/mlp/test.json"


# ── Cross-referencing assertions ─────────────────────────────────────────


class TestCrossReferencingAssertions:
    def test_consistent_frozen_config_refs_pass(self, mock_outputs: Path) -> None:
        with (
            patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs),
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
        ):
            errors = cross_reference_assertions(ModelID.MLP)
        assert not any("frozen config" in e.lower() for e in errors)

    def test_inconsistent_frozen_config_refs_detected(self, mock_outputs: Path) -> None:
        run2 = mock_outputs / "outputs" / "mlp" / FINAL_RUNS_SUBDIR / "run_02_seed_1337" / "run_metadata.json"
        data = _read_json(run2)
        data["frozen_config_ref"] = "outputs/mlp/different_config.json"
        _write_json(run2, data)
        with (
            patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs),
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
        ):
            errors = cross_reference_assertions(ModelID.MLP)
        assert any("frozen config refs differ" in e.lower() for e in errors)

    def test_label_order_consistency_passes(self, mock_outputs: Path) -> None:
        with (
            patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs),
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
        ):
            errors = cross_reference_assertions(ModelID.MLP)
        assert not any("label" in e.lower() for e in errors)

    def test_label_order_inconsistency_detected(self, mock_outputs: Path) -> None:
        run2_lo = mock_outputs / "outputs" / "mlp" / FINAL_RUNS_SUBDIR / "run_02_seed_1337" / "label_order.json"
        _write(
            run2_lo,
            {
                "schema_version": SCHEMA_VERSION,
                "label_names": [f"different_{i}" for i in range(151)],
            },
        )
        with (
            patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs),
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
        ):
            errors = cross_reference_assertions(ModelID.MLP)
        assert any("label" in e.lower() for e in errors)


# ── Metric range validation ──────────────────────────────────────────────


class TestMetricRangeValidation:
    def test_valid_metrics_pass(self) -> None:
        data = {"accuracy": 0.85, "macro_f1": 0.83}
        errors = _validate_metric_ranges(data, "test")
        assert errors == []

    def test_metric_above_one_rejected(self) -> None:
        data = {"accuracy": 1.5}
        errors = _validate_metric_ranges(data, "test")
        assert len(errors) == 1

    def test_metric_below_zero_rejected(self) -> None:
        data = {"macro_f1": -0.1}
        errors = _validate_metric_ranges(data, "test")
        assert len(errors) == 1

    def test_dict_metrics_validated(self) -> None:
        data = {"test_accuracy": {"mean": 0.85, "std": 0.02}}
        errors = _validate_metric_ranges(data, "test")
        assert errors == []

    def test_dict_metric_out_of_range(self) -> None:
        data = {"test_accuracy": {"mean": 1.5, "std": 0.02}}
        errors = _validate_metric_ranges(data, "test")
        assert len(errors) == 1

    def test_non_ratio_metrics_ignored(self) -> None:
        data = {"training_time_seconds": 150.0, "parameter_count": 5000000}
        errors = _validate_metric_ranges(data, "test")
        assert errors == []


# ── No-reconstruction rule ───────────────────────────────────────────────


class TestNoReconstructionRule:
    def test_missing_per_run_bundle_file_fails_validation(self, mock_outputs: Path) -> None:
        """Missing artifact causes clear failure, not silent rebuild (spec G3)."""
        run_dir = mock_outputs / "outputs" / "mlp" / FINAL_RUNS_SUBDIR / "run_01_seed_42"
        (run_dir / "test_metrics.json").unlink()
        with patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs):
            errors = validate_per_run_bundle(run_dir, ModelID.MLP)
        assert any("test_metrics.json" in e for e in errors)

    def test_missing_frozen_config_fails_validation(self, tmp_path: Path) -> None:
        with patch("src.experiment_tracking.model_output_dir", return_value=tmp_path / "mlp"):
            errors = validate_frozen_config(ModelID.MLP)
        assert any("not found" in e for e in errors)

    def test_missing_comparison_row_fails(self, tmp_path: Path) -> None:
        with patch("src.experiment_tracking.model_output_dir", return_value=tmp_path / "mlp"):
            errors = validate_comparison_row(ModelID.MLP)
        assert any("not found" in e for e in errors)


# ── Frozen config validation ─────────────────────────────────────────────


class TestFrozenConfigValidation:
    def test_valid_frozen_config_passes(self, mock_outputs: Path) -> None:
        with (
            patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs),
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
        ):
            errors = validate_frozen_config(ModelID.MLP)
        assert errors == []

    def test_missing_protocol_manifest_ref_reported(self, mock_outputs: Path) -> None:
        fc = mock_outputs / "outputs" / "mlp" / "frozen_final_config.json"
        data = _read_json(fc)
        del data["protocol_manifest_ref"]
        _write_json(fc, data)
        with (
            patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs),
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
        ):
            errors = validate_frozen_config(ModelID.MLP)
        assert any("protocol_manifest_ref" in e for e in errors)


# ── Tuning artifacts validation ──────────────────────────────────────────


class TestTuningArtifactsValidation:
    def test_valid_tuning_passes(self, mock_outputs: Path) -> None:
        with patch(
            "src.experiment_tracking.model_output_dir",
            side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
        ):
            errors = validate_tuning_artifacts(ModelID.MLP)
        assert errors == []

    def test_missing_tuning_csv_reported(self, mock_outputs: Path) -> None:
        (mock_outputs / "outputs" / "mlp" / TUNING_SUBDIR / "tuning_results.csv").unlink()
        with patch(
            "src.experiment_tracking.model_output_dir",
            side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
        ):
            errors = validate_tuning_artifacts(ModelID.MLP)
        assert any("tuning_results.csv" in e for e in errors)


# ── Enrichment idempotency ───────────────────────────────────────────────


class TestEnrichmentIdempotency:
    def test_enrich_frozen_config_idempotent(self, mock_outputs: Path) -> None:
        with (
            patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs),
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
        ):
            enrich_frozen_config(ModelID.MLP)
            changes2 = enrich_frozen_config(ModelID.MLP)
        assert changes2 == []

    def test_enrich_representative_run_idempotent(self, mock_outputs: Path) -> None:
        with (
            patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs),
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
        ):
            enrich_representative_run(ModelID.MLP)
            changes2 = enrich_representative_run(ModelID.MLP)
        assert changes2 == []

    def test_enrich_comparison_row_idempotent(self, mock_outputs: Path) -> None:
        with (
            patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs),
            patch(
                "src.experiment_tracking.model_output_dir",
                side_effect=lambda mid: mock_outputs / "outputs" / str(mid),
            ),
        ):
            enrich_comparison_row(ModelID.MLP)
            changes2 = enrich_comparison_row(ModelID.MLP)
        assert changes2 == []


# ── Legacy isolation ─────────────────────────────────────────────────────


class TestLegacyIsolation:
    def test_canonical_resolution_independent_of_legacy(self) -> None:
        """Canonical resolution does not depend on legacy filenames (spec A2)."""
        for mid in CANONICAL_MODEL_ORDER:
            model_dir = model_output_dir(mid)
            tuning_path = model_dir / TUNING_SUBDIR / "tuning_results.csv"
            assert "reports" not in str(tuning_path)

    def test_document_legacy_artifacts_records_existing_dirs(self) -> None:
        notes = document_legacy_artifacts()
        for legacy_rel in LEGACY_DIRS:
            legacy_path = PROJECT_ROOT / legacy_rel
            if legacy_path.exists() and any(legacy_path.iterdir()):
                assert any(legacy_rel in n for n in notes)


# ── Evaluation-protocol enrichment ───────────────────────────────────────


class TestProtocolEnrichment:
    def test_adds_final_model_rule_if_missing(self, tmp_path: Path) -> None:
        proto = _minimal_protocol(tmp_path)
        del proto["final_model_rule"]
        _write(tmp_path / "evaluation_protocol.json", proto)
        with patch("src.experiment_tracking.SHARED_DIR", tmp_path):
            changes = enrich_evaluation_protocol()
        assert any("final_model_rule" in c for c in changes)
        enriched = _read_json(tmp_path / "evaluation_protocol.json")
        assert "final_model_rule" in enriched

    def test_no_change_when_already_present(self) -> None:
        changes = enrich_evaluation_protocol()
        assert changes == []


# ── Integration: full tracking against real artifacts ────────────────────


def test_tracking_rejects_mixed_generation_before_enrichment(mock_outputs: Path) -> None:
    metadata_path = mock_outputs / "outputs/mlp/final_runs/run_01_seed_42/run_metadata.json"
    metadata = _read_json(metadata_path)
    metadata["provenance"] = {}
    _write(metadata_path, metadata)
    before = {path: path.read_bytes() for path in mock_outputs.rglob("*") if path.is_file()}
    with (
        patch("src.experiment_tracking.PROJECT_ROOT", mock_outputs),
        patch("src.experiment_tracking.model_output_dir", side_effect=lambda mid: mock_outputs / "outputs" / str(mid)),
    ):
        assert run_experiment_tracking() is False
    after = {path: path.read_bytes() for path in mock_outputs.rglob("*") if path.is_file()}
    assert before == after


@pytest.mark.generated_artifacts
class TestIntegrationRealArtifacts:
    """Run validation against the actual project output artifacts.

    These tests are effectively integration checks that verify Phase 1
    artifacts are correctly structured.  They will be skipped if the
    artifacts don't exist (e.g. in CI without a full run).
    """

    @pytest.fixture(autouse=True)
    def _require_real_artifacts(self) -> None:
        proto = SHARED_DIR / "evaluation_protocol.json"
        if not proto.exists():
            pytest.skip("Real artifacts not present (e.g. CI)")

    def test_protocol_manifest_valid(self) -> None:
        assert validate_protocol_manifest() == []

    @pytest.mark.parametrize("model_id", list(ModelID))
    def test_frozen_config_valid(self, model_id: ModelID) -> None:
        assert validate_frozen_config(model_id) == []

    @pytest.mark.parametrize("model_id", list(ModelID))
    def test_tuning_artifacts_valid(self, model_id: ModelID) -> None:
        assert validate_tuning_artifacts(model_id) == []

    @pytest.mark.parametrize("model_id", list(ModelID))
    def test_aggregate_metrics_valid(self, model_id: ModelID) -> None:
        assert validate_aggregate_metrics(model_id) == []

    @pytest.mark.parametrize("model_id", list(ModelID))
    def test_comparison_row_valid(self, model_id: ModelID) -> None:
        assert validate_comparison_row(model_id) == []

    @pytest.mark.parametrize("model_id", list(ModelID))
    def test_representative_run_valid(self, model_id: ModelID) -> None:
        assert validate_representative_run(model_id) == []

    @pytest.mark.parametrize("model_id", list(ModelID))
    def test_run_ledger_valid(self, model_id: ModelID) -> None:
        assert validate_run_ledger(model_id) == []

    @pytest.mark.parametrize("model_id", list(ModelID))
    def test_per_run_bundles_valid(self, model_id: ModelID) -> None:
        ledger = load_current_run_ledger(model_id)
        for rd in ledger.completed_run_dirs:
            errors = validate_per_run_bundle(rd, model_id)
            assert errors == [], f"Errors in {rd.name}: {errors}"

    @pytest.mark.parametrize("model_id", list(ModelID))
    def test_cross_reference_assertions_pass(self, model_id: ModelID) -> None:
        errors = cross_reference_assertions(model_id)
        assert errors == [], f"Cross-ref errors for {model_id}: {errors}"

    def test_parity_audit_passes(self) -> None:
        assert run_parity_audit() == []

    def test_shared_tables_exist(self) -> None:
        for name in (
            "model_comparison_aggregate.csv",
            "model_comparison_aggregate.json",
            "oos_summary_table.csv",
            "oos_summary_table.json",
            "efficiency_summary_table.csv",
            "efficiency_summary_table.json",
        ):
            assert (SHARED_DIR / name).exists(), f"Missing shared table: {name}"

    def test_model_comparison_has_3_rows_canonical_order(self) -> None:
        data = _read_json(SHARED_DIR / "model_comparison_aggregate.json")
        assert len(data["rows"]) == 3
        ids = [r["model_id"] for r in data["rows"]]
        assert ids == ["mlp", "text_cnn", "bilstm"]

    def test_all_provenance_refs_resolve(self) -> None:
        for model_id in CANONICAL_MODEL_ORDER:
            fc_path = model_output_dir(model_id) / "frozen_final_config.json"
            data = _read_json(fc_path)
            for ref_field in (
                "source_tuning_artifact",
                "preprocessing_manifest_ref",
                "label_order_ref",
                "protocol_manifest_ref",
            ):
                ref = data.get(ref_field, "")
                if ref:
                    assert (PROJECT_ROOT / ref).exists(), (
                        f"{model_id} frozen_config.{ref_field} -> {ref} does not resolve"
                    )

    def test_no_absolute_paths_in_run_metadata(self) -> None:
        for model_id in CANONICAL_MODEL_ORDER:
            for rd in load_current_run_ledger(model_id).completed_run_dirs:
                meta = _read_json(rd / "run_metadata.json")
                for field in ("checkpoint_path", "log_path"):
                    val = meta.get(field, "")
                    assert not Path(val).is_absolute(), f"{model_id}/{rd.name}: {field} is absolute: {val}"

    def test_run_experiment_tracking_succeeds(self) -> None:
        result = run_experiment_tracking()
        assert result is True
