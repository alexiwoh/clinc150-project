"""Tests for src.report_figure_generation — the figure-generation orchestrator.

Covers:
- Preflight validation (missing artifacts produce clear errors)
- Representative-run context resolution
- Representative-run figure generation
- Aggregate cross-model figure generation
- Most-confused-pairs table schema and content
- Representative-examples index structure
- Figure-manifest completeness
- Step 10 handoff bundle path validation
- Representative misclassification extraction
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
    MODEL_FIGURES_SUBDIR,
    PER_RUN_REQUIRED_FILES,
    PROTOCOL_MANIFEST_REF,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
)
from src.enums import ModelID
from src.report_figure_generation import (
    FigureRecord,
    _resolve_representative_context,
    generate_aggregate_figures,
    generate_step10_handoff,
    generate_figure_manifest,
    generate_most_confused_pairs_table,
    generate_representative_examples_index,
    generate_representative_figures,
    generate_representative_misclassifications,
    run_preflight_validation,
    run_report_figure_generation,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _minimal_epoch_history() -> dict[str, Any]:
    epochs = []
    for i in range(1, 16):
        epochs.append(
            {
                "epoch": i,
                "train_loss": 4.0 - i * 0.2,
                "val_loss": 3.5 - i * 0.15,
                "val_accuracy": 0.5 + i * 0.02,
                "val_macro_f1": 0.6 + i * 0.015,
            }
        )
    return {"schema_version": SCHEMA_VERSION, "protocol_version": PROTOCOL_VERSION, "epochs": epochs}


def _minimal_per_class_metrics() -> dict[str, Any]:
    classes = []
    for i in range(5):
        classes.append(
            {
                "label_id": i,
                "label_name": f"label_{i}",
                "support": 30,
                "precision": 0.8 + i * 0.02,
                "recall": 0.7 + i * 0.03,
                "f1": 0.75 + i * 0.02,
                "is_oos": i == 4,
            }
        )
    return {"schema_version": SCHEMA_VERSION, "protocol_version": PROTOCOL_VERSION, "classes": classes}


def _minimal_top_confusions() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "confusions": [
            {"true_label": "label_0", "predicted_label": "label_1", "count": 10},
            {"true_label": "label_4", "predicted_label": "label_2", "count": 8},
        ],
    }


def _minimal_top_errors() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "selection_rule": "test",
        "label_ordering": [f"label_{i}" for i in range(5)],
        "errors": [
            {"text": "sample1", "true_label": "label_0", "predicted_label": "label_1"},
            {"text": "sample2", "true_label": "label_4", "predicted_label": "label_2"},
        ],
    }


def _minimal_test_metrics() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "test_accuracy": 0.85,
        "test_macro_f1": 0.83,
        "oos_precision": 0.70,
        "oos_recall": 0.65,
        "oos_f1": 0.67,
    }


def _minimal_label_order() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "label_names": [f"label_{i}" for i in range(5)],
    }


def _minimal_run_metadata(model_id: str, run_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_name": ModelID(model_id).display_name,
        "model_id": model_id,
        "run_id": run_id,
        "run_index": 1,
        "seed": 42,
        "training_seed": 42,
        "dataloader_seed": 43,
        "status": "completed",
        "best_epoch": 10,
        "stopping_epoch": 15,
        "best_val_metric": 0.91,
        "best_val_loss": 0.3,
        "monitor_metric": "val_macro_f1",
        "checkpoint_path": f"outputs/{model_id}/{FINAL_RUNS_SUBDIR}/{run_id}/checkpoint/best.pt",
        "log_path": f"outputs/{model_id}/{FINAL_RUNS_SUBDIR}/{run_id}/logs/log.json",
        "frozen_config_ref": f"outputs/{model_id}/frozen_final_config.json",
        "protocol_manifest_ref": PROTOCOL_MANIFEST_REF,
    }


def _minimal_representative_run(model_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_name": ModelID(model_id).display_name,
        "model_id": model_id,
        "run_id": "run_01_seed_42",
        "run_index": 1,
        "seed": 42,
        "selection_rule": "highest_validation_macro_f1",
        "selection_metric_value": 0.91,
        "artifact_path": f"outputs/{model_id}/{FINAL_RUNS_SUBDIR}/run_01_seed_42",
        "frozen_config_ref": f"outputs/{model_id}/frozen_final_config.json",
        "protocol_manifest_ref": PROTOCOL_MANIFEST_REF,
    }


def _minimal_comparison_row(model_id: str) -> dict[str, Any]:
    mid = ModelID(model_id)
    return {
        "model_name": mid.display_name,
        "model_id": model_id,
        "display_name": mid.display_name,
        "run_count": 3,
        "test_accuracy_mean": 0.85,
        "test_accuracy_std": 0.01,
        "test_macro_f1_mean": 0.83,
        "test_macro_f1_std": 0.02,
        "oos_f1_mean": 0.67,
        "oos_f1_std": 0.01,
    }


_FINAL_PREDICTIONS_CSV = (
    "example_index,text,true_label_id,true_label_name,predicted_label_id,"
    "predicted_label_name,run_id,max_confidence\n"
    "0,hello,0,label_0,0,label_0,run_01_seed_42,0.95\n"
    "1,world,1,label_1,2,label_2,run_01_seed_42,0.80\n"
    "2,test,4,label_4,3,label_3,run_01_seed_42,0.60\n"
)

_CONFUSION_CSV = "label_0,label_1,label_2\n10,2,0\n1,8,1\n0,0,9\n"


def _build_model_tree(root: Path, model_id: str) -> None:
    """Build a minimal model output tree under *root* (standing in for outputs/)."""
    model_dir = root / model_id
    run_id = "run_01_seed_42"
    run_dir = model_dir / FINAL_RUNS_SUBDIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _write(run_dir / "run_metadata.json", _minimal_run_metadata(model_id, run_id))
    _write(run_dir / "epoch_history.json", _minimal_epoch_history())
    _write_csv(run_dir / "confusion_matrix.csv", _CONFUSION_CSV)
    _write(run_dir / "per_class_metrics.json", _minimal_per_class_metrics())
    _write(run_dir / "top_confusions.json", _minimal_top_confusions())
    _write(run_dir / "top_errors.json", _minimal_top_errors())
    _write(run_dir / "test_metrics.json", _minimal_test_metrics())
    _write(run_dir / "label_order.json", _minimal_label_order())
    _write_csv(run_dir / "final_predictions.csv", _FINAL_PREDICTIONS_CSV)

    for fname in PER_RUN_REQUIRED_FILES:
        fpath = run_dir / fname
        if not fpath.exists():
            if fname.endswith(".json"):
                _write(fpath, {"schema_version": SCHEMA_VERSION})
            else:
                _write_csv(fpath, "col1\n1\n")

    agg_dir = model_dir / AGGREGATE_SUBDIR
    agg_dir.mkdir(parents=True, exist_ok=True)
    _write(agg_dir / "representative_run.json", _minimal_representative_run(model_id))
    _write(
        agg_dir / "run_ledger.json",
        {
            "model_id": model_id,
            "run_count_requested": 1,
            "run_count_completed": 1,
            "run_count_failed": 0,
            "run_count_skipped": 0,
            "requested_run_ids": [run_id],
            "completed_run_ids": [run_id],
            "failed_run_ids": [],
            "skipped_run_ids": [],
            "seed_list_requested": [42],
            "seed_list_completed": [42],
            "per_run_metadata_refs": {run_id: f"outputs/{model_id}/{FINAL_RUNS_SUBDIR}/{run_id}/run_metadata.json"},
        },
    )
    _write(
        agg_dir / "aggregate_metrics.json",
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "model_id": model_id,
        },
    )


def _build_shared_tables(shared_dir: Path) -> None:
    """Write the three shared aggregate tables."""
    shared_dir.mkdir(parents=True, exist_ok=True)

    comp_rows = [_minimal_comparison_row(str(mid)) for mid in ModelID]
    _write(
        shared_dir / "model_comparison_aggregate.json",
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "canonical_model_order": [str(mid) for mid in ModelID],
            "rows": comp_rows,
        },
    )

    oos_rows = []
    for mid in ModelID:
        oos_rows.append(
            {
                "display_name": mid.display_name,
                "model_id": str(mid),
                "run_count": 3,
                "oos_precision_mean": 0.70,
                "oos_precision_std": 0.01,
                "oos_recall_mean": 0.65,
                "oos_recall_std": 0.01,
                "oos_f1_mean": 0.67,
                "oos_f1_std": 0.01,
            }
        )
    _write(
        shared_dir / "oos_summary_table.json",
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "canonical_model_order": [str(mid) for mid in ModelID],
            "rows": oos_rows,
        },
    )

    eff_rows = []
    for mid in ModelID:
        eff_rows.append(
            {
                "display_name": mid.display_name,
                "model_id": str(mid),
                "run_count": 3,
                "training_time_seconds_mean": 100.0,
                "training_time_seconds_std": 10.0,
                "inference_examples_per_sec_mean": 5000.0,
                "inference_examples_per_sec_std": 500.0,
                "trainable_parameter_count": 50000,
            }
        )
    _write(
        shared_dir / "efficiency_summary_table.json",
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "canonical_model_order": [str(mid) for mid in ModelID],
            "rows": eff_rows,
        },
    )


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_outputs(tmp_path: Path) -> Path:
    """Build a complete mock outputs directory that mirrors the real layout."""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    for mid in ModelID:
        _build_model_tree(outputs_dir, str(mid))

    _build_shared_tables(outputs_dir / "shared")

    return tmp_path


def _patch_roots(tmp_path: Path):
    """Return a context manager that patches PROJECT_ROOT and derived dirs."""
    return patch.multiple(
        "src.report_figure_generation",
        PROJECT_ROOT=tmp_path,
        SHARED_DIR=tmp_path / "outputs" / "shared",
        SHARED_FIGURES_DIR=tmp_path / "outputs" / "shared" / "figures",
    )


def _patch_model_output_dir(tmp_path: Path):
    return patch(
        "src.report_figure_generation.model_output_dir",
        side_effect=lambda mid: tmp_path / "outputs" / str(mid),
    )


# ── Preflight validation ─────────────────────────────────────────────────


class TestPreflightValidation:
    def test_inactive_representative_fails_before_figure_outputs(self, mock_outputs: Path) -> None:
        model_dir = mock_outputs / "outputs" / "mlp"
        rep_path = model_dir / AGGREGATE_SUBDIR / "representative_run.json"
        rep = json.loads(rep_path.read_text())
        rep["run_id"] = "run_02_seed_1337"
        rep["artifact_path"] = "outputs/mlp/final_runs/run_02_seed_1337"
        _write(rep_path, rep)
        with (
            _patch_roots(mock_outputs),
            _patch_model_output_dir(mock_outputs),
            pytest.raises(ValueError, match="not a completed member"),
        ):
            run_preflight_validation([ModelID.MLP])
        assert not (model_dir / MODEL_FIGURES_SUBDIR).exists()

    def test_direct_misclassification_generation_requires_active_representative(self, mock_outputs: Path) -> None:
        model_dir = mock_outputs / "outputs" / "mlp"
        rep_path = model_dir / AGGREGATE_SUBDIR / "representative_run.json"
        rep = json.loads(rep_path.read_text())
        rep["artifact_path"] = "outputs/mlp/final_runs/run_02_seed_1337"
        _write(rep_path, rep)
        with (
            _patch_roots(mock_outputs),
            _patch_model_output_dir(mock_outputs),
            pytest.raises(ValueError, match="artifact_path does not match"),
        ):
            generate_representative_misclassifications(ModelID.MLP)
        assert not (model_dir / AGGREGATE_SUBDIR / "representative_misclassifications.csv").exists()

    def test_passes_with_complete_artifacts(self, mock_outputs: Path) -> None:
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            run_preflight_validation(list(ModelID))

    def test_fails_on_missing_shared_table(self, mock_outputs: Path) -> None:
        (mock_outputs / "outputs" / "shared" / "oos_summary_table.json").unlink()
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            with pytest.raises(FileNotFoundError, match="oos_summary_table"):
                run_preflight_validation(list(ModelID))

    def test_fails_on_missing_representative_run(self, mock_outputs: Path) -> None:
        (mock_outputs / "outputs" / "mlp" / AGGREGATE_SUBDIR / "representative_run.json").unlink()
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            with pytest.raises(FileNotFoundError, match="representative_run"):
                run_preflight_validation(list(ModelID))

    def test_fails_on_missing_per_run_artifact(self, mock_outputs: Path) -> None:
        run_dir = mock_outputs / "outputs" / "mlp" / FINAL_RUNS_SUBDIR / "run_01_seed_42"
        (run_dir / "epoch_history.json").unlink()
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            with pytest.raises(FileNotFoundError, match="epoch_history"):
                run_preflight_validation(list(ModelID))


# ── Representative-run context resolution ─────────────────────────────────


class TestRepresentativeContextResolution:
    def test_resolves_run_id(self, mock_outputs: Path) -> None:
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            ctx = _resolve_representative_context(ModelID.MLP)
        assert ctx.run_id == "run_01_seed_42"
        assert ctx.seed == 42
        assert ctx.best_epoch == 10
        assert ctx.stopping_epoch == 15

    def test_loads_epoch_history(self, mock_outputs: Path) -> None:
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            ctx = _resolve_representative_context(ModelID.MLP)
        assert len(ctx.epoch_history) == 15
        assert ctx.epoch_history[0]["epoch"] == 1

    def test_loads_label_names(self, mock_outputs: Path) -> None:
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            ctx = _resolve_representative_context(ModelID.MLP)
        assert ctx.label_names == [f"label_{i}" for i in range(5)]

    def test_label_order_matches_saved_artifact(self, mock_outputs: Path) -> None:
        """Verify the resolved label names match the label_order.json artifact exactly."""
        run_dir = mock_outputs / "outputs" / "mlp" / FINAL_RUNS_SUBDIR / "run_01_seed_42"
        saved = json.loads((run_dir / "label_order.json").read_text())
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            ctx = _resolve_representative_context(ModelID.MLP)
        assert ctx.label_names == saved["label_names"]


# ── Representative figure generation ─────────────────────────────────────


class TestRepresentativeFigures:
    def test_generates_all_expected_figures(self, mock_outputs: Path) -> None:
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            records = generate_representative_figures(ModelID.MLP)

        fig_dir = mock_outputs / "outputs" / "mlp" / MODEL_FIGURES_SUBDIR
        expected_names = {
            "representative_train_val_loss_curve.png",
            "representative_val_macro_f1_curve.png",
            "representative_val_accuracy_curve.png",
            "representative_confusion_matrix.png",
            "representative_top_confused_pairs.png",
            "representative_bottom_classes_f1.png",
            "representative_oos_metrics.png",
            "representative_error_summary.png",
        }
        actual_names = {p.name for p in fig_dir.iterdir() if p.suffix == ".png"}
        assert expected_names == actual_names

        assert len(records) == len(expected_names)
        for rec in records:
            assert rec.scope == "representative"
            assert rec.model_name == "mlp"
            assert rec.representative_run_id == "run_01_seed_42"

    def test_training_figure_records_include_epoch_metadata(self, mock_outputs: Path) -> None:
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            records = generate_representative_figures(ModelID.MLP)

        training_types = {"train_val_loss_curve", "val_macro_f1_curve", "val_accuracy_curve"}
        training_records = [r for r in records if r.figure_type in training_types]
        assert len(training_records) == 3

        for rec in training_records:
            assert rec.seed == 42
            assert rec.best_epoch == 10
            assert rec.stopping_epoch == 15
            assert rec.caption_context["dataset_split"] == "validation"
            assert rec.caption_context["selection_rule"] == "highest_validation_macro_f1"

    def test_figure_files_are_nonempty(self, mock_outputs: Path) -> None:
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            generate_representative_figures(ModelID.MLP)

        fig_dir = mock_outputs / "outputs" / "mlp" / MODEL_FIGURES_SUBDIR
        for png in fig_dir.glob("*.png"):
            assert png.stat().st_size > 0, f"{png.name} is empty"


# ── Aggregate figure generation ──────────────────────────────────────────


class TestAggregateFigures:
    def test_generates_all_shared_figures(self, mock_outputs: Path) -> None:
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            records = generate_aggregate_figures()

        fig_dir = mock_outputs / "outputs" / "shared" / "figures"
        expected_names = {
            "model_comparison_test_accuracy.png",
            "model_comparison_test_macro_f1.png",
            "model_comparison_oos_f1.png",
            "oos_metrics_comparison.png",
            "model_efficiency_comparison.png",
        }
        actual_names = {p.name for p in fig_dir.iterdir() if p.suffix == ".png"}
        assert expected_names == actual_names

        assert len(records) == len(expected_names)
        for rec in records:
            assert rec.scope == "aggregate"

    def test_figure_files_are_nonempty(self, mock_outputs: Path) -> None:
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            generate_aggregate_figures()

        fig_dir = mock_outputs / "outputs" / "shared" / "figures"
        for png in fig_dir.glob("*.png"):
            assert png.stat().st_size > 0, f"{png.name} is empty"


# ── Most confused pairs table ────────────────────────────────────────────


class TestMostConfusedPairsTable:
    def test_creates_csv_and_json(self, mock_outputs: Path) -> None:
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            csv_path, json_path = generate_most_confused_pairs_table(list(ModelID))

        assert csv_path.exists()
        assert json_path.exists()

    def test_schema_fields(self, mock_outputs: Path) -> None:
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            _, json_path = generate_most_confused_pairs_table(list(ModelID))

        data = json.loads(json_path.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["protocol_version"] == PROTOCOL_VERSION

        required_keys = {
            "model_name",
            "model_id",
            "representative_run_id",
            "rank",
            "true_label_name",
            "predicted_label_name",
            "count",
        }
        for row in data["rows"]:
            assert required_keys.issubset(row.keys())

    def test_rows_per_model(self, mock_outputs: Path) -> None:
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            _, json_path = generate_most_confused_pairs_table(list(ModelID))

        data = json.loads(json_path.read_text())
        model_ids = {r["model_id"] for r in data["rows"]}
        assert model_ids == {str(mid) for mid in ModelID}


# ── Representative misclassifications ────────────────────────────────────


class TestRepresentativeMisclassifications:
    def test_extracts_misclassified_rows(self, mock_outputs: Path) -> None:
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            out = generate_representative_misclassifications(ModelID.MLP)

        df = pd.read_csv(out)
        assert len(df) == 2
        assert "true_label_name" in df.columns
        assert "predicted_label_name" in df.columns
        assert "text" in df.columns
        assert "run_id" in df.columns


# ── Representative examples index ────────────────────────────────────────


class TestRepresentativeExamplesIndex:
    def test_maps_all_models(self, mock_outputs: Path) -> None:
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            for mid in ModelID:
                generate_representative_misclassifications(mid)
            out = generate_representative_examples_index(list(ModelID))

        data = json.loads(out.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        assert set(data["models"].keys()) == {str(mid) for mid in ModelID}

        for mid_str, entry in data["models"].items():
            assert "representative_run_id" in entry
            assert "representative_predictions_path" in entry
            assert "representative_misclassifications_path" in entry
            assert "representative_confusion_path" in entry


# ── Figure manifest ──────────────────────────────────────────────────────


class TestFigureManifest:
    def test_includes_all_records(self, mock_outputs: Path) -> None:
        records = [
            FigureRecord(
                figure_path="outputs/mlp/figures/test.png",
                figure_type="test_fig",
                scope="representative",
                source_artifact_paths=["outputs/mlp/test.json"],
                model_name="mlp",
                representative_run_id="run_01_seed_42",
                seed=42,
                best_epoch=10,
                stopping_epoch=15,
                caption_context={"dataset_name": "CLINC150", "dataset_split": "test"},
            ),
            FigureRecord(
                figure_path="outputs/shared/figures/agg.png",
                figure_type="agg_fig",
                scope="aggregate",
                source_artifact_paths=["outputs/shared/table.json"],
            ),
        ]
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            out = generate_figure_manifest(records)

        data = json.loads(out.read_text())
        assert len(data["figures"]) == 2
        assert data["schema_version"] == SCHEMA_VERSION

        rep_entry = data["figures"][0]
        assert rep_entry["model_name"] == "mlp"
        assert rep_entry["representative_run_id"] == "run_01_seed_42"
        assert rep_entry["seed"] == 42
        assert rep_entry["best_epoch"] == 10
        assert rep_entry["stopping_epoch"] == 15
        assert rep_entry["caption_context"]["dataset_split"] == "test"

        agg_entry = data["figures"][1]
        assert "model_name" not in agg_entry
        assert "representative_run_id" not in agg_entry
        assert "seed" not in agg_entry
        assert "best_epoch" not in agg_entry
        assert "stopping_epoch" not in agg_entry


# ── Step 10 handoff ──────────────────────────────────────────────────────


class TestStep10Handoff:
    def test_run_count_comes_from_this_models_current_ledger(self, mock_outputs: Path) -> None:
        shared_path = mock_outputs / "outputs" / "shared" / "model_comparison_aggregate.json"
        shared = json.loads(shared_path.read_text())
        shared["rows"][0]["run_count"] = 9
        _write(shared_path, shared)
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            out = generate_step10_handoff(ModelID.BILSTM)
        assert json.loads(out.read_text())["total_runs_aggregated"] == 1

    def test_creates_handoff_with_required_refs(self, mock_outputs: Path) -> None:
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            generate_representative_figures(ModelID.MLP)
            generate_representative_misclassifications(ModelID.MLP)
            out = generate_step10_handoff(ModelID.MLP)

        data = json.loads(out.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["model_id"] == "mlp"
        assert data["representative_run_id"] == "run_01_seed_42"

        required_artifact_keys = {
            "aggregate_metrics",
            "representative_run_metadata",
            "final_predictions",
            "representative_misclassifications",
            "top_errors",
            "confusion_matrix",
            "top_confusions",
            "per_class_metrics",
            "label_order",
        }
        assert required_artifact_keys.issubset(data["artifacts"].keys())

        required_figure_keys = {
            "train_val_loss_curve",
            "val_macro_f1_curve",
            "val_accuracy_curve",
            "confusion_matrix",
            "top_confused_pairs",
            "bottom_classes_f1",
            "oos_metrics",
            "error_summary",
        }
        assert required_figure_keys.issubset(data["figures"].keys())


# ── Full pipeline integration ────────────────────────────────────────────


class TestFullPipeline:
    def test_succeeds_with_complete_artifacts(self, mock_outputs: Path) -> None:
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            success = run_report_figure_generation(list(ModelID))

        assert success

        shared = mock_outputs / "outputs" / "shared"
        assert (shared / "figure_manifest.json").exists()
        assert (shared / "most_confused_pairs_table.json").exists()
        assert (shared / "most_confused_pairs_table.csv").exists()
        assert (shared / "representative_examples_index.json").exists()

        for mid in ModelID:
            agg = mock_outputs / "outputs" / str(mid) / AGGREGATE_SUBDIR
            assert (agg / "step10_handoff.json").exists()
            assert (agg / "representative_misclassifications.csv").exists()

    def test_fails_on_missing_shared_table(self, mock_outputs: Path) -> None:
        (mock_outputs / "outputs" / "shared" / "model_comparison_aggregate.json").unlink()
        with _patch_roots(mock_outputs), _patch_model_output_dir(mock_outputs):
            success = run_report_figure_generation(list(ModelID))

        assert not success
