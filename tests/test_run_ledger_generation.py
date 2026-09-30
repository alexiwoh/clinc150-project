"""Repeated evaluation publishes current membership without deleting past runs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.config import RepeatedRunProtocol
from src.enums import ModelID
from src.repeated_evaluation import RunResult, compute_aggregate_metrics, save_aggregate_artifacts
from src.run_ledger import load_current_run_ledger


def _result(
    root: Path, index: int, seed: int, *, status: str = "completed", provenance: dict[str, Any] | None = None
) -> RunResult:
    run_id = f"run_{index:02d}_seed_{seed}"
    directory = root / "outputs/mlp/final_runs" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_metadata.json").write_text(
        json.dumps(
            {
                "model_id": "mlp",
                "run_id": run_id,
                "run_index": index,
                "seed": seed,
                "status": status,
                "provenance": provenance,
            }
        )
    )
    return RunResult("mlp", run_id, index, seed, seed, seed + 1, status, run_dir=str(directory), provenance=provenance)


def _provenance() -> dict[str, Any]:
    digest = "a" * 64
    return {
        "source": {"git_commit": "commit", "source_dirty": False, "files_sha256": {"src/train.py": digest}},
        "dataset": {
            "name": "synthetic",
            "subset": "fixture",
            "revision": "pinned",
            "label_order_sha256": digest,
            "splits": {
                split: {"count": 2, "texts_sha256": digest, "labels_sha256": digest, "examples_sha256": digest}
                for split in ("train", "validation", "test")
            },
        },
        "preprocessing_artifact_hashes": {"vocab": digest},
        "model_input_hashes": {split: digest for split in ("train", "validation", "test")},
        "frozen_config_hash": digest,
    }


def test_reducing_run_count_replaces_ledger_and_preserves_directories(tmp_path: Path) -> None:
    model_dir = tmp_path / "outputs/mlp"
    runs = [_result(tmp_path, index, seed) for index, seed in enumerate((42, 1337, 2024), 1)]
    with patch("src.repeated_evaluation.model_output_dir", return_value=model_dir):
        aggregate = compute_aggregate_metrics(runs, RepeatedRunProtocol(), ModelID.MLP)
        save_aggregate_artifacts(aggregate, runs, ModelID.MLP, representative_run_id=runs[0].run_id)
        assert (
            len(load_current_run_ledger(ModelID.MLP, model_dir=model_dir, project_root=tmp_path).completed_run_dirs)
            == 3
        )
        current = runs[:1]
        aggregate = compute_aggregate_metrics(current, RepeatedRunProtocol(run_count=1), ModelID.MLP)
        save_aggregate_artifacts(aggregate, current, ModelID.MLP, representative_run_id=current[0].run_id)
    ledger = load_current_run_ledger(ModelID.MLP, model_dir=model_dir, project_root=tmp_path)
    assert ledger.completed_run_ids == (runs[0].run_id,)
    assert len(list((model_dir / "final_runs").iterdir())) == 3
    assert not (model_dir / "aggregate/run_ledger.json.tmp").exists()


def test_failed_runs_remain_accounted_for(tmp_path: Path) -> None:
    model_dir = tmp_path / "outputs/mlp"
    runs = [_result(tmp_path, 1, 42), _result(tmp_path, 2, 1337, status="failed")]
    aggregate = compute_aggregate_metrics(runs, RepeatedRunProtocol(run_count=2), ModelID.MLP)
    with patch("src.repeated_evaluation.model_output_dir", return_value=model_dir):
        save_aggregate_artifacts(aggregate, runs, ModelID.MLP, representative_run_id=runs[0].run_id)
    ledger = load_current_run_ledger(ModelID.MLP, model_dir=model_dir, project_root=tmp_path)
    assert ledger.failed_run_ids == (runs[1].run_id,)
    assert ledger.completed_run_ids == (runs[0].run_id,)


def test_mismatched_membership_is_rejected_before_writes(tmp_path: Path) -> None:
    runs = [_result(tmp_path, 1, 42)]
    aggregate = compute_aggregate_metrics(runs, RepeatedRunProtocol(run_count=1), ModelID.MLP)
    runs[0].run_id = "stale_run"
    with patch("src.repeated_evaluation.model_output_dir", return_value=tmp_path / "outputs/mlp"):
        with pytest.raises(ValueError, match="membership"):
            save_aggregate_artifacts(aggregate, runs, ModelID.MLP)
    assert not (tmp_path / "outputs/mlp/aggregate").exists()


def test_mixed_generation_is_rejected_before_aggregate_writes(tmp_path: Path) -> None:
    current = _provenance()
    changed = deepcopy(current)
    changed["dataset"]["splits"]["test"]["examples_sha256"] = "b" * 64
    runs = [_result(tmp_path, 1, 42, provenance=current), _result(tmp_path, 2, 1337, provenance=changed)]
    aggregate = compute_aggregate_metrics(runs, RepeatedRunProtocol(run_count=2), ModelID.MLP)
    with patch("src.repeated_evaluation.model_output_dir", return_value=tmp_path / "outputs/mlp"):
        with pytest.raises(ValueError, match="dataset"):
            save_aggregate_artifacts(aggregate, runs, ModelID.MLP)
    assert not (tmp_path / "outputs/mlp/aggregate").exists()


def test_discovery_rejects_changed_recorded_identity(tmp_path: Path) -> None:
    model_dir = tmp_path / "outputs/mlp"
    runs = [_result(tmp_path, 1, 42, provenance=_provenance()), _result(tmp_path, 2, 1337, provenance=_provenance())]
    aggregate = compute_aggregate_metrics(runs, RepeatedRunProtocol(run_count=2), ModelID.MLP)
    with patch("src.repeated_evaluation.model_output_dir", return_value=model_dir):
        save_aggregate_artifacts(aggregate, runs, ModelID.MLP)
    metadata_path = Path(runs[1].run_dir) / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["provenance"]["source"]["files_sha256"]["src/train.py"] = "b" * 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="source"):
        load_current_run_ledger(ModelID.MLP, model_dir=model_dir, project_root=tmp_path)
