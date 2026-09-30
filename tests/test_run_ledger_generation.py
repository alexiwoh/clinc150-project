"""Repeated evaluation publishes current membership without deleting past runs."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import RepeatedRunProtocol
from src.enums import ModelID
from src.repeated_evaluation import RunResult, compute_aggregate_metrics, save_aggregate_artifacts
from src.run_ledger import load_current_run_ledger


def _result(root: Path, index: int, seed: int, *, status: str = "completed") -> RunResult:
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
            }
        )
    )
    return RunResult("mlp", run_id, index, seed, seed, seed + 1, status, run_dir=str(directory))


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
