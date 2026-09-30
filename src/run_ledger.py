"""Authoritative current-run ledger loading and representative resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.constants import AGGREGATE_SUBDIR, FINAL_RUNS_SUBDIR, PROJECT_ROOT, model_output_dir
from src.enums import ModelID


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if type(data) is not dict:
        raise ValueError(f"{path}: expected a JSON object")
    return data


@dataclass(frozen=True)
class CurrentRunLedger:
    """Validated current evaluation membership, independent of leftover directories."""

    requested_run_ids: tuple[str, ...]
    completed_run_ids: tuple[str, ...]
    failed_run_ids: tuple[str, ...]
    skipped_run_ids: tuple[str, ...]
    metadata_paths: dict[str, Path]

    @property
    def completed_run_dirs(self) -> list[Path]:
        """Completed artifact directories in the ledger's recorded order."""
        return [self.metadata_paths[run_id].parent for run_id in self.completed_run_ids]


def load_current_run_ledger(
    model_id: ModelID, *, model_dir: Path | None = None, project_root: Path | None = None
) -> CurrentRunLedger:
    """Load authoritative current membership and reject missing or conflicting identities.

    Every completed run must have a repository-relative metadata reference to its
    canonical directory, with matching model, run, status and seed identities.
    Files in unlisted directories are deliberately ignored and preserved.
    """
    root = PROJECT_ROOT if project_root is None else project_root
    directory = model_output_dir(model_id) if model_dir is None else model_dir
    ledger_path = directory / AGGREGATE_SUBDIR / "run_ledger.json"
    if not ledger_path.is_file():
        raise FileNotFoundError(f"Current run ledger not found for {model_id}: {ledger_path}")
    data = _read_json(ledger_path)
    if type(data) is not dict or data.get("model_id") != model_id:
        raise ValueError(f"{ledger_path}: model_id does not match {model_id}")

    groups: dict[str, tuple[str, ...]] = {}
    for key in ("requested_run_ids", "completed_run_ids", "failed_run_ids", "skipped_run_ids"):
        values = data.get(key)
        if type(values) is not list or any(type(value) is not str or not value for value in values):
            raise ValueError(f"{ledger_path}: {key} must be a list of nonempty run IDs")
        if len(values) != len(set(values)):
            raise ValueError(f"{ledger_path}: duplicate run IDs in {key}")
        if any(Path(value).name != value or value in {".", ".."} for value in values):
            raise ValueError(f"{ledger_path}: invalid run ID in {key}")
        groups[key] = tuple(values)

    requested = groups["requested_run_ids"]
    classified = groups["completed_run_ids"] + groups["failed_run_ids"] + groups["skipped_run_ids"]
    if len(classified) != len(set(classified)) or set(classified) != set(requested):
        raise ValueError(f"{ledger_path}: run statuses must partition requested_run_ids")
    for count_key, group_key in (
        ("run_count_requested", "requested_run_ids"),
        ("run_count_completed", "completed_run_ids"),
        ("run_count_failed", "failed_run_ids"),
        ("run_count_skipped", "skipped_run_ids"),
    ):
        if data.get(count_key) != len(groups[group_key]):
            raise ValueError(f"{ledger_path}: {count_key} does not match {group_key}")

    requested_seeds = data.get("seed_list_requested")
    completed_seeds = data.get("seed_list_completed")
    if type(requested_seeds) is not list or len(requested_seeds) != len(requested):
        raise ValueError(f"{ledger_path}: seed_list_requested does not match requested_run_ids")
    if any(type(seed) is not int for seed in requested_seeds) or len(set(requested_seeds)) != len(requested_seeds):
        raise ValueError(f"{ledger_path}: requested seeds must be unique integers")
    if type(completed_seeds) is not list or any(type(seed) is not int for seed in completed_seeds):
        raise ValueError(f"{ledger_path}: seed_list_completed must be a list of integers")
    if requested != tuple(f"run_{index:02d}_seed_{seed}" for index, seed in enumerate(requested_seeds, 1)):
        raise ValueError(f"{ledger_path}: requested run IDs do not match their run indices and seeds")
    refs = data.get("per_run_metadata_refs")
    if type(refs) is not dict or not set(refs).issubset(requested):
        raise ValueError(f"{ledger_path}: per_run_metadata_refs contains unrequested run IDs")
    if not set(groups["completed_run_ids"]).issubset(refs):
        raise ValueError(f"{ledger_path}: completed run missing per_run_metadata_refs entry")

    metadata_paths: dict[str, Path] = {}
    actual_completed_seeds: list[int] = []
    for index, run_id in enumerate(requested, 1):
        if run_id not in refs:
            continue
        ref = refs[run_id]
        if type(ref) is not str or not ref or Path(ref).is_absolute():
            raise ValueError(f"{ledger_path}: metadata reference for {run_id} must be repository-relative")
        metadata_path = root / ref
        expected_path = directory / FINAL_RUNS_SUBDIR / run_id / "run_metadata.json"
        if metadata_path.resolve() != expected_path.resolve():
            raise ValueError(f"{ledger_path}: metadata reference for {run_id} points to a different run")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"{ledger_path}: run metadata not found for {run_id}: {metadata_path}")
        metadata = _read_json(metadata_path)
        if metadata.get("model_id") != model_id or metadata.get("run_id") != run_id:
            raise ValueError(f"{metadata_path}: model_id/run_id does not match the current ledger")
        if metadata.get("run_index") != index or metadata.get("seed") != requested_seeds[index - 1]:
            raise ValueError(f"{metadata_path}: run_index/seed does not match the current ledger")
        if run_id in groups["completed_run_ids"]:
            if metadata.get("status") != "completed":
                raise ValueError(f"{metadata_path}: completed ledger run has status {metadata.get('status')!r}")
        elif run_id in groups["failed_run_ids"] and metadata.get("status") != "failed":
            raise ValueError(f"{metadata_path}: failed ledger run has status {metadata.get('status')!r}")
        metadata_paths[run_id] = metadata_path

    for run_id in groups["completed_run_ids"]:
        actual_completed_seeds.append(_read_json(metadata_paths[run_id])["seed"])
    if completed_seeds != actual_completed_seeds:
        raise ValueError(f"{ledger_path}: seed_list_completed does not match completed run metadata")

    return CurrentRunLedger(
        requested_run_ids=requested,
        completed_run_ids=groups["completed_run_ids"],
        failed_run_ids=groups["failed_run_ids"],
        skipped_run_ids=groups["skipped_run_ids"],
        metadata_paths=metadata_paths,
    )


def resolve_current_run(
    model_id: ModelID,
    run_id: str,
    *,
    artifact_path: str | None = None,
    model_dir: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    """Resolve an active completed run, optionally checking a representative path."""
    ledger = load_current_run_ledger(model_id, model_dir=model_dir, project_root=project_root)
    if run_id not in ledger.completed_run_ids:
        raise ValueError(
            f"Representative run {run_id!r} is not a completed member of the current ledger for {model_id}"
        )
    run_dir = ledger.metadata_paths[run_id].parent
    if artifact_path is not None:
        root = PROJECT_ROOT if project_root is None else project_root
        if Path(artifact_path).is_absolute() or (root / artifact_path).resolve() != run_dir.resolve():
            raise ValueError(f"Representative artifact_path does not match current run {run_id!r} for {model_id}")
    return run_dir
