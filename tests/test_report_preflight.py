"""Tests for src.report.preflight — artifact existence validation.

Covers missing-artifact detection, clear error messages, and partial missing sets.
Uses a mock artifact tree under tmp_path with patched path constants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.constants import PROTOCOL_VERSION, SCHEMA_VERSION
from src.enums import ModelID
from src.report.preflight import validate_artifacts


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def _build_preflight_tree(root: Path) -> None:
    """Build minimal artifact tree satisfying all preflight checks."""
    # Steps 2-3
    _write(root / "data/artifacts/dataset_summary.json", {"ok": True})
    (root / "data/artifacts/dataset_summary.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "data/artifacts/dataset_summary.md").write_text("# Dataset\n")
    _write(root / "data/artifacts/preprocessing_summary.json", {"ok": True})

    # Steps 7-8 per-model
    for mid in ModelID:
        _write(root / f"outputs/{mid}/frozen_final_config.json", {"model_id": str(mid)})
        _write(root / f"outputs/{mid}/aggregate/aggregate_metrics.json", {"model_id": str(mid)})

    # Step 8 shared
    shared = root / "outputs/shared"
    for fname in (
        "evaluation_protocol.json",
        "model_comparison_aggregate.json",
        "oos_summary_table.json",
        "efficiency_summary_table.json",
        "most_confused_pairs_table.json",
        "representative_examples_index.json",
    ):
        _write(shared / fname, {"ok": True})

    # Step 9
    _write(shared / "figure_manifest.json", {"figures": []})

    # Step 10 handoff
    analysis = shared / "analysis"
    _write(
        analysis / "step11_handoff.json",
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "per_model_artifacts": {str(m): {} for m in ModelID},
            "shared_artifacts": {},
        },
    )


def _patches(root: Path):
    """Context manager patching all path constants used by preflight."""
    return (
        patch("src.report.preflight.PROJECT_ROOT", root),
        patch("src.report.preflight.OUTPUTS_DIR", root / "outputs"),
        patch("src.report.artifact_loader.PROJECT_ROOT", root),
        patch("src.report.artifact_loader.SHARED_DIR", root / "outputs" / "shared"),
        patch("src.report.artifact_loader.ANALYSIS_DIR", root / "outputs" / "shared" / "analysis"),
    )


class TestValidateArtifacts:
    def test_passes_with_all_present(self, tmp_path: Path) -> None:
        _build_preflight_tree(tmp_path)
        p1, p2, p3, p4, p5 = _patches(tmp_path)
        with p1, p2, p3, p4, p5:
            result = validate_artifacts()
        assert isinstance(result, dict)
        assert "per_model_artifacts" in result

    def test_fails_on_missing_dataset_summary(self, tmp_path: Path) -> None:
        _build_preflight_tree(tmp_path)
        (tmp_path / "data/artifacts/dataset_summary.json").unlink()
        p1, p2, p3, p4, p5 = _patches(tmp_path)
        with p1, p2, p3, p4, p5:
            with pytest.raises(FileNotFoundError, match="dataset_summary.json"):
                validate_artifacts()

    def test_fails_on_missing_frozen_config(self, tmp_path: Path) -> None:
        _build_preflight_tree(tmp_path)
        (tmp_path / "outputs/mlp/frozen_final_config.json").unlink()
        p1, p2, p3, p4, p5 = _patches(tmp_path)
        with p1, p2, p3, p4, p5:
            with pytest.raises(FileNotFoundError, match="frozen_final_config"):
                validate_artifacts()

    def test_fails_on_missing_shared_artifact(self, tmp_path: Path) -> None:
        _build_preflight_tree(tmp_path)
        (tmp_path / "outputs/shared/oos_summary_table.json").unlink()
        p1, p2, p3, p4, p5 = _patches(tmp_path)
        with p1, p2, p3, p4, p5:
            with pytest.raises(FileNotFoundError, match="oos_summary_table"):
                validate_artifacts()

    def test_fails_on_missing_figure_manifest(self, tmp_path: Path) -> None:
        _build_preflight_tree(tmp_path)
        (tmp_path / "outputs/shared/figure_manifest.json").unlink()
        p1, p2, p3, p4, p5 = _patches(tmp_path)
        with p1, p2, p3, p4, p5:
            with pytest.raises(FileNotFoundError, match="figure_manifest"):
                validate_artifacts()

    def test_multiple_missing_listed(self, tmp_path: Path) -> None:
        _build_preflight_tree(tmp_path)
        (tmp_path / "data/artifacts/dataset_summary.json").unlink()
        (tmp_path / "data/artifacts/preprocessing_summary.json").unlink()
        p1, p2, p3, p4, p5 = _patches(tmp_path)
        with p1, p2, p3, p4, p5:
            with pytest.raises(FileNotFoundError, match="2 required artifact"):
                validate_artifacts()

    def test_missing_handoff_raises_file_not_found(self, tmp_path: Path) -> None:
        _build_preflight_tree(tmp_path)
        (tmp_path / "outputs/shared/analysis/step11_handoff.json").unlink()
        p1, p2, p3, p4, p5 = _patches(tmp_path)
        with p1, p2, p3, p4, p5:
            with pytest.raises(FileNotFoundError):
                validate_artifacts()
