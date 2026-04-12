"""Tests for src.report.assembler and src.report.validation.

Covers:
  - Report manifest generation (all 14 sections present, total counts, source_artifacts)
  - Assembled report structure (TOC anchors, section order, heading hierarchy)
  - Data Sources note at end of assembled report
  - Validation checks (expected files, JSON parsing, scope correctness)
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch


from src.constants import PROTOCOL_VERSION, SCHEMA_VERSION
from src.enums import ModelID
from src.report.assembler import (
    REPORT_TITLE,
    SECTION_ORDER,
    assemble_full_report,
    generate_report_manifest,
)
from src.report.validation import EXPECTED_FILES, validate_report_outputs


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _patches(report_dir: Path, root: Path):
    return ExitStack()


def _build_report_dir(root: Path) -> Path:
    """Build a minimal set of report section files + upstream artifacts for testing."""
    report_dir = root / "outputs" / "shared" / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    src_artifact = "data/artifacts/dataset_summary.json"
    _write(root / src_artifact, {"ok": True})

    for sc in SECTION_ORDER:
        md_path = report_dir / f"{sc.file_stem}.md"
        md_path.write_text(f"## {sc.title}\n\nSection content for {sc.title}.\n")
        _write(
            report_dir / f"{sc.file_stem}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "source_artifacts": [src_artifact],
            },
        )

    _write(
        report_dir / "figure_catalogue.json",
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "total_figure_count": 2,
            "sections": [
                {
                    "section_name": "Model Comparison",
                    "figures": [
                        {
                            "figure_path": "outputs/shared/figures/comparison.png",
                            "caption": "Test comparison",
                            "scope": "aggregate",
                            "model_name": None,
                            "figure_type": "model_comparison_test_macro_f1",
                        }
                    ],
                },
                {
                    "section_name": "OOS Detection",
                    "figures": [
                        {
                            "figure_path": "outputs/shared/analysis/oos_roc.png",
                            "caption": "Test ROC",
                            "scope": "analysis",
                            "model_name": None,
                            "figure_type": "oos_roc_comparison",
                        }
                    ],
                },
            ],
            "source_artifact": "outputs/shared/figure_manifest.json",
        },
    )

    _write(
        report_dir / "main_results.json",
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "tables": [
                {
                    "table_id": "main_comparison",
                    "title": "Main Model Comparison",
                    "data_scope": "aggregate",
                    "columns": ["Model", "F1"],
                    "rows": [{"model_id": str(m)} for m in ModelID],
                    "source_artifact": "outputs/shared/model_comparison_aggregate.json",
                }
            ],
            "source_artifacts": ["outputs/shared/model_comparison_aggregate.json"],
        },
    )

    _write(
        report_dir / "representative_examples.json",
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "data_scope": "representative",
            "examples": [{"text": "hello", "model_id": "mlp"}],
            "source_artifact": "outputs/shared/analysis/curated_report_examples.json",
        },
    )

    _write(
        report_dir / "dataset_description.json",
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "oos_label_id": 42,
            "source_artifact": src_artifact,
        },
    )

    return report_dir


def _build_upstream(root: Path) -> None:
    """Create the upstream artifacts that validation checks reference."""
    _write(root / "data/artifacts/dataset_summary.json", {"ok": True})

    fig_paths = [
        "outputs/shared/figures/comparison.png",
        "outputs/shared/analysis/oos_roc.png",
    ]
    for fp in fig_paths:
        p = root / fp
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"PNG")

    _write(
        root / "outputs/shared/figure_manifest.json",
        {"figures": [{"figure_type": "t", "figure_path": fp} for fp in fig_paths]},
    )

    _write(
        root / "outputs/shared/model_comparison_aggregate.json",
        {"rows": [{"model_id": str(m)} for m in ModelID]},
    )

    _write(
        root / "outputs/shared/evaluation_protocol.json",
        {"oos_evaluation_policy": {"oos_class_id": 42, "oos_class_name": "oos"}},
    )

    _write(
        root / "outputs/shared/analysis/curated_report_examples.json",
        {"examples": [{"text": "hello", "model_id": "mlp"}]},
    )


# ---------------------------------------------------------------------------
# Report manifest tests
# ---------------------------------------------------------------------------


class TestReportManifest:
    def test_all_14_sections_present(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        manifest = generate_report_manifest(report_dir)
        assert manifest["total_sections"] == 14
        ids = {s["section_id"] for s in manifest["sections"]}
        expected = {"B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"}
        assert ids == expected

    def test_source_artifacts_populated(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        manifest = generate_report_manifest(report_dir)
        for sec in manifest["sections"]:
            assert len(sec["source_artifacts"]) > 0, f"Section {sec['section_id']} has no sources"

    def test_figure_count(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        manifest = generate_report_manifest(report_dir)
        assert manifest["total_figures"] == 2

    def test_table_ids_in_results_section(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        manifest = generate_report_manifest(report_dir)
        results_sec = next(s for s in manifest["sections"] if s["section_id"] == "G")
        assert "main_comparison" in results_sec["tables"]

    def test_manifest_written_to_disk(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        generate_report_manifest(report_dir)
        assert (report_dir / "report_structure.json").exists()


# ---------------------------------------------------------------------------
# Assembled report tests
# ---------------------------------------------------------------------------


class TestAssembledReport:
    def test_title_heading(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        text = assemble_full_report(report_dir)
        assert text.startswith(f"# {REPORT_TITLE}")

    def test_toc_present(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        text = assemble_full_report(report_dir)
        assert "## Table of Contents" in text

    def test_toc_anchors_link_to_sections(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        text = assemble_full_report(report_dir)
        assert "[Abstract](#abstract)" in text
        assert "[Key Findings](#key-findings)" in text

    def test_all_sections_included(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        text = assemble_full_report(report_dir)
        for sc in SECTION_ORDER:
            assert f"## {sc.title}" in text, f"Section '{sc.title}' not found in report"

    def test_heading_hierarchy(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        text = assemble_full_report(report_dir)
        lines = text.split("\n")
        h1_count = sum(1 for ln in lines if ln.startswith("# ") and not ln.startswith("## "))
        h2_count = sum(1 for ln in lines if ln.startswith("## "))
        assert h1_count == 1
        assert h2_count >= 15

    def test_data_sources_note(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        text = assemble_full_report(report_dir)
        assert "## Data Sources" in text
        assert "data/artifacts/dataset_summary.json" in text

    def test_written_to_disk(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        assemble_full_report(report_dir)
        assert (report_dir / "full_report_draft.md").exists()
        assert (report_dir / "full_report_draft.md").stat().st_size > 0

    def test_section_order_matches_spec(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        text = assemble_full_report(report_dir)
        positions = []
        for sc in SECTION_ORDER:
            pos = text.find(f"## {sc.title}")
            assert pos >= 0, f"Section '{sc.title}' not found"
            positions.append(pos)
        assert positions == sorted(positions), "Sections are not in spec Q order"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_all_checks_pass(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        _build_upstream(tmp_path)
        generate_report_manifest(report_dir)
        assemble_full_report(report_dir)
        with ExitStack() as stack:
            stack.enter_context(patch("src.report.validation.REPORT_DIR", report_dir))
            stack.enter_context(patch("src.report.artifact_loader.PROJECT_ROOT", tmp_path))
            failures = validate_report_outputs(report_dir)
        assert failures == [], f"Validation failures: {failures}"

    def test_missing_file_detected(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        _build_upstream(tmp_path)
        generate_report_manifest(report_dir)
        assemble_full_report(report_dir)
        (report_dir / "abstract.md").unlink()
        with ExitStack() as stack:
            stack.enter_context(patch("src.report.validation.REPORT_DIR", report_dir))
            stack.enter_context(patch("src.report.artifact_loader.PROJECT_ROOT", tmp_path))
            failures = validate_report_outputs(report_dir)
        assert any("[R1]" in f and "abstract.md" in f for f in failures)

    def test_expected_file_count(self) -> None:
        assert len(EXPECTED_FILES) == 30

    def test_scope_mismatch_detected(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        _build_upstream(tmp_path)
        _write(
            report_dir / "main_results.json",
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "tables": [
                    {
                        "table_id": "bad_scope",
                        "title": "Test",
                        "data_scope": "aggregate",
                        "columns": ["X"],
                        "rows": [],
                        "source_artifact": "outputs/shared/analysis/calibration_summary.json",
                    }
                ],
                "source_artifacts": [],
            },
        )
        generate_report_manifest(report_dir)
        assemble_full_report(report_dir)
        with ExitStack() as stack:
            stack.enter_context(patch("src.report.validation.REPORT_DIR", report_dir))
            stack.enter_context(patch("src.report.artifact_loader.PROJECT_ROOT", tmp_path))
            failures = validate_report_outputs(report_dir)
        assert any("[R10]" in f for f in failures)

    def test_report_drift_detected(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        _build_upstream(tmp_path)
        generate_report_manifest(report_dir)
        assemble_full_report(report_dir)
        (report_dir / "full_report_draft.md").write_text("drifted report\n")
        with ExitStack() as stack:
            stack.enter_context(patch("src.report.validation.REPORT_DIR", report_dir))
            stack.enter_context(patch("src.report.artifact_loader.PROJECT_ROOT", tmp_path))
            failures = validate_report_outputs(report_dir)
        assert any("[R13]" in f for f in failures)

    def test_related_figure_reference_must_exist_in_manifest(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        _build_upstream(tmp_path)
        _write(
            report_dir / "key_findings.json",
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "findings": [
                    {
                        "finding_id": "headline",
                        "claim_id": "headline",
                        "claim_text": "Test claim",
                        "claim": "Test claim",
                        "source_artifact": "data/artifacts/dataset_summary.json",
                        "source_artifacts": ["data/artifacts/dataset_summary.json"],
                        "related_figure_types": ["model_comparison_test_macro_f1"],
                        "related_figure_paths": ["outputs/shared/figures/missing.png"],
                        "metric_value": 1.0,
                        "metric_std": None,
                    }
                ],
                "source_artifacts": ["data/artifacts/dataset_summary.json"],
            },
        )
        generate_report_manifest(report_dir)
        assemble_full_report(report_dir)
        with ExitStack() as stack:
            stack.enter_context(patch("src.report.validation.REPORT_DIR", report_dir))
            stack.enter_context(patch("src.report.artifact_loader.PROJECT_ROOT", tmp_path))
            failures = validate_report_outputs(report_dir)
        assert any("[R14]" in f for f in failures)

    def test_png_claim_source_detected(self, tmp_path: Path) -> None:
        report_dir = _build_report_dir(tmp_path)
        _build_upstream(tmp_path)
        _write(
            report_dir / "key_findings.json",
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "findings": [
                    {
                        "finding_id": "length",
                        "claim_id": "length",
                        "claim_text": "Short-query accuracy is low",
                        "claim": "Short-query accuracy is low",
                        "source_artifact": "outputs/shared/analysis/length_slice_comparison.png",
                        "source_artifacts": ["outputs/shared/analysis/length_slice_comparison.png"],
                        "related_figure_types": ["length_slice"],
                        "related_figure_paths": [],
                        "metric_value": None,
                        "metric_std": None,
                    }
                ],
                "source_artifacts": ["outputs/shared/analysis/length_slice_comparison.png"],
            },
        )
        (tmp_path / "outputs/shared/analysis").mkdir(parents=True, exist_ok=True)
        (tmp_path / "outputs/shared/analysis/length_slice_comparison.png").write_bytes(b"PNG")
        generate_report_manifest(report_dir)
        assemble_full_report(report_dir)
        with ExitStack() as stack:
            stack.enter_context(patch("src.report.validation.REPORT_DIR", report_dir))
            stack.enter_context(patch("src.report.artifact_loader.PROJECT_ROOT", tmp_path))
            failures = validate_report_outputs(report_dir)
        assert any("[R15]" in f for f in failures)
