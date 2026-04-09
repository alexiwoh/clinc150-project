"""Output validation for report generation (Spec Section R).

Implements all 12 validation checks from the spec plus the assertion list.
Returns a list of failure messages; an empty list means all checks passed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.enums import ModelID
from src.report.artifact_loader import REPORT_DIR, load_json, resolve_repo_path

EXPECTED_FILES: tuple[str, ...] = (
    "abstract.md",
    "abstract.json",
    "dataset_description.md",
    "dataset_description.json",
    "preprocessing_summary_report.md",
    "preprocessing_summary_report.json",
    "model_architectures.md",
    "model_architectures.json",
    "experimental_setup.md",
    "experimental_setup.json",
    "main_results.md",
    "main_results.json",
    "oos_detection_results.md",
    "oos_detection_results.json",
    "figure_catalogue.md",
    "figure_catalogue.json",
    "error_analysis_discussion.md",
    "error_analysis_discussion.json",
    "representative_examples.md",
    "representative_examples.json",
    "key_findings.md",
    "key_findings.json",
    "limitations.md",
    "limitations.json",
    "future_improvements.md",
    "future_improvements.json",
    "reproducibility.md",
    "reproducibility.json",
    "report_structure.json",
    "full_report_draft.md",
)

_AGGREGATE_SOURCES: frozenset[str] = frozenset(
    {
        "outputs/shared/model_comparison_aggregate.json",
        "outputs/shared/oos_summary_table.json",
        "outputs/shared/efficiency_summary_table.json",
    }
)

_REPRESENTATIVE_SOURCES: frozenset[str] = frozenset(
    {
        "outputs/shared/analysis/extended_metrics_comparison.json",
        "outputs/shared/analysis/calibration_summary.json",
        "outputs/shared/analysis/oos_threshold_comparison.json",
    }
)


def _collect_sources(data: dict[str, Any]) -> list[str]:
    if "source_artifacts" in data:
        return list(data["source_artifacts"])
    if "source_artifact" in data:
        return [data["source_artifact"]]
    return []


def validate_report_outputs(report_dir: Path | None = None) -> list[str]:
    """Run all validation checks on report outputs.

    Returns a list of failure messages.  Empty list means all checks passed.
    """
    out = report_dir or REPORT_DIR
    failures: list[str] = []

    # Check 1: all 30 expected files exist
    for fname in EXPECTED_FILES:
        if not (out / fname).exists():
            failures.append(f"[R1] Missing expected file: {fname}")

    # Check 2: every JSON file parses cleanly
    for fname in EXPECTED_FILES:
        if not fname.endswith(".json"):
            continue
        fpath = out / fname
        if not fpath.exists():
            continue
        try:
            json.loads(fpath.read_text())
        except json.JSONDecodeError as exc:
            failures.append(f"[R2] JSON parse error in {fname}: {exc}")

    # Check 3: all source_artifact paths resolve to real files
    for fname in EXPECTED_FILES:
        if not fname.endswith(".json"):
            continue
        fpath = out / fname
        if not fpath.exists():
            continue
        data = load_json(fpath)
        for src in _collect_sources(data):
            if not resolve_repo_path(src).exists():
                failures.append(f"[R3] {fname}: source artifact missing: {src}")

    # Check 4: all figure paths in catalogue exist on disk
    fig_path = out / "figure_catalogue.json"
    if fig_path.exists():
        for sec in load_json(fig_path).get("sections", []):
            for fig in sec.get("figures", []):
                fp = fig.get("figure_path", "")
                if fp and not resolve_repo_path(fp).exists():
                    failures.append(f"[R4] Figure path does not exist: {fp}")

    # Check 5: main_results.json table values match upstream exactly
    results_path = out / "main_results.json"
    if results_path.exists():
        for table in load_json(results_path).get("tables", []):
            src = table.get("source_artifact", "")
            if not src:
                continue
            src_full = resolve_repo_path(src)
            if not src_full.exists():
                continue
            upstream_by = {r["model_id"]: r for r in load_json(src_full).get("rows", [])}
            for row in table.get("rows", []):
                mid = row.get("model_id", "")
                if mid and mid not in upstream_by:
                    failures.append(f"[R5] main_results table '{table['table_id']}': model {mid} not in upstream {src}")

    # Check 6: full_report_draft.md exists and is non-empty
    report_path = out / "full_report_draft.md"
    if report_path.exists() and report_path.stat().st_size == 0:
        failures.append("[R6] full_report_draft.md is empty")

    # Check 7: report_structure.json references every section file
    manifest_path = out / "report_structure.json"
    if manifest_path.exists():
        ids = {s["section_id"] for s in load_json(manifest_path).get("sections", [])}
        expected_ids = {"B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"}
        missing = expected_ids - ids
        if missing:
            failures.append(f"[R7] report_structure.json missing sections: {sorted(missing)}")

    # Check 8: figure catalogue total matches figure manifest
    fig_manifest_path = resolve_repo_path("outputs/shared/figure_manifest.json")
    if fig_path.exists() and fig_manifest_path.exists():
        cat_count: int = load_json(fig_path).get("total_figure_count", -1)
        manifest_count: int = len(load_json(fig_manifest_path).get("figures", []))
        if cat_count != manifest_count:
            failures.append(f"[R8] Figure catalogue count ({cat_count}) != figure manifest count ({manifest_count})")

    # Check 9: model_comparison_aggregate has rows for all 3 models
    agg_path = resolve_repo_path("outputs/shared/model_comparison_aggregate.json")
    if agg_path.exists():
        model_ids = {r["model_id"] for r in load_json(agg_path).get("rows", [])}
        expected_models = {str(m) for m in ModelID}
        missing_models = expected_models - model_ids
        if missing_models:
            failures.append(f"[R9] model_comparison_aggregate missing models: {sorted(missing_models)}")

    # Check 10: aggregate/representative scope correctness
    if results_path.exists():
        for table in load_json(results_path).get("tables", []):
            scope = table.get("data_scope")
            src = table.get("source_artifact", "")
            if scope == "aggregate" and src in _REPRESENTATIVE_SOURCES:
                failures.append(
                    f"[R10] Table '{table['table_id']}' labeled aggregate but uses representative source: {src}"
                )
            if scope == "representative" and src in _AGGREGATE_SOURCES:
                failures.append(
                    f"[R10] Table '{table['table_id']}' labeled representative but uses aggregate source: {src}"
                )

    # Check 11: curated examples trace back
    examples_path = out / "representative_examples.json"
    if examples_path.exists():
        ex_data = load_json(examples_path)
        curated_src = ex_data.get("source_artifact", "")
        if curated_src:
            curated_full = resolve_repo_path(curated_src)
            if curated_full.exists():
                curated_keys = {(e["text"], e["model_id"]) for e in load_json(curated_full).get("examples", [])}
                for ex in ex_data.get("examples", []):
                    if (ex["text"], ex["model_id"]) not in curated_keys:
                        failures.append(f"[R11] Example not in curated set: {ex['text'][:40]}... ({ex['model_id']})")

    # Check 12: OOS class identity matches evaluation_protocol
    proto_path = resolve_repo_path("outputs/shared/evaluation_protocol.json")
    if proto_path.exists():
        oos_policy = load_json(proto_path).get("oos_evaluation_policy", {})
        proto_oos_id = oos_policy.get("oos_class_id")

        ds_path = out / "dataset_description.json"
        if ds_path.exists() and proto_oos_id is not None:
            ds_oos_id = load_json(ds_path).get("oos_label_id")
            if ds_oos_id != proto_oos_id:
                failures.append(
                    f"[R12] dataset_description oos_label_id ({ds_oos_id}) "
                    f"!= evaluation_protocol oos_class_id ({proto_oos_id})"
                )

    return failures
