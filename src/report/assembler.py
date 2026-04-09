"""Report structure manifest (Spec Section P) and assembled report draft (Spec Section Q/Q2).

Reads individual section drafts from ``outputs/shared/report/``, builds a
master manifest mapping sections to source artifacts and associated figures,
and assembles ``full_report_draft.md`` with a table of contents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.constants import PROTOCOL_VERSION, SCHEMA_VERSION
from src.report.artifact_loader import REPORT_DIR, load_json, write_json

REPORT_TITLE = "Lightweight Deep Learning Models for Intent Classification and Out-of-Scope Detection on CLINC150"


@dataclass(frozen=True)
class SectionConfig:
    """Metadata for a single report section."""

    section_id: str
    file_stem: str
    title: str


SECTION_ORDER: tuple[SectionConfig, ...] = (
    SectionConfig("B", "abstract", "Abstract"),
    SectionConfig("C", "dataset_description", "Dataset Description"),
    SectionConfig("D", "preprocessing_summary_report", "Preprocessing Summary"),
    SectionConfig("E", "model_architectures", "Model Architectures"),
    SectionConfig("F", "experimental_setup", "Experimental Setup"),
    SectionConfig("G", "main_results", "Main Results"),
    SectionConfig("H", "oos_detection_results", "OOS Detection Results"),
    SectionConfig("J", "error_analysis_discussion", "Error Analysis Discussion"),
    SectionConfig("K", "representative_examples", "Representative Examples"),
    SectionConfig("L", "key_findings", "Key Findings"),
    SectionConfig("M", "limitations", "Limitations"),
    SectionConfig("N", "future_improvements", "Future Improvements"),
    SectionConfig("O", "reproducibility", "Reproducibility"),
    SectionConfig("I", "figure_catalogue", "Figure Catalogue"),
)

_FIGURE_SECTION_MAP: dict[str, str] = {
    "Training Diagnostics": "G",
    "Model Comparison": "G",
    "OOS Detection": "H",
    "Confusion Analysis": "J",
    "Error Analysis": "J",
    "Efficiency": "E",
}


def _slugify(title: str) -> str:
    """Convert a section title to a GitHub-style anchor slug."""
    return title.lower().replace(" ", "-").replace("(", "").replace(")", "")


def _extract_heading(content: str) -> str | None:
    """Extract the first ``##`` heading from markdown content."""
    for line in content.split("\n"):
        if line.startswith("## "):
            return line[3:].strip()
    return None


def _collect_sources(data: dict[str, Any]) -> list[str]:
    """Extract source artifact paths from a section JSON."""
    if "source_artifacts" in data:
        return list(data["source_artifacts"])
    if "source_artifact" in data:
        return [data["source_artifact"]]
    return []


def generate_report_manifest(report_dir: Path | None = None) -> dict[str, Any]:
    """Build ``report_structure.json`` (Spec Section P).

    Reads each section's JSON to collect source_artifacts, maps figures
    from the figure catalogue, and maps tables from main_results.
    """
    out = report_dir or REPORT_DIR

    fig_sections_map: dict[str, list[str]] = {}
    total_figures = 0
    fig_cat_path = out / "figure_catalogue.json"
    if fig_cat_path.exists():
        fig_cat = load_json(fig_cat_path)
        total_figures = fig_cat.get("total_figure_count", 0)
        for sec in fig_cat.get("sections", []):
            report_sid = _FIGURE_SECTION_MAP.get(sec["section_name"], "J")
            paths = [f["figure_path"] for f in sec.get("figures", [])]
            fig_sections_map.setdefault(report_sid, []).extend(paths)

    table_ids: list[str] = []
    results_path = out / "main_results.json"
    if results_path.exists():
        table_ids = [t["table_id"] for t in load_json(results_path).get("tables", [])]

    sections: list[dict[str, Any]] = []
    total_tables = 0

    for sc in SECTION_ORDER:
        json_path = out / f"{sc.file_stem}.json"

        section_sources: list[str] = []
        if json_path.exists():
            section_sources = _collect_sources(load_json(json_path))

        section_figures = fig_sections_map.get(sc.section_id, [])
        section_tables = table_ids if sc.section_id == "G" else []
        total_tables += len(section_tables)

        sections.append(
            {
                "section_id": sc.section_id,
                "section_title": sc.title,
                "output_md": f"outputs/shared/report/{sc.file_stem}.md",
                "output_json": f"outputs/shared/report/{sc.file_stem}.json",
                "source_artifacts": section_sources,
                "figures": section_figures,
                "tables": section_tables,
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "report_title": REPORT_TITLE,
        "sections": sections,
        "assembled_report": "outputs/shared/report/full_report_draft.md",
        "total_sections": len(sections),
        "total_figures": total_figures,
        "total_tables": total_tables,
    }

    write_json(out / "report_structure.json", manifest)
    return manifest


def assemble_full_report(report_dir: Path | None = None) -> str:
    """Build ``full_report_draft.md`` (Spec Sections Q / Q2).

    Assembles all section markdown drafts into one document with a table
    of contents, consistent heading hierarchy, and a Data Sources note.
    """
    out = report_dir or REPORT_DIR

    section_entries: list[tuple[str, str]] = []
    all_sources: set[str] = set()

    for sc in SECTION_ORDER:
        md_path = out / f"{sc.file_stem}.md"
        content = md_path.read_text().strip() if md_path.exists() else ""
        heading = _extract_heading(content) or sc.title
        section_entries.append((heading, content))

        json_path = out / f"{sc.file_stem}.json"
        if json_path.exists():
            all_sources.update(_collect_sources(load_json(json_path)))

    toc_lines: list[str] = ["## Table of Contents", ""]
    for i, (heading, _) in enumerate(section_entries, 1):
        toc_lines.append(f"{i}. [{heading}](#{_slugify(heading)})")
    toc_lines.append("")

    sorted_sources = sorted(all_sources)
    data_sources_lines: list[str] = [
        "---",
        "",
        "## Data Sources",
        "",
        "The following upstream artifacts were consumed to generate this report:",
        "",
        *[f"- `{src}`" for src in sorted_sources],
        "",
    ]

    parts: list[str] = [f"# {REPORT_TITLE}", "", *toc_lines]
    for _, content in section_entries:
        if content:
            parts.append(content)
            parts.append("")
    parts.extend(data_sources_lines)

    report_text = "\n".join(parts)
    report_path = out / "full_report_draft.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text)
    return report_text
