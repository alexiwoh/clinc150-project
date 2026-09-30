"""Artifact loading, path resolution, and section I/O for report generation.

Provides thin wrappers for JSON loading, repo-relative path normalization,
Step 11 handoff resolution, and section (md + json) persistence.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from src.constants import PROJECT_ROOT, PROTOCOL_VERSION, SCHEMA_VERSION, SHARED_DIR

logger = logging.getLogger(__name__)

REPORT_DIR: Path = SHARED_DIR / "report"

ANALYSIS_DIR: Path = SHARED_DIR / "analysis"

HANDOFF_FILENAME: str = "step11_handoff.json"

_REPO_IMAGE_PATTERN = re.compile(r"(!\[[^\]]*\]\()(outputs/[^)]+)(\))")


def render_image_links(markdown: str, output_dir: Path) -> str:
    """Resolve generated repository figure links relative to the saved Markdown.

    Section generators retain repository paths for provenance. Both section
    files and the assembled report live in the same destination directory.
    """

    def relative_link(match: re.Match[str]) -> str:
        target = os.path.relpath(resolve_repo_path(match.group(2)), output_dir)
        return f"{match.group(1)}{Path(target).as_posix()}{match.group(3)}"

    return _REPO_IMAGE_PATTERN.sub(relative_link, markdown)


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    """Read and parse a JSON file, raising ``FileNotFoundError`` if absent."""
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write *data* as pretty-printed JSON with trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_repo_path(relative: str) -> Path:
    """Resolve a repo-relative path string to an absolute ``Path``."""
    return PROJECT_ROOT / relative


def repo_relative(path: Path) -> str:
    """Convert an absolute *path* to a repo-relative string."""
    return str(path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Handoff
# ---------------------------------------------------------------------------


def load_handoff() -> dict[str, Any]:
    """Load and validate ``step11_handoff.json``.

    Returns the parsed dict. Raises ``FileNotFoundError`` if the file is
    absent and ``ValueError`` if the top-level keys are missing.
    """
    handoff_path = ANALYSIS_DIR / HANDOFF_FILENAME
    data = load_json(handoff_path)

    for key in ("per_model_artifacts", "shared_artifacts"):
        if key not in data:
            raise ValueError(f"step11_handoff.json missing required key '{key}'")

    return data


# ---------------------------------------------------------------------------
# Section persistence
# ---------------------------------------------------------------------------


def save_section(
    section_id: str,
    markdown: str,
    metadata: dict[str, Any],
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write a report section's ``.md`` and ``.json`` files.

    *section_id* is used as the file stem (e.g. ``"abstract"``).
    *metadata* is augmented with ``schema_version`` and ``protocol_version``
    if they are not already present.

    Returns ``(md_path, json_path)``.
    """
    out = output_dir or REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)

    metadata.setdefault("schema_version", SCHEMA_VERSION)
    metadata.setdefault("protocol_version", PROTOCOL_VERSION)

    md_path = out / f"{section_id}.md"
    json_path = out / f"{section_id}.json"

    md_path.write_text(render_image_links(markdown, out))
    write_json(json_path, metadata)

    logger.info("Saved report section: %s (.md + .json)", section_id)
    return md_path, json_path
