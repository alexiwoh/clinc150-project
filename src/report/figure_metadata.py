"""Shared figure metadata and structured-claim helpers for Step 11."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from src.report.artifact_loader import load_json, resolve_repo_path

_FIGURE_MANIFEST_PATH = "outputs/shared/figure_manifest.json"


class FigureCaptionContext(TypedDict, total=False):
    """Optional producer-supplied context used to build report captions."""

    analysis_basis: str
    class_count: int
    dataset_name: str
    dataset_split: str
    metric_names: list[str]
    run_count: int
    selection_rule: str
    top_k: int


class FigureManifestEntry(TypedDict, total=False):
    """Typed view over one entry from ``figure_manifest.json``."""

    caption_context: FigureCaptionContext
    figure_path: str
    figure_type: str
    model_name: str | None
    protocol_version: str
    representative_run_id: str
    schema_version: str
    scope: str
    source_artifact_paths: list[str]


class StructuredClaim(TypedDict):
    """Deterministic claim-to-figure linkage metadata for section JSON."""

    claim_id: str
    claim_text: str
    related_figure_paths: list[str]
    related_figure_types: list[str]
    source_artifacts: list[str]
    model_name: NotRequired[str | None]
    scope_hint: NotRequired[str | None]


def load_figure_manifest_entries() -> list[FigureManifestEntry]:
    """Load figure manifest entries in manifest order."""
    manifest = load_json(resolve_repo_path(_FIGURE_MANIFEST_PATH))
    return list(manifest.get("figures", []))


def resolve_related_figure_paths(
    figure_entries: list[FigureManifestEntry],
    *,
    source_artifacts: list[str] | None = None,
    related_figure_types: list[str] | None = None,
    model_name: str | None = None,
    scopes: list[str] | None = None,
) -> list[str]:
    """Resolve figure paths deterministically from claim metadata."""

    requested_sources = set(source_artifacts or [])
    requested_types = set(related_figure_types or [])
    requested_scopes = set(scopes or [])

    def _matches_model(entry: FigureManifestEntry) -> bool:
        entry_model = entry.get("model_name")
        if model_name is None:
            return True
        return entry_model in {model_name, None}

    def _matches_scope(entry: FigureManifestEntry) -> bool:
        if not requested_scopes:
            return True
        return entry.get("scope") in requested_scopes

    def _matches_type(entry: FigureManifestEntry) -> bool:
        if not requested_types:
            return True
        return entry.get("figure_type") in requested_types

    def _matches_source(entry: FigureManifestEntry) -> bool:
        if not requested_sources:
            return False
        entry_sources = set(entry.get("source_artifact_paths", []))
        return bool(entry_sources & requested_sources)

    candidates = [
        entry for entry in figure_entries if _matches_model(entry) and _matches_scope(entry) and _matches_type(entry)
    ]
    if not candidates:
        candidates = [entry for entry in figure_entries if _matches_model(entry) and _matches_scope(entry)]

    source_matches = [entry for entry in candidates if _matches_source(entry)]
    if source_matches:
        return _dedupe_paths(entry["figure_path"] for entry in source_matches if "figure_path" in entry)

    if requested_types:
        typed_matches = [entry for entry in candidates if _matches_type(entry)]
        return _dedupe_paths(entry["figure_path"] for entry in typed_matches if "figure_path" in entry)

    source_only_matches = [
        entry for entry in figure_entries if _matches_model(entry) and _matches_scope(entry) and _matches_source(entry)
    ]
    return _dedupe_paths(entry["figure_path"] for entry in source_only_matches if "figure_path" in entry)


def build_structured_claim(
    *,
    claim_id: str,
    claim_text: str,
    figure_entries: list[FigureManifestEntry],
    source_artifacts: list[str],
    related_figure_types: list[str] | None = None,
    model_name: str | None = None,
    scopes: list[str] | None = None,
) -> StructuredClaim:
    """Create a structured claim with resolved figure references."""

    claim: StructuredClaim = {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "source_artifacts": list(source_artifacts),
        "related_figure_types": list(related_figure_types or []),
        "related_figure_paths": resolve_related_figure_paths(
            figure_entries,
            source_artifacts=source_artifacts,
            related_figure_types=related_figure_types,
            model_name=model_name,
            scopes=scopes,
        ),
    }
    if model_name is not None:
        claim["model_name"] = model_name
    if scopes:
        claim["scope_hint"] = scopes[0]
    return claim


def render_related_figure_note(related_figure_paths: list[str]) -> str | None:
    """Render a compact markdown note for one claim's figure links."""
    if not related_figure_paths:
        return None
    label = "Related figure" if len(related_figure_paths) == 1 else "Related figures"
    joined = ", ".join(f"`{path}`" for path in related_figure_paths)
    return f"{label}: {joined}"


def _dedupe_paths(paths: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered
