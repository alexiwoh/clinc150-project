"""Preflight validation: confirm all upstream artifacts exist before report generation.

Checks Step 10 handoff paths, Step 2-3 preprocessing artifacts,
Step 7-8 per-model and shared evaluation artifacts, and Step 9 figure manifest.
Raises a single ``FileNotFoundError`` listing every missing file.
"""

from __future__ import annotations

import logging

from src.constants import OUTPUTS_DIR, PROJECT_ROOT
from src.enums import ModelID
from src.report.artifact_loader import load_handoff, resolve_repo_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Well-known artifact paths (Steps 2-3, 7-9)
# ---------------------------------------------------------------------------

_STEP_2_3_ARTIFACTS: tuple[str, ...] = (
    "data/artifacts/dataset_summary.json",
    "data/artifacts/dataset_summary.md",
    "data/artifacts/preprocessing_summary.json",
)

_PER_MODEL_ARTIFACTS: tuple[str, ...] = (
    "frozen_final_config.json",
    "aggregate/aggregate_metrics.json",
)

_STEP_8_SHARED_ARTIFACTS: tuple[str, ...] = (
    "outputs/shared/evaluation_protocol.json",
    "outputs/shared/model_comparison_aggregate.json",
    "outputs/shared/oos_summary_table.json",
    "outputs/shared/efficiency_summary_table.json",
    "outputs/shared/most_confused_pairs_table.json",
    "outputs/shared/representative_examples_index.json",
)

_STEP_9_ARTIFACTS: tuple[str, ...] = ("outputs/shared/figure_manifest.json",)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_artifacts() -> dict[str, object]:
    """Validate that all upstream artifacts required for report generation exist.

    Loads ``step11_handoff.json`` and checks every referenced path, then
    checks well-known paths for Steps 2-3, 7-8 (per-model and shared), and
    Step 9.

    Returns the parsed handoff dict on success.

    Raises:
        FileNotFoundError: with a message listing every missing file if any
            required artifact is absent.
    """
    missing: list[str] = []

    # ------------------------------------------------------------------
    # Step 10: handoff paths
    # ------------------------------------------------------------------
    handoff = load_handoff()

    for model_id, paths in handoff.get("per_model_artifacts", {}).items():
        for fname, rel_path in paths.items():
            if isinstance(rel_path, str) and rel_path.startswith("MISSING:"):
                missing.append(f"{model_id}/{fname}")
            elif not (PROJECT_ROOT / rel_path).exists():
                missing.append(rel_path)

    for fname, rel_path in handoff.get("shared_artifacts", {}).items():
        if isinstance(rel_path, str) and rel_path.startswith("MISSING:"):
            missing.append(f"shared/{fname}")
        elif not (PROJECT_ROOT / rel_path).exists():
            missing.append(rel_path)

    # ------------------------------------------------------------------
    # Steps 2-3: dataset & preprocessing
    # ------------------------------------------------------------------
    for rel in _STEP_2_3_ARTIFACTS:
        if not resolve_repo_path(rel).exists():
            missing.append(rel)

    # ------------------------------------------------------------------
    # Steps 7-8: per-model artifacts
    # ------------------------------------------------------------------
    for model_id in ModelID:
        model_dir = OUTPUTS_DIR / str(model_id)
        for fname in _PER_MODEL_ARTIFACTS:
            full = model_dir / fname
            if not full.exists():
                rel = str(full.relative_to(PROJECT_ROOT))
                missing.append(rel)

    # ------------------------------------------------------------------
    # Step 8: shared evaluation artifacts
    # ------------------------------------------------------------------
    for rel in _STEP_8_SHARED_ARTIFACTS:
        if not resolve_repo_path(rel).exists():
            missing.append(rel)

    # ------------------------------------------------------------------
    # Step 9: figure manifest
    # ------------------------------------------------------------------
    for rel in _STEP_9_ARTIFACTS:
        if not resolve_repo_path(rel).exists():
            missing.append(rel)

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------
    if missing:
        deduped = list(dict.fromkeys(missing))
        formatted = "\n  ".join(deduped)
        raise FileNotFoundError(
            f"Preflight validation failed — {len(deduped)} required artifact(s) missing:\n  {formatted}"
        )

    logger.info("Preflight validation passed: all upstream artifacts present.")
    return handoff
