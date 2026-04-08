"""Shared I/O helpers, path resolution, and artifact loaders for error analysis."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.constants import (
    AGGREGATE_SUBDIR,
    ANALYSIS_SUBDIR,
    OOS_LABEL_NAME,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    SHARED_DIR,
    model_output_dir,
)
from src.enums import ModelID

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    """Read and parse a JSON file."""
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    """Write *data* as pretty-printed JSON with trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def repo_relative(path: Path) -> str:
    """Return the repo-relative string for *path*."""
    return str(path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def analysis_output_dir(model_id: ModelID) -> Path:
    """Return ``outputs/<model>/analysis/``, creating if needed."""
    d = model_output_dir(model_id) / ANALYSIS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def shared_analysis_dir() -> Path:
    """Return ``outputs/shared/analysis/``, creating if needed."""
    d = SHARED_DIR / ANALYSIS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Handoff resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HandoffContext:
    """Resolved paths and metadata from a model's error_analysis_handoff.json."""

    model_id: ModelID
    representative_run_id: str
    run_dir: Path
    final_predictions_path: Path
    confidences_path: Path
    per_class_metrics_path: Path
    confusion_matrix_path: Path
    top_confusions_path: Path
    top_errors_path: Path
    label_order_path: Path
    test_metrics_path: Path


def resolve_handoff(model_id: ModelID) -> HandoffContext:
    """Read ``error_analysis_handoff.json`` for *model_id* and resolve all paths."""
    handoff_path = model_output_dir(model_id) / AGGREGATE_SUBDIR / "error_analysis_handoff.json"
    handoff = read_json(handoff_path)
    artifacts: dict[str, str] = handoff["artifacts"]

    run_dir = (PROJECT_ROOT / artifacts["final_predictions"]).parent

    return HandoffContext(
        model_id=model_id,
        representative_run_id=handoff["representative_run_id"],
        run_dir=run_dir,
        final_predictions_path=PROJECT_ROOT / artifacts["final_predictions"],
        confidences_path=run_dir / "confidences.npz",
        per_class_metrics_path=PROJECT_ROOT / artifacts["per_class_metrics"],
        confusion_matrix_path=PROJECT_ROOT / artifacts["confusion_matrix"],
        top_confusions_path=PROJECT_ROOT / artifacts["top_confusions"],
        top_errors_path=PROJECT_ROOT / artifacts["top_errors"],
        label_order_path=PROJECT_ROOT / artifacts["label_order"],
        test_metrics_path=run_dir / "test_metrics.json",
    )


# ---------------------------------------------------------------------------
# Artifact loaders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceData:
    """Loaded contents of a ``confidences.npz`` file."""

    probabilities: npt.NDArray[np.floating[Any]]
    predictions: npt.NDArray[np.intp]
    targets: npt.NDArray[np.intp]
    label_names: npt.NDArray[np.str_]
    oos_class_idx: int


def load_confidences(npz_path: Path) -> ConfidenceData:
    """Load ``confidences.npz`` with shape and key validation."""
    data = np.load(npz_path, allow_pickle=False)

    required_keys = {"probabilities", "predictions", "targets", "label_names"}
    missing = required_keys - set(data.files)
    if missing:
        raise ValueError(f"{npz_path}: missing keys {missing}")

    probs: npt.NDArray[np.floating[Any]] = data["probabilities"]
    preds: npt.NDArray[np.intp] = data["predictions"]
    targets: npt.NDArray[np.intp] = data["targets"]
    label_names: npt.NDArray[np.str_] = data["label_names"]

    n_examples, n_classes = probs.shape
    if preds.shape[0] != n_examples:
        raise ValueError(f"predictions length {preds.shape[0]} != probabilities rows {n_examples}")
    if targets.shape[0] != n_examples:
        raise ValueError(f"targets length {targets.shape[0]} != probabilities rows {n_examples}")
    if label_names.shape[0] != n_classes:
        raise ValueError(f"label_names length {label_names.shape[0]} != probabilities cols {n_classes}")

    oos_indices = [i for i, name in enumerate(label_names) if name == OOS_LABEL_NAME]
    if len(oos_indices) != 1:
        raise ValueError(f"Expected exactly one OOS label, found {len(oos_indices)}")

    return ConfidenceData(
        probabilities=probs,
        predictions=preds,
        targets=targets,
        label_names=label_names,
        oos_class_idx=oos_indices[0],
    )


def load_predictions(csv_path: Path) -> pd.DataFrame:
    """Load ``final_predictions.csv`` with column validation."""
    df = pd.read_csv(csv_path)
    required_cols = {
        "text",
        "true_label_id",
        "true_label_name",
        "predicted_label_id",
        "predicted_label_name",
        "run_id",
        "max_confidence",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    return df


def validate_consistency(conf: ConfidenceData, preds_df: pd.DataFrame) -> None:
    """Assert that confidences.npz and final_predictions.csv are consistent."""
    n_conf = conf.probabilities.shape[0]
    n_csv = len(preds_df)
    if n_conf != n_csv:
        raise ValueError(f"confidences.npz has {n_conf} examples but predictions CSV has {n_csv}")

    csv_pred_ids = preds_df["predicted_label_id"].values
    npz_pred_ids = conf.predictions
    mismatches = int(np.sum(csv_pred_ids != npz_pred_ids))
    if mismatches > 0:
        raise ValueError(f"{mismatches} prediction ID mismatches between confidences.npz and predictions CSV")


# ---------------------------------------------------------------------------
# Common artifact envelope
# ---------------------------------------------------------------------------


def artifact_envelope(
    model_id: ModelID | None = None,
    representative_run_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build the common metadata envelope for analysis artifacts."""
    env: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
    }
    if model_id is not None:
        env["model_id"] = str(model_id)
        env["model_name"] = model_id.display_name
    if representative_run_id is not None:
        env["representative_run_id"] = representative_run_id
    env.update(extra)
    return env


# ---------------------------------------------------------------------------
# All-run directory discovery
# ---------------------------------------------------------------------------


def discover_all_run_dirs(model_id: ModelID) -> list[Path]:
    """Return paths to all final-run directories for *model_id*, sorted by name."""
    final_runs_dir = OUTPUTS_DIR / str(model_id) / "final_runs"
    if not final_runs_dir.is_dir():
        return []
    return sorted(p for p in final_runs_dir.iterdir() if p.is_dir())
