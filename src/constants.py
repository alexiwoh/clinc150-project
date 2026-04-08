"""Shared constants such as label names, special tokens, and default paths."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project directory structure
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = DATA_DIR / "artifacts"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

# Legacy flat output directories (backward-compatible with Steps 4-6 scripts)
CHECKPOINTS_DIR: Path = OUTPUTS_DIR / "checkpoints"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"

# ---------------------------------------------------------------------------
# Run-first directory layout
# ---------------------------------------------------------------------------
SHARED_DIR: Path = OUTPUTS_DIR / "shared"
SHARED_FIGURES_DIR: Path = SHARED_DIR / "figures"

# Sub-directory names within each model's output tree
TUNING_SUBDIR: str = "tuning"
FINAL_RUNS_SUBDIR: str = "final_runs"
AGGREGATE_SUBDIR: str = "aggregate"
MODEL_FIGURES_SUBDIR: str = "figures"
RUN_CHECKPOINT_SUBDIR: str = "checkpoint"
RUN_LOGS_SUBDIR: str = "logs"


def model_output_dir(model_id: str) -> Path:
    """Return the root output directory for a given model: ``outputs/{model_id}``."""
    return OUTPUTS_DIR / model_id


def run_dir_name(index: int, seed: int) -> str:
    """Build the canonical run-directory name: ``run_01_seed_42``."""
    return f"run_{index:02d}_seed_{seed}"


# ---------------------------------------------------------------------------
# CLINC150 label constants
# ---------------------------------------------------------------------------
OOS_LABEL_NAME: str = "oos"
OOS_LABEL_ID: int = 42
NUM_CLASSES: int = 151
NUM_IN_SCOPE_CLASSES: int = 150

# ---------------------------------------------------------------------------
# Protocol versioning and evaluation defaults
# ---------------------------------------------------------------------------
SCHEMA_VERSION: str = "1.0"
PROTOCOL_VERSION: str = "1.0"

DEFAULT_SEED_LIST: tuple[int, ...] = (42, 1337, 2024)

DEFAULT_AGGREGATE_METRICS: tuple[str, ...] = (
    "val_accuracy",
    "val_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "test_precision",
    "test_recall",
    "oos_precision",
    "oos_recall",
    "oos_f1",
    "training_time_seconds",
    "inference_total_seconds",
    "inference_avg_ms_per_example",
    "inference_examples_per_sec",
    "parameter_count",
    "trainable_parameter_count",
)

# Repo-relative paths to shared preprocessing artifacts
PREPROCESSING_MANIFEST_REF: str = "data/artifacts/preprocessing_summary.json"
LABEL_ORDER_REF: str = "data/artifacts/id_to_label.json"

# Repo-relative path to the canonical protocol manifest
PROTOCOL_MANIFEST_REF: str = "outputs/shared/evaluation_protocol.json"

# ---------------------------------------------------------------------------
# Named summary groups for aggregate_metrics.json
# ---------------------------------------------------------------------------
SUMMARY_GROUPS: dict[str, tuple[str, ...]] = {
    "validation_summary": ("val_accuracy", "val_macro_f1"),
    "test_summary": ("test_accuracy", "test_macro_f1", "test_precision", "test_recall"),
    "oos_summary": ("oos_precision", "oos_recall", "oos_f1"),
    "efficiency_summary": (
        "training_time_seconds",
        "inference_total_seconds",
        "inference_avg_ms_per_example",
        "inference_examples_per_sec",
    ),
    "parameter_count_summary": ("parameter_count", "trainable_parameter_count"),
}

# Per-run required artifact filenames
PER_RUN_REQUIRED_FILES: tuple[str, ...] = (
    "run_metadata.json",
    "validation_metrics.json",
    "test_metrics.json",
    "epoch_history.json",
    "final_predictions.csv",
    "confusion_matrix.csv",
    "top_confusions.json",
    "top_errors.json",
    "per_class_metrics.json",
    "label_order.json",
)
