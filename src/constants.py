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
CHECKPOINTS_DIR: Path = OUTPUTS_DIR / "checkpoints"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"

# ---------------------------------------------------------------------------
# CLINC150 label constants
# ---------------------------------------------------------------------------
OOS_LABEL_NAME: str = "oos"
OOS_LABEL_ID: int = 42
NUM_CLASSES: int = 151
NUM_IN_SCOPE_CLASSES: int = 150
