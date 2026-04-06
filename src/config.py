"""Central configuration for hyperparameters, file paths, training settings, and model options."""

from dataclasses import dataclass
from pathlib import Path

import torch

from src import constants


def get_device() -> torch.device:
    """Auto-detect the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Number of DataLoader worker processes.
# 0 = main process only (required for MPS; safe default for all platforms).
# Increase on CUDA/CPU setups with many cores for faster data loading.
NUM_WORKERS: int = 0


@dataclass(frozen=True)
class DatasetConfig:
    """Experiment-level dataset settings."""

    name: str = "clinc/clinc_oos"
    subset: str = "small"
    cache_dir: Path = constants.RAW_DIR


DATASET_CONFIG: DatasetConfig = DatasetConfig()


@dataclass(frozen=True)
class PreprocessingConfig:
    """Settings for the text preprocessing pipeline."""

    lowercase: bool = True
    max_seq_length: int = 20
    min_token_freq: int = 1
    batch_size: int = 64
    tfidf_max_features: int = 10_000
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    padding_side: str = "right"
    truncation_side: str = "right"
    random_seed: int = 42
    artifacts_dir: Path = constants.ARTIFACTS_DIR


PREPROCESSING_CONFIG: PreprocessingConfig = PreprocessingConfig()
