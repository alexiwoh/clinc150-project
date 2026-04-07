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
    subset: str = "plus"
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


@dataclass(frozen=True)
class MLPBaselineConfig:
    """Hyperparameters for the TF-IDF + MLP baseline model."""

    hidden_dim: int = 512
    second_hidden_dim: int | None = None
    dropout_rate: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10
    optimizer: str = "adam"
    activation: str = "relu"
    use_class_weights: bool = False
    use_lr_scheduler: bool = False
    random_seed: int = 42
    monitor_metric: str = "val_macro_f1"
    oos_strategy: str = "explicit_class"

    def to_dict(self) -> dict:
        """Serialize config to a plain dict for checkpoints and run summaries."""
        return {
            "hidden_dim": self.hidden_dim,
            "second_hidden_dim": self.second_hidden_dim,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "optimizer": self.optimizer,
            "activation": self.activation,
            "use_class_weights": self.use_class_weights,
            "use_lr_scheduler": self.use_lr_scheduler,
            "random_seed": self.random_seed,
            "monitor_metric": self.monitor_metric,
            "oos_strategy": self.oos_strategy,
        }


@dataclass(frozen=True)
class TextCNNConfig:
    """Hyperparameters for the Text CNN (Kim-style) model."""

    vocab_size: int = 0
    embedding_dim: int = 128
    num_filters: int = 100
    kernel_sizes: tuple[int, ...] = (3, 4, 5)
    dropout_rate: float = 0.3
    activation: str = "relu"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10
    optimizer: str = "adam"
    trainable_embeddings: bool = True
    use_class_weights: bool = False
    use_lr_scheduler: bool = False
    random_seed: int = 42
    dataloader_seed: int = 42
    monitor_metric: str = "val_macro_f1"
    oos_strategy: str = "explicit_class"
    max_seq_length: int = 20

    def to_dict(self) -> dict:
        """Serialize config to a plain dict for checkpoints and run summaries."""
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_filters": self.num_filters,
            "kernel_sizes": list(self.kernel_sizes),
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "optimizer": self.optimizer,
            "trainable_embeddings": self.trainable_embeddings,
            "use_class_weights": self.use_class_weights,
            "use_lr_scheduler": self.use_lr_scheduler,
            "random_seed": self.random_seed,
            "dataloader_seed": self.dataloader_seed,
            "monitor_metric": self.monitor_metric,
            "oos_strategy": self.oos_strategy,
            "max_seq_length": self.max_seq_length,
        }
