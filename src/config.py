"""Central configuration for hyperparameters, file paths, training settings, and model options."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src import constants
from src.enums import ModelID
from src.training_contracts import MONITOR_METRIC_DIRECTIONS

SUPPORTED_OPTIMIZER: str = "adam"
SUPPORTED_OOS_STRATEGY: str = "explicit_class"
SUPPORTED_SUMMARIZATION_MODE: str = "concat_final_hidden"
SUPPORTED_REPRESENTATIVE_RULE: str = "highest_validation_macro_f1"
SUPPORTED_REPEATED_MONITOR: str = "val_macro_f1"
ALL_RUNS_ARTIFACT_POLICY: str = "all_runs"


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


# ---------------------------------------------------------------------------
# Model config hierarchy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseModelConfig:
    """Shared training hyperparameters common to all models."""

    dropout_rate: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10
    optimizer: str = SUPPORTED_OPTIMIZER
    use_class_weights: bool = False
    use_lr_scheduler: bool = False
    random_seed: int = 42
    dataloader_seed: int = 42
    monitor_metric: str = SUPPORTED_REPEATED_MONITOR
    oos_strategy: str = SUPPORTED_OOS_STRATEGY

    def __post_init__(self) -> None:
        """Reject settings that the training functions do not implement."""
        if self.optimizer != SUPPORTED_OPTIMIZER:
            raise ValueError(f"Only optimizer={SUPPORTED_OPTIMIZER!r} is supported, got {self.optimizer!r}")
        if self.oos_strategy != SUPPORTED_OOS_STRATEGY:
            raise ValueError(f"Only oos_strategy={SUPPORTED_OOS_STRATEGY!r} is supported, got {self.oos_strategy!r}")
        if self.monitor_metric not in MONITOR_METRIC_DIRECTIONS:
            raise ValueError(f"Unsupported monitor_metric: {self.monitor_metric!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize all fields to a JSON-safe dict."""
        return {k: list(v) if type(v) is tuple else v for k, v in dataclasses.asdict(self).items()}


@dataclass(frozen=True)
class BaseNeuralConfig(BaseModelConfig):
    """Shared hyperparameters for neural (embedding-based) models."""

    vocab_size: int = 0
    embedding_dim: int = 128
    trainable_embeddings: bool = True
    max_seq_length: int = 20


@dataclass(frozen=True)
class MLPBaselineConfig(BaseModelConfig):
    """Hyperparameters for the TF-IDF + MLP baseline model."""

    hidden_dim: int = 512
    second_hidden_dim: int | None = None
    activation: str = "relu"


@dataclass(frozen=True)
class TextCNNConfig(BaseNeuralConfig):
    """Hyperparameters for the Text CNN (Kim-style) model."""

    num_filters: int = 100
    kernel_sizes: tuple[int, ...] = (3, 4, 5)
    activation: str = "relu"


@dataclass(frozen=True)
class BiLSTMConfig(BaseNeuralConfig):
    """Hyperparameters for the BiLSTM sentence classifier."""

    hidden_dim: int = 128
    num_layers: int = 1
    bidirectional: bool = True
    learning_rate: float = 5e-4
    summarization_mode: str = SUPPORTED_SUMMARIZATION_MODE
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.summarization_mode != SUPPORTED_SUMMARIZATION_MODE:
            raise ValueError(
                f"Only summarization_mode={SUPPORTED_SUMMARIZATION_MODE!r} is supported, "
                f"got {self.summarization_mode!r}"
            )


# ---------------------------------------------------------------------------
# Model config registry
# ---------------------------------------------------------------------------

MODEL_CONFIG_CLASSES: dict[ModelID, type[BaseModelConfig]] = {
    ModelID.MLP: MLPBaselineConfig,
    ModelID.TEXT_CNN: TextCNNConfig,
    ModelID.BILSTM: BiLSTMConfig,
}


# ---------------------------------------------------------------------------
# Repeated-run evaluation protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepeatedRunProtocol:
    """Shared, immutable configuration for the repeated-run evaluation protocol."""

    schema_version: str = constants.SCHEMA_VERSION
    protocol_version: str = constants.PROTOCOL_VERSION
    run_count: int = 3
    seed_list: tuple[int, ...] = constants.DEFAULT_SEED_LIST
    representative_run_rule: str = SUPPORTED_REPRESENTATIVE_RULE
    probability_saving_policy: str = ALL_RUNS_ARTIFACT_POLICY
    confusion_artifact_policy: str = ALL_RUNS_ARTIFACT_POLICY
    aggregate_metric_list: tuple[str, ...] = constants.DEFAULT_AGGREGATE_METRICS
    test_evaluation_enabled: bool = True
    timing_includes_dataloader_overhead: bool = True

    def __post_init__(self) -> None:
        """Enforce the single evaluation/artifact contract implemented by this project."""
        if self.schema_version != constants.SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {self.schema_version!r}")
        if self.protocol_version != constants.PROTOCOL_VERSION:
            raise ValueError(f"Unsupported protocol_version: {self.protocol_version!r}")
        if self.representative_run_rule != SUPPORTED_REPRESENTATIVE_RULE:
            raise ValueError(f"Only representative_run_rule={SUPPORTED_REPRESENTATIVE_RULE!r} is supported")
        if self.probability_saving_policy != ALL_RUNS_ARTIFACT_POLICY:
            raise ValueError(f"Only probability_saving_policy={ALL_RUNS_ARTIFACT_POLICY!r} is supported")
        if self.confusion_artifact_policy != ALL_RUNS_ARTIFACT_POLICY:
            raise ValueError(f"Only confusion_artifact_policy={ALL_RUNS_ARTIFACT_POLICY!r} is supported")
        if self.aggregate_metric_list != constants.DEFAULT_AGGREGATE_METRICS:
            raise ValueError("Only the full DEFAULT_AGGREGATE_METRICS list is supported")
        if self.test_evaluation_enabled is not True:
            raise ValueError("Only test_evaluation_enabled=True is supported")
        if self.timing_includes_dataloader_overhead is not True:
            raise ValueError("Only timing_includes_dataloader_overhead=True is supported")

    def effective_seed_list(self) -> list[int]:
        """Return the prefix of seed_list actually used for this run_count."""
        return list(self.seed_list[: self.run_count])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for evaluation_protocol.json."""
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "run_count": self.run_count,
            "seed_list": list(self.seed_list),
            "effective_seed_list": self.effective_seed_list(),
            "representative_run_rule": self.representative_run_rule,
            "probability_saving_policy": self.probability_saving_policy,
            "confusion_artifact_policy": self.confusion_artifact_policy,
            "aggregate_metric_list": list(self.aggregate_metric_list),
            "test_evaluation_enabled": self.test_evaluation_enabled,
            "timing_includes_dataloader_overhead": self.timing_includes_dataloader_overhead,
        }


@dataclass(frozen=True)
class FrozenModelConfig:
    """Fully-resolved, immutable configuration for one model's repeated-run evaluation.

    Contains all training-relevant hyperparameters plus provenance fields that
    trace back to the tuning artifact and winning row.
    """

    schema_version: str
    protocol_version: str
    model_id: str
    model_name: str
    hyperparameters: dict[str, Any]
    source_tuning_artifact: str
    winning_row_id: str
    selection_metric: str
    selection_value: float
    vocab_size: int | None
    max_seq_length: int | None
    monitor_metric: str
    preprocessing_manifest_ref: str
    label_order_ref: str
    oos_strategy: str
    oos_class_id: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for frozen_final_config.json."""
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "hyperparameters": self.hyperparameters,
            "source_tuning_artifact": self.source_tuning_artifact,
            "winning_row_id": self.winning_row_id,
            "selection_metric": self.selection_metric,
            "selection_value": self.selection_value,
            "vocab_size": self.vocab_size,
            "max_seq_length": self.max_seq_length,
            "monitor_metric": self.monitor_metric,
            "preprocessing_manifest_ref": self.preprocessing_manifest_ref,
            "label_order_ref": self.label_order_ref,
            "oos_strategy": self.oos_strategy,
            "oos_class_id": self.oos_class_id,
        }

    def to_model_config(self) -> BaseModelConfig:
        """Reconstruct the concrete model config dataclass from stored hyperparameters."""
        config_cls = MODEL_CONFIG_CLASSES[ModelID(self.model_id)]
        config = config_cls(**self.hyperparameters)
        if self.monitor_metric != config.monitor_metric:
            raise ValueError("Frozen monitor_metric must match hyperparameters.monitor_metric")
        if config.monitor_metric != SUPPORTED_REPEATED_MONITOR:
            raise ValueError(f"Repeated evaluation only supports monitor_metric={SUPPORTED_REPEATED_MONITOR!r}")
        if self.oos_strategy != config.oos_strategy:
            raise ValueError("Frozen oos_strategy must match hyperparameters.oos_strategy")
        if self.oos_class_id != constants.OOS_LABEL_ID:
            raise ValueError(f"Only oos_class_id={constants.OOS_LABEL_ID} is supported")
        return config
