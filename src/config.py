"""Central configuration for hyperparameters, file paths, training settings, and model options."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src import constants
from src.enums import ModelID

SUPPORTED_OPTIMIZER: str = "adam"
SUPPORTED_OOS_STRATEGY: str = "explicit_class"
SUPPORTED_SUMMARIZATION_MODE: str = "concat_final_hidden"
SUPPORTED_REPRESENTATIVE_RULE: str = "highest_validation_macro_f1"
SUPPORTED_REPEATED_MONITOR: str = "val_macro_f1"
ALL_RUNS_ARTIFACT_POLICY: str = "all_runs"
SUPPORTED_ACTIVATIONS: tuple[str, ...] = ("relu", "gelu", "tanh")
MAX_RANDOM_SEED: int = 2**32 - 1
MAX_DATALOADER_SEED: int = 2**64 - 1


def _validate_integer(name: str, value: int, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _validate_finite_number(name: str, value: float, minimum: float = 0.0) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}, got {value!r}")


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
        _validate_integer("batch_size", self.batch_size)
        _validate_integer("max_epochs", self.max_epochs)
        _validate_integer("early_stopping_patience", self.early_stopping_patience)
        _validate_integer("random_seed", self.random_seed, minimum=0)
        _validate_integer("dataloader_seed", self.dataloader_seed, minimum=0)
        if self.random_seed > MAX_RANDOM_SEED:
            raise ValueError(f"random_seed must be <= {MAX_RANDOM_SEED}")
        if self.dataloader_seed > MAX_DATALOADER_SEED:
            raise ValueError(f"dataloader_seed must be <= {MAX_DATALOADER_SEED}")
        _validate_finite_number("dropout_rate", self.dropout_rate)
        if self.dropout_rate > 1:
            raise ValueError("dropout_rate must be <= 1")
        _validate_finite_number("learning_rate", self.learning_rate)
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be > 0")
        _validate_finite_number("weight_decay", self.weight_decay)
        if type(self.use_class_weights) is not bool or type(self.use_lr_scheduler) is not bool:
            raise ValueError("use_class_weights and use_lr_scheduler must be booleans")
        if self.optimizer != SUPPORTED_OPTIMIZER:
            raise ValueError(f"Only optimizer={SUPPORTED_OPTIMIZER!r} is supported, got {self.optimizer!r}")
        if self.oos_strategy != SUPPORTED_OOS_STRATEGY:
            raise ValueError(f"Only oos_strategy={SUPPORTED_OOS_STRATEGY!r} is supported, got {self.oos_strategy!r}")
        if self.monitor_metric != SUPPORTED_REPEATED_MONITOR:
            raise ValueError(f"Model workflows only support monitor_metric={SUPPORTED_REPEATED_MONITOR!r}")

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

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_integer("vocab_size", self.vocab_size, minimum=0)
        _validate_integer("embedding_dim", self.embedding_dim)
        _validate_integer("max_seq_length", self.max_seq_length)
        if type(self.trainable_embeddings) is not bool:
            raise ValueError("trainable_embeddings must be a boolean")


@dataclass(frozen=True)
class MLPBaselineConfig(BaseModelConfig):
    """Hyperparameters for the TF-IDF + MLP baseline model."""

    hidden_dim: int = 512
    second_hidden_dim: int | None = None
    activation: str = "relu"

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_integer("hidden_dim", self.hidden_dim)
        if self.second_hidden_dim is not None:
            _validate_integer("second_hidden_dim", self.second_hidden_dim)
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {self.activation!r}")


@dataclass(frozen=True)
class TextCNNConfig(BaseNeuralConfig):
    """Hyperparameters for the Text CNN (Kim-style) model."""

    num_filters: int = 100
    kernel_sizes: tuple[int, ...] = (3, 4, 5)
    activation: str = "relu"

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_integer("num_filters", self.num_filters)
        if type(self.kernel_sizes) is not tuple or not self.kernel_sizes:
            raise ValueError("kernel_sizes must be a nonempty tuple of positive integers")
        for kernel_size in self.kernel_sizes:
            _validate_integer("kernel_sizes element", kernel_size)
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {self.activation!r}")


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
        _validate_integer("hidden_dim", self.hidden_dim)
        _validate_integer("num_layers", self.num_layers)
        _validate_finite_number("max_grad_norm", self.max_grad_norm)
        if self.max_grad_norm == 0:
            raise ValueError("max_grad_norm must be > 0")
        if type(self.bidirectional) is not bool or type(self.gradient_clipping) is not bool:
            raise ValueError("bidirectional and gradient_clipping must be booleans")
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
        _validate_integer("run_count", self.run_count)
        if type(self.seed_list) is not tuple or not self.seed_list:
            raise ValueError("seed_list must be a nonempty tuple")
        if self.run_count > len(self.seed_list):
            raise ValueError(f"run_count must not exceed the available seeds ({len(self.seed_list)})")
        for seed in self.seed_list:
            _validate_integer("seed_list element", seed, minimum=0)
            if seed > MAX_RANDOM_SEED:
                raise ValueError(f"seed_list elements must be <= {MAX_RANDOM_SEED}")
        if len(self.seed_list) != len(set(self.seed_list)):
            raise ValueError("seed_list contains duplicates")
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
    protocol_manifest_ref: str = constants.PROTOCOL_MANIFEST_REF

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrozenModelConfig:
        """Read the supported frozen-config schema, including historical protocol references."""
        if type(data) is not dict:
            raise ValueError("Frozen config must be a JSON object")
        required_fields = {
            "schema_version",
            "protocol_version",
            "model_id",
            "model_name",
            "hyperparameters",
            "source_tuning_artifact",
            "winning_row_id",
            "selection_metric",
            "selection_value",
            "vocab_size",
            "max_seq_length",
            "monitor_metric",
            "preprocessing_manifest_ref",
            "label_order_ref",
            "oos_strategy",
            "oos_class_id",
        }
        unknown_fields = data.keys() - required_fields - {"protocol_manifest_ref"}
        missing_fields = required_fields - data.keys()
        if missing_fields or unknown_fields:
            raise ValueError(
                f"Invalid frozen config fields: missing={sorted(missing_fields)}, unknown={sorted(unknown_fields)}"
            )
        frozen = cls(**data)
        if type(frozen.hyperparameters) is not dict:
            raise ValueError("hyperparameters must be a JSON object")
        expected_hyperparameters = MODEL_CONFIG_CLASSES[ModelID(frozen.model_id)]().to_dict().keys()
        if frozen.hyperparameters.keys() != expected_hyperparameters:
            raise ValueError("Frozen hyperparameters must contain all and only the model configuration fields")
        frozen.to_model_config()
        return frozen

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for frozen_final_config.json."""
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "hyperparameters": self.to_model_config().to_dict(),
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
            "protocol_manifest_ref": self.protocol_manifest_ref,
        }

    def to_model_config(self) -> BaseModelConfig:
        """Reconstruct the concrete model config dataclass from stored hyperparameters."""
        if self.schema_version != constants.SCHEMA_VERSION or self.protocol_version != constants.PROTOCOL_VERSION:
            raise ValueError("Unsupported frozen config schema_version or protocol_version")
        model_id = ModelID(self.model_id)
        if self.model_name != model_id.display_name:
            raise ValueError("Frozen model_name must match model_id")
        if self.selection_metric != "best_val_metric":
            raise ValueError("Only selection_metric='best_val_metric' is supported")
        _validate_finite_number("selection_value", self.selection_value)
        if self.selection_value > 1:
            raise ValueError("selection_value must be <= 1")
        for name, value in (
            ("source_tuning_artifact", self.source_tuning_artifact),
            ("winning_row_id", self.winning_row_id),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.protocol_manifest_ref != constants.PROTOCOL_MANIFEST_REF:
            raise ValueError("Unsupported protocol_manifest_ref")
        if self.preprocessing_manifest_ref != constants.PREPROCESSING_MANIFEST_REF:
            raise ValueError("Unsupported preprocessing_manifest_ref")
        if self.label_order_ref != constants.LABEL_ORDER_REF:
            raise ValueError("Unsupported label_order_ref")
        if type(self.hyperparameters) is not dict:
            raise ValueError("hyperparameters must be a JSON object")
        hyperparameters = self.hyperparameters.copy()
        if model_id is ModelID.TEXT_CNN and "kernel_sizes" in hyperparameters:
            kernels = hyperparameters["kernel_sizes"]
            if type(kernels) not in (tuple, list):
                raise ValueError("kernel_sizes must be an array of positive integers")
            hyperparameters["kernel_sizes"] = tuple(kernels)
        config_cls = MODEL_CONFIG_CLASSES[model_id]
        try:
            config = config_cls(**hyperparameters)
        except TypeError as error:
            raise ValueError(f"Invalid hyperparameters for {model_id}: {error}") from error
        resolved_hyperparameters = config.to_dict()
        if (
            model_id is ModelID.TEXT_CNN
            and max(resolved_hyperparameters["kernel_sizes"]) > resolved_hyperparameters["max_seq_length"]
        ):
            raise ValueError("Frozen kernel_sizes must not exceed max_seq_length")
        if self.monitor_metric != config.monitor_metric:
            raise ValueError("Frozen monitor_metric must match hyperparameters.monitor_metric")
        if config.monitor_metric != SUPPORTED_REPEATED_MONITOR:
            raise ValueError(f"Repeated evaluation only supports monitor_metric={SUPPORTED_REPEATED_MONITOR!r}")
        if self.oos_strategy != config.oos_strategy:
            raise ValueError("Frozen oos_strategy must match hyperparameters.oos_strategy")
        if type(self.oos_class_id) is not int or self.oos_class_id != constants.OOS_LABEL_ID:
            raise ValueError(f"Only oos_class_id={constants.OOS_LABEL_ID} is supported")
        if model_id is ModelID.MLP:
            if self.vocab_size is not None or self.max_seq_length is not None:
                raise ValueError("MLP frozen vocab_size and max_seq_length must be null")
        else:
            _validate_integer("frozen vocab_size", self.vocab_size)
            _validate_integer("frozen max_seq_length", self.max_seq_length)
            if self.vocab_size != hyperparameters.get("vocab_size"):
                raise ValueError("Frozen vocab_size must match hyperparameters.vocab_size")
            if self.max_seq_length != hyperparameters.get("max_seq_length"):
                raise ValueError("Frozen max_seq_length must match hyperparameters.max_seq_length")
        return config
