"""Repeated-run evaluation protocol for CLINC150 models.

Frozen-config extraction, tuning normalization, protocol saving,
core repeated-run execution, per-run artifact bundle saving,
aggregate metric computation, and representative-run selection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import time
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.config import (
    BaseModelConfig,
    BiLSTMConfig,
    FrozenModelConfig,
    MLPBaselineConfig,
    RepeatedRunProtocol,
    TextCNNConfig,
    get_device,
)
from src.constants import (
    AGGREGATE_SUBDIR,
    ARTIFACTS_DIR,
    FINAL_RUNS_SUBDIR,
    LABEL_ORDER_REF,
    MODEL_FIGURES_SUBDIR,
    OOS_LABEL_ID,
    OOS_LABEL_NAME,
    OUTPUTS_DIR,
    PREPROCESSING_MANIFEST_REF,
    PROJECT_ROOT,
    PROTOCOL_MANIFEST_REF,
    PROTOCOL_VERSION,
    REPORTS_DIR,
    RUN_CHECKPOINT_SUBDIR,
    RUN_LOGS_SUBDIR,
    SCHEMA_VERSION,
    SHARED_DIR,
    SUMMARY_GROUPS,
    TUNING_SUBDIR,
    model_output_dir,
    run_dir_name,
)
from src.enums import ModelID
from src.metrics import (
    build_confusion_matrix,
    compute_classification_metrics,
    compute_oos_metrics,
    find_top_confusions,
    find_top_errors,
)
from src.utils import count_parameters, set_seed

logger = logging.getLogger(__name__)


def _to_repo_relative(path: str | Path) -> str:
    """Convert an absolute path to repo-relative. Pass-through if already relative."""
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p)
    return str(p)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Outcome of a single repeated-run execution."""

    model_id: str
    run_id: str
    run_index: int
    seed: int
    training_seed: int
    dataloader_seed: int
    status: str  # "completed" or "failed"

    # Validation metrics
    val_accuracy: float = 0.0
    val_macro_f1: float = 0.0
    val_precision: float = 0.0
    val_recall: float = 0.0
    val_loss: float = 0.0

    # Test metrics
    test_accuracy: float = 0.0
    test_macro_f1: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0

    # OOS metrics
    oos_precision: float = 0.0
    oos_recall: float = 0.0
    oos_f1: float = 0.0

    # Efficiency metrics
    training_time_seconds: float = 0.0
    inference_total_seconds: float = 0.0
    inference_avg_ms_per_example: float = 0.0
    inference_examples_per_sec: float = 0.0
    parameter_count: int = 0
    trainable_parameter_count: int = 0

    # Training details
    best_epoch: int = 0
    stopping_epoch: int = 0
    best_val_metric: float = 0.0
    best_val_loss: float = 0.0
    monitor_metric: str = "val_macro_f1"
    checkpoint_path: str = ""
    log_path: str = ""

    # Artifacts
    run_dir: str = ""
    epoch_history: list[dict[str, Any]] = field(default_factory=list)

    # Failure info (for failed runs)
    failure_reason: str = ""
    stage_reached: str = ""
    partial_artifacts: list[str] = field(default_factory=list)


@dataclass
class AggregateMetrics:
    """Aggregated mean/std metrics across completed runs for one model."""

    model_id: str
    run_count_requested: int
    run_count_completed: int
    seed_list_requested: list[int]
    seed_list_completed: list[int]
    metrics: dict[str, dict[str, float]]  # {metric_name: {"mean": ..., "std": ...}}


@dataclass
class RepresentativeRunInfo:
    """Metadata for the selected representative run."""

    run_id: str
    run_index: int
    seed: int
    selection_rule: str
    selection_metric_value: float
    artifact_path: str


@dataclass
class ModelEvaluationResult:
    """Complete result of repeated-run evaluation for one model."""

    model_id: str
    frozen_config: FrozenModelConfig
    run_results: list[RunResult]
    aggregate: AggregateMetrics
    representative: RepresentativeRunInfo


# ---------------------------------------------------------------------------
# Frozen-config extraction
# ---------------------------------------------------------------------------


def _select_winning_row(df: pd.DataFrame, model_id: str) -> pd.Series:
    """Select the best tuning row using validation-only criteria.

    Selection: max ``best_val_metric``, tie-break by ``best_val_loss`` (lower is
    better, if available), then by ``run_name`` (lexicographic).
    """
    assert not df.empty, f"Tuning CSV for '{model_id}' is empty"

    sorted_df = df.copy()

    has_val_loss = "best_val_loss" in sorted_df.columns
    if has_val_loss:
        sorted_df = sorted_df.sort_values(
            by=["best_val_metric", "best_val_loss", "run_name"],
            ascending=[False, True, True],
        )
    else:
        sorted_df = sorted_df.sort_values(
            by=["best_val_metric", "run_name"],
            ascending=[False, True],
        )

    return sorted_df.iloc[0]


def _build_mlp_hyperparameters(row: pd.Series) -> dict[str, Any]:
    """Extract MLP hyperparameters from the winning tuning row."""
    second_hidden = row.get("second_hidden_dim")
    if pd.isna(second_hidden) or second_hidden == "":
        second_hidden = None
    else:
        second_hidden = int(second_hidden)

    return {
        "hidden_dim": int(row["hidden_dim"]),
        "second_hidden_dim": second_hidden,
        "dropout_rate": float(row["dropout_rate"]),
        "learning_rate": float(row["learning_rate"]),
        "weight_decay": float(row["weight_decay"]),
        "batch_size": int(row.get("batch_size", 64)),
        "max_epochs": int(row.get("max_epochs", 100)),
        "early_stopping_patience": int(row.get("early_stopping_patience", 10)),
        "optimizer": str(row.get("optimizer", "adam")),
        "activation": str(row.get("activation", "relu")),
        "use_class_weights": bool(row.get("use_class_weights", False)),
        "use_lr_scheduler": bool(row.get("use_lr_scheduler", False)),
        "random_seed": int(row.get("random_seed", 42)),
        "dataloader_seed": int(row.get("random_seed", 42)),
        "monitor_metric": str(row.get("monitor_metric", "val_macro_f1")),
        "oos_strategy": str(row.get("oos_strategy", "explicit_class")),
    }


def _build_text_cnn_hyperparameters(row: pd.Series) -> dict[str, Any]:
    """Extract TextCNN hyperparameters from the winning tuning row."""
    kernel_sizes_raw = row.get("kernel_sizes", "[3, 4, 5]")
    if isinstance(kernel_sizes_raw, str):
        kernel_sizes = tuple(json.loads(kernel_sizes_raw))
    else:
        kernel_sizes = tuple(kernel_sizes_raw)

    return {
        "vocab_size": 0,  # resolved at runtime from preprocessing artifacts
        "embedding_dim": int(row["embedding_dim"]),
        "num_filters": int(row["num_filters"]),
        "kernel_sizes": kernel_sizes,
        "dropout_rate": float(row["dropout_rate"]),
        "activation": "relu",
        "learning_rate": float(row["learning_rate"]),
        "weight_decay": float(row["weight_decay"]),
        "batch_size": 64,
        "max_epochs": 100,
        "early_stopping_patience": 10,
        "optimizer": "adam",
        "trainable_embeddings": True,
        "use_class_weights": False,
        "use_lr_scheduler": False,
        "random_seed": 42,
        "dataloader_seed": 42,
        "monitor_metric": "val_macro_f1",
        "oos_strategy": "explicit_class",
        "max_seq_length": 20,
    }


def _build_bilstm_hyperparameters(row: pd.Series) -> dict[str, Any]:
    """Extract BiLSTM hyperparameters from the winning tuning row."""
    return {
        "vocab_size": 0,  # resolved at runtime from preprocessing artifacts
        "embedding_dim": int(row["embedding_dim"]),
        "hidden_dim": int(row["hidden_dim"]),
        "num_layers": int(row["num_layers"]),
        "bidirectional": bool(row["bidirectional"]),
        "dropout_rate": float(row["dropout_rate"]),
        "learning_rate": float(row["learning_rate"]),
        "weight_decay": float(row["weight_decay"]),
        "batch_size": 64,
        "max_epochs": 100,
        "early_stopping_patience": 10,
        "optimizer": "adam",
        "trainable_embeddings": True,
        "use_class_weights": False,
        "use_lr_scheduler": False,
        "random_seed": 42,
        "dataloader_seed": 42,
        "monitor_metric": "val_macro_f1",
        "oos_strategy": "explicit_class",
        "max_seq_length": 20,
        "summarization_mode": str(row.get("summarization_mode", "concat_final_hidden")),
        "gradient_clipping": True,
        "max_grad_norm": float(row.get("max_grad_norm", 1.0)),
    }


_HYPERPARAM_BUILDERS: dict[ModelID, Any] = {
    ModelID.MLP: _build_mlp_hyperparameters,
    ModelID.TEXT_CNN: _build_text_cnn_hyperparameters,
    ModelID.BILSTM: _build_bilstm_hyperparameters,
}


def _resolve_vocab_size(model_id: ModelID) -> int | None:
    """Resolve vocab_size from preprocessing artifacts for neural models."""
    if model_id is ModelID.MLP:
        return None

    vocab_path = ARTIFACTS_DIR / "vocab.json"
    assert vocab_path.exists(), f"Vocab artifact not found: {vocab_path}"
    vocab_data: dict[str, Any] = json.loads(vocab_path.read_text())
    return int(vocab_data["size"])


def _resolve_max_seq_length(model_id: ModelID) -> int | None:
    """Resolve actual max_seq_length from preprocessing summary."""
    if model_id is ModelID.MLP:
        return None

    summary_path = ARTIFACTS_DIR / "preprocessing_summary.json"
    assert summary_path.exists(), f"Preprocessing summary not found: {summary_path}"
    summary: dict[str, Any] = json.loads(summary_path.read_text())
    return int(summary["max_seq_length"])


def extract_frozen_config(
    model_id: ModelID,
    tuning_csv_path: Path,
) -> FrozenModelConfig:
    """Extract a frozen config from an existing tuning CSV.

    Reads the tuning results, selects the winning row using validation-only
    criteria, resolves all runtime values, and returns a typed FrozenModelConfig.
    """
    assert tuning_csv_path.exists(), f"Tuning CSV not found: {tuning_csv_path}"

    df = pd.read_csv(tuning_csv_path)
    winning_row = _select_winning_row(df, model_id)

    build_hyperparams = _HYPERPARAM_BUILDERS[model_id]
    hyperparameters: dict[str, Any] = build_hyperparams(winning_row)

    vocab_size = _resolve_vocab_size(model_id)
    max_seq_length = _resolve_max_seq_length(model_id)

    # For neural models, set the resolved vocab_size in hyperparameters
    if vocab_size is not None:
        hyperparameters["vocab_size"] = vocab_size

    # Compute repo-relative path for the source tuning artifact
    try:
        source_ref = str(tuning_csv_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        source_ref = str(tuning_csv_path)

    frozen = FrozenModelConfig(
        schema_version=SCHEMA_VERSION,
        protocol_version=PROTOCOL_VERSION,
        model_id=model_id,
        model_name=model_id.display_name,
        hyperparameters=hyperparameters,
        source_tuning_artifact=source_ref,
        winning_row_id=str(winning_row["run_name"]),
        selection_metric="best_val_metric",
        selection_value=float(winning_row["best_val_metric"]),
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        monitor_metric=hyperparameters.get("monitor_metric", "val_macro_f1"),
        preprocessing_manifest_ref=PREPROCESSING_MANIFEST_REF,
        label_order_ref=LABEL_ORDER_REF,
        oos_strategy=hyperparameters.get("oos_strategy", "explicit_class"),
        oos_class_id=OOS_LABEL_ID,
    )

    logger.info(
        "Frozen config for %s: winning_row=%s, selection_value=%.6f",
        model_id,
        frozen.winning_row_id,
        frozen.selection_value,
    )

    return frozen


def save_frozen_config(frozen: FrozenModelConfig) -> Path:
    """Save a FrozenModelConfig to ``outputs/{model_id}/frozen_final_config.json``."""
    out_dir = model_output_dir(frozen.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frozen_final_config.json"
    data = frozen.to_dict()
    data["protocol_manifest_ref"] = PROTOCOL_MANIFEST_REF
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    logger.info("Saved frozen config: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Tuning artifact normalization
# ---------------------------------------------------------------------------


def _find_tuning_csv(model_id: ModelID) -> Path:
    """Locate the tuning CSV, checking canonical then legacy locations."""
    canonical = model_output_dir(model_id) / TUNING_SUBDIR / "tuning_results.csv"
    if canonical.exists():
        return canonical

    legacy = REPORTS_DIR / model_id.legacy_tuning_filename
    assert legacy.exists(), (
        f"Tuning CSV not found for '{model_id}' at canonical ({canonical}) or legacy ({legacy}) locations"
    )
    return legacy


def normalize_tuning_artifacts(model_id: ModelID) -> Path:
    """Copy legacy tuning CSV to canonical location and write selection_summary.json.

    Returns the path to the canonical tuning CSV.
    """
    tuning_dir = model_output_dir(model_id) / TUNING_SUBDIR
    tuning_dir.mkdir(parents=True, exist_ok=True)
    canonical_csv = tuning_dir / "tuning_results.csv"

    legacy_csv = REPORTS_DIR / model_id.legacy_tuning_filename

    if not canonical_csv.exists() and legacy_csv.exists():
        shutil.copy2(legacy_csv, canonical_csv)
        logger.info("Copied tuning CSV: %s -> %s", legacy_csv, canonical_csv)
    elif canonical_csv.exists():
        logger.info("Canonical tuning CSV already exists: %s", canonical_csv)
    else:
        raise FileNotFoundError(
            f"No tuning CSV found for '{model_id}' at legacy ({legacy_csv}) or canonical ({canonical_csv})"
        )

    # Extract winning row and write selection summary
    df = pd.read_csv(canonical_csv)
    winning_row = _select_winning_row(df, model_id)

    has_val_loss = "best_val_loss" in df.columns
    tie_break_fields = ["best_val_loss", "run_name"] if has_val_loss else ["run_name"]

    selection_summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_id": model_id,
        "model_name": model_id.display_name,
        "source_tuning_artifact": str(canonical_csv.relative_to(OUTPUTS_DIR)),
        "total_configurations_evaluated": len(df),
        "selection_metric": "best_val_metric",
        "selection_rule": "max best_val_metric",
        "tie_break_fields": tie_break_fields,
        "winning_row_id": str(winning_row["run_name"]),
        "winning_best_val_metric": float(winning_row["best_val_metric"]),
        "winning_best_val_loss": float(winning_row["best_val_loss"]) if has_val_loss else None,
        "winning_best_epoch": int(winning_row["best_epoch"]),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    summary_path = tuning_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(selection_summary, indent=2) + "\n")
    logger.info("Saved selection summary: %s", summary_path)

    return canonical_csv


# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------


def save_evaluation_protocol(
    protocol: RepeatedRunProtocol,
    models: list[ModelID],
) -> Path:
    """Write ``outputs/shared/evaluation_protocol.json``.

    Contains the full protocol config, model list, canonical model order,
    OOS evaluation policy, metric definitions, and efficiency timing policy.
    """
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SHARED_DIR / "evaluation_protocol.json"

    protocol_data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_config": protocol.to_dict(),
        "models": models,
        "canonical_model_order": [m.value for m in ModelID],
        "canonical_model_display_names": {m.value: m.display_name for m in ModelID},
        "oos_evaluation_policy": {
            "oos_class_name": OOS_LABEL_NAME,
            "oos_class_id": OOS_LABEL_ID,
            "evaluation_method": "explicit_class",
            "one_vs_rest_rule": "oos is the positive class; all in-scope labels are negative",
            "note": "OOS is included in multiclass macro F1 by default",
        },
        "metric_definitions": {
            "accuracy": "overall accuracy (correct / total)",
            "macro_f1": "macro-averaged F1 across all classes including OOS",
            "precision": "macro-averaged precision across all classes including OOS",
            "recall": "macro-averaged recall across all classes including OOS",
            "oos_precision": "one-vs-rest precision for OOS class",
            "oos_recall": "one-vs-rest recall for OOS class",
            "oos_f1": "one-vs-rest F1 for OOS class",
            "zero_division_policy": "zero_division=0 (sklearn convention)",
            "value_range": "[0.0, 1.0] (raw ratios, not percentages)",
        },
        "efficiency_timing_policy": {
            "timing_includes_dataloader_overhead": protocol.timing_includes_dataloader_overhead,
            "fields": [
                "training_time_seconds",
                "inference_total_seconds",
                "inference_avg_ms_per_example",
                "inference_examples_per_sec",
                "parameter_count",
                "trainable_parameter_count",
            ],
        },
        "seed_policy": {
            "seed_list": list(protocol.seed_list),
            "effective_seed_list": protocol.effective_seed_list(),
            "derivation_rule": "training_seed = seed, dataloader_seed = seed + 1",
            "report_note": (
                "Each run copies the frozen model configuration with `random_seed = seed` "
                "and `dataloader_seed = seed + 1`. The training seed controls initialization "
                "and training randomness; the dataloader seed controls train-batch shuffling."
            ),
            "note": "Same seed list reused across all models",
        },
        "final_model_rule": (
            "best checkpoint by the configured validation monitor, with lower validation loss as tie-break; "
            "representative run selected separately by validation macro F1"
        ),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    out_path.write_text(json.dumps(protocol_data, indent=2) + "\n")
    logger.info("Saved evaluation protocol: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Foundation orchestration
# ---------------------------------------------------------------------------


def ensure_model_output_dirs(model_id: ModelID) -> dict[str, Path]:
    """Create the canonical output directory tree for a model and return paths."""
    root = model_output_dir(model_id)
    dirs: dict[str, Path] = {
        "root": root,
        "tuning": root / TUNING_SUBDIR,
        "final_runs": root / FINAL_RUNS_SUBDIR,
        "aggregate": root / AGGREGATE_SUBDIR,
        "figures": root / MODEL_FIGURES_SUBDIR,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def extract_and_freeze_configs(
    model_ids: list[ModelID],
    protocol: RepeatedRunProtocol,
) -> dict[ModelID, FrozenModelConfig]:
    """Extract frozen configs from tuning artifacts for all requested models.

    Saves the shared evaluation protocol, normalises tuning CSVs into the
    canonical directory layout, selects the winning row per model using
    validation-only criteria, and serialises each ``FrozenModelConfig``.

    Returns a dict mapping ModelID -> FrozenModelConfig.
    """
    effective_seeds = protocol.effective_seed_list()
    assert len(effective_seeds) > 0, "Seed list must be non-empty"
    assert len(effective_seeds) == len(set(effective_seeds)), f"Seed list contains duplicates: {effective_seeds}"

    print("=" * 60)
    print("  Frozen-Config Extraction")
    print("=" * 60)
    print(f"  Models:         {model_ids}")
    print(f"  Run count:      {protocol.run_count}")
    print(f"  Effective seeds: {effective_seeds}")
    print()

    protocol_path = save_evaluation_protocol(protocol, model_ids)
    print(f"  Evaluation protocol: {protocol_path}")

    frozen_configs: dict[ModelID, FrozenModelConfig] = {}

    for model_id in model_ids:
        print(f"\n--- {model_id.display_name} ({model_id}) ---")

        dirs = ensure_model_output_dirs(model_id)
        print(f"  Output root:   {dirs['root']}")

        canonical_csv = normalize_tuning_artifacts(model_id)
        print(f"  Tuning CSV:    {canonical_csv}")

        frozen = extract_frozen_config(model_id, canonical_csv)
        frozen_path = save_frozen_config(frozen)
        print(f"  Frozen config: {frozen_path}")
        print(f"  Winning row:   {frozen.winning_row_id}")
        print(f"  Val metric:    {frozen.selection_value:.6f}")

        frozen_configs[model_id] = frozen

    print("\n" + "=" * 60)
    print("  Frozen-config extraction complete.")
    print("=" * 60)

    return frozen_configs


# ---------------------------------------------------------------------------
# Core repeated-run execution and aggregation
# ---------------------------------------------------------------------------


def _frozen_config_hash(frozen: FrozenModelConfig) -> str:
    """Deterministic hash of the frozen config for provenance tracking."""
    raw = json.dumps(frozen.to_dict(), sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _derive_seeds(base_seed: int) -> tuple[int, int]:
    """Derive training_seed and dataloader_seed deterministically from a base seed.

    Convention: training_seed = seed, dataloader_seed = seed + 1.
    """
    return base_seed, base_seed + 1


def _load_data_for_model(
    model_id: ModelID,
    config: BaseModelConfig,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Load datasets and metadata for a model. Returns (data_bundle, texts_by_split).

    The data_bundle contains everything needed to create DataLoaders and evaluate:
    loaders, metadata, input_dim (MLP only), num_classes.
    """
    from src.train import load_neural_data, load_tfidf_data

    texts_by_split: dict[str, list[str]] = {}

    if model_id is ModelID.MLP:
        assert isinstance(config, MLPBaselineConfig)
        loaders, input_dim, num_classes, metadata = load_tfidf_data(config)
        texts_by_split = metadata["texts_by_split"]
        return {
            "loaders": loaders,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "metadata": metadata,
        }, texts_by_split
    else:
        assert isinstance(config, (TextCNNConfig, BiLSTMConfig))
        loaders, metadata = load_neural_data(config)

        from src.config import DATASET_CONFIG
        from src.dataset import CLINCDataset
        from src.preprocessing import clean_text

        dataset = CLINCDataset.load(DATASET_CONFIG)
        for split in ("train", "validation", "test"):
            raw = dataset[split]
            texts_by_split[split] = [clean_text(t) for t in raw["text"]]

        return {
            "loaders": loaders,
            "num_classes": metadata["num_classes"],
            "metadata": metadata,
        }, texts_by_split


def _rebuild_dataloaders(
    data_bundle: dict[str, Any],
    model_id: ModelID,
    config: BaseModelConfig,
    dataloader_seed: int,
) -> dict[str, DataLoader]:
    """Rebuild DataLoaders with a new dataloader_seed for reproducibility."""
    from src.dataset import create_dataloaders

    existing_loaders: dict[str, DataLoader] = data_bundle["loaders"]
    datasets_dict = {split: loader.dataset for split, loader in existing_loaders.items()}

    return create_dataloaders(
        datasets_dict,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=False,
        random_seed=dataloader_seed,
    )


def _run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], np.ndarray, float]:
    """Run inference on a data loader. Returns (preds, targets, probs, inference_seconds)."""
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_probs: list[np.ndarray] = []

    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.tolist())
    inference_time = time.time() - start_time

    probs_array = np.concatenate(all_probs, axis=0)

    return all_preds, all_targets, probs_array, inference_time


def _compute_run_metrics(
    preds: list[int],
    targets: list[int],
    inference_time: float,
) -> dict[str, float]:
    """Compute all metrics for a single run's test or validation split."""
    cls_metrics = compute_classification_metrics(targets, preds)
    oos_metrics = compute_oos_metrics(targets, preds, OOS_LABEL_ID)

    n_examples = len(targets)
    avg_ms = (inference_time / max(n_examples, 1)) * 1000
    examples_per_sec = n_examples / max(inference_time, 1e-9)

    return {
        "accuracy": cls_metrics["accuracy"],
        "macro_f1": cls_metrics["macro_f1"],
        "precision": cls_metrics["macro_precision"],
        "recall": cls_metrics["macro_recall"],
        "oos_precision": oos_metrics["oos_precision"],
        "oos_recall": oos_metrics["oos_recall"],
        "oos_f1": oos_metrics["oos_f1"],
        "inference_total_seconds": inference_time,
        "inference_avg_ms_per_example": avg_ms,
        "inference_examples_per_sec": examples_per_sec,
    }


# ---------------------------------------------------------------------------
# Per-run artifact bundle saving
# ---------------------------------------------------------------------------


def _save_run_metadata(
    run_result: RunResult,
    frozen: FrozenModelConfig,
    run_dir: Path,
    protocol: RepeatedRunProtocol | None = None,
) -> None:
    """Save run_metadata.json."""
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_name": frozen.model_name,
        "model_id": frozen.model_id,
        "run_id": run_result.run_id,
        "run_index": run_result.run_index,
        "seed": run_result.seed,
        "training_seed": run_result.training_seed,
        "dataloader_seed": run_result.dataloader_seed,
        "status": run_result.status,
        "best_epoch": run_result.best_epoch,
        "stopping_epoch": run_result.stopping_epoch,
        "best_val_metric": run_result.best_val_metric,
        "best_val_loss": run_result.best_val_loss,
        "monitor_metric": run_result.monitor_metric,
        "checkpoint_path": _to_repo_relative(run_result.checkpoint_path),
        "log_path": _to_repo_relative(run_result.log_path),
        "frozen_config_ref": str(
            Path(model_output_dir(frozen.model_id) / "frozen_final_config.json").relative_to(PROJECT_ROOT)
        ),
        "protocol_manifest_ref": str(Path(SHARED_DIR / "evaluation_protocol.json").relative_to(PROJECT_ROOT)),
        "preprocessing_manifest_ref": frozen.preprocessing_manifest_ref,
        "label_order_ref": frozen.label_order_ref,
        "probability_saving_policy": protocol.probability_saving_policy if protocol else "unknown",
        "device": str(get_device()),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "frozen_config_hash": _frozen_config_hash(frozen),
    }
    if run_result.status == "failed":
        metadata["failure_reason"] = run_result.failure_reason
        metadata["stage_reached"] = run_result.stage_reached
        metadata["partial_artifacts"] = run_result.partial_artifacts

    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _save_metrics_json(
    metrics: dict[str, float],
    model: torch.nn.Module,
    training_time: float,
    inference_time: float,
    n_examples: int,
    filename: str,
    run_dir: Path,
) -> None:
    """Save validation_metrics.json or test_metrics.json."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)

    avg_ms = (inference_time / max(n_examples, 1)) * 1000
    examples_per_sec = n_examples / max(inference_time, 1e-9)

    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "oos_precision": metrics["oos_precision"],
        "oos_recall": metrics["oos_recall"],
        "oos_f1": metrics["oos_f1"],
        "training_time_seconds": training_time,
        "inference_total_seconds": inference_time,
        "inference_avg_ms_per_example": avg_ms,
        "inference_examples_per_sec": examples_per_sec,
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
    }
    (run_dir / filename).write_text(json.dumps(out, indent=2) + "\n")


def _save_epoch_history(
    epoch_history: list[dict[str, Any]],
    run_dir: Path,
) -> None:
    """Save epoch_history.json."""
    records: list[dict[str, Any]] = []
    for record in epoch_history:
        entry: dict[str, Any] = {
            "epoch": record["epoch"],
            "train_loss": record["train_loss"],
            "val_loss": record["val_loss"],
            "val_accuracy": record["val_accuracy"],
            "val_macro_f1": record["val_macro_f1"],
            "val_macro_precision": record.get("val_precision", 0.0),
            "val_macro_recall": record.get("val_recall", 0.0),
            "val_oos_f1": record.get("val_oos_f1", 0.0),
            "learning_rate": record.get("learning_rate", 0.0),
            "checkpoint_updated": record.get("checkpoint_updated", False),
        }
        if "epoch_duration_seconds" in record:
            entry["epoch_duration_seconds"] = record["epoch_duration_seconds"]
        records.append(entry)

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "epochs": records,
    }
    (run_dir / "epoch_history.json").write_text(json.dumps(artifact, indent=2) + "\n")


def _save_final_predictions(
    preds: list[int],
    targets: list[int],
    texts: list[str],
    probs: np.ndarray,
    label_names: list[str],
    run_id: str,
    run_dir: Path,
) -> None:
    """Save final_predictions.csv."""
    max_confidences = probs.max(axis=1).tolist()
    records: list[dict[str, Any]] = []
    for i in range(len(targets)):
        records.append(
            {
                "example_index": i,
                "text": texts[i] if i < len(texts) else "",
                "true_label_id": targets[i],
                "true_label_name": label_names[targets[i]],
                "predicted_label_id": preds[i],
                "predicted_label_name": label_names[preds[i]],
                "run_id": run_id,
                "max_confidence": max_confidences[i],
            }
        )
    pd.DataFrame(records).to_csv(run_dir / "final_predictions.csv", index=False)


def _save_per_class_metrics(
    preds: list[int],
    targets: list[int],
    label_names: list[str],
    run_dir: Path,
) -> None:
    """Save per_class_metrics.json."""
    report = classification_report(
        targets,
        preds,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    entries: list[dict[str, Any]] = []
    for label_id, name in enumerate(label_names):
        if name in report:
            class_data = report[name]
            entries.append(
                {
                    "label_id": label_id,
                    "label_name": name,
                    "support": int(class_data["support"]),
                    "precision": float(class_data["precision"]),
                    "recall": float(class_data["recall"]),
                    "f1": float(class_data["f1-score"]),
                    "is_oos": name == OOS_LABEL_NAME,
                }
            )
    per_class_artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "classes": entries,
    }
    (run_dir / "per_class_metrics.json").write_text(json.dumps(per_class_artifact, indent=2) + "\n")


def _save_confusion_artifacts(
    preds: list[int],
    targets: list[int],
    texts: list[str],
    label_names: list[str],
    run_dir: Path,
) -> None:
    """Save confusion_matrix.csv, top_confusions.json, top_errors.json."""
    confusion_df = build_confusion_matrix(targets, preds, label_names)
    confusion_df.to_csv(run_dir / "confusion_matrix.csv")

    top_confusions = find_top_confusions(confusion_df)
    top_confusions_artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "confusions": top_confusions,
    }
    (run_dir / "top_confusions.json").write_text(json.dumps(top_confusions_artifact, indent=2) + "\n")

    top_errors = find_top_errors(targets, preds, texts, label_names)
    top_errors_artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "selection_rule": (
            "All misclassified examples sorted by (true_label, predicted_label, text); first top_k returned."
        ),
        "label_ordering": label_names,
        "errors": top_errors,
    }
    (run_dir / "top_errors.json").write_text(json.dumps(top_errors_artifact, indent=2) + "\n")


def _save_confidences(
    probs: np.ndarray,
    preds: list[int],
    targets: list[int],
    label_names: list[str],
    run_dir: Path,
) -> None:
    """Save confidences.npz."""
    np.savez_compressed(
        run_dir / "confidences.npz",
        probabilities=probs,
        predictions=np.array(preds),
        targets=np.array(targets),
        label_names=np.array(label_names),
    )


def save_per_run_bundle(
    run_result: RunResult,
    frozen: FrozenModelConfig,
    model: torch.nn.Module,
    val_metrics: dict[str, float],
    test_preds: list[int],
    test_targets: list[int],
    test_probs: np.ndarray,
    texts: list[str],
    label_names: list[str],
    training_time: float,
    test_inference_time: float,
    val_inference_time: float,
    n_val_examples: int,
    protocol: RepeatedRunProtocol,
    run_dir: Path,
) -> None:
    """Save the complete per-run artifact bundle."""
    n_test = len(test_targets)

    _save_run_metadata(run_result, frozen, run_dir, protocol)

    # Validation metrics (with OOS from validation inference)
    _save_metrics_json(
        val_metrics, model, training_time, val_inference_time, n_val_examples, "validation_metrics.json", run_dir
    )

    # Test metrics
    test_met = _compute_run_metrics(test_preds, test_targets, test_inference_time)
    test_metrics_full: dict[str, float] = {
        "accuracy": test_met["accuracy"],
        "macro_f1": test_met["macro_f1"],
        "precision": test_met["precision"],
        "recall": test_met["recall"],
        "oos_precision": test_met["oos_precision"],
        "oos_recall": test_met["oos_recall"],
        "oos_f1": test_met["oos_f1"],
    }
    _save_metrics_json(
        test_metrics_full, model, training_time, test_inference_time, n_test, "test_metrics.json", run_dir
    )

    _save_epoch_history(run_result.epoch_history, run_dir)
    _save_final_predictions(test_preds, test_targets, texts, test_probs, label_names, run_result.run_id, run_dir)
    _save_per_class_metrics(test_preds, test_targets, label_names, run_dir)
    _save_confusion_artifacts(test_preds, test_targets, texts, label_names, run_dir)

    label_order_artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "label_names": label_names,
    }
    (run_dir / "label_order.json").write_text(json.dumps(label_order_artifact, indent=2) + "\n")

    # The supported all_runs policy saves probabilities for every completed run.
    _save_confidences(test_probs, test_preds, test_targets, label_names, run_dir)


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------


def execute_single_run(
    model_id: ModelID,
    frozen: FrozenModelConfig,
    seed: int,
    run_index: int,
    protocol: RepeatedRunProtocol,
    run_dir: Path,
    data_bundle: dict[str, Any],
    texts_by_split: dict[str, list[str]],
) -> RunResult:
    """Execute one complete train-evaluate cycle for a single seed.

    Returns a RunResult with all metrics and artifact paths.
    """
    from src.train import (
        build_bilstm,
        build_mlp_baseline,
        build_text_cnn,
        train_bilstm,
        train_mlp_baseline,
        train_text_cnn,
    )
    from src.utils import load_checkpoint

    training_seed, dataloader_seed = _derive_seeds(seed)
    if frozen.model_id != model_id:
        raise ValueError(f"Frozen config model_id {frozen.model_id!r} does not match {model_id!r}")
    model_config = replace(frozen.to_model_config(), random_seed=training_seed, dataloader_seed=dataloader_seed)
    run_id = run_dir_name(run_index, seed)

    checkpoint_subdir = run_dir / RUN_CHECKPOINT_SUBDIR
    log_subdir = run_dir / RUN_LOGS_SUBDIR
    checkpoint_subdir.mkdir(parents=True, exist_ok=True)
    log_subdir.mkdir(parents=True, exist_ok=True)

    result = RunResult(
        model_id=model_id,
        run_id=run_id,
        run_index=run_index,
        seed=seed,
        training_seed=model_config.random_seed,
        dataloader_seed=model_config.dataloader_seed,
        status="in_progress",
        run_dir=str(run_dir),
        monitor_metric=frozen.monitor_metric,
    )

    try:
        # --- Stage 1: Seed and rebuild DataLoaders ---
        set_seed(training_seed)

        loaders = _rebuild_dataloaders(data_bundle, model_id, model_config, dataloader_seed)

        result.stage_reached = "data_loaded"

        # --- Stage 2: Train ---
        metadata = data_bundle["metadata"]
        num_classes: int = data_bundle["num_classes"]

        checkpoint_filename = f"best_{run_id}.pt"

        if model_id is ModelID.MLP:
            assert isinstance(model_config, MLPBaselineConfig)
            input_dim: int = data_bundle["input_dim"]
            fit_result = train_mlp_baseline(
                model_config,
                loaders,
                input_dim,
                num_classes,
                metadata,
                checkpoint_dir=checkpoint_subdir,
                log_dir=log_subdir,
            )
        elif model_id is ModelID.TEXT_CNN:
            assert isinstance(model_config, TextCNNConfig)
            fit_result = train_text_cnn(
                model_config,
                loaders,
                num_classes,
                metadata,
                checkpoint_dir=checkpoint_subdir,
                log_dir=log_subdir,
            )
        elif model_id is ModelID.BILSTM:
            assert isinstance(model_config, BiLSTMConfig)
            fit_result = train_bilstm(
                model_config,
                loaders,
                num_classes,
                metadata,
                checkpoint_dir=checkpoint_subdir,
                log_dir=log_subdir,
            )
        else:
            raise ValueError(f"Unknown model_id: {model_id}")

        if fit_result.get("failure_reason"):
            result.status = "failed"
            result.failure_reason = fit_result["failure_reason"]
            result.stage_reached = "training_failed"
            _save_run_metadata(result, frozen, run_dir, protocol)
            return result

        result.stage_reached = "training_complete"
        result.best_epoch = fit_result["best_epoch"]
        result.stopping_epoch = fit_result["stopping_epoch"]
        result.training_time_seconds = fit_result["training_duration"]
        result.checkpoint_path = fit_result["checkpoint_path"]
        result.log_path = fit_result.get("log_path", "")
        result.epoch_history = fit_result["epoch_history"]

        # Extract validation metrics from best epoch
        best_epoch_idx = result.best_epoch - 1
        if 0 <= best_epoch_idx < len(result.epoch_history):
            best_record = result.epoch_history[best_epoch_idx]
            result.val_accuracy = best_record.get("val_accuracy", 0.0)
            result.val_macro_f1 = best_record.get("val_macro_f1", 0.0)
            result.val_precision = best_record.get("val_precision", 0.0)
            result.val_recall = best_record.get("val_recall", 0.0)
            result.val_loss = best_record.get("val_loss", 0.0)
        result.best_val_metric = fit_result["best_val_metric"]
        result.best_val_loss = result.val_loss

        # --- Stage 3: Rename checkpoint to canonical name ---
        old_checkpoint = Path(fit_result["checkpoint_path"])
        canonical_checkpoint = checkpoint_subdir / checkpoint_filename
        if old_checkpoint.exists() and old_checkpoint != canonical_checkpoint:
            old_checkpoint.rename(canonical_checkpoint)
            result.checkpoint_path = str(canonical_checkpoint)

        # --- Stage 4: Evaluation (validation + test) ---
        device = get_device()
        label_names: list[str] = metadata["label_names"]

        if model_id is ModelID.MLP:
            assert isinstance(model_config, MLPBaselineConfig)
            input_dim = data_bundle["input_dim"]
            model = build_mlp_baseline(model_config, input_dim, num_classes)
        elif model_id is ModelID.TEXT_CNN:
            assert isinstance(model_config, TextCNNConfig)
            model = build_text_cnn(model_config, num_classes)
        else:
            assert isinstance(model_config, BiLSTMConfig)
            model = build_bilstm(model_config, num_classes)

        load_checkpoint(result.checkpoint_path, model, device)
        model.to(device)

        # Validation inference for complete metrics including OOS
        val_preds, val_targets, _val_probs, val_inf_time = _run_inference(model, loaders["validation"], device)
        val_metrics = _compute_run_metrics(val_preds, val_targets, val_inf_time)
        result.val_accuracy = val_metrics["accuracy"]
        result.val_macro_f1 = val_metrics["macro_f1"]
        result.val_precision = val_metrics["precision"]
        result.val_recall = val_metrics["recall"]

        # Test inference
        test_preds, test_targets, test_probs, test_inference_time = _run_inference(model, loaders["test"], device)
        test_metrics = _compute_run_metrics(test_preds, test_targets, test_inference_time)
        result.test_accuracy = test_metrics["accuracy"]
        result.test_macro_f1 = test_metrics["macro_f1"]
        result.test_precision = test_metrics["precision"]
        result.test_recall = test_metrics["recall"]
        result.oos_precision = test_metrics["oos_precision"]
        result.oos_recall = test_metrics["oos_recall"]
        result.oos_f1 = test_metrics["oos_f1"]
        result.inference_total_seconds = test_metrics["inference_total_seconds"]
        result.inference_avg_ms_per_example = test_metrics["inference_avg_ms_per_example"]
        result.inference_examples_per_sec = test_metrics["inference_examples_per_sec"]

        total_params = sum(p.numel() for p in model.parameters())
        result.parameter_count = total_params
        result.trainable_parameter_count = count_parameters(model)

        result.stage_reached = "evaluation_complete"
        result.status = "completed"

        # Save per-run artifact bundle
        test_texts = texts_by_split.get("test", [])
        save_per_run_bundle(
            run_result=result,
            frozen=frozen,
            model=model,
            val_metrics=val_metrics,
            test_preds=test_preds,
            test_targets=test_targets,
            test_probs=test_probs,
            texts=test_texts,
            label_names=label_names,
            training_time=result.training_time_seconds,
            test_inference_time=test_inference_time,
            val_inference_time=val_inf_time,
            n_val_examples=len(val_targets),
            protocol=protocol,
            run_dir=run_dir,
        )

    except Exception as e:
        result.status = "failed"
        result.failure_reason = str(e)
        if not result.stage_reached:
            result.stage_reached = "initialization"
        logger.error("Run %s failed at stage '%s': %s", run_id, result.stage_reached, e)
        logger.debug(traceback.format_exc())
        try:
            _save_run_metadata(result, frozen, run_dir, protocol)
        except Exception:
            logger.warning("Could not save run_metadata.json for failed run %s", run_id)

    return result


# ---------------------------------------------------------------------------
# Aggregate metric computation
# ---------------------------------------------------------------------------


def compute_aggregate_metrics(
    run_results: list[RunResult],
    protocol: RepeatedRunProtocol,
    model_id: str,
) -> AggregateMetrics:
    """Compute mean/std across completed runs for all tracked metrics."""
    completed = [r for r in run_results if r.status == "completed"]

    metric_fields: list[tuple[str, str]] = [
        ("val_accuracy", "val_accuracy"),
        ("val_macro_f1", "val_macro_f1"),
        ("test_accuracy", "test_accuracy"),
        ("test_macro_f1", "test_macro_f1"),
        ("test_precision", "test_precision"),
        ("test_recall", "test_recall"),
        ("oos_precision", "oos_precision"),
        ("oos_recall", "oos_recall"),
        ("oos_f1", "oos_f1"),
        ("training_time_seconds", "training_time_seconds"),
        ("inference_total_seconds", "inference_total_seconds"),
        ("inference_avg_ms_per_example", "inference_avg_ms_per_example"),
        ("inference_examples_per_sec", "inference_examples_per_sec"),
        ("parameter_count", "parameter_count"),
        ("trainable_parameter_count", "trainable_parameter_count"),
    ]

    metrics: dict[str, dict[str, float]] = {}
    for metric_name, attr_name in metric_fields:
        values = [float(getattr(r, attr_name)) for r in completed]
        if values:
            metrics[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
            }
        else:
            metrics[metric_name] = {"mean": 0.0, "std": 0.0}

    return AggregateMetrics(
        model_id=model_id,
        run_count_requested=protocol.run_count,
        run_count_completed=len(completed),
        seed_list_requested=list(protocol.seed_list[: protocol.run_count]),
        seed_list_completed=[r.seed for r in completed],
        metrics=metrics,
    )


def save_aggregate_artifacts(
    aggregate: AggregateMetrics,
    run_results: list[RunResult],
    model_id: str,
    representative_run_id: str = "",
    frozen_config_ref: str = "",
) -> Path:
    """Save aggregate artifacts to ``outputs/{model}/aggregate/``."""
    agg_dir = model_output_dir(model_id) / AGGREGATE_SUBDIR
    agg_dir.mkdir(parents=True, exist_ok=True)

    mid = ModelID(model_id)
    completed = [r for r in run_results if r.status == "completed"]
    failed = [r for r in run_results if r.status == "failed"]

    if not frozen_config_ref:
        frozen_config_ref = _to_repo_relative(model_output_dir(model_id) / "frozen_final_config.json")

    # 1. aggregate_metrics.json (named summary groups)
    agg_json: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_name": mid.display_name,
        "model_id": aggregate.model_id,
        "run_count_requested": aggregate.run_count_requested,
        "run_count_completed": aggregate.run_count_completed,
        "seed_list_requested": aggregate.seed_list_requested,
        "seed_list_completed": aggregate.seed_list_completed,
        "all_runs_succeeded": aggregate.run_count_completed == aggregate.run_count_requested,
        "successful_run_ids": [r.run_id for r in completed],
        "failed_run_ids": [r.run_id for r in failed],
        "representative_run_id": representative_run_id,
        "frozen_config_ref": frozen_config_ref,
        "protocol_manifest_ref": PROTOCOL_MANIFEST_REF,
    }
    for group_name, metric_keys in SUMMARY_GROUPS.items():
        group: dict[str, dict[str, float]] = {}
        for mk in metric_keys:
            group[mk] = aggregate.metrics.get(mk, {"mean": 0.0, "std": 0.0})
        agg_json[group_name] = group

    (agg_dir / "aggregate_metrics.json").write_text(json.dumps(agg_json, indent=2) + "\n")

    # 2. aggregate_metrics.csv
    rows: list[dict[str, Any]] = []
    for metric_name, values in aggregate.metrics.items():
        rows.append({"metric": metric_name, "mean": values["mean"], "std": values["std"]})
    pd.DataFrame(rows).to_csv(agg_dir / "aggregate_metrics.csv", index=False)

    # 3. per_run_metrics.csv
    per_run_rows: list[dict[str, Any]] = []
    for r in run_results:
        per_run_rows.append(
            {
                "run_id": r.run_id,
                "run_index": r.run_index,
                "seed": r.seed,
                "status": r.status,
                "val_accuracy": r.val_accuracy,
                "val_macro_f1": r.val_macro_f1,
                "test_accuracy": r.test_accuracy,
                "test_macro_f1": r.test_macro_f1,
                "test_precision": r.test_precision,
                "test_recall": r.test_recall,
                "oos_precision": r.oos_precision,
                "oos_recall": r.oos_recall,
                "oos_f1": r.oos_f1,
                "training_time_seconds": r.training_time_seconds,
                "inference_total_seconds": r.inference_total_seconds,
                "inference_avg_ms_per_example": r.inference_avg_ms_per_example,
                "inference_examples_per_sec": r.inference_examples_per_sec,
                "parameter_count": r.parameter_count,
                "trainable_parameter_count": r.trainable_parameter_count,
                "best_epoch": r.best_epoch,
                "stopping_epoch": r.stopping_epoch,
            }
        )
    pd.DataFrame(per_run_rows).to_csv(agg_dir / "per_run_metrics.csv", index=False)

    # 4. aggregate_comparison_row.json
    # parameter_count and trainable_parameter_count are scalar (constant across runs)
    param_count = completed[0].parameter_count if completed else 0
    trainable_param_count = completed[0].trainable_parameter_count if completed else 0

    comparison_row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_name": mid.display_name,
        "model_id": model_id,
        "display_name": mid.display_name,
        "input_type": mid.input_type,
        "run_count": aggregate.run_count_completed,
        "primary_val_metric": "val_macro_f1",
    }
    for metric_name, values in aggregate.metrics.items():
        if metric_name in ("parameter_count", "trainable_parameter_count"):
            continue
        comparison_row[f"{metric_name}_mean"] = values["mean"]
        comparison_row[f"{metric_name}_std"] = values["std"]
    comparison_row["parameter_count"] = param_count
    comparison_row["trainable_parameter_count"] = trainable_param_count
    comparison_row["representative_run_id"] = representative_run_id
    comparison_row["frozen_config_ref"] = frozen_config_ref
    (agg_dir / "aggregate_comparison_row.json").write_text(json.dumps(comparison_row, indent=2) + "\n")

    return agg_dir


# ---------------------------------------------------------------------------
# Representative-run selection
# ---------------------------------------------------------------------------


def select_representative_run(
    run_results: list[RunResult],
    protocol: RepeatedRunProtocol,
) -> RepresentativeRunInfo:
    """Select the representative run per model.

    Default rule: highest validation macro F1.
    Tie-break: lower validation loss -> earlier run index -> lexicographic run ID.
    """
    completed = [r for r in run_results if r.status == "completed"]
    assert completed, "No completed runs to select representative from"

    def sort_key(r: RunResult) -> tuple[float, float, int, str]:
        return (-r.val_macro_f1, r.val_loss, r.run_index, r.run_id)

    completed.sort(key=sort_key)
    best = completed[0]

    return RepresentativeRunInfo(
        run_id=best.run_id,
        run_index=best.run_index,
        seed=best.seed,
        selection_rule=protocol.representative_run_rule,
        selection_metric_value=best.val_macro_f1,
        artifact_path=best.run_dir,
    )


def save_representative_run(
    representative: RepresentativeRunInfo,
    model_id: str,
) -> Path:
    """Save representative_run.json to ``outputs/{model}/aggregate/``."""
    agg_dir = model_output_dir(model_id) / AGGREGATE_SUBDIR
    agg_dir.mkdir(parents=True, exist_ok=True)

    mid = ModelID(model_id)
    frozen_config_ref = _to_repo_relative(model_output_dir(model_id) / "frozen_final_config.json")

    rep_json: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_name": mid.display_name,
        "model_id": model_id,
        "run_id": representative.run_id,
        "run_index": representative.run_index,
        "seed": representative.seed,
        "selection_rule": representative.selection_rule,
        "selection_metric_value": representative.selection_metric_value,
        "artifact_path": _to_repo_relative(representative.artifact_path),
        "frozen_config_ref": frozen_config_ref,
        "protocol_manifest_ref": PROTOCOL_MANIFEST_REF,
        "note": "This run is representative only. Aggregate metrics come from all completed runs.",
    }
    out_path = agg_dir / "representative_run.json"
    out_path.write_text(json.dumps(rep_json, indent=2) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Repeated-run orchestration
# ---------------------------------------------------------------------------


def run_repeated_evaluation(
    model_id: ModelID,
    frozen: FrozenModelConfig,
    protocol: RepeatedRunProtocol,
) -> ModelEvaluationResult:
    """Execute the full repeated-run evaluation for one model.

    1. Load data once
    2. For each seed: execute a full train+evaluate cycle
    3. Compute aggregate metrics
    4. Select representative run
    5. Save all aggregate artifacts
    """
    effective_seeds = protocol.effective_seed_list()

    assert frozen.model_id == model_id, (
        f"Frozen config model_id '{frozen.model_id}' doesn't match requested model_id '{model_id}'"
    )
    assert len(effective_seeds) > 0, "Seed list must be non-empty"
    assert len(effective_seeds) == len(set(effective_seeds)), f"Seed list contains duplicates: {effective_seeds}"

    model_config = frozen.to_model_config()
    final_runs_dir = model_output_dir(model_id) / FINAL_RUNS_SUBDIR
    final_runs_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  Repeated-Run Evaluation — {frozen.model_name}")
    print("=" * 60)
    print(f"  Model ID:        {model_id}")
    print(f"  Frozen config:   {frozen.winning_row_id}")
    print(f"  Source tuning:   {frozen.source_tuning_artifact}")
    print(f"  Run count:       {protocol.run_count}")
    print(f"  Seed list:       {effective_seeds}")
    print(f"  Monitor metric:  {frozen.monitor_metric}")
    print()

    # Load data once
    data_bundle, texts_by_split = _load_data_for_model(model_id, model_config)

    run_results: list[RunResult] = []

    for idx, seed in enumerate(effective_seeds, start=1):
        run_id = run_dir_name(idx, seed)
        run_dir = final_runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"  --- Run {idx}/{len(effective_seeds)}: {run_id} ---")

        result = execute_single_run(
            model_id=model_id,
            frozen=frozen,
            seed=seed,
            run_index=idx,
            protocol=protocol,
            run_dir=run_dir,
            data_bundle=data_bundle,
            texts_by_split=texts_by_split,
        )
        run_results.append(result)

        if result.status == "completed":
            print(f"    Best val metric: {result.best_val_metric:.4f} (epoch {result.best_epoch})")
            print(f"    Test accuracy:   {result.test_accuracy:.4f}")
            print(f"    Test macro F1:   {result.test_macro_f1:.4f}")
            print(f"    OOS F1:          {result.oos_f1:.4f}")
            print(f"    Training time:   {result.training_time_seconds:.1f}s")
        else:
            print(f"    FAILED at stage '{result.stage_reached}': {result.failure_reason}")

    # Compute aggregates and select representative run
    completed = [r for r in run_results if r.status == "completed"]
    assert completed, f"All runs failed for model '{model_id}'"

    aggregate = compute_aggregate_metrics(run_results, protocol, model_id)
    representative = select_representative_run(run_results, protocol)

    frozen_config_ref = _to_repo_relative(model_output_dir(model_id) / "frozen_final_config.json")
    save_aggregate_artifacts(
        aggregate,
        run_results,
        model_id,
        representative_run_id=representative.run_id,
        frozen_config_ref=frozen_config_ref,
    )
    save_representative_run(representative, model_id)

    # --- Validation assertions ---

    # Assert label ordering consistency across runs
    label_orders: list[list[str]] = []
    for r in completed:
        label_path = Path(r.run_dir) / "label_order.json"
        if label_path.exists():
            raw = json.loads(label_path.read_text())
            names = raw["label_names"] if isinstance(raw, dict) else raw
            label_orders.append(names)
    if len(label_orders) > 1:
        for lo in label_orders[1:]:
            assert lo == label_orders[0], "Label ordering inconsistency across runs"

    # Assert preprocessing manifest references are identical across runs
    manifest_refs: set[str] = set()
    for r in completed:
        meta_path = Path(r.run_dir) / "run_metadata.json"
        if meta_path.exists():
            meta: dict[str, Any] = json.loads(meta_path.read_text())
            manifest_refs.add(meta.get("preprocessing_manifest_ref", ""))
    assert len(manifest_refs) <= 1, f"Preprocessing manifest refs differ across runs: {manifest_refs}"

    # Assert all runs use the same monitoring metric
    monitor_metrics: set[str] = {r.monitor_metric for r in completed}
    assert len(monitor_metrics) == 1, f"Monitor metrics differ across runs: {monitor_metrics}"

    # Assert per-run artifact schemas validate before aggregates are built
    for r in completed:
        rdir = Path(r.run_dir)
        for required_file in (
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
        ):
            assert (rdir / required_file).exists(), f"Required per-run artifact missing: {rdir / required_file}"

    # Print aggregate summary
    print()
    config_hash = _frozen_config_hash(frozen)
    print(f"  === Aggregate Results: {frozen.model_name} ===")
    print(f"  Frozen config hash: {config_hash}")
    print(f"  Completed: {aggregate.run_count_completed}/{aggregate.run_count_requested} runs")
    for metric_name, vals in aggregate.metrics.items():
        print(f"    {metric_name}: {vals['mean']:.4f} ± {vals['std']:.4f}")
    print(f"  Representative run: {representative.run_id} (val_macro_f1={representative.selection_metric_value:.4f})")
    print()

    return ModelEvaluationResult(
        model_id=model_id,
        frozen_config=frozen,
        run_results=run_results,
        aggregate=aggregate,
        representative=representative,
    )


# ---------------------------------------------------------------------------
# Multi-model orchestration
# ---------------------------------------------------------------------------


def run_all_repeated_evaluations(
    model_ids: list[ModelID],
    frozen_configs: dict[ModelID, FrozenModelConfig],
    protocol: RepeatedRunProtocol,
) -> dict[ModelID, ModelEvaluationResult]:
    """Run repeated-evaluation for every requested model using its frozen config.

    Each model is trained + evaluated ``protocol.run_count`` times (one per seed).
    After all models finish, cross-model seed consistency is validated.

    Requires frozen configs to have been extracted first.
    """
    effective_seeds = protocol.effective_seed_list()
    assert len(effective_seeds) > 0, "Seed list must be non-empty"

    results: dict[ModelID, ModelEvaluationResult] = {}

    for model_id in model_ids:
        assert model_id in frozen_configs, (
            f"No frozen config found for '{model_id}'. Run extract_and_freeze_configs() first."
        )
        frozen = frozen_configs[model_id]
        result = run_repeated_evaluation(model_id, frozen, protocol)
        results[model_id] = result

    # Cross-model validation: same seeds used
    all_seed_lists = [r.aggregate.seed_list_completed for r in results.values()]
    if len(all_seed_lists) > 1:
        for sl in all_seed_lists[1:]:
            assert sl == all_seed_lists[0], f"Seed lists differ across models: {all_seed_lists}"

    return results
