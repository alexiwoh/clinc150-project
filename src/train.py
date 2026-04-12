"""Model training entry logic for launching experiments."""

from __future__ import annotations

import json
import logging
import platform
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import DataLoader

from src.config import DATASET_CONFIG, BiLSTMConfig, MLPBaselineConfig, TextCNNConfig, get_device
from src.enums import ModelID
from src.constants import (
    ARTIFACTS_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    NUM_CLASSES,
    OOS_LABEL_ID,
    OOS_LABEL_NAME,
    PROJECT_ROOT,
    REPORTS_DIR,
)
from src.dataset import CLINCDataset, IntentDataset, TFIDFDataset, create_dataloaders
from src.models.bilstm import BiLSTMClassifier
from src.models.mlp import MLPClassifier
from src.models.text_cnn import TextCNN
from src.preprocessing import (
    PAD_ID,
    UNK_ID,
    PreprocessingArtifacts,
    Vocabulary,
    clean_text,
    compute_oov_stats,
    compute_truncation_stats,
    load_preprocessing_artifacts,
    numericalize,
    pad_or_truncate,
    tokenize_text,
)
from src.trainers.trainer import Trainer
from src.utils import count_parameters, ensure_dir, load_checkpoint, set_seed

logger = logging.getLogger(__name__)

_SPLIT_NAMES = ("train", "validation", "test")

_TUNING_SEARCH_SPACE: dict[str, list[int | float]] = {
    "hidden_dim": [256, 512, 1024],
    "dropout_rate": [0.2, 0.3, 0.5],
    "learning_rate": [1e-3, 5e-4],
    "weight_decay": [0.0, 1e-4],
}

_TWO_LAYER_CONFIGS: list[dict[str, int | float]] = [
    {"hidden_dim": 512, "second_hidden_dim": 256, "dropout_rate": 0.3, "learning_rate": 1e-3, "weight_decay": 0.0},
    {"hidden_dim": 512, "second_hidden_dim": 256, "dropout_rate": 0.3, "learning_rate": 5e-4, "weight_decay": 1e-4},
    {"hidden_dim": 1024, "second_hidden_dim": 256, "dropout_rate": 0.2, "learning_rate": 1e-3, "weight_decay": 0.0},
    {"hidden_dim": 1024, "second_hidden_dim": 256, "dropout_rate": 0.2, "learning_rate": 5e-4, "weight_decay": 1e-4},
]


class TFIDFMetadata(TypedDict):
    """Metadata from loading TF-IDF data for the MLP baseline pipeline."""

    artifact_refs: dict[str, str]
    label_names: list[str]
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    texts_by_split: dict[str, list[str]]


class NeuralMetadata(TypedDict):
    """Metadata from loading neural (token-sequence) data for the Text CNN pipeline."""

    artifact_refs: dict[str, str]
    label_names: list[str]
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    num_classes: int
    vocab_size: int
    oov_stats: dict[str, dict[str, int | float]]
    truncation_stats: dict[str, dict[str, int | float]]
    preprocessing_policy: dict[str, bool | str]
    manifest_timestamp: str | None


def load_tfidf_data(
    config: MLPBaselineConfig,
) -> tuple[dict[str, DataLoader], int, int, TFIDFMetadata]:
    """Load TF-IDF features from saved artifacts and build DataLoaders.

    Returns (loaders, input_dim, num_classes, metadata).
    """
    artifacts: PreprocessingArtifacts = load_preprocessing_artifacts(ARTIFACTS_DIR)
    vectorizer: TfidfVectorizer = artifacts["vectorizer"]
    summary: dict[str, Any] = artifacts["summary"]

    label_to_id_path = ARTIFACTS_DIR / "label_to_id.json"
    id_to_label_path = ARTIFACTS_DIR / "id_to_label.json"
    label_to_id: dict[str, int] = json.loads(label_to_id_path.read_text())
    id_to_label: dict[int, str] = {int(k): v for k, v in json.loads(id_to_label_path.read_text()).items()}

    num_classes: int = len(label_to_id)
    assert num_classes == NUM_CLASSES, (
        f"Label mapping has {num_classes} classes but constants.NUM_CLASSES={NUM_CLASSES}"
    )
    expected_ids: set[int] = set(range(num_classes))
    actual_ids: set[int] = set(id_to_label.keys())
    assert actual_ids == expected_ids, f"id_to_label keys must span [0, {num_classes}), got {sorted(actual_ids)[:5]}..."

    label_names: list[str] = [id_to_label[i] for i in range(num_classes)]

    dataset = CLINCDataset.load(DATASET_CONFIG)
    assert dataset.label_names == label_names, "CLINCDataset label names don't match saved label mapping"

    texts_by_split: dict[str, list[str]] = {}
    labels_by_split: dict[str, list[int]] = {}
    for split in _SPLIT_NAMES:
        raw = dataset[split]
        texts_by_split[split] = [clean_text(t) for t in raw["text"]]
        labels_by_split[split] = raw["intent"]

    tfidf_matrices = {}
    for split in _SPLIT_NAMES:
        tfidf_matrices[split] = vectorizer.transform(texts_by_split[split])

    input_dim: int = tfidf_matrices["train"].shape[1]
    expected_dim: int | None = summary.get("tfidf_fitted_feature_dim")
    if expected_dim is not None:
        assert input_dim == expected_dim, (
            f"TF-IDF input_dim={input_dim} doesn't match preprocessing_summary ({expected_dim})"
        )

    n_train: int = tfidf_matrices["train"].shape[0]
    dense_bytes: int = n_train * input_dim * 4
    dense_mb: float = dense_bytes / (1024 * 1024)
    logger.info("Estimated dense train memory: %.1f MB (%d x %d x 4 bytes)", dense_mb, n_train, input_dim)
    assert dense_mb < 2048, f"Dense TF-IDF memory ({dense_mb:.0f} MB) exceeds 2 GB safety limit"

    datasets_dict: dict[str, TFIDFDataset] = {}
    for split in _SPLIT_NAMES:
        label_tensor = torch.tensor(labels_by_split[split], dtype=torch.long)

        mat = tfidf_matrices[split]
        arr = mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)
        assert not np.isnan(arr).any(), f"NaN in TF-IDF features for {split}"
        assert not np.isinf(arr).any(), f"Inf in TF-IDF features for {split}"
        assert label_tensor.dtype == torch.long, f"Label dtype is {label_tensor.dtype}, expected torch.long"
        assert (label_tensor >= 0).all() and (label_tensor < num_classes).all(), (
            f"Label ids out of range [0, {num_classes}) in {split}"
        )

        datasets_dict[split] = TFIDFDataset(tfidf_matrices[split], label_tensor)

    loaders = create_dataloaders(
        datasets_dict,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=False,
        random_seed=config.random_seed,
    )

    for split in _SPLIT_NAMES:
        assert len(loaders[split]) > 0, f"DataLoader for {split} has zero batches"

    for split in _SPLIT_NAMES:
        batch_x, batch_y = next(iter(loaders[split]))
        logger.info(
            "First batch %s: input shape=%s dtype=%s | label shape=%s dtype=%s",
            split,
            list(batch_x.shape),
            batch_x.dtype,
            list(batch_y.shape),
            batch_y.dtype,
        )

    logger.info("Model consumes sparse-origin inputs converted to dense float32 per-row in TFIDFDataset.__getitem__")

    artifact_refs: dict[str, str] = {
        "tfidf_vectorizer": str((ARTIFACTS_DIR / "tfidf_vectorizer.pkl").relative_to(PROJECT_ROOT)),
        "label_to_id": str(label_to_id_path.relative_to(PROJECT_ROOT)),
        "id_to_label": str(id_to_label_path.relative_to(PROJECT_ROOT)),
        "preprocessing_summary": str((ARTIFACTS_DIR / "preprocessing_summary.json").relative_to(PROJECT_ROOT)),
    }

    metadata: TFIDFMetadata = {
        "artifact_refs": artifact_refs,
        "label_names": label_names,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "texts_by_split": texts_by_split,
    }

    return loaders, input_dim, num_classes, metadata


def compute_class_weights(
    train_labels: list[int] | torch.Tensor, num_classes: int, device: torch.device
) -> torch.Tensor:
    """Compute inverse-frequency class weights from training labels only."""
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.tolist()
    counts = Counter(train_labels)
    total = len(train_labels)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls_id in range(num_classes):
        c = counts.get(cls_id, 0)
        weights[cls_id] = total / (num_classes * max(c, 1))
    return weights.to(device)


def build_mlp_baseline(config: MLPBaselineConfig, input_dim: int, num_classes: int) -> MLPClassifier:
    """Construct an MLPClassifier from config, with dimension assertions."""
    assert input_dim > 0, f"input_dim must be positive, got {input_dim}"
    assert num_classes == NUM_CLASSES, f"num_classes={num_classes} doesn't match NUM_CLASSES={NUM_CLASSES}"
    return MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        second_hidden_dim=config.second_hidden_dim,
        dropout_rate=config.dropout_rate,
        activation=config.activation,
    )


def _make_run_name(config: MLPBaselineConfig) -> str:
    """Build a deterministic, machine-readable run name from config."""
    layers = "1layer" if config.second_hidden_dim is None else f"2layer_h2{config.second_hidden_dim}"
    lr_str = f"lr{config.learning_rate}".replace(".", "")
    wd_str = f"wd{config.weight_decay}".replace(".", "")
    return f"h{config.hidden_dim}_d{config.dropout_rate}_{lr_str}_{wd_str}_{layers}_s{config.random_seed}"


def train_mlp_baseline(
    config: MLPBaselineConfig,
    loaders: dict[str, DataLoader],
    input_dim: int,
    num_classes: int,
    metadata: TFIDFMetadata,
    checkpoint_dir: Path | str | None = None,
    log_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Train a single MLP baseline run. Never iterates the test loader.

    Args:
        checkpoint_dir: Directory for saving checkpoints. Defaults to CHECKPOINTS_DIR.
        log_dir: Directory for saving training logs. Defaults to LOGS_DIR.
    """
    effective_checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else CHECKPOINTS_DIR
    effective_log_dir = Path(log_dir) if log_dir is not None else LOGS_DIR

    device = get_device()
    set_seed(config.random_seed)

    model = build_mlp_baseline(config, input_dim, num_classes)
    model.to(device)

    param_count = count_parameters(model)
    run_name = _make_run_name(config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    class_weights = None
    if config.use_class_weights:
        train_ds = loaders["train"].dataset
        assert hasattr(train_ds, "_labels"), "Expected TFIDFDataset with _labels attribute"
        class_weights = compute_class_weights(train_ds._labels, num_classes, device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    scheduler = None
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    trainer = Trainer(model, optimizer, criterion, device, scheduler=scheduler)

    print(f"\n{'=' * 60}")
    print(f"  MLP Baseline Run: {run_name}")
    print(f"{'=' * 60}")
    print(f"  Artifact refs: {metadata['artifact_refs']}")
    print(f"  Input dim: {input_dim}")
    print(f"  Num classes: {num_classes}")
    print(f"  Architecture: {model}")
    print(f"  Parameter count: {param_count:,}")
    print(f"  Optimizer: {config.optimizer} | lr={config.learning_rate} | wd={config.weight_decay}")
    print(f"  Dropout: {config.dropout_rate}")
    print(f"  Monitor metric: {config.monitor_metric}")
    print("  val_macro_f1 is the primary early-stopping / model-selection metric")

    fit_result = trainer.fit(
        train_loader=loaders["train"],
        val_loader=loaders["validation"],
        max_epochs=config.max_epochs,
        patience=config.early_stopping_patience,
        checkpoint_dir=effective_checkpoint_dir,
        run_name=run_name,
        config=config.to_dict(),
        artifact_refs=metadata["artifact_refs"],
        log_dir=effective_log_dir,
        monitor_metric=config.monitor_metric,
    )

    for record in fit_result["epoch_history"]:
        print(
            f"  Epoch {record['epoch']:3d} | "
            f"train_loss={record['train_loss']:.4f} | "
            f"val_loss={record['val_loss']:.4f} | "
            f"val_accuracy={record['val_accuracy']:.4f} | "
            f"val_macro_f1={record['val_macro_f1']:.4f}"
        )

    print(f"  Stopped: {fit_result['reason_for_stopping']} at epoch {fit_result['stopping_epoch']}")
    print(f"  Best epoch: {fit_result['best_epoch']} (val metric={fit_result['best_val_metric']:.4f})")
    print(f"  Checkpoint: {fit_result['checkpoint_path']}")
    print(f"  Training time: {fit_result['training_duration']:.1f}s")

    fit_result["parameter_count"] = param_count
    fit_result["input_dim"] = input_dim
    fit_result["num_classes"] = num_classes
    return fit_result


def run_mlp_tuning(
    configs: list[MLPBaselineConfig],
    loaders: dict[str, DataLoader],
    input_dim: int,
    num_classes: int,
    metadata: TFIDFMetadata,
) -> pd.DataFrame:
    """Run all tuning configs, save results to CSV. Never touches test loader."""
    ensure_dir(REPORTS_DIR)
    results: list[dict[str, Any]] = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n--- Tuning run {i}/{len(configs)} ---")
        run_result = train_mlp_baseline(cfg, loaders, input_dim, num_classes, metadata)
        row = {
            "run_name": run_result["run_name"],
            "best_epoch": run_result["best_epoch"],
            "best_val_metric": run_result["best_val_metric"],
            "training_duration": run_result["training_duration"],
            "checkpoint_path": run_result["checkpoint_path"],
            "reason_for_stopping": run_result["reason_for_stopping"],
            **run_result["config"],
        }
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(REPORTS_DIR / "mlp_tuning_results.csv", index=False)
    return df


def verify_mlp_checkpoint(
    checkpoint_path: str,
    config: MLPBaselineConfig,
    val_loader: DataLoader,
    input_dim: int,
    num_classes: int,
    expected_best_metric: float,
) -> None:
    """Fresh-process checkpoint reload verification for MLP.

    Loads a saved checkpoint, reconstructs the model, evaluates on the
    validation set, and asserts the metric matches the saved best value.
    """
    device: torch.device = get_device()

    model: MLPClassifier = build_mlp_baseline(config, input_dim, num_classes)
    meta: dict[str, Any] = load_checkpoint(checkpoint_path, model, device)
    model.to(device)
    model.eval()

    assert "artifact_refs" in meta, "Checkpoint missing artifact_refs"
    assert "config" in meta, "Checkpoint missing config"

    criterion: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    trainer: Trainer = Trainer(model, torch.optim.Adam(model.parameters()), criterion, device)
    val_metrics: dict[str, Any] = trainer.evaluate(val_loader)

    reloaded_f1: float = val_metrics["macro_f1"]
    tolerance: float = 1e-3
    assert abs(reloaded_f1 - expected_best_metric) < tolerance, (
        f"Reloaded val macro_f1={reloaded_f1:.6f} differs from saved best={expected_best_metric:.6f} "
        f"(tolerance={tolerance})"
    )
    logger.info(
        "MLP checkpoint reload PASSED: val macro_f1=%.6f (saved=%.6f)",
        reloaded_f1,
        expected_best_metric,
    )


def run_mlp_experiment() -> dict[str, Any]:
    """Full workflow: load data, default run, checkpoint verify, tuning, select best, final test eval."""
    from src.evaluate import evaluate_mlp_baseline

    default_config = MLPBaselineConfig()
    loaders, input_dim, num_classes, metadata = load_tfidf_data(default_config)

    # --- Phase 1 verification: default run + checkpoint reload ---
    print("\n=== Default Baseline Run ===")
    default_result = train_mlp_baseline(default_config, loaders, input_dim, num_classes, metadata)

    print("\n=== Fresh-Process Checkpoint Reload Test ===")
    verify_mlp_checkpoint(
        default_result["checkpoint_path"],
        default_config,
        loaders["validation"],
        input_dim,
        num_classes,
        default_result["best_val_metric"],
    )
    print("  Reload test PASSED")

    # --- Phase 2: staged tuning ---
    tuning_grid = _build_tuning_grid()
    print(f"\n=== Hyperparameter Tuning ({len(tuning_grid)} configs) ===")
    tuning_df = run_mlp_tuning(tuning_grid, loaders, input_dim, num_classes, metadata)

    # --- Select best config ---
    best_idx = int(tuning_df["best_val_metric"].idxmax())
    best_row = tuning_df.iloc[best_idx]
    best_checkpoint = best_row["checkpoint_path"]
    best_run_name: str = best_row["run_name"]
    best_config = MLPBaselineConfig(
        hidden_dim=int(best_row["hidden_dim"]),
        second_hidden_dim=int(best_row["second_hidden_dim"]) if pd.notna(best_row.get("second_hidden_dim")) else None,
        dropout_rate=float(best_row["dropout_rate"]),
        learning_rate=float(best_row["learning_rate"]),
        weight_decay=float(best_row["weight_decay"]),
        batch_size=int(best_row["batch_size"]),
        max_epochs=int(best_row["max_epochs"]),
        early_stopping_patience=int(best_row["early_stopping_patience"]),
        random_seed=int(best_row["random_seed"]),
    )

    print(f"\n=== Best config: {best_run_name} (val_metric={best_row['best_val_metric']:.4f}) ===")

    # --- Test-eval ordering guard ---
    assert best_checkpoint is not None, "Model selection not complete before test evaluation"

    # --- Final test evaluation ---
    print("=== Final Test Evaluation ===")

    best_training_result: dict[str, Any] = {
        "run_name": best_run_name,
        "training_duration": (
            float(best_row["training_duration"]) if pd.notna(best_row["training_duration"]) else None
        ),
        "best_epoch": int(best_row["best_epoch"]) if pd.notna(best_row["best_epoch"]) else None,
        "best_val_metric": float(best_row["best_val_metric"]),
        "checkpoint_path": best_checkpoint,
        "config": best_config.to_dict(),
        "device": str(get_device()),
        "training_seed": best_config.random_seed,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }

    log_path = LOGS_DIR / f"mlp_training_log_{best_run_name}.json"
    if log_path.exists():
        best_training_result["epoch_history"] = json.loads(log_path.read_text())

    test_results = evaluate_mlp_baseline(
        best_checkpoint,
        best_config,
        training_result=best_training_result,
    )

    print(f"\n{'=' * 60}")
    print("  Final MLP Test Results")
    print(f"{'=' * 60}")
    print(f"  Test accuracy:    {test_results['test_accuracy']:.4f}")
    print(f"  Test macro F1:    {test_results['test_macro_f1']:.4f}")
    print(f"  Test precision:   {test_results['test_precision']:.4f}")
    print(f"  Test recall:      {test_results['test_recall']:.4f}")
    print(f"  OOS precision:    {test_results['oos_precision']:.4f}")
    print(f"  OOS recall:       {test_results['oos_recall']:.4f}")
    print(f"  OOS F1:           {test_results['oos_f1']:.4f}")
    print(f"  Inference:        {test_results['inference_latency']['avg_ms_per_example']:.2f} ms/example")

    return {
        "default_result": default_result,
        "tuning_df": tuning_df,
        "best_config": best_config,
        "best_run_name": best_run_name,
        "test_results": test_results,
    }


def _build_tuning_grid() -> list[MLPBaselineConfig]:
    """Construct the tuning grid: 36 single-layer + 4 two-layer = 40 configs."""
    keys = list(_TUNING_SEARCH_SPACE.keys())
    configs = [
        MLPBaselineConfig(**dict(zip(keys, values, strict=True))) for values in product(*_TUNING_SEARCH_SPACE.values())
    ]
    configs.extend(MLPBaselineConfig(**overrides) for overrides in _TWO_LAYER_CONFIGS)
    return configs


# ======================================================================
# Text CNN — frozen-artifact data loading, training, and experiment
# ======================================================================


def _validate_preprocessing_manifest(
    summary: dict[str, Any],
    config: TextCNNConfig | BiLSTMConfig,
    label_to_id: dict[str, int],
) -> None:
    """Validate the full preprocessing manifest against a neural model config.

    Fails loudly on any mismatch so stale artifacts are never used silently.
    vocab_size is read from the manifest as the source of truth rather than
    cross-checked against a config default.
    """
    assert isinstance(summary["vocabulary_size"], int) and summary["vocabulary_size"] > 0, (
        f"Manifest vocabulary_size must be a positive int, got {summary['vocabulary_size']!r}"
    )

    special_tokens = summary["special_tokens"]
    assert special_tokens["<PAD>"] == PAD_ID, f"PAD_ID mismatch: {special_tokens['<PAD>']} != {PAD_ID}"
    assert special_tokens["<UNK>"] == UNK_ID, f"UNK_ID mismatch: {special_tokens['<UNK>']} != {UNK_ID}"

    assert summary["max_seq_length"] == config.max_seq_length, (
        f"Manifest max_seq_length={summary['max_seq_length']} != config.max_seq_length={config.max_seq_length}"
    )

    manifest_num_classes = len(label_to_id)
    assert manifest_num_classes == NUM_CLASSES, (
        f"Label mapping has {manifest_num_classes} classes but constants.NUM_CLASSES={NUM_CLASSES}"
    )

    assert summary["tokenizer"] == "whitespace split", (
        f"Expected tokenizer='whitespace split', got {summary['tokenizer']!r}"
    )

    cleaning_policy = summary["text_cleaning_policy"]
    assert cleaning_policy["lowercase"] is True, "Expected lowercase=true in manifest"

    assert OOS_LABEL_NAME in label_to_id, f"OOS label {OOS_LABEL_NAME!r} missing from label_to_id"
    assert label_to_id[OOS_LABEL_NAME] == OOS_LABEL_ID, (
        f"OOS label id mismatch: {label_to_id[OOS_LABEL_NAME]} != {OOS_LABEL_ID}"
    )

    assert "timestamp" in summary, "Preprocessing manifest missing timestamp (freshness indicator)"


def load_neural_data(
    config: TextCNNConfig | BiLSTMConfig,
) -> tuple[dict[str, DataLoader], NeuralMetadata]:
    """Load neural data from frozen Step 3 artifacts — never rebuilds vocab or TF-IDF.

    Returns ``(loaders, metadata)`` where metadata contains artifact_refs,
    label_names, preprocessing policy, OOV/truncation stats, and manifest info.
    """
    vocab_path = ARTIFACTS_DIR / "vocab.json"
    label_to_id_path = ARTIFACTS_DIR / "label_to_id.json"
    id_to_label_path = ARTIFACTS_DIR / "id_to_label.json"
    summary_path = ARTIFACTS_DIR / "preprocessing_summary.json"

    vocab_data: dict[str, Any] = json.loads(vocab_path.read_text())
    vocab: Vocabulary = Vocabulary.from_dict(vocab_data)
    label_to_id: dict[str, int] = json.loads(label_to_id_path.read_text())
    id_to_label: dict[int, str] = {int(k): v for k, v in json.loads(id_to_label_path.read_text()).items()}
    summary: dict[str, Any] = json.loads(summary_path.read_text())

    _validate_preprocessing_manifest(summary, config, label_to_id)

    num_classes: int = len(label_to_id)
    label_names: list[str] = [id_to_label[i] for i in range(num_classes)]

    logger.info("Inherited text preprocessing policy:")
    logger.info("  lowercase: yes")
    logger.info("  punctuation: kept")
    logger.info("  numbers: kept")
    logger.info("  whitespace: normalized")

    dataset = CLINCDataset.load(DATASET_CONFIG)
    assert dataset.label_names == label_names, "CLINCDataset label names don't match saved label mapping"

    oov_stats: dict[str, dict[str, int | float]] = {}
    truncation_stats: dict[str, dict[str, int | float]] = {}
    datasets_dict: dict[str, IntentDataset] = {}

    for split in _SPLIT_NAMES:
        raw = dataset[split]
        texts = [clean_text(t) for t in raw["text"]]
        tokenized = [tokenize_text(t) for t in texts]

        oov_stats[split] = compute_oov_stats(tokenized, vocab)

        numericalized = [numericalize(toks, vocab) for toks in tokenized]
        lengths = [len(seq) for seq in numericalized]
        truncation_stats[split] = compute_truncation_stats(lengths, config.max_seq_length)

        padded = [pad_or_truncate(seq, config.max_seq_length) for seq in numericalized]
        seq_tensor = torch.tensor(padded, dtype=torch.long)
        label_list: list[int] = raw["intent"]
        label_tensor = torch.tensor(label_list, dtype=torch.long)

        assert (label_tensor >= 0).all() and (label_tensor < num_classes).all(), (
            f"Label ids out of range [0, {num_classes}) in {split}"
        )

        datasets_dict[split] = IntentDataset(seq_tensor, label_tensor)

    for split, stats in oov_stats.items():
        logger.info(
            "OOV %s: %d/%d (%.4f%%)",
            split,
            stats["unknown_tokens"],
            stats["total_tokens"],
            stats["oov_rate"] * 100,
        )
    for split, stats in truncation_stats.items():
        logger.info(
            "Truncation %s: %d/%d (%.4f%%)",
            split,
            stats["truncated"],
            stats["total_sequences"],
            stats["truncation_rate"] * 100,
        )

    loaders = create_dataloaders(
        datasets_dict,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=False,
        random_seed=config.dataloader_seed,
    )

    ensure_dir(CHECKPOINTS_DIR)

    for split in _SPLIT_NAMES:
        assert len(loaders[split]) > 0, f"DataLoader for {split} has zero batches"

    logger.info(
        "DataLoader settings: train=shuffle(seed=%d), val/test=no shuffle, num_workers=0, pin_memory=False",
        config.dataloader_seed,
    )

    for split in _SPLIT_NAMES:
        batch_x, batch_y = next(iter(loaders[split]))
        logger.info(
            "First batch %s: input shape=%s dtype=%s | label shape=%s dtype=%s",
            split,
            list(batch_x.shape),
            batch_x.dtype,
            list(batch_y.shape),
            batch_y.dtype,
        )

    artifact_refs: dict[str, str] = {
        "vocab": str(vocab_path.relative_to(PROJECT_ROOT)),
        "label_to_id": str(label_to_id_path.relative_to(PROJECT_ROOT)),
        "id_to_label": str(id_to_label_path.relative_to(PROJECT_ROOT)),
        "preprocessing_summary": str(summary_path.relative_to(PROJECT_ROOT)),
    }

    preprocessing_policy: dict[str, bool | str] = {
        "lowercase": True,
        "punctuation": "kept",
        "numbers": "kept",
        "whitespace": "normalized",
    }

    metadata: NeuralMetadata = {
        "artifact_refs": artifact_refs,
        "label_names": label_names,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "num_classes": num_classes,
        "vocab_size": summary["vocabulary_size"],
        "oov_stats": oov_stats,
        "truncation_stats": truncation_stats,
        "preprocessing_policy": preprocessing_policy,
        "manifest_timestamp": summary.get("timestamp"),
    }

    return loaders, metadata


def build_text_cnn(config: TextCNNConfig, num_classes: int) -> TextCNN:
    """Construct a TextCNN model from config, with dimension assertions."""
    assert config.vocab_size > 0, (
        f"vocab_size must be positive (got {config.vocab_size}); "
        "set it from NeuralMetadata['vocab_size'] before building the model"
    )
    assert num_classes == NUM_CLASSES, f"num_classes={num_classes} doesn't match NUM_CLASSES={NUM_CLASSES}"
    return TextCNN(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        num_filters=config.num_filters,
        kernel_sizes=config.kernel_sizes,
        num_classes=num_classes,
        dropout_rate=config.dropout_rate,
        activation=config.activation,
        max_seq_length=config.max_seq_length,
        trainable_embeddings=config.trainable_embeddings,
    )


def _text_cnn_run_name(config: TextCNNConfig) -> str:
    """Build a deterministic, machine-readable run name for a Text CNN config."""
    ks = "x".join(str(k) for k in config.kernel_sizes)
    lr_str = f"lr{config.learning_rate}".replace(".", "")
    wd_str = f"wd{config.weight_decay}".replace(".", "")
    return (
        f"emb{config.embedding_dim}_nf{config.num_filters}_ks{ks}"
        f"_d{config.dropout_rate}_{lr_str}_{wd_str}_s{config.random_seed}"
    )


def _one_batch_smoke_test(
    model: TextCNN,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TextCNNConfig,
    num_classes: int,
) -> None:
    """One-batch forward/backward smoke test before full training.

    Verifies input shape, embedding output, pre-conv transpose, logits shape,
    loss computation, backward pass, and optimizer step.
    """
    batch_x, batch_y = next(iter(loader))
    assert batch_x.dtype == torch.long, f"Expected integer input, got {batch_x.dtype}"
    assert batch_x.shape[1] == config.max_seq_length, f"seq_len mismatch: {batch_x.shape[1]} != {config.max_seq_length}"
    batch_size = batch_x.shape[0]

    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    model.train()

    embedded = model.embedding(batch_x)
    assert embedded.shape == (batch_size, config.max_seq_length, config.embedding_dim), (
        f"Embedding shape {embedded.shape} != expected ({batch_size}, {config.max_seq_length}, {config.embedding_dim})"
    )

    transposed = embedded.transpose(1, 2)
    assert transposed.shape == (batch_size, config.embedding_dim, config.max_seq_length), (
        f"Pre-conv shape {transposed.shape} != expected ({batch_size}, {config.embedding_dim}, {config.max_seq_length})"
    )

    optimizer.zero_grad()
    logits = model(batch_x)
    assert logits.shape == (batch_size, num_classes), (
        f"Logits shape {logits.shape} != expected ({batch_size}, {num_classes})"
    )

    loss = criterion(logits, batch_y)
    assert not torch.isnan(loss), "NaN loss in smoke test"
    loss.backward()
    optimizer.step()

    logger.info("One-batch smoke test PASSED")


def train_text_cnn(
    config: TextCNNConfig,
    loaders: dict[str, DataLoader],
    num_classes: int,
    metadata: NeuralMetadata,
    checkpoint_dir: Path | str | None = None,
    log_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Train a single Text CNN run. Never iterates the test loader.

    Includes a one-batch smoke test before the full training loop.

    Args:
        checkpoint_dir: Directory for saving checkpoints. Defaults to CHECKPOINTS_DIR.
        log_dir: Directory for saving training logs. Defaults to LOGS_DIR.
    """
    effective_checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else CHECKPOINTS_DIR
    effective_log_dir = Path(log_dir) if log_dir is not None else LOGS_DIR

    device: torch.device = get_device()
    set_seed(config.random_seed)

    model: TextCNN = build_text_cnn(config, num_classes)
    model.to(device)

    total_params: int = sum(p.numel() for p in model.parameters())
    trainable_params: int = count_parameters(model)
    run_name: str = _text_cnn_run_name(config)

    optimizer: torch.optim.Adam = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    class_weights: torch.Tensor | None = None
    if config.use_class_weights:
        train_ds = loaders["train"].dataset
        assert hasattr(train_ds, "_labels"), "Expected IntentDataset with _labels attribute"
        class_weights = compute_class_weights(train_ds._labels, num_classes, device)

    criterion: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(weight=class_weights)

    scheduler: torch.optim.lr_scheduler.StepLR | None = None
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    print(f"\n{'=' * 60}")
    print(f"  Text CNN Run: {run_name}")
    print(f"{'=' * 60}")
    print(f"  Preprocessing refs: {metadata['artifact_refs']}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Max seq length: {config.max_seq_length}")
    print(f"  Num classes: {num_classes}")
    print("  Architecture: TextCNN (Kim-style)")
    print(f"  Embedding dim: {config.embedding_dim} | trainable={config.trainable_embeddings}")
    print(f"  Kernel sizes: {config.kernel_sizes}")
    print(f"  Num filters: {config.num_filters}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Optimizer: {config.optimizer} | lr={config.learning_rate} | wd={config.weight_decay}")
    print(f"  Dropout: {config.dropout_rate}")
    print(f"  Activation: {config.activation}")
    print(f"  Monitor metric: {config.monitor_metric}")
    print("  val_macro_f1 is the primary early-stopping / model-selection metric")

    _one_batch_smoke_test(model, loaders["train"], criterion, optimizer, device, config, num_classes)
    set_seed(config.random_seed)
    model = build_text_cnn(config, num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    trainer = Trainer(model, optimizer, criterion, device, scheduler=scheduler)

    try:
        fit_result = trainer.fit(
            train_loader=loaders["train"],
            val_loader=loaders["validation"],
            max_epochs=config.max_epochs,
            patience=config.early_stopping_patience,
            checkpoint_dir=effective_checkpoint_dir,
            run_name=run_name,
            config=config.to_dict(),
            artifact_refs=metadata["artifact_refs"],
            log_dir=effective_log_dir,
            monitor_metric=config.monitor_metric,
            model_prefix=ModelID.TEXT_CNN,
        )
    except Exception as e:
        logger.error("Text CNN run %s FAILED: %s", run_name, e)
        return {
            "run_name": run_name,
            "config": config.to_dict(),
            "failure_reason": str(e),
            "epochs_completed": 0,
            "checkpoint_path": None,
        }

    for record in fit_result["epoch_history"]:
        print(
            f"  Epoch {record['epoch']:3d} | "
            f"train_loss={record['train_loss']:.4f} | "
            f"val_loss={record['val_loss']:.4f} | "
            f"val_accuracy={record['val_accuracy']:.4f} | "
            f"val_macro_f1={record['val_macro_f1']:.4f}"
        )

    print(f"  Stopped: {fit_result['reason_for_stopping']} at epoch {fit_result['stopping_epoch']}")
    print(f"  Best epoch: {fit_result['best_epoch']} (val metric={fit_result['best_val_metric']:.4f})")
    print(f"  Checkpoint: {fit_result['checkpoint_path']}")
    print(f"  Training time: {fit_result['training_duration']:.1f}s")

    fit_result["total_parameters"] = total_params
    fit_result["trainable_parameters"] = trainable_params
    fit_result["num_classes"] = num_classes
    fit_result["device"] = str(device)
    fit_result["training_seed"] = config.random_seed
    fit_result["dataloader_seed"] = config.dataloader_seed
    fit_result["python_version"] = platform.python_version()
    fit_result["pytorch_version"] = torch.__version__

    return fit_result


def verify_text_cnn_checkpoint(
    checkpoint_path: str,
    config: TextCNNConfig,
    val_loader: DataLoader,
    num_classes: int,
    expected_best_metric: float,
) -> None:
    """Fresh-process checkpoint reload verification.

    Loads a saved checkpoint, reconstructs the model, evaluates on the
    validation set, and asserts the metric matches the saved best value.
    """
    device: torch.device = get_device()

    model: TextCNN = build_text_cnn(config, num_classes)
    meta: dict[str, Any] = load_checkpoint(checkpoint_path, model, device)
    model.to(device)
    model.eval()

    assert "artifact_refs" in meta, "Checkpoint missing artifact_refs"
    assert "config" in meta, "Checkpoint missing config"

    criterion: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    trainer: Trainer = Trainer(model, torch.optim.Adam(model.parameters()), criterion, device)
    val_metrics: dict[str, Any] = trainer.evaluate(val_loader)

    reloaded_f1: float = val_metrics["macro_f1"]
    tolerance: float = 1e-3
    assert abs(reloaded_f1 - expected_best_metric) < tolerance, (
        f"Reloaded val macro_f1={reloaded_f1:.6f} differs from saved best={expected_best_metric:.6f} "
        f"(tolerance={tolerance})"
    )
    logger.info(
        "Checkpoint reload PASSED: val macro_f1=%.6f (saved=%.6f)",
        reloaded_f1,
        expected_best_metric,
    )


def run_text_cnn_default(
    config: TextCNNConfig | None = None,
) -> dict[str, Any]:
    """Run one default Text CNN training end-to-end and verify checkpoint reload.

    Returns the full run result dict including metadata.
    """
    loading_config = config if config is not None else TextCNNConfig()
    loaders: dict[str, DataLoader]
    metadata: NeuralMetadata
    loaders, metadata = load_neural_data(loading_config)
    num_classes: int = metadata["num_classes"]
    if config is None or config.vocab_size == 0:
        config = TextCNNConfig(vocab_size=metadata["vocab_size"])

    print("\n=== Default Text CNN Run ===")
    result: dict[str, Any] = train_text_cnn(config, loaders, num_classes, metadata)

    if result.get("failure_reason"):
        raise RuntimeError(f"Default Text CNN run failed: {result['failure_reason']}")

    print("\n=== Fresh-Process Checkpoint Reload Test ===")
    verify_text_cnn_checkpoint(
        result["checkpoint_path"],
        config,
        loaders["validation"],
        num_classes,
        result["best_val_metric"],
    )
    print("  Reload test PASSED")

    result["metadata"] = metadata
    return result


# ======================================================================
# Text CNN — staged hyperparameter tuning
# ======================================================================

_CNN_TUNING_STAGE_1: dict[str, list[float]] = {
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "dropout_rate": [0.2, 0.3, 0.5],
}

_CNN_TUNING_STAGE_2: dict[str, list[int]] = {
    "embedding_dim": [64, 128, 256],
    "num_filters": [64, 100, 128],
}

_CNN_TUNING_STAGE_3_KERNELS: list[tuple[int, ...]] = [
    (3, 4, 5),
    (2, 3, 4),
    (3, 4, 5, 6),
]

_CNN_TUNING_STAGE_3_WD: list[float] = [0.0, 1e-4]


def _build_cnn_tuning_grid() -> list[list[TextCNNConfig]]:
    """Build a staged tuning grid for the Text CNN.

    Returns a list of stages, each a list of configs to evaluate.
    Stage 1: lr x dropout (9 runs)
    Stage 2: embedding_dim x num_filters (9 runs, using best lr/dropout from stage 1)
    Stage 3: kernel combos x weight_decay (6 runs, using best settings from stages 1-2)
    """
    stage_1: list[TextCNNConfig] = [
        TextCNNConfig(learning_rate=lr, dropout_rate=dr)
        for lr in _CNN_TUNING_STAGE_1["learning_rate"]
        for dr in _CNN_TUNING_STAGE_1["dropout_rate"]
    ]
    return [stage_1]


def run_text_cnn_tuning(
    loaders: dict[str, DataLoader],
    num_classes: int,
    metadata: NeuralMetadata,
) -> pd.DataFrame:
    """Run staged Text CNN hyperparameter tuning. Returns results as a DataFrame."""
    ensure_dir(REPORTS_DIR)
    all_results: list[dict[str, Any]] = []
    vocab_size: int = metadata["vocab_size"]

    # --- Stage 1: lr x dropout ---
    stage_1_configs: list[TextCNNConfig] = [
        TextCNNConfig(vocab_size=vocab_size, learning_rate=lr, dropout_rate=dr)
        for lr in _CNN_TUNING_STAGE_1["learning_rate"]
        for dr in _CNN_TUNING_STAGE_1["dropout_rate"]
    ]
    print(f"\n=== Text CNN Tuning Stage 1: lr x dropout ({len(stage_1_configs)} runs) ===")
    stage_1_results: list[dict[str, Any]] = []
    for i, cfg in enumerate(stage_1_configs, 1):
        print(f"\n--- Stage 1 run {i}/{len(stage_1_configs)} ---")
        result = train_text_cnn(cfg, loaders, num_classes, metadata)
        row = _tuning_row(result)
        stage_1_results.append(row)
        all_results.append(row)

    best_s1 = _select_best_from_rows(stage_1_results)
    best_lr = best_s1["learning_rate"]
    best_dr = best_s1["dropout_rate"]
    print(f"\n  Stage 1 best: lr={best_lr}, dropout={best_dr}")

    # --- Stage 2: embedding_dim x num_filters ---
    stage_2_configs: list[TextCNNConfig] = [
        TextCNNConfig(
            vocab_size=vocab_size,
            learning_rate=best_lr,
            dropout_rate=best_dr,
            embedding_dim=ed,
            num_filters=nf,
        )
        for ed in _CNN_TUNING_STAGE_2["embedding_dim"]
        for nf in _CNN_TUNING_STAGE_2["num_filters"]
    ]
    print(f"\n=== Text CNN Tuning Stage 2: emb_dim x num_filters ({len(stage_2_configs)} runs) ===")
    stage_2_results: list[dict[str, Any]] = []
    for i, cfg in enumerate(stage_2_configs, 1):
        print(f"\n--- Stage 2 run {i}/{len(stage_2_configs)} ---")
        result = train_text_cnn(cfg, loaders, num_classes, metadata)
        row = _tuning_row(result)
        stage_2_results.append(row)
        all_results.append(row)

    best_s2 = _select_best_from_rows(stage_2_results)
    best_ed = best_s2["embedding_dim"]
    best_nf = best_s2["num_filters"]
    print(f"\n  Stage 2 best: emb_dim={best_ed}, num_filters={best_nf}")

    # --- Stage 3: kernel combos x weight_decay ---
    stage_3_configs: list[TextCNNConfig] = [
        TextCNNConfig(
            vocab_size=vocab_size,
            learning_rate=best_lr,
            dropout_rate=best_dr,
            embedding_dim=best_ed,
            num_filters=best_nf,
            kernel_sizes=ks,
            weight_decay=wd,
        )
        for ks in _CNN_TUNING_STAGE_3_KERNELS
        for wd in _CNN_TUNING_STAGE_3_WD
    ]
    already_run: set[str] = {r["run_name"] for r in all_results}
    stage_3_new = [c for c in stage_3_configs if _text_cnn_run_name(c) not in already_run]
    if len(stage_3_new) < len(stage_3_configs):
        print(f"  Skipping {len(stage_3_configs) - len(stage_3_new)} stage-3 configs already run in earlier stages")
    print(f"\n=== Text CNN Tuning Stage 3: kernels x wd ({len(stage_3_new)} new runs) ===")
    for i, cfg in enumerate(stage_3_new, 1):
        print(f"\n--- Stage 3 run {i}/{len(stage_3_new)} ---")
        result = train_text_cnn(cfg, loaders, num_classes, metadata)
        all_results.append(_tuning_row(result))

    df = pd.DataFrame(all_results)
    df.to_csv(REPORTS_DIR / "text_cnn_tuning_results.csv", index=False)
    return df


def _tuning_row(result: dict[str, Any]) -> dict[str, Any]:
    """Extract a tuning results row from a training run result."""
    cfg = result.get("config", {})
    return {
        "run_name": result.get("run_name", "unknown"),
        "best_epoch": result.get("best_epoch"),
        "best_val_metric": result.get("best_val_metric"),
        "best_val_loss": _get_best_val_loss(result),
        "training_duration": result.get("training_duration"),
        "checkpoint_path": result.get("checkpoint_path"),
        "reason_for_stopping": result.get("reason_for_stopping"),
        "failure_reason": result.get("failure_reason"),
        "epochs_completed": result.get("stopping_epoch", result.get("epochs_completed", 0)),
        "total_parameters": result.get("total_parameters"),
        "trainable_parameters": result.get("trainable_parameters"),
        "embedding_dim": cfg.get("embedding_dim"),
        "num_filters": cfg.get("num_filters"),
        "kernel_sizes": str(cfg.get("kernel_sizes")),
        "dropout_rate": cfg.get("dropout_rate"),
        "learning_rate": cfg.get("learning_rate"),
        "weight_decay": cfg.get("weight_decay"),
    }


def _get_best_val_loss(result: dict[str, Any]) -> float | None:
    """Extract the validation loss at the best epoch."""
    history = result.get("epoch_history", [])
    best_epoch = result.get("best_epoch")
    if not history or best_epoch is None:
        return None
    for record in history:
        if record["epoch"] == best_epoch:
            return record.get("val_loss")
    return None


def _select_best_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the best tuning row using val macro F1 with tie-break policy.

    Tie-break (within 1e-4): lower val loss -> fewer trainable params ->
    faster training -> lexicographic run_name.
    """
    valid = [r for r in rows if r.get("best_val_metric") is not None]
    assert valid, "No valid tuning results to select from"

    def sort_key(r: dict[str, Any]) -> tuple[float, float, int, float, str]:
        metric = r["best_val_metric"]
        val_loss = r.get("best_val_loss") or float("inf")
        params = r.get("trainable_parameters") or 0
        duration = r.get("training_duration") or float("inf")
        name = r.get("run_name", "")
        return (-metric, val_loss, params, duration, name)

    valid.sort(key=sort_key)
    return valid[0]


def _get_test_texts(config: TextCNNConfig) -> list[str]:
    """Load and clean test-split texts for error analysis."""
    from src.preprocessing import clean_text as _clean

    dataset = CLINCDataset.load(DATASET_CONFIG)
    raw = dataset["test"]
    return [_clean(t) for t in raw["text"]]


def run_text_cnn_experiment() -> dict[str, Any]:
    """Full Text CNN workflow: load data, default run, tuning, select best, final test eval.

    Final-model selection rule: use the single best validation-selected tuning run
    directly; do NOT retrain.
    Tie-break: (1) lower val loss, (2) fewer trainable params, (3) faster training,
    (4) lexicographic run_name.
    """
    from src.evaluate import evaluate_text_cnn

    initial_config = TextCNNConfig()
    loaders, metadata = load_neural_data(initial_config)
    num_classes: int = metadata["num_classes"]
    vocab_size: int = metadata["vocab_size"]
    default_config = TextCNNConfig(vocab_size=vocab_size)

    # --- Phase 1 verification: default run + checkpoint reload ---
    print("\n=== Default Text CNN Run ===")
    default_result = train_text_cnn(default_config, loaders, num_classes, metadata)
    if default_result.get("failure_reason"):
        raise RuntimeError(f"Default Text CNN run failed: {default_result['failure_reason']}")

    print("\n=== Fresh-Process Checkpoint Reload Test ===")
    verify_text_cnn_checkpoint(
        default_result["checkpoint_path"],
        default_config,
        loaders["validation"],
        num_classes,
        default_result["best_val_metric"],
    )
    print("  Reload test PASSED")

    # --- Phase 2: staged tuning ---
    print("\n=== Text CNN Hyperparameter Tuning ===")
    tuning_df = run_text_cnn_tuning(loaders, num_classes, metadata)

    # --- Select best config ---
    valid_mask = tuning_df["best_val_metric"].notna()
    valid_df = tuning_df[valid_mask].copy()
    assert len(valid_df) > 0, "No valid tuning runs completed"

    valid_df = valid_df.sort_values(
        by=["best_val_metric", "best_val_loss", "trainable_parameters", "training_duration", "run_name"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)
    best_row = valid_df.iloc[0]
    best_checkpoint: str = best_row["checkpoint_path"]
    best_run_name: str = best_row["run_name"]

    best_config = TextCNNConfig(
        vocab_size=vocab_size,
        embedding_dim=int(best_row["embedding_dim"]),
        num_filters=int(best_row["num_filters"]),
        kernel_sizes=tuple(int(x) for x in best_row["kernel_sizes"].strip("[]()").split(",")),
        dropout_rate=float(best_row["dropout_rate"]),
        learning_rate=float(best_row["learning_rate"]),
        weight_decay=float(best_row["weight_decay"]),
    )

    print(f"\n=== Best config: {best_run_name} (val_macro_f1={best_row['best_val_metric']:.4f}) ===")

    # --- Test-eval ordering guard: model selection must be finished before test eval ---
    assert best_checkpoint is not None, "Model selection not complete before test evaluation"

    # --- Final test evaluation ---
    print("=== Final Test Evaluation ===")
    test_texts = _get_test_texts(best_config)

    best_training_result = {
        "run_name": best_run_name,
        "training_duration": (
            float(best_row["training_duration"]) if pd.notna(best_row["training_duration"]) else None
        ),
        "total_parameters": (int(best_row["total_parameters"]) if pd.notna(best_row.get("total_parameters")) else None),
        "trainable_parameters": (
            int(best_row["trainable_parameters"]) if pd.notna(best_row.get("trainable_parameters")) else None
        ),
        "best_epoch": int(best_row["best_epoch"]) if pd.notna(best_row["best_epoch"]) else None,
        "best_val_metric": float(best_row["best_val_metric"]),
        "checkpoint_path": best_checkpoint,
        "config": best_config.to_dict(),
        "device": str(get_device()),
        "training_seed": best_config.random_seed,
        "dataloader_seed": best_config.dataloader_seed,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }

    # Find the epoch history from the training log for this run
    log_path = LOGS_DIR / f"text_cnn_training_log_{best_run_name}.json"
    if log_path.exists():
        best_training_result["epoch_history"] = json.loads(log_path.read_text())

    test_results = evaluate_text_cnn(
        best_checkpoint,
        best_config,
        best_training_result,
        metadata,
        loaders["test"],
        test_texts,
    )

    print(f"\n{'=' * 60}")
    print("  Final Text CNN Test Results")
    print(f"{'=' * 60}")
    print(f"  Test accuracy:    {test_results['test_accuracy']:.4f}")
    print(f"  Test macro F1:    {test_results['test_macro_f1']:.4f}")
    print(f"  Test precision:   {test_results['test_precision']:.4f}")
    print(f"  Test recall:      {test_results['test_recall']:.4f}")
    print(f"  OOS precision:    {test_results['oos_precision']:.4f}")
    print(f"  OOS recall:       {test_results['oos_recall']:.4f}")
    print(f"  OOS F1:           {test_results['oos_f1']:.4f}")
    print(f"  Inference:        {test_results['inference_latency']['avg_ms_per_example']:.2f} ms/example")

    return {
        "default_result": default_result,
        "tuning_df": tuning_df,
        "best_config": best_config,
        "best_run_name": best_run_name,
        "test_results": test_results,
    }


# ======================================================================
# BiLSTM — training, tuning, and experiment pipeline
# ======================================================================

_BILSTM_TUNING_STAGE_1: dict[str, list[float]] = {
    "learning_rate": [5e-4, 1e-3, 1e-4],
    "dropout_rate": [0.3, 0.5],
}

_BILSTM_TUNING_STAGE_2: dict[str, list[int]] = {
    "hidden_dim": [128, 256],
    "embedding_dim": [64, 128, 256],
}

_BILSTM_TUNING_STAGE_3: list[dict[str, int | float]] = [
    {"num_layers": 1, "weight_decay": 0.0},
    {"num_layers": 2, "weight_decay": 0.0},
    {"num_layers": 1, "weight_decay": 1e-4},
]


def build_bilstm(config: BiLSTMConfig, num_classes: int) -> BiLSTMClassifier:
    """Construct a BiLSTMClassifier from config, with dimension assertions."""
    assert config.vocab_size > 0, (
        f"vocab_size must be positive (got {config.vocab_size}); "
        "set it from NeuralMetadata['vocab_size'] before building the model"
    )
    assert num_classes == NUM_CLASSES, f"num_classes={num_classes} doesn't match NUM_CLASSES={NUM_CLASSES}"
    return BiLSTMClassifier(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_classes=num_classes,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
        bidirectional=config.bidirectional,
        trainable_embeddings=config.trainable_embeddings,
    )


def _bilstm_run_name(config: BiLSTMConfig) -> str:
    """Build a deterministic, machine-readable run name for a BiLSTM config."""
    lr_str = f"lr{config.learning_rate}".replace(".", "")
    wd_str = f"wd{config.weight_decay}".replace(".", "")
    return (
        f"emb{config.embedding_dim}_h{config.hidden_dim}_L{config.num_layers}"
        f"_d{config.dropout_rate}_{lr_str}_{wd_str}_s{config.random_seed}"
    )


def _one_batch_bilstm_smoke_test(
    model: BiLSTMClassifier,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: BiLSTMConfig,
    num_classes: int,
) -> None:
    """One-batch forward/backward smoke test verifying all BiLSTM shapes."""
    batch_x, batch_y = next(iter(loader))
    assert batch_x.dtype == torch.long, f"Expected integer input, got {batch_x.dtype}"
    assert batch_x.shape[1] == config.max_seq_length, f"seq_len mismatch: {batch_x.shape[1]} != {config.max_seq_length}"
    batch_size: int = batch_x.shape[0]

    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    model.train()

    embedded: torch.Tensor = model.embedding(batch_x)
    assert embedded.shape == (batch_size, config.max_seq_length, config.embedding_dim), (
        f"Embedding shape {embedded.shape} != expected ({batch_size}, {config.max_seq_length}, {config.embedding_dim})"
    )

    num_directions: int = 2 if config.bidirectional else 1
    lstm_output: torch.Tensor
    h_n: torch.Tensor
    lstm_output, (h_n, _c_n) = model.lstm(embedded)
    assert lstm_output.shape == (batch_size, config.max_seq_length, config.hidden_dim * num_directions), (
        f"LSTM output shape {lstm_output.shape} != expected "
        f"({batch_size}, {config.max_seq_length}, {config.hidden_dim * num_directions})"
    )
    assert h_n.shape == (config.num_layers * num_directions, batch_size, config.hidden_dim), (
        f"h_n shape {h_n.shape} != expected ({config.num_layers * num_directions}, {batch_size}, {config.hidden_dim})"
    )

    summarized: torch.Tensor
    if config.bidirectional:
        summarized = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=-1)
    else:
        summarized = h_n[-1, :, :]
    expected_classifier_dim: int = config.hidden_dim * num_directions
    assert summarized.shape == (batch_size, expected_classifier_dim), (
        f"Summarization shape {summarized.shape} != expected ({batch_size}, {expected_classifier_dim})"
    )

    optimizer.zero_grad()
    logits: torch.Tensor = model(batch_x)
    assert logits.shape == (batch_size, num_classes), (
        f"Logits shape {logits.shape} != expected ({batch_size}, {num_classes})"
    )

    loss: torch.Tensor = criterion(logits, batch_y)
    assert not torch.isnan(loss), "NaN loss in smoke test"
    loss.backward()

    if config.gradient_clipping:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

    optimizer.step()

    print("  Smoke test batch shapes:")
    print(f"    input:         {list(batch_x.shape)} dtype={batch_x.dtype}")
    print(f"    embedding:     {list(embedded.shape)}")
    print(f"    lstm_output:   {list(lstm_output.shape)}")
    print(f"    h_n:           {list(h_n.shape)}")
    print(f"    summarized:    {list(summarized.shape)}")
    print(f"    logits:        {list(logits.shape)}")
    logger.info("One-batch BiLSTM smoke test PASSED")


def _bilstm_tuning_row(result: dict[str, Any]) -> dict[str, Any]:
    """Extract a BiLSTM-specific tuning results row from a training run result."""
    cfg = result.get("config", {})
    return {
        "run_name": result.get("run_name", "unknown"),
        "best_epoch": result.get("best_epoch"),
        "best_val_metric": result.get("best_val_metric"),
        "best_val_loss": _get_best_val_loss(result),
        "training_duration": result.get("training_duration"),
        "checkpoint_path": result.get("checkpoint_path"),
        "reason_for_stopping": result.get("reason_for_stopping"),
        "failure_reason": result.get("failure_reason"),
        "epochs_completed": result.get("stopping_epoch", result.get("epochs_completed", 0)),
        "total_parameters": result.get("total_parameters"),
        "trainable_parameters": result.get("trainable_parameters"),
        "hidden_dim": cfg.get("hidden_dim"),
        "num_layers": cfg.get("num_layers"),
        "bidirectional": cfg.get("bidirectional"),
        "embedding_dim": cfg.get("embedding_dim"),
        "max_grad_norm": cfg.get("max_grad_norm"),
        "summarization_mode": cfg.get("summarization_mode"),
        "dropout_rate": cfg.get("dropout_rate"),
        "learning_rate": cfg.get("learning_rate"),
        "weight_decay": cfg.get("weight_decay"),
    }


def train_bilstm(
    config: BiLSTMConfig,
    loaders: dict[str, DataLoader],
    num_classes: int,
    metadata: NeuralMetadata,
    checkpoint_dir: Path | str | None = None,
    log_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Train a single BiLSTM run. Never iterates the test loader.

    Includes a one-batch smoke test before the full training loop.

    Args:
        checkpoint_dir: Directory for saving checkpoints. Defaults to CHECKPOINTS_DIR.
        log_dir: Directory for saving training logs. Defaults to LOGS_DIR.
    """
    effective_checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else CHECKPOINTS_DIR
    effective_log_dir = Path(log_dir) if log_dir is not None else LOGS_DIR

    device: torch.device = get_device()
    set_seed(config.random_seed)

    model: BiLSTMClassifier = build_bilstm(config, num_classes)
    model.to(device)

    total_params: int = sum(p.numel() for p in model.parameters())
    trainable_params: int = count_parameters(model)
    run_name: str = _bilstm_run_name(config)

    optimizer: torch.optim.Adam = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    class_weights: torch.Tensor | None = None
    if config.use_class_weights:
        train_ds = loaders["train"].dataset
        assert hasattr(train_ds, "_labels"), "Expected IntentDataset with _labels attribute"
        class_weights = compute_class_weights(train_ds._labels, num_classes, device)

    criterion: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(weight=class_weights)

    scheduler: torch.optim.lr_scheduler.StepLR | None = None
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    num_directions: int = 2 if config.bidirectional else 1
    lstm_dropout_note: str = (
        "no-op when num_layers=1; classifier-path dropout active"
        if config.num_layers == 1
        else f"LSTM inter-layer dropout={config.dropout_rate}; classifier-path dropout also active"
    )

    print(f"\n{'=' * 60}")
    print(f"  BiLSTM Run: {run_name}")
    print(f"{'=' * 60}")
    print(f"  Preprocessing refs: {metadata['artifact_refs']}")
    print(f"  Preprocessing manifest: {metadata['artifact_refs'].get('preprocessing_summary', 'N/A')}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Max seq length: {config.max_seq_length}")
    print(f"  Num classes: {num_classes}")
    print("  Architecture: BiLSTM sentence classifier")
    print("  batch_first: True")
    print(f"  Embedding dim: {config.embedding_dim} | trainable={config.trainable_embeddings}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num layers: {config.num_layers} | bidirectional={config.bidirectional}")
    print(f"  Summarization: {config.summarization_mode} (h_n[-2,:,:] || h_n[-1,:,:])")
    clip_status: str = f"enabled (max_norm={config.max_grad_norm})" if config.gradient_clipping else "disabled"
    print(f"  Gradient clipping: {clip_status}")
    print(f"  LSTM dropout: {lstm_dropout_note}")
    print(f"  Classifier input dim: {config.hidden_dim * num_directions}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Optimizer: {config.optimizer} | lr={config.learning_rate} | wd={config.weight_decay}")
    print(f"  Dropout: {config.dropout_rate}")
    print(f"  Monitor metric: {config.monitor_metric}")
    print("  val_macro_f1 is the primary early-stopping / model-selection metric")
    print(f"  Class weights: {'enabled' if config.use_class_weights else 'disabled (default)'}")
    print(f"  DataLoader: train=shuffle(seed={config.dataloader_seed}), val/test=no shuffle")
    print("  DataLoader: drop_last=False, num_workers=0, pin_memory=False")

    _one_batch_bilstm_smoke_test(model, loaders["train"], criterion, optimizer, device, config, num_classes)

    set_seed(config.random_seed)
    model = build_bilstm(config, num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    max_grad_norm: float | None = config.max_grad_norm if config.gradient_clipping else None
    trainer: Trainer = Trainer(model, optimizer, criterion, device, scheduler=scheduler, max_grad_norm=max_grad_norm)

    try:
        fit_result = trainer.fit(
            train_loader=loaders["train"],
            val_loader=loaders["validation"],
            max_epochs=config.max_epochs,
            patience=config.early_stopping_patience,
            checkpoint_dir=effective_checkpoint_dir,
            run_name=run_name,
            config=config.to_dict(),
            artifact_refs=metadata["artifact_refs"],
            log_dir=effective_log_dir,
            monitor_metric=config.monitor_metric,
            model_prefix=ModelID.BILSTM,
        )
    except Exception as e:
        logger.error("BiLSTM run %s FAILED: %s", run_name, e)
        return {
            "run_name": run_name,
            "config": config.to_dict(),
            "failure_reason": str(e),
            "epochs_completed": 0,
            "checkpoint_path": None,
        }

    for record in fit_result["epoch_history"]:
        print(
            f"  Epoch {record['epoch']:3d} | "
            f"train_loss={record['train_loss']:.4f} | "
            f"val_loss={record['val_loss']:.4f} | "
            f"val_accuracy={record['val_accuracy']:.4f} | "
            f"val_macro_f1={record['val_macro_f1']:.4f}"
        )

    print(f"  Stopped: {fit_result['reason_for_stopping']} at epoch {fit_result['stopping_epoch']}")
    print(f"  Best epoch: {fit_result['best_epoch']} (val metric={fit_result['best_val_metric']:.4f})")
    print(f"  Checkpoint: {fit_result['checkpoint_path']}")
    print(f"  Training time: {fit_result['training_duration']:.1f}s")

    fit_result["total_parameters"] = total_params
    fit_result["trainable_parameters"] = trainable_params
    fit_result["num_classes"] = num_classes
    fit_result["device"] = str(device)
    fit_result["training_seed"] = config.random_seed
    fit_result["dataloader_seed"] = config.dataloader_seed
    fit_result["python_version"] = platform.python_version()
    fit_result["pytorch_version"] = torch.__version__

    return fit_result


def verify_bilstm_checkpoint(
    checkpoint_path: str,
    config: BiLSTMConfig,
    val_loader: DataLoader,
    num_classes: int,
    expected_best_metric: float,
) -> None:
    """Fresh-process checkpoint reload verification for BiLSTM.

    Loads a saved checkpoint, reconstructs the model, evaluates on the
    validation set, and asserts the metric matches the saved best value.
    """
    device: torch.device = get_device()

    model: BiLSTMClassifier = build_bilstm(config, num_classes)
    meta: dict[str, Any] = load_checkpoint(checkpoint_path, model, device)
    model.to(device)
    model.eval()

    assert "artifact_refs" in meta, "Checkpoint missing artifact_refs"
    assert "config" in meta, "Checkpoint missing config"

    criterion: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    trainer: Trainer = Trainer(model, torch.optim.Adam(model.parameters()), criterion, device)
    val_metrics: dict[str, Any] = trainer.evaluate(val_loader)

    reloaded_f1: float = val_metrics["macro_f1"]
    tolerance: float = 1e-3
    assert abs(reloaded_f1 - expected_best_metric) < tolerance, (
        f"Reloaded val macro_f1={reloaded_f1:.6f} differs from saved best={expected_best_metric:.6f} "
        f"(tolerance={tolerance})"
    )
    logger.info(
        "BiLSTM checkpoint reload PASSED: val macro_f1=%.6f (saved=%.6f)",
        reloaded_f1,
        expected_best_metric,
    )


def run_bilstm_default(
    config: BiLSTMConfig | None = None,
) -> dict[str, Any]:
    """Run one default BiLSTM training end-to-end and verify checkpoint reload.

    Returns the full run result dict including metadata.
    """
    loading_config: BiLSTMConfig = config if config is not None else BiLSTMConfig()
    loaders: dict[str, DataLoader]
    metadata: NeuralMetadata
    loaders, metadata = load_neural_data(loading_config)
    num_classes: int = metadata["num_classes"]
    if config is None or config.vocab_size == 0:
        config = BiLSTMConfig(vocab_size=metadata["vocab_size"])

    print("\n=== Default BiLSTM Run ===")
    result: dict[str, Any] = train_bilstm(config, loaders, num_classes, metadata)

    if result.get("failure_reason"):
        raise RuntimeError(f"Default BiLSTM run failed: {result['failure_reason']}")

    print("\n=== Fresh-Process Checkpoint Reload Test ===")
    verify_bilstm_checkpoint(
        result["checkpoint_path"],
        config,
        loaders["validation"],
        num_classes,
        result["best_val_metric"],
    )
    print("  Reload test PASSED")

    result["metadata"] = metadata
    return result


def run_bilstm_tuning(
    loaders: dict[str, DataLoader],
    num_classes: int,
    metadata: NeuralMetadata,
) -> pd.DataFrame:
    """Run staged BiLSTM hyperparameter tuning. Returns results as a DataFrame."""
    ensure_dir(REPORTS_DIR)
    all_results: list[dict[str, Any]] = []
    already_run: set[str] = set()
    vocab_size: int = metadata["vocab_size"]

    def _run_stage(configs: list[BiLSTMConfig], stage_name: str) -> list[dict[str, Any]]:
        new_configs: list[BiLSTMConfig] = [c for c in configs if _bilstm_run_name(c) not in already_run]
        skipped: int = len(configs) - len(new_configs)
        if skipped > 0:
            print(f"  Skipping {skipped} {stage_name} configs already run in earlier stages")
        print(f"\n=== BiLSTM Tuning {stage_name} ({len(new_configs)} new runs) ===")
        stage_results: list[dict[str, Any]] = []
        for i, cfg in enumerate(new_configs, 1):
            print(f"\n--- {stage_name} run {i}/{len(new_configs)} ---")
            result: dict[str, Any] = train_bilstm(cfg, loaders, num_classes, metadata)
            row: dict[str, Any] = _bilstm_tuning_row(result)
            stage_results.append(row)
            all_results.append(row)
            already_run.add(row["run_name"])
        return stage_results

    # --- Stage 1: lr x dropout ---
    stage_1_configs: list[BiLSTMConfig] = [
        BiLSTMConfig(vocab_size=vocab_size, learning_rate=lr, dropout_rate=dr)
        for lr in _BILSTM_TUNING_STAGE_1["learning_rate"]
        for dr in _BILSTM_TUNING_STAGE_1["dropout_rate"]
    ]
    stage_1_results: list[dict[str, Any]] = _run_stage(stage_1_configs, "Stage 1: lr x dropout")
    best_s1: dict[str, Any] = _select_best_from_rows(stage_1_results)
    best_lr: float = best_s1["learning_rate"]
    best_dr: float = best_s1["dropout_rate"]
    print(f"\n  Stage 1 best: lr={best_lr}, dropout={best_dr}")

    # --- Stage 2: hidden_dim x embedding_dim ---
    stage_2_configs: list[BiLSTMConfig] = [
        BiLSTMConfig(
            vocab_size=vocab_size,
            learning_rate=best_lr,
            dropout_rate=best_dr,
            hidden_dim=hd,
            embedding_dim=ed,
        )
        for hd in _BILSTM_TUNING_STAGE_2["hidden_dim"]
        for ed in _BILSTM_TUNING_STAGE_2["embedding_dim"]
    ]
    stage_2_results: list[dict[str, Any]] = _run_stage(stage_2_configs, "Stage 2: hidden_dim x emb_dim")
    best_s2: dict[str, Any] = _select_best_from_rows(stage_2_results)
    best_hd: int = best_s2["hidden_dim"]
    best_ed: int = best_s2["embedding_dim"]
    print(f"\n  Stage 2 best: hidden_dim={best_hd}, emb_dim={best_ed}")

    # --- Stage 3: num_layers x weight_decay ---
    stage_3_configs: list[BiLSTMConfig] = [
        BiLSTMConfig(
            vocab_size=vocab_size,
            learning_rate=best_lr,
            dropout_rate=best_dr,
            hidden_dim=best_hd,
            embedding_dim=best_ed,
            num_layers=overrides["num_layers"],
            weight_decay=overrides["weight_decay"],
        )
        for overrides in _BILSTM_TUNING_STAGE_3
    ]
    _run_stage(stage_3_configs, "Stage 3: num_layers x wd")

    df = pd.DataFrame(all_results)
    df.to_csv(REPORTS_DIR / "bilstm_tuning_results.csv", index=False)
    return df


def _get_bilstm_test_texts() -> list[str]:
    """Load and clean test-split texts for BiLSTM error analysis."""
    dataset = CLINCDataset.load(DATASET_CONFIG)
    raw = dataset["test"]
    return [clean_text(t) for t in raw["text"]]


def run_bilstm_experiment() -> dict[str, Any]:
    """Full BiLSTM workflow: load data, default run, tuning, select best, final test eval.

    Final-model selection rule: use the single best validation-selected tuning run
    directly; do NOT retrain.
    Tie-break: (1) lower val loss, (2) fewer trainable params, (3) faster training,
    (4) lexicographic run_name.
    """
    from src.evaluate import evaluate_bilstm

    initial_config = BiLSTMConfig()
    loaders, metadata = load_neural_data(initial_config)
    num_classes: int = metadata["num_classes"]
    vocab_size: int = metadata["vocab_size"]
    default_config = BiLSTMConfig(vocab_size=vocab_size)

    # --- Phase 1 verification: default run + checkpoint reload ---
    print("\n=== Default BiLSTM Run ===")
    default_result = train_bilstm(default_config, loaders, num_classes, metadata)
    if default_result.get("failure_reason"):
        raise RuntimeError(f"Default BiLSTM run failed: {default_result['failure_reason']}")

    print("\n=== Fresh-Process Checkpoint Reload Test ===")
    verify_bilstm_checkpoint(
        default_result["checkpoint_path"],
        default_config,
        loaders["validation"],
        num_classes,
        default_result["best_val_metric"],
    )
    print("  Reload test PASSED")

    # --- Phase 2: staged tuning ---
    print("\n=== BiLSTM Hyperparameter Tuning ===")
    tuning_df = run_bilstm_tuning(loaders, num_classes, metadata)

    # --- Select best config ---
    valid_mask = tuning_df["best_val_metric"].notna()
    valid_df = tuning_df[valid_mask].copy()
    assert len(valid_df) > 0, "No valid tuning runs completed"

    valid_df = valid_df.sort_values(
        by=["best_val_metric", "best_val_loss", "trainable_parameters", "training_duration", "run_name"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)
    best_row = valid_df.iloc[0]
    best_checkpoint: str = best_row["checkpoint_path"]
    best_run_name: str = best_row["run_name"]

    best_config = BiLSTMConfig(
        vocab_size=vocab_size,
        embedding_dim=int(best_row["embedding_dim"]),
        hidden_dim=int(best_row["hidden_dim"]),
        num_layers=int(best_row["num_layers"]),
        bidirectional=bool(best_row["bidirectional"]),
        dropout_rate=float(best_row["dropout_rate"]),
        learning_rate=float(best_row["learning_rate"]),
        weight_decay=float(best_row["weight_decay"]),
        max_grad_norm=float(best_row["max_grad_norm"]),
    )

    print(f"\n=== Best config: {best_run_name} (val_macro_f1={best_row['best_val_metric']:.4f}) ===")

    assert best_checkpoint is not None, "Model selection not complete before test evaluation"

    # --- Final test evaluation ---
    print("=== Final Test Evaluation ===")
    test_texts = _get_bilstm_test_texts()

    best_training_result: dict[str, Any] = {
        "run_name": best_run_name,
        "training_duration": (
            float(best_row["training_duration"]) if pd.notna(best_row["training_duration"]) else None
        ),
        "total_parameters": int(best_row["total_parameters"]) if pd.notna(best_row.get("total_parameters")) else None,
        "trainable_parameters": (
            int(best_row["trainable_parameters"]) if pd.notna(best_row.get("trainable_parameters")) else None
        ),
        "best_epoch": int(best_row["best_epoch"]) if pd.notna(best_row["best_epoch"]) else None,
        "best_val_metric": float(best_row["best_val_metric"]),
        "checkpoint_path": best_checkpoint,
        "config": best_config.to_dict(),
        "device": str(get_device()),
        "training_seed": best_config.random_seed,
        "dataloader_seed": best_config.dataloader_seed,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }

    log_path = LOGS_DIR / f"bilstm_training_log_{best_run_name}.json"
    if log_path.exists():
        best_training_result["epoch_history"] = json.loads(log_path.read_text())

    test_results = evaluate_bilstm(
        best_checkpoint,
        best_config,
        best_training_result,
        metadata,
        loaders["test"],
        test_texts,
    )

    print(f"\n{'=' * 60}")
    print("  Final BiLSTM Test Results")
    print(f"{'=' * 60}")
    print(f"  Test accuracy:    {test_results['test_accuracy']:.4f}")
    print(f"  Test macro F1:    {test_results['test_macro_f1']:.4f}")
    print(f"  Test precision:   {test_results['test_precision']:.4f}")
    print(f"  Test recall:      {test_results['test_recall']:.4f}")
    print(f"  OOS precision:    {test_results['oos_precision']:.4f}")
    print(f"  OOS recall:       {test_results['oos_recall']:.4f}")
    print(f"  OOS F1:           {test_results['oos_f1']:.4f}")
    print(f"  Inference:        {test_results['inference_latency']['avg_ms_per_example']:.2f} ms/example")

    return {
        "default_result": default_result,
        "tuning_df": tuning_df,
        "best_config": best_config,
        "best_run_name": best_run_name,
        "test_results": test_results,
    }
