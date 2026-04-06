"""Model training entry logic for launching experiments."""

from __future__ import annotations

import json
import logging
from collections import Counter
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import DataLoader

from src.config import DATASET_CONFIG, MLPBaselineConfig, get_device
from src.constants import (
    ARTIFACTS_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    NUM_CLASSES,
    REPORTS_DIR,
)
from src.dataset import CLINCDataset, TFIDFDataset, create_dataloaders
from src.models.mlp import MLPClassifier
from src.preprocessing import PreprocessingArtifacts, clean_text, load_preprocessing_artifacts
from src.trainers.trainer import Trainer
from src.utils import count_parameters, ensure_dir, set_seed

logger = logging.getLogger(__name__)

_SPLIT_NAMES = ("train", "validation", "test")

_TUNING_SEARCH_SPACE: dict[str, list[Any]] = {
    "hidden_dim": [256, 512, 1024],
    "dropout_rate": [0.2, 0.3, 0.5],
    "learning_rate": [1e-3, 5e-4],
    "weight_decay": [0.0, 1e-4],
}

_TWO_LAYER_CONFIGS: list[dict[str, Any]] = [
    {"hidden_dim": 512, "second_hidden_dim": 256, "dropout_rate": 0.3, "learning_rate": 1e-3, "weight_decay": 0.0},
    {"hidden_dim": 512, "second_hidden_dim": 256, "dropout_rate": 0.3, "learning_rate": 5e-4, "weight_decay": 1e-4},
    {"hidden_dim": 1024, "second_hidden_dim": 256, "dropout_rate": 0.2, "learning_rate": 1e-3, "weight_decay": 0.0},
    {"hidden_dim": 1024, "second_hidden_dim": 256, "dropout_rate": 0.2, "learning_rate": 5e-4, "weight_decay": 1e-4},
]


def load_tfidf_data(
    config: MLPBaselineConfig,
) -> tuple[dict[str, DataLoader], int, int, dict[str, Any]]:
    """Load TF-IDF features from saved artifacts and build DataLoaders.

    Returns (loaders, input_dim, num_classes, metadata).
    """
    artifacts: PreprocessingArtifacts = load_preprocessing_artifacts(ARTIFACTS_DIR)
    vectorizer: TfidfVectorizer = artifacts["vectorizer"]
    summary: dict[str, Any] = artifacts["summary"]

    label_to_id_path = ARTIFACTS_DIR / "label_to_id.json"
    id_to_label_path = ARTIFACTS_DIR / "id_to_label.json"
    label_to_id: dict[str, int] = json.loads(label_to_id_path.read_text())
    id_to_label: dict[str, str] = {int(k): v for k, v in json.loads(id_to_label_path.read_text()).items()}

    num_classes = len(label_to_id)
    assert num_classes == NUM_CLASSES, (
        f"Label mapping has {num_classes} classes but constants.NUM_CLASSES={NUM_CLASSES}"
    )
    expected_ids = set(range(num_classes))
    actual_ids = set(id_to_label.keys())
    assert actual_ids == expected_ids, f"id_to_label keys must span [0, {num_classes}), got {sorted(actual_ids)[:5]}..."

    label_names = [id_to_label[i] for i in range(num_classes)]

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

    input_dim = tfidf_matrices["train"].shape[1]
    expected_dim = summary.get("tfidf_fitted_feature_dim")
    if expected_dim is not None:
        assert input_dim == expected_dim, (
            f"TF-IDF input_dim={input_dim} doesn't match preprocessing_summary ({expected_dim})"
        )

    n_train = tfidf_matrices["train"].shape[0]
    dense_bytes = n_train * input_dim * 4
    dense_mb = dense_bytes / (1024 * 1024)
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

    artifact_refs = {
        "tfidf_vectorizer": str(ARTIFACTS_DIR / "tfidf_vectorizer.pkl"),
        "label_to_id": str(label_to_id_path),
        "id_to_label": str(id_to_label_path),
        "preprocessing_summary": str(ARTIFACTS_DIR / "preprocessing_summary.json"),
    }

    metadata = {
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
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Train a single MLP baseline run. Never iterates the test loader."""
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
        checkpoint_dir=CHECKPOINTS_DIR,
        run_name=run_name,
        config=config.to_dict(),
        artifact_refs=metadata["artifact_refs"],
        log_dir=LOGS_DIR,
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
    metadata: dict[str, Any],
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


def run_mlp_experiment() -> dict[str, Any]:
    """Full workflow: load data, default run, tuning, select best, final test eval."""
    from src.evaluate import evaluate_mlp_baseline

    default_config = MLPBaselineConfig()
    loaders, input_dim, num_classes, metadata = load_tfidf_data(default_config)

    print("\n=== Default Baseline Run ===")
    default_result = train_mlp_baseline(default_config, loaders, input_dim, num_classes, metadata)

    tuning_grid = _build_tuning_grid()
    print(f"\n=== Hyperparameter Tuning ({len(tuning_grid)} configs) ===")
    tuning_df = run_mlp_tuning(tuning_grid, loaders, input_dim, num_classes, metadata)

    best_idx = int(tuning_df["best_val_metric"].idxmax())
    best_row = tuning_df.iloc[best_idx]
    best_checkpoint = best_row["checkpoint_path"]
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

    best_training_duration = float(best_row["training_duration"])

    print(f"\n=== Best config: {best_row['run_name']} (val_metric={best_row['best_val_metric']:.4f}) ===")
    print("=== Final Test Evaluation ===")

    test_results = evaluate_mlp_baseline(best_checkpoint, best_config, training_duration=best_training_duration)

    return {
        "default_result": default_result,
        "tuning_df": tuning_df,
        "best_config": best_config,
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
