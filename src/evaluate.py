"""Final model evaluation, metric computation, and confusion matrix generation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import DataLoader

from src.config import DATASET_CONFIG, MLPBaselineConfig, get_device
from src.constants import ARTIFACTS_DIR, NUM_CLASSES, OOS_LABEL_ID, REPORTS_DIR
from src.dataset import CLINCDataset, TFIDFDataset
from src.metrics import (
    build_confusion_matrix,
    compute_classification_metrics,
    compute_oos_metrics,
    find_top_confusions,
    find_top_errors,
)
from src.models.mlp import MLPClassifier
from src.preprocessing import PreprocessingArtifacts, clean_text, load_preprocessing_artifacts
from src.utils import count_parameters, ensure_dir, load_checkpoint


def evaluate_mlp_baseline(
    checkpoint_path: str | Path,
    config: MLPBaselineConfig,
    training_duration: float | None = None,
) -> dict[str, Any]:
    """Evaluate an MLP baseline from a saved checkpoint (standalone, no re-training).

    Loads the vectorizer, dataset, and label mapping from disk artifacts.
    """
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    device = get_device()

    artifacts: PreprocessingArtifacts = load_preprocessing_artifacts(ARTIFACTS_DIR)
    vectorizer: TfidfVectorizer = artifacts["vectorizer"]

    id_to_label: dict[int, str] = {
        int(k): v for k, v in json.loads((ARTIFACTS_DIR / "id_to_label.json").read_text()).items()
    }
    num_classes = len(id_to_label)
    label_names = [id_to_label[i] for i in range(num_classes)]

    dataset: CLINCDataset = CLINCDataset.load(DATASET_CONFIG)
    test_raw: dict[str, list[str]] = dataset["test"]
    test_texts: list[str] = [clean_text(t) for t in test_raw["text"]]
    test_labels: list[int] = test_raw["intent"]

    tfidf_test: spmatrix = vectorizer.transform(test_texts)

    input_dim: int = tfidf_test.shape[1]

    label_tensor: torch.Tensor = torch.tensor(test_labels, dtype=torch.long)
    test_ds = TFIDFDataset(tfidf_test, label_tensor)
    test_loader: DataLoader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    model: MLPClassifier = MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        second_hidden_dim=config.second_hidden_dim,
        dropout_rate=config.dropout_rate,
        activation=config.activation,
    )
    meta = load_checkpoint(checkpoint_path, model, device)
    model.to(device)
    model.eval()

    all_preds: list[int] = []
    all_targets: list[int] = []

    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.tolist())
    inference_time = time.time() - start_time

    n_examples = len(all_targets)
    avg_ms_per_example = (inference_time / max(n_examples, 1)) * 1000
    examples_per_sec = n_examples / max(inference_time, 1e-9)

    cls_metrics = compute_classification_metrics(all_targets, all_preds)
    oos_metrics = compute_oos_metrics(all_targets, all_preds, OOS_LABEL_ID)

    confusion_df = build_confusion_matrix(all_targets, all_preds, label_names)
    top_confusions = find_top_confusions(confusion_df)
    top_errors = find_top_errors(all_targets, all_preds, test_texts, label_names)

    param_count = count_parameters(model)

    results: dict[str, Any] = {
        "test_accuracy": cls_metrics["accuracy"],
        "test_macro_f1": cls_metrics["macro_f1"],
        "test_precision": cls_metrics["macro_precision"],
        "test_recall": cls_metrics["macro_recall"],
        "oos_precision": oos_metrics["oos_precision"],
        "oos_recall": oos_metrics["oos_recall"],
        "oos_f1": oos_metrics["oos_f1"],
        "inference_latency": {
            "total_seconds": inference_time,
            "avg_ms_per_example": avg_ms_per_example,
            "examples_per_sec": examples_per_sec,
        },
        "parameter_count": param_count,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": meta["epoch"],
        "checkpoint_best_metric": meta["best_metric"],
        "confusion_matrix": confusion_df,
        "top_confusions": top_confusions,
        "top_errors": top_errors,
        "label_names": label_names,
        "config": config.to_dict(),
        "predictions": all_preds,
        "targets": all_targets,
        "input_dim": input_dim,
        "training_duration": training_duration,
    }

    save_test_artifacts(results, REPORTS_DIR)

    return results


def save_test_artifacts(results: dict[str, Any], reports_dir: Path | str) -> None:
    """Save all test evaluation artifacts to disk."""
    reports_dir = ensure_dir(reports_dir)

    test_metrics = {
        "test_accuracy": results["test_accuracy"],
        "test_macro_f1": results["test_macro_f1"],
        "test_precision": results["test_precision"],
        "test_recall": results["test_recall"],
        "oos_precision": results["oos_precision"],
        "oos_recall": results["oos_recall"],
        "oos_f1": results["oos_f1"],
        "inference_latency": results["inference_latency"],
        "parameter_count": results["parameter_count"],
        "checkpoint_path": results["checkpoint_path"],
    }
    (reports_dir / "mlp_test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    results["confusion_matrix"].to_csv(reports_dir / "mlp_confusion_matrix.csv")

    (reports_dir / "mlp_top_confusions.json").write_text(json.dumps(results["top_confusions"], indent=2))

    top_errors_artifact = {
        "selection_rule": (
            "All misclassified examples sorted by (true_label, predicted_label, text); first top_k returned."
        ),
        "errors": results["top_errors"],
    }
    (reports_dir / "mlp_top_errors.json").write_text(json.dumps(top_errors_artifact, indent=2))

    config = results.get("config", {})
    run_name = _run_name_from_config(config)

    comparison_row = {
        "model_name": "mlp_baseline",
        "input_type": "tfidf",
        "primary_val_metric": config.get("monitor_metric", "val_macro_f1"),
        "best_val_macro_f1": results.get("checkpoint_best_metric"),
        "test_accuracy": results["test_accuracy"],
        "test_macro_f1": results["test_macro_f1"],
        "test_precision": results["test_precision"],
        "test_recall": results["test_recall"],
        "oos_precision": results["oos_precision"],
        "oos_recall": results["oos_recall"],
        "oos_f1": results["oos_f1"],
        "training_time": results.get("training_duration"),
        "inference_latency": results["inference_latency"],
        "parameter_count": results["parameter_count"],
        "checkpoint_path": results["checkpoint_path"],
        "preprocessing_artifact_refs": str(ARTIFACTS_DIR),
        "notes": "TF-IDF + MLP baseline; best tuning run used directly (no retraining).",
    }
    (reports_dir / "mlp_comparison_row.json").write_text(json.dumps(comparison_row, indent=2))

    run_summary = _build_run_summary(results, run_name)
    (reports_dir / f"mlp_run_summary_{run_name}.json").write_text(json.dumps(run_summary, indent=2))


def _run_name_from_config(config: dict[str, Any]) -> str:
    hd = config.get("hidden_dim", 512)
    dr = config.get("dropout_rate", 0.3)
    lr = config.get("learning_rate", 1e-3)
    wd = config.get("weight_decay", 0.0)
    shd = config.get("second_hidden_dim")
    seed = config.get("random_seed", 42)
    layers = "1layer" if shd is None else f"2layer_h2{shd}"
    lr_str = f"lr{lr}".replace(".", "")
    wd_str = f"wd{wd}".replace(".", "")
    return f"h{hd}_d{dr}_{lr_str}_{wd_str}_{layers}_s{seed}"


def _build_run_summary(results: dict[str, Any], run_name: str) -> dict[str, Any]:
    config = results.get("config", {})
    return {
        "run_name": run_name,
        "config_snapshot": config,
        "preprocessing_artifact_refs": str(ARTIFACTS_DIR),
        "tfidf_input_dim": results.get("input_dim"),
        "num_classes": NUM_CLASSES,
        "model_architecture": "MLPClassifier",
        "parameter_count": results["parameter_count"],
        "optimizer_settings": {
            "type": config.get("optimizer", "adam"),
            "lr": config.get("learning_rate"),
            "weight_decay": config.get("weight_decay"),
        },
        "dropout_rate": config.get("dropout_rate"),
        "weight_decay": config.get("weight_decay"),
        "max_epochs": config.get("max_epochs"),
        "early_stopping_patience": config.get("early_stopping_patience"),
        "best_epoch": results.get("checkpoint_epoch"),
        "best_val_metric": results.get("checkpoint_best_metric"),
        "primary_model_selection_metric": config.get("monitor_metric", "val_macro_f1"),
        "training_duration": results.get("training_duration"),
        "checkpoint_path": results["checkpoint_path"],
        "oos_strategy": config.get("oos_strategy", "explicit_class"),
        "class_weight_policy": "enabled" if config.get("use_class_weights") else "disabled (default)",
        "tfidf_input_format": "sparse-origin, dense float32 per-row in TFIDFDataset.__getitem__",
        "inference_latency": results["inference_latency"],
        "lr_scheduling_used": config.get("use_lr_scheduler", False),
        "shuffle_policy": "train=shuffled (seeded Generator), val/test=no shuffle",
        "preprocessing_refs_for_comparison": str(ARTIFACTS_DIR),
        "label_name_ordering": results.get("label_names"),
    }
