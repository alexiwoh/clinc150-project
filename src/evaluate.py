"""Final model evaluation, metric computation, and confusion matrix generation."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader

from src.config import DATASET_CONFIG, BiLSTMConfig, MLPBaselineConfig, TextCNNConfig, get_device
from src.constants import ARTIFACTS_DIR, NUM_CLASSES, OOS_LABEL_ID, REPORTS_DIR
from src.dataset import CLINCDataset, TFIDFDataset
from src.metrics import (
    build_confusion_matrix,
    compute_classification_metrics,
    compute_oos_metrics,
    find_top_confusions,
    find_top_errors,
)
from src.models.bilstm import BiLSTMClassifier
from src.models.mlp import MLPClassifier
from src.models.text_cnn import TextCNN
from src.preprocessing import PreprocessingArtifacts, clean_text, load_preprocessing_artifacts
from src.train import NeuralMetadata, build_bilstm, build_text_cnn
from src.utils import count_parameters, ensure_dir, load_checkpoint


def evaluate_mlp_baseline(
    checkpoint_path: str | Path,
    config: MLPBaselineConfig,
    training_result: dict[str, Any] | None = None,
    training_duration: float | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate an MLP baseline from a saved checkpoint (standalone, no re-training).

    Loads the vectorizer, dataset, and label mapping from disk artifacts.

    Args:
        training_result: Optional training context dict with run_name, epoch_history,
            device info, etc. When provided, enables saving epoch_history.csv and
            richer run summaries (matching Text CNN / BiLSTM evaluate patterns).
        training_duration: Legacy parameter kept for backward compat; ignored when
            training_result is provided.
        output_dir: Directory for saving artifacts. Defaults to REPORTS_DIR.
    """
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    reports_dir: Path | str = output_dir if output_dir is not None else REPORTS_DIR
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
    all_logits: list[np.ndarray] = []

    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu().numpy())
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

    logits_array = np.concatenate(all_logits, axis=0)
    probs_array = _softmax(logits_array)

    per_class_report = classification_report(
        all_targets,
        all_preds,
        labels=list(range(num_classes)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    total_params: int = sum(p.numel() for p in model.parameters())
    trainable_params: int = count_parameters(model)

    effective_duration = training_result.get("training_duration") if training_result else training_duration

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
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
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
        "per_class_report": per_class_report,
        "probabilities": probs_array,
        "input_dim": input_dim,
        "training_duration": effective_duration,
        "training_result": training_result or {},
        "id_to_label": id_to_label,
        "test_texts": test_texts,
    }

    save_test_artifacts(results, reports_dir)

    return results


def save_test_artifacts(results: dict[str, Any], reports_dir: Path | str) -> None:
    """Save all MLP test evaluation artifacts to disk (12 files, matching Text CNN / BiLSTM parity)."""
    import pandas as pd

    reports_dir = ensure_dir(reports_dir)
    config = results.get("config", {})
    label_names: list[str] = results["label_names"]
    training_result: dict[str, Any] = results.get("training_result", {})

    # 1. Test metrics (schema matches Text CNN / BiLSTM)
    test_metrics = {
        "test_accuracy": results["test_accuracy"],
        "test_macro_f1": results["test_macro_f1"],
        "test_precision": results["test_precision"],
        "test_recall": results["test_recall"],
        "oos_precision": results["oos_precision"],
        "oos_recall": results["oos_recall"],
        "oos_f1": results["oos_f1"],
        "inference_latency": results["inference_latency"],
        "total_parameters": results["total_parameters"],
        "trainable_parameters": results["trainable_parameters"],
        "checkpoint_path": results["checkpoint_path"],
    }
    (reports_dir / "mlp_test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    # 2. Confusion matrix
    results["confusion_matrix"].to_csv(reports_dir / "mlp_confusion_matrix.csv")

    # 3. Top confusions
    (reports_dir / "mlp_top_confusions.json").write_text(json.dumps(results["top_confusions"], indent=2))

    # 4. Top errors (with label_ordering, matching Text CNN / BiLSTM)
    top_errors_artifact = {
        "selection_rule": (
            "All misclassified examples sorted by (true_label, predicted_label, text); first top_k returned."
        ),
        "label_ordering": label_names,
        "errors": results["top_errors"],
    }
    (reports_dir / "mlp_top_errors.json").write_text(json.dumps(top_errors_artifact, indent=2))

    # 5. Per-class metrics with embedded label ordering
    per_class_out = {
        "label_ordering": label_names,
        "per_class_metrics": {
            name: results["per_class_report"][name] for name in label_names if name in results["per_class_report"]
        },
        "macro_avg": results["per_class_report"].get("macro avg"),
        "weighted_avg": results["per_class_report"].get("weighted avg"),
    }
    (reports_dir / "mlp_per_class_metrics.json").write_text(json.dumps(per_class_out, indent=2))

    # 6. Label order standalone artifact
    (reports_dir / "mlp_label_order.json").write_text(json.dumps(label_names, indent=2))

    # 7. Confidence / probability outputs
    np.savez_compressed(
        reports_dir / "mlp_confidences.npz",
        probabilities=results["probabilities"],
        predictions=np.array(results["predictions"]),
        targets=np.array(results["targets"]),
        label_names=np.array(label_names),
    )

    # 8. Comparison row (canonical model_name = "mlp", matches Text CNN / BiLSTM schema)
    run_name = _run_name_from_config(config)
    comparison_row = {
        "model_name": "mlp",
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
        "parameter_count": results["total_parameters"],
        "trainable_parameter_count": results["trainable_parameters"],
        "checkpoint_path": results["checkpoint_path"],
        "preprocessing_artifact_refs": str(ARTIFACTS_DIR),
        "notes": "TF-IDF + MLP baseline; best tuning run used directly (no retraining).",
    }
    (reports_dir / "mlp_comparison_row.json").write_text(json.dumps(comparison_row, indent=2))

    # 9. Run summary
    run_summary = _build_run_summary(results, run_name)
    (reports_dir / f"mlp_run_summary_{run_name}.json").write_text(json.dumps(run_summary, indent=2))

    # 10. Epoch history CSV
    epoch_history = training_result.get("epoch_history", [])
    if epoch_history:
        pd.DataFrame(epoch_history).to_csv(reports_dir / "mlp_epoch_history.csv", index=False)

    # 11. Final predictions CSV (matching BiLSTM pattern)
    preds = results["predictions"]
    targets = results["targets"]
    texts = results.get("test_texts", [])
    probs = results["probabilities"]
    max_confidences = probs.max(axis=1).tolist()

    id_to_label: dict[int, str] = results.get("id_to_label", {i: label_names[i] for i in range(len(label_names))})
    pred_df = pd.DataFrame(
        {
            "y_true": targets,
            "y_pred": preds,
            "true_label_name": [id_to_label[t] for t in targets],
            "pred_label_name": [id_to_label[p] for p in preds],
            "text": texts if len(texts) == len(targets) else [""] * len(targets),
            "max_confidence": max_confidences,
        }
    )
    pred_df.to_csv(reports_dir / "mlp_final_predictions.csv", index=False)


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
    training_result = results.get("training_result", {})
    return {
        "run_name": run_name,
        "config_snapshot": config,
        "preprocessing_artifact_refs": str(ARTIFACTS_DIR),
        "tfidf_input_dim": results.get("input_dim"),
        "num_classes": NUM_CLASSES,
        "model_architecture": "MLPClassifier",
        "total_parameters": results["total_parameters"],
        "trainable_parameters": results["trainable_parameters"],
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
        "lr_scheduling_policy": {
            "used": config.get("use_lr_scheduler", False),
            "justification": (
                "not used; fixed learning rate; compact search space and early stopping control training length"
            ),
        },
        "shuffle_policy": {
            "train": f"shuffled (seeded Generator, seed={config.get('random_seed', 42)})",
            "validation": "no shuffle",
            "test": "no shuffle",
        },
        "preprocessing_refs_for_comparison": str(ARTIFACTS_DIR),
        "label_name_ordering": results.get("label_names"),
        "device_and_environment": {
            "device": training_result.get("device", str(get_device())),
            "python_version": training_result.get("python_version", platform.python_version()),
            "pytorch_version": training_result.get("pytorch_version", torch.__version__),
        },
        "final_model_selection_rule": "single best validation run used directly; no retraining",
        "confidence_saving_decision": "test-split softmax probabilities saved to mlp_confidences.npz",
        "training_seed": config.get("random_seed", 42),
        "test_accuracy": results["test_accuracy"],
        "test_macro_f1": results["test_macro_f1"],
        "test_precision": results["test_precision"],
        "test_recall": results["test_recall"],
        "oos_precision": results["oos_precision"],
        "oos_recall": results["oos_recall"],
        "oos_f1": results["oos_f1"],
    }


# ======================================================================
# Text CNN evaluation (Step 5)
# ======================================================================


def evaluate_text_cnn(
    checkpoint_path: str | Path,
    config: TextCNNConfig,
    training_result: dict[str, Any],
    metadata: NeuralMetadata,
    test_loader: DataLoader,
    test_texts: list[str],
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate a Text CNN from a saved checkpoint on the test split.

    Loads the best checkpoint (not last epoch state), computes test metrics,
    confusion matrix, per-class metrics, top errors, and saves all artifacts.

    Args:
        output_dir: Directory for saving artifacts. Defaults to REPORTS_DIR.
    """
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    device = get_device()
    num_classes: int = metadata["num_classes"]
    label_names: list[str] = metadata["label_names"]

    model: TextCNN = build_text_cnn(config, num_classes)
    meta = load_checkpoint(checkpoint_path, model, device)
    model.to(device)
    model.eval()

    all_preds: list[int] = []
    all_targets: list[int] = []
    all_logits: list[np.ndarray] = []

    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu().numpy())
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

    logits_array = np.concatenate(all_logits, axis=0)
    probs_array = _softmax(logits_array)

    per_class_report = classification_report(
        all_targets,
        all_preds,
        labels=list(range(num_classes)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    total_params: int = sum(p.numel() for p in model.parameters())
    trainable_params: int = count_parameters(model)

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
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
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
        "per_class_report": per_class_report,
        "probabilities": probs_array,
        "training_result": training_result,
        "metadata": metadata,
        "test_texts": test_texts,
    }

    _save_text_cnn_artifacts(results, output_dir if output_dir is not None else REPORTS_DIR)

    return results


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _save_text_cnn_artifacts(results: dict[str, Any], reports_dir: Path | str) -> None:
    """Save all Text CNN test evaluation artifacts to disk."""
    reports_dir = ensure_dir(reports_dir)
    config = results["config"]
    label_names: list[str] = results["label_names"]
    training_result: dict[str, Any] = results["training_result"]
    metadata: NeuralMetadata = results["metadata"]

    # Test metrics
    test_metrics = {
        "test_accuracy": results["test_accuracy"],
        "test_macro_f1": results["test_macro_f1"],
        "test_precision": results["test_precision"],
        "test_recall": results["test_recall"],
        "oos_precision": results["oos_precision"],
        "oos_recall": results["oos_recall"],
        "oos_f1": results["oos_f1"],
        "inference_latency": results["inference_latency"],
        "total_parameters": results["total_parameters"],
        "trainable_parameters": results["trainable_parameters"],
        "checkpoint_path": results["checkpoint_path"],
    }
    (reports_dir / "text_cnn_test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    # Confusion matrix
    results["confusion_matrix"].to_csv(reports_dir / "text_cnn_confusion_matrix.csv")

    # Top confusions
    (reports_dir / "text_cnn_top_confusions.json").write_text(json.dumps(results["top_confusions"], indent=2))

    # Top errors
    top_errors_artifact = {
        "selection_rule": (
            "All misclassified examples sorted by (true_label, predicted_label, text); first top_k returned."
        ),
        "label_ordering": label_names,
        "errors": results["top_errors"],
    }
    (reports_dir / "text_cnn_top_errors.json").write_text(json.dumps(top_errors_artifact, indent=2))

    # Per-class metrics with embedded label ordering
    per_class_out = {
        "label_ordering": label_names,
        "per_class_metrics": {
            name: results["per_class_report"][name] for name in label_names if name in results["per_class_report"]
        },
        "macro_avg": results["per_class_report"].get("macro avg"),
        "weighted_avg": results["per_class_report"].get("weighted avg"),
    }
    (reports_dir / "text_cnn_per_class_metrics.json").write_text(json.dumps(per_class_out, indent=2))

    # Label order standalone artifact
    (reports_dir / "text_cnn_label_order.json").write_text(json.dumps(label_names, indent=2))

    # Confidence / probability outputs
    np.savez_compressed(
        reports_dir / "text_cnn_confidences.npz",
        probabilities=results["probabilities"],
        predictions=np.array(results["predictions"]),
        targets=np.array(results["targets"]),
        label_names=np.array(label_names),
    )

    # Comparison row
    comparison_row = {
        "model_name": "text_cnn",
        "input_type": "token_sequences",
        "primary_val_metric": config.get("monitor_metric", "val_macro_f1"),
        "best_val_macro_f1": results["checkpoint_best_metric"],
        "test_accuracy": results["test_accuracy"],
        "test_macro_f1": results["test_macro_f1"],
        "test_precision": results["test_precision"],
        "test_recall": results["test_recall"],
        "oos_precision": results["oos_precision"],
        "oos_recall": results["oos_recall"],
        "oos_f1": results["oos_f1"],
        "training_time": training_result.get("training_duration"),
        "inference_latency": results["inference_latency"],
        "parameter_count": results["total_parameters"],
        "trainable_parameter_count": results["trainable_parameters"],
        "checkpoint_path": results["checkpoint_path"],
        "preprocessing_artifact_refs": str(ARTIFACTS_DIR),
        "notes": "Kim-style Text CNN; best tuning run used directly (no retraining).",
    }
    (reports_dir / "text_cnn_comparison_row.json").write_text(json.dumps(comparison_row, indent=2))

    # Run summary (all 36 spec Q fields)
    run_name = training_result.get("run_name", "unknown")
    run_summary = _build_text_cnn_run_summary(results, training_result, metadata, run_name)
    (reports_dir / f"text_cnn_run_summary_{run_name}.json").write_text(json.dumps(run_summary, indent=2))

    # Epoch history CSV
    epoch_history = training_result.get("epoch_history", [])
    if epoch_history:
        import pandas as pd

        pd.DataFrame(epoch_history).to_csv(reports_dir / "text_cnn_epoch_history.csv", index=False)

    # Final predictions CSV (matching BiLSTM pattern)
    import pandas as pd

    preds = results["predictions"]
    targets = results["targets"]
    texts = results.get("test_texts", [])
    probs = results["probabilities"]
    max_confidences = probs.max(axis=1).tolist()

    id_to_label: dict[int, str] = metadata["id_to_label"]
    pred_df = pd.DataFrame(
        {
            "y_true": targets,
            "y_pred": preds,
            "true_label_name": [id_to_label[t] for t in targets],
            "pred_label_name": [id_to_label[p] for p in preds],
            "text": texts if len(texts) == len(targets) else [""] * len(targets),
            "max_confidence": max_confidences,
        }
    )
    pred_df.to_csv(reports_dir / "text_cnn_final_predictions.csv", index=False)


def _build_text_cnn_run_summary(
    results: dict[str, Any],
    training_result: dict[str, Any],
    metadata: NeuralMetadata,
    run_name: str,
) -> dict[str, Any]:
    """Build the full 36-field run summary for spec Q."""
    config = results["config"]
    return {
        # 1-3
        "run_name": run_name,
        "config_snapshot": config,
        "preprocessing_artifact_refs": metadata["artifact_refs"],
        # 4-6
        "vocab_size": config["vocab_size"],
        "max_sequence_length": config["max_seq_length"],
        "number_of_classes": metadata["num_classes"],
        # 7-10
        "model_architecture": "TextCNN (Kim-style sentence CNN)",
        "embedding_policy": {
            "initialization": "random (PyTorch default)",
            "trainable": config["trainable_embeddings"],
            "padding_idx": 0,
            "unk_initialization": "default random init (not zeroed)",
        },
        "kernel_sizes": config["kernel_sizes"],
        "num_filters": config["num_filters"],
        # 11
        "total_parameters": results["total_parameters"],
        "trainable_parameters": results["trainable_parameters"],
        # 12-14
        "optimizer_settings": {
            "type": config.get("optimizer", "adam"),
            "lr": config["learning_rate"],
            "weight_decay": config["weight_decay"],
        },
        "dropout_rate": config["dropout_rate"],
        "weight_decay": config["weight_decay"],
        # 15-16
        "max_epochs": config["max_epochs"],
        "early_stopping_patience": config["early_stopping_patience"],
        # 17-18
        "best_epoch": results["checkpoint_epoch"],
        "best_val_metric": results["checkpoint_best_metric"],
        # 19
        "primary_model_selection_metric": config.get("monitor_metric", "val_macro_f1"),
        # 20-21
        "training_duration": training_result.get("training_duration"),
        "checkpoint_path": results["checkpoint_path"],
        # 22-23
        "oos_strategy": config.get("oos_strategy", "explicit_class"),
        "class_weight_policy": "enabled" if config.get("use_class_weights") else "disabled (default)",
        # 24
        "inference_latency": results["inference_latency"],
        # 25
        "lr_scheduling_policy": {
            "used": config.get("use_lr_scheduler", False),
            "justification": (
                "not used; fixed learning rate; compact search space and early stopping control training length"
            ),
        },
        # 26
        "shuffle_policy": {
            "train": f"shuffled (seeded Generator, seed={config.get('dataloader_seed', 42)})",
            "validation": "no shuffle",
            "test": "no shuffle",
        },
        # 27
        "preprocessing_refs_for_comparison": metadata["artifact_refs"],
        # 28
        "label_name_ordering": results["label_names"],
        # 29
        "preprocessing_manifest_reference": str(ARTIFACTS_DIR / "preprocessing_summary.json"),
        # 30
        "preprocessing_config_hash_or_version": metadata.get("manifest_timestamp"),
        # 31
        "inherited_text_preprocessing_policy": metadata["preprocessing_policy"],
        # 32
        "sequence_truncation_percentages_by_split": metadata["truncation_stats"],
        # 33
        "unk_coverage_statistics_by_split": metadata["oov_stats"],
        # 34
        "embedding_policy_machine_readable": {
            "type": "random",
            "trainable": config["trainable_embeddings"],
            "padding_idx": 0,
            "unk_init": "default_random",
        },
        # 35
        "lr_scheduling_policy_machine_readable": {
            "used": False,
            "type": None,
        },
        # 36
        "device_and_environment": {
            "device": training_result.get("device", str(get_device())),
            "python_version": training_result.get("python_version", platform.python_version()),
            "pytorch_version": training_result.get("pytorch_version", torch.__version__),
        },
        # Extra fields from spec
        "final_model_selection_rule": "single best validation run used directly; no retraining",
        "tie_break_policy": (
            "when val macro F1 within 1e-4: (1) lower val loss, (2) fewer trainable params, "
            "(3) faster training, (4) lexicographic run_name"
        ),
        "confidence_saving_decision": "test-split softmax probabilities saved to text_cnn_confidences.npz",
        "dataloader_generator_seed": config.get("dataloader_seed", 42),
        "training_seed": config.get("random_seed", 42),
        "activation": config.get("activation", "relu"),
        # Test metrics for completeness
        "test_accuracy": results["test_accuracy"],
        "test_macro_f1": results["test_macro_f1"],
        "test_precision": results["test_precision"],
        "test_recall": results["test_recall"],
        "oos_precision": results["oos_precision"],
        "oos_recall": results["oos_recall"],
        "oos_f1": results["oos_f1"],
    }


# ======================================================================
# BiLSTM evaluation (Step 6)
# ======================================================================


def evaluate_bilstm(
    checkpoint_path: str | Path,
    config: BiLSTMConfig,
    training_result: dict[str, Any],
    metadata: NeuralMetadata,
    test_loader: DataLoader,
    test_texts: list[str],
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate a BiLSTM from a saved checkpoint on the test split.

    Loads the best checkpoint (not last epoch state), computes test metrics,
    confusion matrix, per-class metrics, top errors, and saves all artifacts.

    Args:
        output_dir: Directory for saving artifacts. Defaults to REPORTS_DIR.
    """
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    device = get_device()
    num_classes: int = metadata["num_classes"]
    label_names: list[str] = metadata["label_names"]

    model: BiLSTMClassifier = build_bilstm(config, num_classes)
    meta = load_checkpoint(checkpoint_path, model, device)
    model.to(device)
    model.eval()

    all_preds: list[int] = []
    all_targets: list[int] = []
    all_logits: list[np.ndarray] = []

    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu().numpy())
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

    logits_array = np.concatenate(all_logits, axis=0)
    probs_array = _softmax(logits_array)

    per_class_report = classification_report(
        all_targets,
        all_preds,
        labels=list(range(num_classes)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    total_params: int = sum(p.numel() for p in model.parameters())
    trainable_params: int = count_parameters(model)

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
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
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
        "per_class_report": per_class_report,
        "probabilities": probs_array,
        "training_result": training_result,
        "metadata": metadata,
        "test_texts": test_texts,
    }

    _save_bilstm_artifacts(results, output_dir if output_dir is not None else REPORTS_DIR)

    return results


def _save_bilstm_artifacts(results: dict[str, Any], reports_dir: Path | str) -> None:
    """Save all BiLSTM test evaluation artifacts to disk (12 files)."""
    import pandas as pd

    reports_dir = ensure_dir(reports_dir)
    config = results["config"]
    label_names: list[str] = results["label_names"]
    training_result: dict[str, Any] = results["training_result"]
    metadata: NeuralMetadata = results["metadata"]

    # 1. Test metrics
    test_metrics = {
        "test_accuracy": results["test_accuracy"],
        "test_macro_f1": results["test_macro_f1"],
        "test_precision": results["test_precision"],
        "test_recall": results["test_recall"],
        "oos_precision": results["oos_precision"],
        "oos_recall": results["oos_recall"],
        "oos_f1": results["oos_f1"],
        "inference_latency": results["inference_latency"],
        "total_parameters": results["total_parameters"],
        "trainable_parameters": results["trainable_parameters"],
        "checkpoint_path": results["checkpoint_path"],
    }
    (reports_dir / "bilstm_test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    # 2. Confusion matrix
    results["confusion_matrix"].to_csv(reports_dir / "bilstm_confusion_matrix.csv")

    # 3. Top confusions
    (reports_dir / "bilstm_top_confusions.json").write_text(json.dumps(results["top_confusions"], indent=2))

    # 4. Top errors (with label_ordering, matching Text CNN pattern)
    top_errors_artifact = {
        "selection_rule": (
            "All misclassified examples sorted by (true_label, predicted_label, text); first top_k returned."
        ),
        "label_ordering": label_names,
        "errors": results["top_errors"],
    }
    (reports_dir / "bilstm_top_errors.json").write_text(json.dumps(top_errors_artifact, indent=2))

    # 5. Per-class metrics with embedded label ordering
    per_class_out = {
        "label_ordering": label_names,
        "per_class_metrics": {
            name: results["per_class_report"][name] for name in label_names if name in results["per_class_report"]
        },
        "macro_avg": results["per_class_report"].get("macro avg"),
        "weighted_avg": results["per_class_report"].get("weighted avg"),
    }
    (reports_dir / "bilstm_per_class_metrics.json").write_text(json.dumps(per_class_out, indent=2))

    # 6. Label order standalone artifact
    (reports_dir / "bilstm_label_order.json").write_text(json.dumps(label_names, indent=2))

    # 7. Confidence / probability outputs
    np.savez_compressed(
        reports_dir / "bilstm_confidences.npz",
        probabilities=results["probabilities"],
        predictions=np.array(results["predictions"]),
        targets=np.array(results["targets"]),
        label_names=np.array(label_names),
    )

    # 8. Comparison row (validates against both MLP and Text CNN schemas)
    comparison_row = {
        "model_name": "bilstm",
        "input_type": "token_sequences",
        "primary_val_metric": config.get("monitor_metric", "val_macro_f1"),
        "best_val_macro_f1": results["checkpoint_best_metric"],
        "test_accuracy": results["test_accuracy"],
        "test_macro_f1": results["test_macro_f1"],
        "test_precision": results["test_precision"],
        "test_recall": results["test_recall"],
        "oos_precision": results["oos_precision"],
        "oos_recall": results["oos_recall"],
        "oos_f1": results["oos_f1"],
        "training_time": training_result.get("training_duration"),
        "inference_latency": results["inference_latency"],
        "parameter_count": results["total_parameters"],
        "trainable_parameter_count": results["trainable_parameters"],
        "checkpoint_path": results["checkpoint_path"],
        "preprocessing_artifact_refs": str(ARTIFACTS_DIR),
        "notes": "BiLSTM sentence classifier; best tuning run used directly (no retraining).",
    }
    (reports_dir / "bilstm_comparison_row.json").write_text(json.dumps(comparison_row, indent=2))

    # 9. Run summary (43+ fields)
    run_name = training_result.get("run_name", "unknown")
    run_summary = _build_bilstm_run_summary(results, training_result, metadata, run_name)
    (reports_dir / f"bilstm_run_summary_{run_name}.json").write_text(json.dumps(run_summary, indent=2))

    # 10. Epoch history CSV
    epoch_history = training_result.get("epoch_history", [])
    if epoch_history:
        pd.DataFrame(epoch_history).to_csv(reports_dir / "bilstm_epoch_history.csv", index=False)

    # 11. Final predictions CSV (NEW: not in Text CNN template)
    preds = results["predictions"]
    targets = results["targets"]
    texts = results.get("test_texts", [])
    probs = results["probabilities"]
    max_confidences = probs.max(axis=1).tolist()

    id_to_label: dict[int, str] = metadata["id_to_label"]
    pred_df = pd.DataFrame(
        {
            "y_true": targets,
            "y_pred": preds,
            "true_label_name": [id_to_label[t] for t in targets],
            "pred_label_name": [id_to_label[p] for p in preds],
            "text": texts if len(texts) == len(targets) else [""] * len(targets),
            "max_confidence": max_confidences,
        }
    )
    pred_df.to_csv(reports_dir / "bilstm_final_predictions.csv", index=False)


def _build_bilstm_run_summary(
    results: dict[str, Any],
    training_result: dict[str, Any],
    metadata: NeuralMetadata,
    run_name: str,
) -> dict[str, Any]:
    """Build the full 43+ field run summary for BiLSTM (spec R)."""
    config = results["config"]
    num_layers = config.get("num_layers", 1)
    bidirectional = config.get("bidirectional", True)
    dropout_rate = config.get("dropout_rate", 0.3)

    lstm_dropout_note = (
        "no-op when num_layers=1; classifier-path dropout active"
        if num_layers == 1
        else f"LSTM inter-layer dropout={dropout_rate}; classifier-path dropout also active"
    )

    return {
        # Shared fields (matching Text CNN run summary)
        "run_name": run_name,
        "config_snapshot": config,
        "preprocessing_artifact_refs": metadata["artifact_refs"],
        "vocab_size": config["vocab_size"],
        "max_sequence_length": config["max_seq_length"],
        "number_of_classes": metadata["num_classes"],
        "embedding_policy": {
            "initialization": "random (PyTorch default)",
            "trainable": config["trainable_embeddings"],
            "padding_idx": 0,
            "unk_initialization": "default random init (not zeroed)",
        },
        "total_parameters": results["total_parameters"],
        "trainable_parameters": results["trainable_parameters"],
        "optimizer_settings": {
            "type": config.get("optimizer", "adam"),
            "lr": config["learning_rate"],
            "weight_decay": config["weight_decay"],
        },
        "dropout_rate": dropout_rate,
        "weight_decay": config.get("weight_decay", 0.0),
        "max_epochs": config["max_epochs"],
        "early_stopping_patience": config["early_stopping_patience"],
        "best_epoch": results["checkpoint_epoch"],
        "best_val_metric": results["checkpoint_best_metric"],
        "primary_model_selection_metric": config.get("monitor_metric", "val_macro_f1"),
        "training_duration": training_result.get("training_duration"),
        "checkpoint_path": results["checkpoint_path"],
        "oos_strategy": config.get("oos_strategy", "explicit_class"),
        "class_weight_policy": "enabled" if config.get("use_class_weights") else "disabled (default)",
        "inference_latency": results["inference_latency"],
        "lr_scheduling_policy": {
            "used": config.get("use_lr_scheduler", False),
            "justification": (
                "not used; fixed learning rate; compact search space and early stopping control training length"
            ),
        },
        "shuffle_policy": {
            "train": f"shuffled (seeded Generator, seed={config.get('dataloader_seed', 42)})",
            "validation": "no shuffle",
            "test": "no shuffle",
        },
        "preprocessing_refs_for_comparison": metadata["artifact_refs"],
        "label_name_ordering": results["label_names"],
        "preprocessing_manifest_reference": str(ARTIFACTS_DIR / "preprocessing_summary.json"),
        "preprocessing_config_hash_or_version": metadata.get("manifest_timestamp"),
        "inherited_text_preprocessing_policy": metadata["preprocessing_policy"],
        "sequence_truncation_percentages_by_split": metadata["truncation_stats"],
        "unk_coverage_statistics_by_split": metadata["oov_stats"],
        "embedding_policy_machine_readable": {
            "type": "random",
            "trainable": config["trainable_embeddings"],
            "padding_idx": 0,
            "unk_init": "default_random",
        },
        "lr_scheduling_policy_machine_readable": {
            "used": False,
            "type": None,
        },
        "device_and_environment": {
            "device": training_result.get("device", str(get_device())),
            "python_version": training_result.get("python_version", platform.python_version()),
            "pytorch_version": training_result.get("pytorch_version", torch.__version__),
        },
        "final_model_selection_rule": "single best validation run used directly; no retraining",
        "tie_break_policy": (
            "when val macro F1 within 1e-4: (1) lower val loss, (2) fewer trainable params, "
            "(3) faster training, (4) lexicographic run_name"
        ),
        "confidence_saving_decision": "test-split softmax probabilities saved to bilstm_confidences.npz",
        "dataloader_generator_seed": config.get("dataloader_seed", 42),
        "training_seed": config.get("random_seed", 42),
        # BiLSTM-specific fields (spec R additions)
        "model_architecture": "BiLSTM sentence classifier",
        "hidden_dim": config.get("hidden_dim", 128),
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "summarization_policy": "concat_final_hidden: h_n[-2,:,:] || h_n[-1,:,:]",
        "gradient_clipping_policy": {
            "enabled": config.get("gradient_clipping", True),
            "max_grad_norm": config.get("max_grad_norm", 1.0),
        },
        "batch_first_convention": True,
        "lstm_internal_dropout_note": lstm_dropout_note,
        "hidden_state_initialization_policy": "default zeros (PyTorch default)",
        "sequence_handling_policy": "fixed-length padded batches, no packed sequences, padding_idx=0",
        "deterministic_algorithms_policy": (
            "not enforced (torch.use_deterministic_algorithms not called); reproducibility via manual seeding only"
        ),
        "dataloader_reproducibility": {
            "train_shuffle": True,
            "val_shuffle": False,
            "test_shuffle": False,
            "drop_last": False,
            "num_workers": 0,
            "pin_memory": False,
            "generator_seed": config.get("dataloader_seed", 42),
        },
        # Test metrics for completeness
        "test_accuracy": results["test_accuracy"],
        "test_macro_f1": results["test_macro_f1"],
        "test_precision": results["test_precision"],
        "test_recall": results["test_recall"],
        "oos_precision": results["oos_precision"],
        "oos_recall": results["oos_recall"],
        "oos_f1": results["oos_f1"],
    }
