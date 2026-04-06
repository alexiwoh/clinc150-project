"""Accuracy, precision, recall, macro F1, and OOS metric helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Compute overall classification metrics (model-agnostic)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_oos_metrics(y_true: ArrayLike, y_pred: ArrayLike, oos_label_id: int) -> dict[str, float]:
    """Compute OOS-specific precision, recall, F1 (binary: positive = oos_label_id)."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    binary_true = (y_true_arr == oos_label_id).astype(int)
    binary_pred = (y_pred_arr == oos_label_id).astype(int)
    return {
        "oos_precision": float(precision_score(binary_true, binary_pred, zero_division=0)),
        "oos_recall": float(recall_score(binary_true, binary_pred, zero_division=0)),
        "oos_f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
    }


def build_confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, label_names: list[str]) -> pd.DataFrame:
    """Build a full confusion matrix with label-name row/column indices."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    return pd.DataFrame(cm, index=label_names, columns=label_names)


def find_top_confusions(confusion_df: pd.DataFrame, top_k: int = 10) -> list[dict]:
    """Return the top-K most-confused intent pairs (off-diagonal, descending)."""
    label_names = list(confusion_df.index)
    n = len(label_names)
    pairs: list[tuple[int, str, str]] = []
    values = confusion_df.values
    for i in range(n):
        for j in range(n):
            if i != j and values[i, j] > 0:
                pairs.append((int(values[i, j]), label_names[i], label_names[j]))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [{"true_label": true, "predicted_label": pred, "count": count} for count, true, pred in pairs[:top_k]]


def find_top_errors(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    texts: list[str],
    label_names: list[str],
    top_k: int = 20,
) -> list[dict]:
    """Return the top-K misclassified examples with deterministic ordering.

    Selection rule: collect all misclassified examples, sort by
    (true_label_name, predicted_label_name, text) for stable ordering,
    return the first ``top_k``.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    errors: list[dict] = []
    for i in range(len(y_true_arr)):
        if y_true_arr[i] != y_pred_arr[i]:
            true_id = int(y_true_arr[i])
            pred_id = int(y_pred_arr[i])
            errors.append(
                {
                    "text": texts[i],
                    "true_label": label_names[true_id],
                    "predicted_label": label_names[pred_id],
                    "true_id": true_id,
                    "predicted_id": pred_id,
                }
            )
    errors.sort(key=lambda e: (e["true_label"], e["predicted_label"], e["text"]))
    return errors[:top_k]
