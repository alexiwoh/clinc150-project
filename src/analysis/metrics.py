"""Pure metric computation functions for error analysis.

All functions accept numpy arrays and return floats or structured dicts.
No file I/O is performed here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import f1_score, precision_score, recall_score

from src.analysis.constants import ECE_DEFAULT_BINS


# ---------------------------------------------------------------------------
# Extended classification metrics (Section B)
# ---------------------------------------------------------------------------


def compute_extended_classification_metrics(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> dict[str, float]:
    """Compute micro- and weighted-averaged precision, recall, and F1.

    These complement the macro-averaged metrics already in ``test_metrics.json``.
    """
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Calibration metrics (Section C)
# ---------------------------------------------------------------------------


def compute_ece(
    probabilities: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.intp],
    n_bins: int = ECE_DEFAULT_BINS,
) -> tuple[float, list[dict[str, Any]]]:
    """Expected Calibration Error with equal-width bins.

    Returns ``(ece_value, bin_data)`` where each bin dict contains
    ``accuracy``, ``confidence``, ``count``, ``bin_lower``, ``bin_upper``.
    """
    max_probs = np.max(probabilities, axis=1)
    preds = np.argmax(probabilities, axis=1)
    correct = (preds == targets).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data: list[dict[str, Any]] = []
    ece = 0.0
    n_total = len(targets)

    for b in range(n_bins):
        lower, upper = float(bin_edges[b]), float(bin_edges[b + 1])
        if b < n_bins - 1:
            mask = (max_probs >= lower) & (max_probs < upper)
        else:
            mask = (max_probs >= lower) & (max_probs <= upper)

        count = int(mask.sum())
        if count > 0:
            bin_acc = float(correct[mask].mean())
            bin_conf = float(max_probs[mask].mean())
            ece += (count / n_total) * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = 0.0

        bin_data.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
            }
        )

    return float(ece), bin_data


def compute_mce(
    probabilities: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.intp],
    n_bins: int = ECE_DEFAULT_BINS,
) -> float:
    """Maximum Calibration Error: worst-bin |accuracy - confidence| gap."""
    _, bin_data = compute_ece(probabilities, targets, n_bins)
    return max(
        (abs(b["accuracy"] - b["confidence"]) for b in bin_data if b["count"] > 0),
        default=0.0,
    )


def compute_brier_score(
    probabilities: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.intp],
) -> float:
    """Multi-class Brier score: mean of sum of squared diffs vs one-hot targets."""
    n_examples, n_classes = probabilities.shape
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(n_examples), targets] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def compute_nll(
    probabilities: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.intp],
) -> float:
    """Negative log-likelihood: mean of -log(p_true_class)."""
    eps = 1e-12
    true_class_probs = probabilities[np.arange(len(targets)), targets]
    return float(-np.mean(np.log(np.clip(true_class_probs, eps, None))))


# ---------------------------------------------------------------------------
# Confidence distribution statistics (Section C3)
# ---------------------------------------------------------------------------


def compute_confidence_statistics(
    probabilities: npt.NDArray[np.floating[Any]],
    predictions: npt.NDArray[np.intp],
    targets: npt.NDArray[np.intp],
) -> dict[str, float]:
    """Compute confidence distribution statistics for correct vs incorrect predictions.

    Returns a dict with mean/std of max confidence, prediction entropy,
    and confidence margin split by correct/incorrect.
    """
    max_probs = np.max(probabilities, axis=1)
    correct_mask = predictions == targets

    sorted_probs = np.sort(probabilities, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]

    eps = 1e-12
    log_probs = np.log(np.clip(probabilities, eps, None))
    entropies = -np.sum(probabilities * log_probs, axis=1)

    def _stats(values: npt.NDArray[np.floating[Any]]) -> tuple[float, float]:
        if len(values) == 0:
            return 0.0, 0.0
        return float(np.mean(values)), float(np.std(values))

    correct_conf_mean, correct_conf_std = _stats(max_probs[correct_mask])
    incorrect_conf_mean, incorrect_conf_std = _stats(max_probs[~correct_mask])
    entropy_mean, entropy_std = _stats(entropies)
    correct_margin_mean, correct_margin_std = _stats(margins[correct_mask])
    incorrect_margin_mean, incorrect_margin_std = _stats(margins[~correct_mask])

    return {
        "correct_max_confidence_mean": correct_conf_mean,
        "correct_max_confidence_std": correct_conf_std,
        "incorrect_max_confidence_mean": incorrect_conf_mean,
        "incorrect_max_confidence_std": incorrect_conf_std,
        "prediction_entropy_mean": entropy_mean,
        "prediction_entropy_std": entropy_std,
        "correct_margin_mean": correct_margin_mean,
        "correct_margin_std": correct_margin_std,
        "incorrect_margin_mean": incorrect_margin_mean,
        "incorrect_margin_std": incorrect_margin_std,
    }


# ---------------------------------------------------------------------------
# OOS threshold metrics (Section D)
# ---------------------------------------------------------------------------


def compute_oos_detection_metrics(
    probabilities: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.intp],
    oos_class_idx: int,
) -> dict[str, Any]:
    """OOS detection metrics using the explicit OOS class probability.

    Returns AUROC, AUPR, FPR@95TPR, FPR@90TPR, and the full ROC/PR curves.
    """
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    binary_targets = (targets == oos_class_idx).astype(int)
    oos_scores = probabilities[:, oos_class_idx]

    fpr, tpr, _roc_thresholds = roc_curve(binary_targets, oos_scores)
    auroc = float(auc(fpr, tpr))

    pr_precision, pr_recall, _pr_thresholds = precision_recall_curve(binary_targets, oos_scores)
    aupr = float(auc(pr_recall, pr_precision))

    fpr_at_95tpr = _fpr_at_tpr(fpr, tpr, target_tpr=0.95)
    fpr_at_90tpr = _fpr_at_tpr(fpr, tpr, target_tpr=0.90)

    return {
        "auroc": auroc,
        "aupr": aupr,
        "fpr_at_95tpr": fpr_at_95tpr,
        "fpr_at_90tpr": fpr_at_90tpr,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": pr_precision.tolist(), "recall": pr_recall.tolist()},
    }


def compute_msp_oos_detection_metrics(
    probabilities: npt.NDArray[np.floating[Any]],
    targets: npt.NDArray[np.intp],
    oos_class_idx: int,
) -> dict[str, float]:
    """Maximum Softmax Probability (MSP) baseline for OOS detection.

    OOS is predicted when max(softmax) is low, so we use ``1 - max(softmax)``
    as the OOS score (higher score = more likely OOS).
    """
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    binary_targets = (targets == oos_class_idx).astype(int)
    msp_oos_scores = 1.0 - np.max(probabilities, axis=1)

    fpr, tpr, _ = roc_curve(binary_targets, msp_oos_scores)
    auroc = float(auc(fpr, tpr))

    pr_precision, pr_recall, _ = precision_recall_curve(binary_targets, msp_oos_scores)
    aupr = float(auc(pr_recall, pr_precision))

    return {"msp_auroc": auroc, "msp_aupr": aupr}


def _fpr_at_tpr(
    fpr: npt.NDArray[np.floating[Any]],
    tpr: npt.NDArray[np.floating[Any]],
    target_tpr: float,
) -> float:
    """Find the FPR at which TPR first reaches *target_tpr*."""
    indices = np.where(tpr >= target_tpr)[0]
    if len(indices) == 0:
        return 1.0
    return float(fpr[indices[0]])


# ---------------------------------------------------------------------------
# Confidence stratification (Section G)
# ---------------------------------------------------------------------------


def stratify_by_confidence(
    max_confidences: npt.NDArray[np.floating[Any]],
    correct_mask: npt.NDArray[np.bool_],
    bin_edges: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Stratify predictions into confidence bins and compute accuracy per bin.

    Returns one dict per bin with ``bin_label``, ``count``, ``accuracy``,
    ``error_count``, ``error_fraction_of_total``.
    """
    from src.analysis.constants import CONFIDENCE_BIN_EDGES, CONFIDENCE_STRATA_LABELS

    if bin_edges is None:
        bin_edges = CONFIDENCE_BIN_EDGES

    total_errors = int((~correct_mask).sum())
    strata: list[dict[str, Any]] = []

    for i in range(len(bin_edges) - 1):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        if i < len(bin_edges) - 2:
            mask = (max_confidences >= lower) & (max_confidences < upper)
        else:
            mask = (max_confidences >= lower) & (max_confidences <= upper)

        count = int(mask.sum())
        n_correct = int(correct_mask[mask].sum())
        n_errors = count - n_correct

        strata.append(
            {
                "bin_label": CONFIDENCE_STRATA_LABELS[i]
                if i < len(CONFIDENCE_STRATA_LABELS)
                else f"[{lower}, {upper}]",
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "accuracy": float(n_correct / count) if count > 0 else 0.0,
                "error_count": n_errors,
                "error_fraction_of_total": float(n_errors / total_errors) if total_errors > 0 else 0.0,
            }
        )

    return strata
