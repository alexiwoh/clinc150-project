"""Tests for pure metric functions in src/analysis/metrics.py and related modules."""

from __future__ import annotations


import numpy as np

from src.analysis.enums import ErrorCategory
from src.analysis.metrics import (
    compute_brier_score,
    compute_confidence_statistics,
    compute_ece,
    compute_extended_classification_metrics,
    compute_mce,
    compute_msp_oos_detection_metrics,
    compute_nll,
    compute_oos_detection_metrics,
    stratify_by_confidence,
)
from src.analysis.taxonomy import classify_single_error


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _perfect_probabilities(n: int, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities where argmax == targets (perfectly calibrated)."""
    targets = np.random.RandomState(42).randint(0, n_classes, size=n)
    probs = np.full((n, n_classes), 0.01 / (n_classes - 1))
    probs[np.arange(n), targets] = 0.99
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / row_sums
    return probs, targets


def _random_probabilities(n: int, n_classes: int, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    """Return random softmax probabilities and random targets."""
    rng = np.random.RandomState(seed)
    raw = rng.dirichlet(np.ones(n_classes), size=n)
    targets = rng.randint(0, n_classes, size=n)
    return raw.astype(np.float64), targets.astype(np.intp)


# ---------------------------------------------------------------------------
# TestComputeECE
# ---------------------------------------------------------------------------


class TestComputeECE:
    def test_perfect_calibration_low_ece(self) -> None:
        probs, targets = _perfect_probabilities(1000, 10)
        ece, bin_data = compute_ece(probs, targets)
        assert ece < 0.05, f"ECE should be near 0 for perfect predictions, got {ece}"

    def test_bin_weights_sum_to_one(self) -> None:
        probs, targets = _random_probabilities(500, 10)
        _, bin_data = compute_ece(probs, targets)
        total_weight = sum(b["count"] for b in bin_data) / len(targets)
        assert abs(total_weight - 1.0) < 1e-9

    def test_bin_count_equals_n_bins(self) -> None:
        probs, targets = _random_probabilities(200, 5)
        _, bin_data = compute_ece(probs, targets, n_bins=10)
        assert len(bin_data) == 10

    def test_ece_in_unit_interval(self) -> None:
        probs, targets = _random_probabilities(300, 20)
        ece, _ = compute_ece(probs, targets)
        assert 0.0 <= ece <= 1.0

    def test_ece_increases_with_miscalibration(self) -> None:
        probs_good, targets = _perfect_probabilities(500, 10)
        probs_bad, _ = _random_probabilities(500, 10, seed=99)
        ece_good, _ = compute_ece(probs_good, targets)
        ece_bad, _ = compute_ece(probs_bad, targets)
        assert ece_good < ece_bad


# ---------------------------------------------------------------------------
# TestComputeMCE
# ---------------------------------------------------------------------------


class TestComputeMCE:
    def test_mce_gte_ece(self) -> None:
        probs, targets = _random_probabilities(500, 10)
        ece, _ = compute_ece(probs, targets)
        mce = compute_mce(probs, targets)
        assert mce >= ece - 1e-9

    def test_perfect_calibration_low_mce(self) -> None:
        probs, targets = _perfect_probabilities(1000, 10)
        mce = compute_mce(probs, targets)
        assert mce < 0.1


# ---------------------------------------------------------------------------
# TestComputeBrierScore
# ---------------------------------------------------------------------------


class TestComputeBrierScore:
    def test_perfect_predictions_near_zero(self) -> None:
        n, nc = 200, 5
        targets = np.array([i % nc for i in range(n)], dtype=np.intp)
        probs = np.zeros((n, nc))
        probs[np.arange(n), targets] = 1.0
        brier = compute_brier_score(probs, targets)
        assert abs(brier) < 1e-9

    def test_uniform_predictions_known_value(self) -> None:
        n, nc = 100, 4
        targets = np.zeros(n, dtype=np.intp)
        probs = np.full((n, nc), 1.0 / nc)
        brier = compute_brier_score(probs, targets)
        expected = 1.0 - 1.0 / nc
        assert abs(brier - expected) < 1e-6, f"Expected {expected}, got {brier}"

    def test_brier_non_negative(self) -> None:
        probs, targets = _random_probabilities(300, 10)
        assert compute_brier_score(probs, targets) >= 0.0


# ---------------------------------------------------------------------------
# TestComputeNLL
# ---------------------------------------------------------------------------


class TestComputeNLL:
    def test_perfect_predictions_near_zero(self) -> None:
        n, nc = 200, 5
        targets = np.array([i % nc for i in range(n)], dtype=np.intp)
        probs = np.full((n, nc), 1e-8)
        probs[np.arange(n), targets] = 1.0 - (nc - 1) * 1e-8
        nll = compute_nll(probs, targets)
        assert nll < 0.001

    def test_nll_non_negative(self) -> None:
        probs, targets = _random_probabilities(300, 10)
        assert compute_nll(probs, targets) >= 0.0

    def test_uniform_nll_matches_log_classes(self) -> None:
        n, nc = 100, 8
        targets = np.zeros(n, dtype=np.intp)
        probs = np.full((n, nc), 1.0 / nc)
        nll = compute_nll(probs, targets)
        expected = np.log(nc)
        assert abs(nll - expected) < 1e-6


# ---------------------------------------------------------------------------
# TestComputeExtendedMetrics
# ---------------------------------------------------------------------------


class TestComputeExtendedMetrics:
    def test_perfect_predictions(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        result = compute_extended_classification_metrics(y, y)
        assert abs(result["micro_f1"] - 1.0) < 1e-9
        assert abs(result["weighted_f1"] - 1.0) < 1e-9

    def test_keys_present(self) -> None:
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 1])
        result = compute_extended_classification_metrics(y_true, y_pred)
        expected_keys = {
            "micro_f1",
            "weighted_f1",
            "micro_precision",
            "micro_recall",
            "weighted_precision",
            "weighted_recall",
        }
        assert expected_keys == set(result.keys())

    def test_micro_f1_equals_accuracy_for_multiclass(self) -> None:
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 5, size=200)
        y_pred = rng.randint(0, 5, size=200)
        result = compute_extended_classification_metrics(y_true, y_pred)
        accuracy = float(np.mean(y_true == y_pred))
        assert abs(result["micro_f1"] - accuracy) < 1e-9


# ---------------------------------------------------------------------------
# TestComputeConfidenceStatistics
# ---------------------------------------------------------------------------


class TestComputeConfidenceStatistics:
    def test_keys_present(self) -> None:
        probs, targets = _random_probabilities(100, 5)
        preds = np.argmax(probs, axis=1)
        stats = compute_confidence_statistics(probs, preds, targets)
        assert "correct_max_confidence_mean" in stats
        assert "prediction_entropy_mean" in stats
        assert "correct_margin_mean" in stats

    def test_correct_higher_confidence_than_incorrect(self) -> None:
        """When high-confidence predictions are mostly correct, correct mean >= incorrect mean."""
        rng = np.random.RandomState(42)
        n, nc = 500, 5
        targets = rng.randint(0, nc, size=n).astype(np.intp)
        probs = rng.dirichlet(np.ones(nc) * 0.3, size=n).astype(np.float64)
        # Boost the true-class probability so argmax is mostly correct
        probs[np.arange(n), targets] += 2.0
        probs = probs / probs.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        stats = compute_confidence_statistics(probs, preds, targets)
        assert stats["correct_max_confidence_mean"] >= stats["incorrect_max_confidence_mean"] - 1e-9


# ---------------------------------------------------------------------------
# TestOOSDetectionMetrics
# ---------------------------------------------------------------------------


class TestOOSDetectionMetrics:
    def test_auroc_in_unit_interval(self) -> None:
        probs, targets = _random_probabilities(500, 10)
        oos_idx = 3
        targets[:50] = oos_idx
        result = compute_oos_detection_metrics(probs, targets, oos_idx)
        assert 0.0 <= result["auroc"] <= 1.0

    def test_aupr_in_unit_interval(self) -> None:
        probs, targets = _random_probabilities(500, 10)
        oos_idx = 3
        targets[:50] = oos_idx
        result = compute_oos_detection_metrics(probs, targets, oos_idx)
        assert 0.0 <= result["aupr"] <= 1.0

    def test_perfect_oos_detection(self) -> None:
        n, nc, oos_idx = 200, 5, 2
        targets = np.array([oos_idx if i < 50 else i % nc for i in range(n)], dtype=np.intp)
        probs = np.full((n, nc), 0.01)
        probs[targets == oos_idx, oos_idx] = 0.99
        probs[targets != oos_idx, oos_idx] = 0.001
        probs = probs / probs.sum(axis=1, keepdims=True)
        result = compute_oos_detection_metrics(probs, targets, oos_idx)
        assert result["auroc"] > 0.95

    def test_fpr_at_95tpr_present(self) -> None:
        probs, targets = _random_probabilities(500, 10)
        oos_idx = 0
        targets[:100] = oos_idx
        result = compute_oos_detection_metrics(probs, targets, oos_idx)
        assert "fpr_at_95tpr" in result
        assert 0.0 <= result["fpr_at_95tpr"] <= 1.0

    def test_roc_pr_curves_present(self) -> None:
        probs, targets = _random_probabilities(200, 5)
        oos_idx = 1
        targets[:40] = oos_idx
        result = compute_oos_detection_metrics(probs, targets, oos_idx)
        assert "roc_curve" in result
        assert "pr_curve" in result
        assert len(result["roc_curve"]["fpr"]) > 1


class TestMSPOOSDetection:
    def test_msp_metrics_present(self) -> None:
        probs, targets = _random_probabilities(200, 5)
        oos_idx = 1
        targets[:40] = oos_idx
        result = compute_msp_oos_detection_metrics(probs, targets, oos_idx)
        assert "msp_auroc" in result
        assert "msp_aupr" in result
        assert 0.0 <= result["msp_auroc"] <= 1.0


# ---------------------------------------------------------------------------
# TestConfidenceStratification
# ---------------------------------------------------------------------------


class TestConfidenceStratification:
    def test_five_strata_returned(self) -> None:
        max_confs = np.linspace(0.1, 0.95, 100)
        correct = np.ones(100, dtype=bool)
        strata = stratify_by_confidence(max_confs, correct)
        assert len(strata) == 5

    def test_counts_sum_to_total(self) -> None:
        rng = np.random.RandomState(42)
        max_confs = rng.uniform(0, 1, size=500)
        correct = rng.choice([True, False], size=500)
        strata = stratify_by_confidence(max_confs, correct)
        assert sum(s["count"] for s in strata) == 500

    def test_perfect_accuracy_in_high_bin(self) -> None:
        max_confs = np.array([0.85, 0.9, 0.95, 0.99])
        correct = np.array([True, True, True, True])
        strata = stratify_by_confidence(max_confs, correct)
        high_bin = strata[-1]
        assert high_bin["accuracy"] == 1.0

    def test_error_fraction_sums_to_one(self) -> None:
        rng = np.random.RandomState(42)
        max_confs = rng.uniform(0, 1, size=500)
        correct = rng.choice([True, False], size=500)
        strata = stratify_by_confidence(max_confs, correct)
        total_frac = sum(s["error_fraction_of_total"] for s in strata)
        assert abs(total_frac - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# TestClassifySingleError (taxonomy)
# ---------------------------------------------------------------------------


class TestClassifySingleError:
    _domain_map = {
        "balance": "banking",
        "transfer": "banking",
        "recipe": "kitchen_and_dining",
        "calories": "kitchen_and_dining",
        "book_flight": "travel",
    }

    def test_oos_as_inscope(self) -> None:
        primary, secondary = classify_single_error("what is this", "oos", "balance", domain_map=self._domain_map)
        assert primary == ErrorCategory.OOS_AS_INSCOPE

    def test_inscope_as_oos(self) -> None:
        primary, secondary = classify_single_error("check my balance", "balance", "oos", domain_map=self._domain_map)
        assert primary == ErrorCategory.INSCOPE_AS_OOS

    def test_near_semantic_confusion(self) -> None:
        primary, _ = classify_single_error("how much money", "balance", "transfer", domain_map=self._domain_map)
        assert primary == ErrorCategory.NEAR_SEMANTIC_CONFUSION

    def test_cross_domain_confusion(self) -> None:
        primary, _ = classify_single_error(
            "i need a recipe for travel",
            "recipe",
            "book_flight",
            domain_map=self._domain_map,
        )
        assert primary == ErrorCategory.CROSS_DOMAIN_CONFUSION

    def test_short_query_secondary(self) -> None:
        primary, secondary = classify_single_error("my bal", "balance", "transfer", domain_map=self._domain_map)
        assert primary == ErrorCategory.NEAR_SEMANTIC_CONFUSION
        assert ErrorCategory.SHORT_QUERY_AMBIGUITY in secondary

    def test_short_oos_primary_is_oos(self) -> None:
        primary, secondary = classify_single_error("hi", "oos", "balance", domain_map=self._domain_map)
        assert primary == ErrorCategory.OOS_AS_INSCOPE
        assert ErrorCategory.SHORT_QUERY_AMBIGUITY in secondary

    def test_long_query_no_short_tag(self) -> None:
        long_text = "i want to check the balance in my savings account please"
        primary, secondary = classify_single_error(long_text, "balance", "transfer", domain_map=self._domain_map)
        assert ErrorCategory.SHORT_QUERY_AMBIGUITY not in secondary
