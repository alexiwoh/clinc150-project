"""Tests for classification metrics, OOS metrics, confusion matrix, top confusions/errors."""

import pandas as pd

from src.metrics import (
    build_confusion_matrix,
    compute_classification_metrics,
    compute_oos_metrics,
    find_top_confusions,
    find_top_errors,
)


class TestComputeClassificationMetrics:
    def test_perfect_predictions(self) -> None:
        y = [0, 1, 2, 0, 1, 2]
        m = compute_classification_metrics(y, y)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0
        assert m["macro_precision"] == 1.0
        assert m["macro_recall"] == 1.0

    def test_all_wrong(self) -> None:
        y_true = [0, 0, 0]
        y_pred = [1, 1, 1]
        m = compute_classification_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0

    def test_partial(self) -> None:
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]
        m = compute_classification_metrics(y_true, y_pred)
        assert 0.0 < m["accuracy"] < 1.0
        assert 0.0 < m["macro_f1"] < 1.0


class TestComputeOOSMetrics:
    def test_perfect_oos(self) -> None:
        oos_id = 2
        y_true = [0, 1, 2, 2, 0]
        y_pred = [0, 1, 2, 2, 0]
        m = compute_oos_metrics(y_true, y_pred, oos_id)
        assert m["oos_precision"] == 1.0
        assert m["oos_recall"] == 1.0
        assert m["oos_f1"] == 1.0

    def test_no_oos_in_data(self) -> None:
        m = compute_oos_metrics([0, 1, 0], [0, 1, 0], oos_label_id=5)
        assert m["oos_precision"] == 0.0
        assert m["oos_recall"] == 0.0
        assert m["oos_f1"] == 0.0

    def test_oos_missed_completely(self) -> None:
        m = compute_oos_metrics([2, 2, 0], [0, 0, 0], oos_label_id=2)
        assert m["oos_recall"] == 0.0

    def test_oos_false_positives(self) -> None:
        m = compute_oos_metrics([0, 0, 0], [2, 0, 0], oos_label_id=2)
        assert m["oos_precision"] == 0.0
        assert m["oos_recall"] == 0.0


class TestBuildConfusionMatrix:
    def test_shape_and_labels(self) -> None:
        labels = ["a", "b", "c"]
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 2, 1, 0]
        cm = build_confusion_matrix(y_true, y_pred, labels)
        assert isinstance(cm, pd.DataFrame)
        assert cm.shape == (3, 3)
        assert list(cm.index) == labels
        assert list(cm.columns) == labels

    def test_diagonal_for_perfect(self) -> None:
        labels = ["x", "y"]
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        cm = build_confusion_matrix(y_true, y_pred, labels)
        assert cm.values[0, 1] == 0
        assert cm.values[1, 0] == 0
        assert cm.values[0, 0] == 2
        assert cm.values[1, 1] == 2


class TestFindTopConfusions:
    def test_ordering(self) -> None:
        labels = ["a", "b", "c"]
        y_true = [0, 0, 0, 1, 1, 2]
        y_pred = [1, 1, 1, 0, 2, 0]
        cm = build_confusion_matrix(y_true, y_pred, labels)
        top = find_top_confusions(cm, top_k=5)
        assert top[0]["count"] >= top[-1]["count"]
        assert top[0]["true_label"] == "a"
        assert top[0]["predicted_label"] == "b"
        assert top[0]["count"] == 3

    def test_empty_for_perfect(self) -> None:
        labels = ["a", "b"]
        cm = build_confusion_matrix([0, 1], [0, 1], labels)
        top = find_top_confusions(cm, top_k=5)
        assert len(top) == 0


class TestFindTopErrors:
    def test_deterministic_ordering(self) -> None:
        """Errors should be sorted by (true_label, predicted_label, text)."""
        labels = ["alpha", "beta", "gamma"]
        y_true = [0, 1, 2, 0]
        y_pred = [1, 0, 0, 2]
        texts = ["zebra", "apple", "mango", "cat"]

        errors = find_top_errors(y_true, y_pred, texts, labels, top_k=10)
        assert len(errors) == 4

        sort_keys = [(e["true_label"], e["predicted_label"], e["text"]) for e in errors]
        assert sort_keys == sorted(sort_keys)

    def test_top_k_limit(self) -> None:
        labels = ["a", "b"]
        y_true = [0] * 10
        y_pred = [1] * 10
        texts = [f"text_{i}" for i in range(10)]
        errors = find_top_errors(y_true, y_pred, texts, labels, top_k=3)
        assert len(errors) == 3

    def test_correct_fields(self) -> None:
        labels = ["a", "b"]
        errors = find_top_errors([0], [1], ["hello"], labels, top_k=5)
        assert len(errors) == 1
        e = errors[0]
        assert e["text"] == "hello"
        assert e["true_label"] == "a"
        assert e["predicted_label"] == "b"
        assert e["true_id"] == 0
        assert e["predicted_id"] == 1

    def test_no_errors_for_perfect(self) -> None:
        errors = find_top_errors([0, 1], [0, 1], ["a", "b"], ["x", "y"], top_k=5)
        assert len(errors) == 0

    def test_determinism_repeated_calls(self) -> None:
        labels = ["a", "b", "c"]
        y_true = [0, 1, 2, 0, 1]
        y_pred = [2, 0, 1, 1, 2]
        texts = ["e", "d", "c", "b", "a"]
        first = find_top_errors(y_true, y_pred, texts, labels)
        second = find_top_errors(y_true, y_pred, texts, labels)
        assert first == second
