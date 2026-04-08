"""Tests for src.visualizers.results.ResultsVisualizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.visualizers import ResultsVisualizer

_NUM_CLASSES = 5
_LABEL_NAMES = ["alpha", "beta", "gamma", "delta", "oos"]


@pytest.fixture()
def figures_dir(tmp_path: Path) -> Path:
    d = tmp_path / "figures"
    d.mkdir()
    return d


@pytest.fixture()
def tuning_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_name": [f"run_{i}" for i in range(6)],
            "best_val_metric": [0.85, 0.87, 0.83, 0.90, 0.88, 0.86],
        }
    )


@pytest.fixture()
def confusion_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    matrix = rng.integers(0, 10, size=(_NUM_CLASSES, _NUM_CLASSES))
    np.fill_diagonal(matrix, rng.integers(20, 40, size=_NUM_CLASSES))
    return pd.DataFrame(matrix, index=_LABEL_NAMES, columns=_LABEL_NAMES)


@pytest.fixture()
def targets_and_predictions() -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(42)
    targets = rng.integers(0, _NUM_CLASSES, size=200).tolist()
    predictions = targets.copy()
    for i in range(0, 200, 5):
        predictions[i] = (predictions[i] + 1) % _NUM_CLASSES
    return targets, predictions


@pytest.fixture()
def top_confusions() -> list[dict]:
    return [
        {"true_label": "alpha", "predicted_label": "beta", "count": 15},
        {"true_label": "gamma", "predicted_label": "delta", "count": 10},
        {"true_label": "oos", "predicted_label": "alpha", "count": 8},
    ]


@pytest.fixture()
def top_errors() -> list[dict]:
    errors = []
    pairs = [("alpha", "beta"), ("gamma", "delta"), ("oos", "alpha"), ("alpha", "beta"), ("gamma", "delta")]
    for true, pred in pairs * 3:
        errors.append({"text": f"sample from {true}", "true_label": true, "predicted_label": pred})
    return errors


@pytest.fixture()
def oos_metrics() -> dict[str, float]:
    return {"oos_precision": 0.85, "oos_recall": 0.40, "oos_f1": 0.54}


class TestPlotTuningSummary:
    def test_creates_file(self, figures_dir: Path, tuning_df: pd.DataFrame) -> None:
        out = figures_dir / "tuning_summary.png"
        ResultsVisualizer.plot_tuning_summary(tuning_df, out)
        assert out.exists() and out.stat().st_size > 0


class TestPlotConfusionMatrix:
    def test_creates_file(self, figures_dir: Path, confusion_df: pd.DataFrame) -> None:
        out = figures_dir / "confusion.png"
        ResultsVisualizer.plot_confusion_matrix(confusion_df, out)
        assert out.exists() and out.stat().st_size > 0


class TestPlotTopConfusedPairs:
    def test_creates_file(self, figures_dir: Path, top_confusions: list[dict]) -> None:
        out = figures_dir / "top_confused.png"
        ResultsVisualizer.plot_top_confused_pairs(top_confusions, out)
        assert out.exists() and out.stat().st_size > 0


class TestPlotBottomClassesF1:
    def test_creates_file(
        self,
        figures_dir: Path,
        targets_and_predictions: tuple[list[int], list[int]],
    ) -> None:
        targets, predictions = targets_and_predictions
        out = figures_dir / "bottom_f1.png"
        ResultsVisualizer.plot_bottom_classes_f1(targets, predictions, _LABEL_NAMES, out, bottom_n=3)
        assert out.exists() and out.stat().st_size > 0


class TestPlotOosMetrics:
    def test_creates_file(self, figures_dir: Path, oos_metrics: dict[str, float]) -> None:
        out = figures_dir / "oos_metrics.png"
        ResultsVisualizer.plot_oos_metrics(oos_metrics, out)
        assert out.exists() and out.stat().st_size > 0


class TestPlotErrorSummary:
    def test_creates_file(self, figures_dir: Path, top_errors: list[dict]) -> None:
        out = figures_dir / "error_summary.png"
        ResultsVisualizer.plot_error_summary(top_errors, out)
        assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Aggregate / cross-model plotting methods
# ---------------------------------------------------------------------------

_MODEL_STATS: list[tuple[str, float, float]] = [
    ("TF-IDF + MLP", 0.92, 0.01),
    ("Text CNN", 0.91, 0.02),
    ("BiLSTM", 0.89, 0.015),
]


class TestPlotModelComparisonBar:
    def test_creates_file(self, figures_dir: Path) -> None:
        out = figures_dir / "model_comparison.png"
        ResultsVisualizer.plot_model_comparison_bar(_MODEL_STATS, "Test Accuracy", out)
        assert out.exists() and out.stat().st_size > 0

    def test_single_model(self, figures_dir: Path) -> None:
        out = figures_dir / "model_comparison_single.png"
        ResultsVisualizer.plot_model_comparison_bar([_MODEL_STATS[0]], "Test F1", out)
        assert out.exists() and out.stat().st_size > 0


class TestPlotOosMetricsComparison:
    @pytest.fixture()
    def oos_rows(self) -> list[dict]:
        return [
            {
                "display_name": "TF-IDF + MLP",
                "oos_precision_mean": 0.88,
                "oos_precision_std": 0.01,
                "oos_recall_mean": 0.49,
                "oos_recall_std": 0.04,
                "oos_f1_mean": 0.63,
                "oos_f1_std": 0.03,
            },
            {
                "display_name": "Text CNN",
                "oos_precision_mean": 0.94,
                "oos_precision_std": 0.01,
                "oos_recall_mean": 0.36,
                "oos_recall_std": 0.04,
                "oos_f1_mean": 0.52,
                "oos_f1_std": 0.04,
            },
            {
                "display_name": "BiLSTM",
                "oos_precision_mean": 0.88,
                "oos_precision_std": 0.01,
                "oos_recall_mean": 0.29,
                "oos_recall_std": 0.02,
                "oos_f1_mean": 0.44,
                "oos_f1_std": 0.03,
            },
        ]

    def test_creates_file(self, figures_dir: Path, oos_rows: list[dict]) -> None:
        out = figures_dir / "oos_comparison.png"
        ResultsVisualizer.plot_oos_metrics_comparison(oos_rows, out)
        assert out.exists() and out.stat().st_size > 0


class TestPlotEfficiencyComparison:
    @pytest.fixture()
    def eff_rows(self) -> list[dict]:
        return [
            {
                "display_name": "TF-IDF + MLP",
                "training_time_seconds_mean": 60.6,
                "training_time_seconds_std": 10.8,
                "inference_examples_per_sec_mean": 17783.0,
                "inference_examples_per_sec_std": 391.0,
                "trainable_parameter_count": 5197975,
            },
            {
                "display_name": "Text CNN",
                "training_time_seconds_mean": 136.5,
                "training_time_seconds_std": 7.7,
                "inference_examples_per_sec_mean": 32082.0,
                "inference_examples_per_sec_std": 2139.0,
                "trainable_parameter_count": 1930167,
            },
            {
                "display_name": "BiLSTM",
                "training_time_seconds_mean": 91.7,
                "training_time_seconds_std": 23.2,
                "inference_examples_per_sec_mean": 8432.0,
                "inference_examples_per_sec_std": 1080.0,
                "trainable_parameter_count": 4284311,
            },
        ]

    def test_creates_file(self, figures_dir: Path, eff_rows: list[dict]) -> None:
        out = figures_dir / "efficiency.png"
        ResultsVisualizer.plot_efficiency_comparison(eff_rows, out)
        assert out.exists() and out.stat().st_size > 0


class TestPlotBottomClassesF1FromMetrics:
    @pytest.fixture()
    def class_metrics(self) -> list[dict]:
        return [
            {"label_name": "alpha", "f1": 0.72},
            {"label_name": "beta", "f1": 0.95},
            {"label_name": "gamma", "f1": 0.60},
            {"label_name": "delta", "f1": 0.88},
            {"label_name": "oos", "f1": 0.45},
        ]

    def test_creates_file(self, figures_dir: Path, class_metrics: list[dict]) -> None:
        out = figures_dir / "bottom_f1_metrics.png"
        ResultsVisualizer.plot_bottom_classes_f1_from_metrics(class_metrics, out, bottom_n=3)
        assert out.exists() and out.stat().st_size > 0

    def test_respects_bottom_n(self, figures_dir: Path, class_metrics: list[dict]) -> None:
        out = figures_dir / "bottom_f1_all.png"
        ResultsVisualizer.plot_bottom_classes_f1_from_metrics(class_metrics, out, bottom_n=10)
        assert out.exists() and out.stat().st_size > 0
