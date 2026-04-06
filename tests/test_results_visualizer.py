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
