"""Tests for src.visualizers.training.TrainingVisualizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.visualizers import TrainingVisualizer


@pytest.fixture()
def figures_dir(tmp_path: Path) -> Path:
    d = tmp_path / "figures"
    d.mkdir()
    return d


@pytest.fixture()
def epoch_history() -> list[dict]:
    """Synthetic 10-epoch training history."""
    return [
        {
            "epoch": i,
            "train_loss": 5.0 - i * 0.4,
            "val_loss": 4.5 - i * 0.35,
            "val_accuracy": 0.3 + i * 0.06,
            "val_macro_f1": 0.25 + i * 0.065,
        }
        for i in range(1, 11)
    ]


class TestPlotLossCurves:
    def test_creates_file(self, figures_dir: Path, epoch_history: list[dict]) -> None:
        out = figures_dir / "loss_curve.png"
        TrainingVisualizer.plot_loss_curves(epoch_history, out)
        assert out.exists() and out.stat().st_size > 0

    def test_with_best_epoch(self, figures_dir: Path, epoch_history: list[dict]) -> None:
        out = figures_dir / "loss_curve_best.png"
        TrainingVisualizer.plot_loss_curves(epoch_history, out, best_epoch=7)
        assert out.exists() and out.stat().st_size > 0


class TestPlotValMetricCurve:
    def test_creates_file(self, figures_dir: Path, epoch_history: list[dict]) -> None:
        out = figures_dir / "val_f1.png"
        TrainingVisualizer.plot_val_metric_curve(epoch_history, out)
        assert out.exists() and out.stat().st_size > 0

    def test_with_best_epoch_marker(self, figures_dir: Path, epoch_history: list[dict]) -> None:
        out = figures_dir / "val_f1_best.png"
        TrainingVisualizer.plot_val_metric_curve(epoch_history, out, best_epoch=8)
        assert out.exists() and out.stat().st_size > 0

    def test_best_epoch_equals_stop_epoch(self, figures_dir: Path, epoch_history: list[dict]) -> None:
        out = figures_dir / "val_f1_no_early_stop.png"
        TrainingVisualizer.plot_val_metric_curve(epoch_history, out, best_epoch=10)
        assert out.exists() and out.stat().st_size > 0

    def test_custom_metric_key(self, figures_dir: Path, epoch_history: list[dict]) -> None:
        out = figures_dir / "val_acc.png"
        TrainingVisualizer.plot_val_metric_curve(epoch_history, out, metric_key="val_accuracy")
        assert out.exists() and out.stat().st_size > 0
