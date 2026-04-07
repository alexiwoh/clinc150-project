"""Tests for src.visualizers.dataset.DatasetVisualizer."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.visualizers import DatasetVisualizer


@pytest.fixture()
def figures_dir(tmp_path: Path) -> Path:
    d = tmp_path / "figures"
    d.mkdir()
    return d


class TestPlotSplitSizes:
    def test_creates_file(self, figures_dir: Path) -> None:
        sizes = {"train": 15250, "validation": 3100, "test": 5500}
        out = figures_dir / "split_sizes.png"
        DatasetVisualizer.plot_split_sizes(sizes, out)
        assert out.exists() and out.stat().st_size > 0


class TestPlotClassDistribution:
    @pytest.fixture()
    def class_data(self) -> tuple[list[str], list[int]]:
        names = [f"intent_{i}" for i in range(10)] + ["oos"]
        counts = [50] * 10 + [100]
        return names, counts

    def test_full_chart(self, figures_dir: Path, class_data: tuple[list[str], list[int]]) -> None:
        names, counts = class_data
        out = figures_dir / "class_dist_full.png"
        DatasetVisualizer.plot_class_distribution(names, counts, out)
        assert out.exists() and out.stat().st_size > 0

    def test_top_n_chart(self, figures_dir: Path, class_data: tuple[list[str], list[int]]) -> None:
        names, counts = class_data
        out = figures_dir / "class_dist_top5.png"
        DatasetVisualizer.plot_class_distribution(names, counts, out, top_n=5)
        assert out.exists() and out.stat().st_size > 0


class TestPlotOosVsInscope:
    def test_creates_file(self, figures_dir: Path) -> None:
        oos = {"train": 250, "validation": 100, "test": 1000}
        ins = {"train": 15000, "validation": 3000, "test": 4500}
        out = figures_dir / "oos_vs_inscope.png"
        DatasetVisualizer.plot_oos_vs_inscope(oos, ins, out)
        assert out.exists() and out.stat().st_size > 0


class TestPlotQueryLengthHistogram:
    def test_creates_file(self, figures_dir: Path) -> None:
        lengths = [5, 7, 8, 6, 10, 3, 4, 9, 7, 6, 8, 5, 11, 7, 6]
        out = figures_dir / "query_length_histogram.png"
        DatasetVisualizer.plot_query_length_histogram(lengths, out)
        assert out.exists() and out.stat().st_size > 0


class TestPlotQueryLengthBoxplot:
    def test_creates_file(self, figures_dir: Path) -> None:
        lengths_by_split = {
            "train": [5, 7, 8, 6, 10, 3, 4, 9],
            "validation": [6, 7, 5, 8, 9],
            "test": [4, 6, 7, 8, 5, 10, 11],
        }
        out = figures_dir / "query_length_boxplot.png"
        DatasetVisualizer.plot_query_length_boxplot(lengths_by_split, out)
        assert out.exists() and out.stat().st_size > 0


class TestSaveRepresentativeQueries:
    def test_creates_csv_with_correct_columns(self, figures_dir: Path) -> None:
        rows = [
            {"split": "train", "label": "greeting", "scope": "in_scope", "query": "hello there"},
            {"split": "train", "label": "oos", "scope": "oos", "query": "what is the meaning of life"},
        ]
        out = figures_dir / "representative_queries.csv"
        DatasetVisualizer.save_representative_queries(rows, out)
        assert out.exists() and out.stat().st_size > 0

        with out.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["split", "label", "scope", "query"]
            read_rows = list(reader)
        assert len(read_rows) == 2
        assert read_rows[0]["label"] == "greeting"
        assert read_rows[1]["scope"] == "oos"
