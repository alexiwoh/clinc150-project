"""Tests for src.dataset.CLINCDataset."""

from __future__ import annotations

import pytest

from src.config import DATASET_CONFIG
from src.constants import NUM_CLASSES, OOS_LABEL_ID, OOS_LABEL_NAME
from src.dataset import CLINCDataset

pytestmark = pytest.mark.network

EXPECTED_SPLIT_NAMES = {"train", "validation", "test"}
EXPECTED_COLUMNS = {"text", "intent"}
EXPECTED_TRAIN_SIZE = 15250


@pytest.fixture(scope="session")
def ds() -> CLINCDataset:
    """Load the dataset once for the entire test session."""
    return CLINCDataset.load(DATASET_CONFIG)


class TestLoad:
    def test_splits_present(self, ds: CLINCDataset) -> None:
        sizes = ds.split_sizes()
        assert set(sizes.keys()) == EXPECTED_SPLIT_NAMES

    def test_columns(self, ds: CLINCDataset) -> None:
        for split in EXPECTED_SPLIT_NAMES:
            assert set(ds[split].column_names) == EXPECTED_COLUMNS


class TestLabelMetadata:
    def test_label_count(self, ds: CLINCDataset) -> None:
        assert ds.num_classes == NUM_CLASSES
        assert len(ds.label_names) == NUM_CLASSES

    def test_oos_in_labels(self, ds: CLINCDataset) -> None:
        assert OOS_LABEL_NAME in ds.label_names

    def test_round_trip_by_name(self, ds: CLINCDataset) -> None:
        for name in ds.label_names:
            assert ds.label_name(ds.label_id(name)) == name

    def test_round_trip_by_id(self, ds: CLINCDataset) -> None:
        for label_id in range(ds.num_classes):
            assert ds.label_id(ds.label_name(label_id)) == label_id

    def test_cross_split_label_space(self, ds: CLINCDataset) -> None:
        """All splits must share the same set of label names."""
        train_labels = set(ds["train"].features["intent"].names)
        for split in ("validation", "test"):
            assert set(ds[split].features["intent"].names) == train_labels


class TestSplitQueries:
    def test_class_distribution_sums(self, ds: CLINCDataset) -> None:
        dist = ds.class_distribution("train")
        assert sum(dist.values()) == EXPECTED_TRAIN_SIZE

    def test_sample_examples_count(self, ds: CLINCDataset) -> None:
        examples = ds.sample_examples("train", OOS_LABEL_ID, n=3)
        assert len(examples) == 3
        assert all(isinstance(t, str) for t in examples)

    def test_oos_counts_keys_and_values(self, ds: CLINCDataset) -> None:
        oos = ds.oos_counts()
        assert set(oos.keys()) == EXPECTED_SPLIT_NAMES
        assert all(count > 0 for count in oos.values())

    def test_in_scope_counts_consistent(self, ds: CLINCDataset) -> None:
        sizes = ds.split_sizes()
        oos = ds.oos_counts()
        in_scope = ds.in_scope_counts()
        for split in EXPECTED_SPLIT_NAMES:
            assert in_scope[split] == sizes[split] - oos[split]


class TestDataQuality:
    def test_no_null_or_empty_texts(self, ds: CLINCDataset) -> None:
        report = ds.validate()
        assert all(v == 0 for v in report.null_text_counts.values())
        assert all(v == 0 for v in report.empty_text_counts.values())

    def test_no_out_of_range_labels(self, ds: CLINCDataset) -> None:
        report = ds.validate()
        assert all(v == 0 for v in report.out_of_range_label_counts.values())

    def test_validation_report_clean(self, ds: CLINCDataset) -> None:
        assert ds.validate().is_clean
