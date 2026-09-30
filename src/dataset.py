"""Dataset loading, split handling, label mapping, and PyTorch dataset wrappers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Value, load_dataset
from scipy.sparse import issparse, spmatrix
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from src.config import DatasetConfig
from src.constants import OOS_LABEL_ID

_SPLIT_NAMES = ("train", "validation", "test")
_TEXT_FIELD = "text"
_LABEL_FIELD = "intent"


@dataclass(frozen=True)
class ValidationReport:
    """Result of null / malformed-data checks across all splits."""

    null_text_counts: dict[str, int]
    empty_text_counts: dict[str, int]
    out_of_range_label_counts: dict[str, int]

    @property
    def is_clean(self) -> bool:
        return (
            all(v == 0 for v in self.null_text_counts.values())
            and all(v == 0 for v in self.empty_text_counts.values())
            and all(v == 0 for v in self.out_of_range_label_counts.values())
        )


class CLINCDataset:
    """Wrapper around the HuggingFace CLINC150 dataset.

    Owns the loaded splits and derived label metadata.  Instantiate via the
    ``load`` classmethod, then query through properties and methods.
    """

    text_field: str = _TEXT_FIELD
    label_field: str = _LABEL_FIELD

    def __init__(
        self,
        config: DatasetConfig,
        dataset: DatasetDict,
        label_names: list[str],
        name_to_id: dict[str, int],
        id_to_name: dict[int, str],
    ) -> None:
        self._config = config
        self._dataset = dataset
        self._label_names = label_names
        self._name_to_id = name_to_id
        self._id_to_name = id_to_name

    @classmethod
    def load(cls, config: DatasetConfig) -> CLINCDataset:
        """Download (or load from cache) the CLINC150 dataset and build label maps."""
        dataset = load_dataset(config.name, config.subset, revision=config.revision, cache_dir=str(config.cache_dir))
        assert isinstance(dataset, DatasetDict)

        cls._validate_schema(dataset)
        cls._validate_label_space(dataset)

        class_label = dataset["train"].features[_LABEL_FIELD]
        names: list[str] = class_label.names
        name_to_id: dict[str, int] = {n: i for i, n in enumerate(names)}
        id_to_name: dict[int, str] = dict(enumerate(names))

        return cls(config, dataset, names, name_to_id, id_to_name)

    # ------------------------------------------------------------------
    # Schema / label-space validation (called during load)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_schema(dataset: DatasetDict) -> None:
        """Assert expected columns and types exist in every split."""
        for split_name in _SPLIT_NAMES:
            features = dataset[split_name].features
            assert _TEXT_FIELD in features, f"Missing '{_TEXT_FIELD}' column in {split_name}"
            assert _LABEL_FIELD in features, f"Missing '{_LABEL_FIELD}' column in {split_name}"
            assert isinstance(features[_TEXT_FIELD], Value), f"'{_TEXT_FIELD}' is not a Value feature in {split_name}"
            assert isinstance(features[_LABEL_FIELD], ClassLabel), (
                f"'{_LABEL_FIELD}' is not a ClassLabel feature in {split_name}"
            )

    @staticmethod
    def _validate_label_space(dataset: DatasetDict) -> None:
        """Assert all splits share the same ClassLabel names."""
        train_names = dataset["train"].features[_LABEL_FIELD].names
        for split_name in ("validation", "test"):
            split_names = dataset[split_name].features[_LABEL_FIELD].names
            assert split_names == train_names, (
                f"Label names in {split_name} differ from train: "
                f"train has {len(train_names)}, {split_name} has {len(split_names)}"
            )

    # ------------------------------------------------------------------
    # Data quality validation (on demand)
    # ------------------------------------------------------------------

    def validate(self) -> ValidationReport:
        """Check for null texts, empty texts, and out-of-range labels."""
        null_texts: dict[str, int] = {}
        empty_texts: dict[str, int] = {}
        bad_labels: dict[str, int] = {}
        for split_name in _SPLIT_NAMES:
            split = self._dataset[split_name]
            texts = split[_TEXT_FIELD]
            labels = split[_LABEL_FIELD]
            null_texts[split_name] = sum(1 for t in texts if t is None)
            empty_texts[split_name] = sum(1 for t in texts if t is not None and t.strip() == "")
            bad_labels[split_name] = sum(1 for lbl in labels if lbl < 0 or lbl >= self.num_classes)
        return ValidationReport(null_texts, empty_texts, bad_labels)

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------

    @property
    def source(self) -> str:
        """Full HuggingFace identifier (e.g. 'clinc/clinc_oos')."""
        return self._config.name

    @property
    def subset(self) -> str:
        return self._config.subset

    @property
    def column_names(self) -> list[str]:
        return self._dataset["train"].column_names

    @property
    def feature_types(self) -> dict[str, str]:
        """Mapping of column name to human-readable feature type string."""
        return {name: str(feat) for name, feat in self._dataset["train"].features.items()}

    # ------------------------------------------------------------------
    # Label metadata
    # ------------------------------------------------------------------

    @property
    def label_names(self) -> list[str]:
        """Ordered list of all label names (length 151)."""
        return list(self._label_names)

    @property
    def num_classes(self) -> int:
        return len(self._label_names)

    def label_name(self, label_id: int) -> str:
        return self._id_to_name[label_id]

    def label_id(self, name: str) -> int:
        return self._name_to_id[name]

    # ------------------------------------------------------------------
    # Split-level queries
    # ------------------------------------------------------------------

    def split_sizes(self) -> dict[str, int]:
        return {s: len(self._dataset[s]) for s in _SPLIT_NAMES}

    def class_distribution(self, split: str) -> Counter[int]:
        return Counter(self._dataset[split][_LABEL_FIELD])

    def sample_examples(self, split: str, label_id: int, n: int = 5) -> list[str]:
        """Return up to *n* example texts for *label_id* from *split*."""
        filtered = self._dataset[split].filter(lambda row: row[_LABEL_FIELD] == label_id)
        return filtered[_TEXT_FIELD][:n]

    def oos_counts(self) -> dict[str, int]:
        """Number of OOS examples in each split."""
        return {s: self.class_distribution(s)[OOS_LABEL_ID] for s in _SPLIT_NAMES}

    def in_scope_counts(self) -> dict[str, int]:
        """Number of in-scope (non-OOS) examples in each split."""
        sizes = self.split_sizes()
        oos = self.oos_counts()
        return {s: sizes[s] - oos[s] for s in _SPLIT_NAMES}

    # ------------------------------------------------------------------
    # Escape hatches
    # ------------------------------------------------------------------

    def to_dataframe(self, split: str) -> pd.DataFrame:
        """Convert a split to a pandas DataFrame for ad-hoc analysis."""
        return self._dataset[split].to_pandas()

    def __getitem__(self, split: str) -> Dataset:
        """Direct access to the raw HuggingFace Dataset for a split."""
        return self._dataset[split]


# ======================================================================
# PyTorch Dataset wrappers (Step 3)
# ======================================================================


class IntentDataset(TorchDataset):
    """PyTorch Dataset wrapping padded token-id sequences and labels."""

    def __init__(self, sequences: torch.Tensor, labels: torch.Tensor) -> None:
        assert sequences.shape[0] == labels.shape[0], "sequence/label count mismatch"
        self._sequences = sequences
        self._labels = labels

    def __len__(self) -> int:
        return self._sequences.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._sequences[idx], self._labels[idx]


class TFIDFDataset(TorchDataset):
    """PyTorch Dataset wrapping TF-IDF sparse matrix rows and labels."""

    def __init__(self, features: spmatrix, labels: torch.Tensor) -> None:
        assert features.shape[0] == labels.shape[0], "feature/label count mismatch"
        self._features = features
        self._labels = labels

    def __len__(self) -> int:
        return self._features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self._features[idx]
        if issparse(row):
            row = row.toarray()
        return torch.tensor(row, dtype=torch.float32).squeeze(0), self._labels[idx]


def create_dataloaders(
    datasets: dict[str, TorchDataset],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    random_seed: int = 42,
) -> dict[str, DataLoader]:
    """Create DataLoaders for each split. Only the train loader shuffles."""
    loaders: dict[str, DataLoader] = {}
    for split, ds in datasets.items():
        shuffle = split == "train"
        generator = torch.Generator().manual_seed(random_seed) if shuffle else None
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
        )
    return loaders
