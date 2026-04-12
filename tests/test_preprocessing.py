"""Tests for the Step 3 preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.sparse import issparse

from src.config import PreprocessingConfig
from src.dataset import IntentDataset, TFIDFDataset, create_dataloaders
from src.preprocessing import (
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
    Vocabulary,
    build_tfidf_features,
    build_vocab,
    clean_text,
    compute_oov_stats,
    compute_sequence_length_stats,
    compute_truncation_stats,
    export_preprocessing_artifacts,
    load_preprocessing_artifacts,
    numericalize,
    pad_or_truncate,
    run_preprocessing,
    tokenize_text,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def sample_texts() -> list[str]:
    return [
        "What is the weather like today?",
        "Book me a flight to Paris",
        "  Hello   world  ",
        "I'd like to check my balance",
        "Set a timer for 10 minutes",
    ]


@pytest.fixture()
def sample_tokenized() -> list[list[str]]:
    return [
        ["what", "is", "the", "weather", "like", "today?"],
        ["book", "me", "a", "flight", "to", "paris"],
        ["hello", "world"],
        ["i'd", "like", "to", "check", "my", "balance"],
        ["set", "a", "timer", "for", "10", "minutes"],
    ]


@pytest.fixture()
def vocab(sample_tokenized: list[list[str]]) -> Vocabulary:
    return build_vocab(sample_tokenized)


@pytest.fixture()
def config() -> PreprocessingConfig:
    return PreprocessingConfig(max_seq_length=8, batch_size=2)


# ======================================================================
# A. clean_text
# ======================================================================


class TestCleanText:
    def test_strips_whitespace(self) -> None:
        assert clean_text("  hello  ") == "hello"

    def test_normalizes_internal_whitespace(self) -> None:
        assert clean_text("hello   world") == "hello world"

    def test_lowercases_by_default(self) -> None:
        assert clean_text("Hello World") == "hello world"

    def test_no_lowercase_when_disabled(self) -> None:
        assert clean_text("Hello World", lowercase=False) == "Hello World"

    def test_preserves_punctuation(self) -> None:
        assert clean_text("what's up?") == "what's up?"

    def test_preserves_digits(self) -> None:
        assert clean_text("Set timer for 10 min") == "set timer for 10 min"

    def test_preserves_contractions(self) -> None:
        assert clean_text("I'd like to go") == "i'd like to go"

    def test_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_whitespace_only(self) -> None:
        assert clean_text("   ") == ""


# ======================================================================
# C. tokenize_text
# ======================================================================


class TestTokenizeText:
    def test_basic_split(self) -> None:
        assert tokenize_text("hello world") == ["hello", "world"]

    def test_empty_string(self) -> None:
        assert tokenize_text("") == []

    def test_single_token(self) -> None:
        assert tokenize_text("hello") == ["hello"]

    def test_preserves_punctuation_attached(self) -> None:
        assert tokenize_text("today?") == ["today?"]


# ======================================================================
# D. build_vocab
# ======================================================================


class TestBuildVocab:
    def test_pad_and_unk_ids_fixed(self, vocab: Vocabulary) -> None:
        assert vocab.token_to_id[PAD_TOKEN] == PAD_ID
        assert vocab.token_to_id[UNK_TOKEN] == UNK_ID
        assert vocab.pad_id == PAD_ID
        assert vocab.unk_id == UNK_ID

    def test_all_training_tokens_present(self, sample_tokenized: list[list[str]], vocab: Vocabulary) -> None:
        for tokens in sample_tokenized:
            for tok in tokens:
                assert tok in vocab.token_to_id

    def test_deterministic_ordering(self, sample_tokenized: list[list[str]]) -> None:
        v1 = build_vocab(sample_tokenized)
        v2 = build_vocab(sample_tokenized)
        assert v1.token_to_id == v2.token_to_id
        assert v1.id_to_token == v2.id_to_token

    def test_frequency_ordering(self) -> None:
        texts = [["a", "b", "a"], ["a", "c"]]
        v = build_vocab(texts)
        assert v.token_to_id["a"] < v.token_to_id["b"]
        assert v.token_to_id["a"] < v.token_to_id["c"]

    def test_tie_breaking_alphabetical(self) -> None:
        texts = [["banana", "apple"]]
        v = build_vocab(texts)
        assert v.token_to_id["apple"] < v.token_to_id["banana"]

    def test_min_freq_filters(self) -> None:
        texts = [["a", "b", "a", "c", "a", "b"]]
        v = build_vocab(texts, min_freq=2)
        assert "a" in v.token_to_id
        assert "b" in v.token_to_id
        assert "c" not in v.token_to_id

    def test_round_trip(self, vocab: Vocabulary) -> None:
        for tok, idx in vocab.token_to_id.items():
            assert vocab.id_to_token[idx] == tok

    def test_size_property(self, sample_tokenized: list[list[str]], vocab: Vocabulary) -> None:
        unique_tokens = set()
        for tokens in sample_tokenized:
            unique_tokens.update(tokens)
        assert vocab.size == len(unique_tokens) + 2  # +2 for PAD, UNK


# ======================================================================
# E. numericalize
# ======================================================================


class TestNumericalize:
    def test_known_tokens(self, vocab: Vocabulary) -> None:
        tokens = ["what", "is", "the"]
        ids = numericalize(tokens, vocab)
        assert all(i != UNK_ID for i in ids)
        assert len(ids) == 3

    def test_unknown_tokens_map_to_unk(self, vocab: Vocabulary) -> None:
        tokens = ["xyzzy", "what"]
        ids = numericalize(tokens, vocab)
        assert ids[0] == UNK_ID
        assert ids[1] != UNK_ID

    def test_empty_sequence(self, vocab: Vocabulary) -> None:
        assert numericalize([], vocab) == []


# ======================================================================
# F-G. Sequence length stats, pad_or_truncate
# ======================================================================


class TestSequenceLengthStats:
    def test_basic_stats(self) -> None:
        lengths = [3, 5, 7, 9, 11]
        stats = compute_sequence_length_stats(lengths)
        assert stats["mean"] == 7.0
        assert stats["max"] == 11
        assert stats["min"] == 3
        assert stats["count"] == 5


class TestPadOrTruncate:
    def test_pads_short_sequence_right(self) -> None:
        result = pad_or_truncate([1, 2], 5)
        assert result == [1, 2, 0, 0, 0]

    def test_pads_short_sequence_left(self) -> None:
        result = pad_or_truncate([1, 2], 5, padding_side="left")
        assert result == [0, 0, 0, 1, 2]

    def test_truncates_long_sequence_right(self) -> None:
        result = pad_or_truncate([1, 2, 3, 4, 5], 3)
        assert result == [1, 2, 3]

    def test_truncates_long_sequence_left(self) -> None:
        result = pad_or_truncate([1, 2, 3, 4, 5], 3, truncation_side="left")
        assert result == [3, 4, 5]

    def test_exact_length_unchanged(self) -> None:
        result = pad_or_truncate([1, 2, 3], 3)
        assert result == [1, 2, 3]

    def test_empty_sequence(self) -> None:
        result = pad_or_truncate([], 3)
        assert result == [0, 0, 0]

    def test_custom_pad_id(self) -> None:
        result = pad_or_truncate([1], 3, pad_id=99)
        assert result == [1, 99, 99]


# ======================================================================
# OOV & truncation stats
# ======================================================================


class TestOOVStats:
    def test_no_oov_on_training_data(self, sample_tokenized: list[list[str]], vocab: Vocabulary) -> None:
        stats = compute_oov_stats(sample_tokenized, vocab)
        assert stats["unknown_tokens"] == 0
        assert stats["oov_rate"] == 0.0

    def test_oov_on_unseen_tokens(self, vocab: Vocabulary) -> None:
        unseen = [["xyzzy", "plugh", "what"]]
        stats = compute_oov_stats(unseen, vocab)
        assert stats["unknown_tokens"] == 2
        assert stats["total_tokens"] == 3


class TestTruncationStats:
    def test_no_truncation(self) -> None:
        stats = compute_truncation_stats([3, 5, 7], 10)
        assert stats["truncated"] == 0

    def test_some_truncation(self) -> None:
        stats = compute_truncation_stats([3, 5, 7, 12], 10)
        assert stats["truncated"] == 1
        assert stats["total_sequences"] == 4


# ======================================================================
# K. TF-IDF features
# ======================================================================


class TestBuildTfidfFeatures:
    def test_shapes_consistent(self, config: PreprocessingConfig) -> None:
        train = ["hello world", "book flight", "set timer"]
        val = ["check balance"]
        test = ["what weather", "flight status"]
        result = build_tfidf_features(train, val, test, config)
        assert result.train.shape[0] == 3
        assert result.validation.shape[0] == 1
        assert result.test.shape[0] == 2
        assert result.train.shape[1] == result.validation.shape[1] == result.test.shape[1]

    def test_feature_dim_matches_vocab_size(self, config: PreprocessingConfig) -> None:
        train = ["hello world", "book flight"]
        val = ["check balance"]
        test = ["what weather"]
        result = build_tfidf_features(train, val, test, config)
        assert result.feature_dim == result.train.shape[1]

    def test_no_nan_or_inf(self, config: PreprocessingConfig) -> None:
        train = ["hello world", "book flight"]
        val = ["check balance"]
        test = ["what weather"]
        result = build_tfidf_features(train, val, test, config)
        for mat in [result.train, result.validation, result.test]:
            arr = mat.toarray()
            assert not np.isnan(arr).any()
            assert not np.isinf(arr).any()


# ======================================================================
# I. IntentDataset
# ======================================================================


class TestIntentDataset:
    def test_len(self) -> None:
        seqs = torch.randint(0, 100, (10, 8))
        labels = torch.randint(0, 5, (10,))
        ds = IntentDataset(seqs, labels)
        assert len(ds) == 10

    def test_getitem(self) -> None:
        seqs = torch.randint(0, 100, (5, 8))
        labels = torch.randint(0, 5, (5,))
        ds = IntentDataset(seqs, labels)
        x, y = ds[0]
        assert x.shape == (8,)
        assert y.shape == ()

    def test_mismatch_raises(self) -> None:
        seqs = torch.randint(0, 100, (5, 8))
        labels = torch.randint(0, 5, (3,))
        with pytest.raises(AssertionError):
            IntentDataset(seqs, labels)


# ======================================================================
# L. TFIDFDataset
# ======================================================================


class TestTFIDFDataset:
    def test_len(self, config: PreprocessingConfig) -> None:
        result = build_tfidf_features(["hello world", "check balance"], ["set timer"], ["book flight"], config)
        labels = torch.tensor([0, 1])
        ds = TFIDFDataset(result.train, labels)
        assert len(ds) == 2

    def test_getitem_returns_dense(self, config: PreprocessingConfig) -> None:
        result = build_tfidf_features(["hello world", "check balance"], ["set timer"], ["book flight"], config)
        labels = torch.tensor([0, 1])
        ds = TFIDFDataset(result.train, labels)
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert x.dtype == torch.float32
        assert not issparse(x)


# ======================================================================
# J. create_dataloaders
# ======================================================================


class TestCreateDataloaders:
    def test_batch_shapes(self) -> None:
        seqs = torch.randint(0, 100, (10, 8))
        labels = torch.randint(0, 5, (10,))
        datasets = {"train": IntentDataset(seqs, labels), "validation": IntentDataset(seqs, labels)}
        loaders = create_dataloaders(datasets, batch_size=4, num_workers=0, pin_memory=False)
        assert "train" in loaders
        assert "validation" in loaders
        for loader in loaders.values():
            assert len(loader) > 0
            batch = next(iter(loader))
            assert len(batch) == 2

    def test_train_shuffles(self) -> None:
        seqs = torch.arange(100).reshape(100, 1)
        labels = torch.zeros(100, dtype=torch.long)
        ds = IntentDataset(seqs, labels)
        loaders = create_dataloaders({"train": ds}, batch_size=100, num_workers=0, pin_memory=False, random_seed=42)
        batch_x, _ = next(iter(loaders["train"]))
        assert not torch.equal(batch_x, seqs)


# ======================================================================
# O. Artifact round-trip
# ======================================================================


class TestArtifactRoundTrip:
    def test_vocab_save_load(self, vocab: Vocabulary, tmp_path: Path) -> None:
        config = PreprocessingConfig(artifacts_dir=tmp_path, max_seq_length=8)
        tfidf = build_tfidf_features(["hello world"], ["check balance"], ["set timer"], config)
        export_preprocessing_artifacts(
            vocab=vocab,
            tfidf_result=tfidf,
            seq_length_stats={"mean": 5.0},
            summary={"test": True},
            sample_examples=[],
            artifacts_dir=tmp_path,
        )
        loaded = load_preprocessing_artifacts(tmp_path)
        assert loaded["vocab"].token_to_id == vocab.token_to_id
        assert loaded["vocab"].id_to_token == vocab.id_to_token
        assert loaded["vocab"].pad_id == PAD_ID
        assert loaded["vocab"].unk_id == UNK_ID

    def test_all_files_created(self, vocab: Vocabulary, tmp_path: Path) -> None:
        config = PreprocessingConfig(artifacts_dir=tmp_path, max_seq_length=8)
        tfidf = build_tfidf_features(["hello world"], ["check balance"], ["set timer"], config)
        paths = export_preprocessing_artifacts(
            vocab=vocab,
            tfidf_result=tfidf,
            seq_length_stats={"mean": 5.0},
            summary={"test": True},
            sample_examples=[],
            artifacts_dir=tmp_path,
        )
        for name, path in paths.items():
            assert path.exists(), f"Missing artifact: {name} at {path}"


# ======================================================================
# U. Determinism
# ======================================================================


class TestDeterminism:
    def test_vocab_deterministic(self, sample_tokenized: list[list[str]]) -> None:
        v1 = build_vocab(sample_tokenized)
        v2 = build_vocab(sample_tokenized)
        assert v1.token_to_id == v2.token_to_id

    def test_full_pipeline_deterministic(self) -> None:
        texts = {
            "train": ["hello world", "book a flight", "check my balance"],
            "validation": ["set a timer"],
            "test": ["what is the weather"],
        }
        labels = {
            "train": [0, 1, 2],
            "validation": [3],
            "test": [0],
        }
        config = PreprocessingConfig(max_seq_length=8)
        r1 = run_preprocessing(texts, labels, config)
        r2 = run_preprocessing(texts, labels, config)
        assert r1.vocab.token_to_id == r2.vocab.token_to_id
        for split in ("train", "validation", "test"):
            assert torch.equal(r1.sequences[split], r2.sequences[split])
            assert torch.equal(r1.labels[split], r2.labels[split])
