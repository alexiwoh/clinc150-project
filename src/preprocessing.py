"""Text cleaning, tokenization, vocabulary building, sequence padding, and TF-IDF feature preparation."""

from __future__ import annotations

import json
import pickle
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import PreprocessingConfig
from src.constants import ARTIFACTS_DIR

# ---------------------------------------------------------------------------
# Special token constants
# ---------------------------------------------------------------------------
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1

_MULTI_SPACE = re.compile(r"\s+")
_SPLIT_NAMES = ("train", "validation", "test")


# ---------------------------------------------------------------------------
# Typed containers
# ---------------------------------------------------------------------------
class PreprocessingArtifacts(TypedDict):
    """Artifacts loaded from disk for inference-time reuse."""

    vocab: Vocabulary
    vectorizer: TfidfVectorizer
    summary: dict[str, Any]
    seq_length_stats: dict[str, Any]


@dataclass(frozen=True)
class Vocabulary:
    """Immutable vocabulary mapping tokens <-> integer ids."""

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def to_dict(self) -> dict:
        return {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Vocabulary:
        token_to_id = data["token_to_id"]
        id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)


@dataclass(frozen=True)
class TFIDFResult:
    """Container for TF-IDF feature matrices and the fitted vectorizer."""

    train: spmatrix
    validation: spmatrix
    test: spmatrix
    vectorizer: TfidfVectorizer
    vocab_size: int
    feature_dim: int


@dataclass
class PreprocessingResult:
    """All outputs of the preprocessing pipeline."""

    # Neural branch
    sequences: dict[str, torch.Tensor]
    labels: dict[str, torch.Tensor]
    vocab: Vocabulary

    # TF-IDF branch
    tfidf: TFIDFResult

    # Stats / metadata
    seq_length_stats: dict
    oov_stats: dict[str, dict]
    truncation_stats: dict[str, dict]
    config: PreprocessingConfig
    summary: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# A. Text cleaning (spec section B)
# ---------------------------------------------------------------------------
def clean_text(text: str, *, lowercase: bool = True) -> str:
    """Conservative text cleaning: strip, normalize whitespace, optional lowercase."""
    text = text.strip()
    text = _MULTI_SPACE.sub(" ", text)
    if lowercase:
        text = text.lower()
    return text


# ---------------------------------------------------------------------------
# C. Tokenization (spec section C)
# ---------------------------------------------------------------------------
def tokenize_text(text: str) -> list[str]:
    """Whitespace tokenization on already-cleaned text."""
    return text.split()


# ---------------------------------------------------------------------------
# D. Vocabulary (spec section D)
# ---------------------------------------------------------------------------
def build_vocab(tokenized_texts: list[list[str]], *, min_freq: int = 1) -> Vocabulary:
    """Build vocabulary from training tokens only.

    Deterministic ordering: descending frequency, then alphabetical for ties.
    PAD=0 and UNK=1 are always first.
    """
    freq: Counter[str] = Counter()
    for tokens in tokenized_texts:
        freq.update(tokens)

    sorted_tokens = sorted(
        (tok for tok, count in freq.items() if count >= min_freq),
        key=lambda tok: (-freq[tok], tok),
    )

    token_to_id: dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    next_id = 2
    for tok in sorted_tokens:
        if tok not in token_to_id:
            token_to_id[tok] = next_id
            next_id += 1

    id_to_token = {v: k for k, v in token_to_id.items()}
    return Vocabulary(token_to_id=token_to_id, id_to_token=id_to_token)


# ---------------------------------------------------------------------------
# E. Numericalization (spec section E)
# ---------------------------------------------------------------------------
def numericalize(tokens: list[str], vocab: Vocabulary) -> list[int]:
    """Map tokens to integer ids; unknown tokens become UNK."""
    unk = vocab.unk_id
    return [vocab.token_to_id.get(tok, unk) for tok in tokens]


# ---------------------------------------------------------------------------
# F–G. Sequence length stats, padding, truncation (spec sections F–G)
# ---------------------------------------------------------------------------
def compute_sequence_length_stats(lengths: list[int]) -> dict:
    """Compute summary statistics over token-sequence lengths."""
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    return {
        "mean": round(statistics.mean(lengths), 2),
        "median": round(statistics.median(lengths), 2),
        "p90": sorted_lengths[int(n * 0.90)] if n else 0,
        "p95": sorted_lengths[int(n * 0.95)] if n else 0,
        "max": max(lengths) if lengths else 0,
        "min": min(lengths) if lengths else 0,
        "count": n,
    }


def pad_or_truncate(
    ids: list[int],
    max_len: int,
    pad_id: int = PAD_ID,
    *,
    padding_side: str = "right",
    truncation_side: str = "right",
) -> list[int]:
    """Pad or truncate a sequence to exactly *max_len*."""
    if len(ids) > max_len:
        if truncation_side == "right":
            ids = ids[:max_len]
        else:
            ids = ids[-max_len:]

    pad_count = max_len - len(ids)
    if pad_count > 0:
        padding = [pad_id] * pad_count
        if padding_side == "right":
            ids = ids + padding
        else:
            ids = padding + ids

    return ids


# ---------------------------------------------------------------------------
# OOV / truncation stats helpers
# ---------------------------------------------------------------------------
def compute_oov_stats(tokenized_texts: list[list[str]], vocab: Vocabulary) -> dict[str, int | float]:
    """Compute OOV (unknown-token which are out of vocabulary) statistics for a set of tokenized texts."""
    total = 0
    unknown = 0
    for tokens in tokenized_texts:
        for tok in tokens:
            total += 1
            if tok not in vocab.token_to_id:
                unknown += 1
    rate = unknown / total if total > 0 else 0.0
    return {"total_tokens": total, "unknown_tokens": unknown, "oov_rate": round(rate, 6)}


def compute_truncation_stats(lengths: list[int], max_len: int) -> dict[str, int | float]:
    """Compute how many sequences exceed *max_len*."""
    n = len(lengths)
    truncated = sum(1 for length in lengths if length > max_len)
    rate = truncated / n if n > 0 else 0.0
    return {"total_sequences": n, "truncated": truncated, "truncation_rate": round(rate, 6)}


# ---------------------------------------------------------------------------
# K. TF-IDF features (spec section K)
# ---------------------------------------------------------------------------
def build_tfidf_features(
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
    config: PreprocessingConfig,
) -> TFIDFResult:
    """Fit TF-IDF on training texts only, then transform all splits.

    Lowercase is handled by the shared cleaning step, so the vectorizer's
    own lowercasing is disabled to avoid double-lowering.
    """
    vectorizer = TfidfVectorizer(
        max_features=config.tfidf_max_features,
        ngram_range=config.tfidf_ngram_range,
        lowercase=False,
        stop_words=None,
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    x_test = vectorizer.transform(test_texts)

    vocab_size = len(vectorizer.vocabulary_)
    feature_dim = x_train.shape[1]

    return TFIDFResult(
        train=x_train,
        validation=x_val,
        test=x_test,
        vectorizer=vectorizer,
        vocab_size=vocab_size,
        feature_dim=feature_dim,
    )


# ---------------------------------------------------------------------------
# O. Artifact I/O (spec section O)
# ---------------------------------------------------------------------------
def export_preprocessing_artifacts(
    vocab: Vocabulary,
    tfidf_result: TFIDFResult,
    seq_length_stats: dict,
    summary: dict,
    sample_examples: list[dict],
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict[str, Path]:
    """Save all preprocessing artifacts to disk. Returns mapping of name -> path."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    vocab_path = artifacts_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab.to_dict(), indent=2))
    paths["vocab"] = vocab_path

    stats_path = artifacts_dir / "sequence_length_stats.json"
    stats_path.write_text(json.dumps(seq_length_stats, indent=2))
    paths["sequence_length_stats"] = stats_path

    summary_json_path = artifacts_dir / "preprocessing_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2))
    paths["preprocessing_summary_json"] = summary_json_path

    summary_md_path = artifacts_dir / "preprocessing_summary.md"
    summary_md_path.write_text(_build_summary_markdown(summary))
    paths["preprocessing_summary_md"] = summary_md_path

    vectorizer_path = artifacts_dir / "tfidf_vectorizer.pkl"
    with vectorizer_path.open("wb") as f:
        pickle.dump(tfidf_result.vectorizer, f)
    paths["tfidf_vectorizer"] = vectorizer_path

    samples_path = artifacts_dir / "sample_preprocessed_examples.json"
    samples_path.write_text(json.dumps(sample_examples, indent=2))
    paths["sample_preprocessed_examples"] = samples_path

    return paths


def load_preprocessing_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> PreprocessingArtifacts:
    """Load saved preprocessing artifacts for inference-time reuse."""
    vocab_path = artifacts_dir / "vocab.json"
    vocab_data = json.loads(vocab_path.read_text())
    vocab = Vocabulary.from_dict(vocab_data)

    vectorizer_path = artifacts_dir / "tfidf_vectorizer.pkl"
    with vectorizer_path.open("rb") as f:
        vectorizer: TfidfVectorizer = pickle.load(f)  # noqa: S301

    summary_path = artifacts_dir / "preprocessing_summary.json"
    summary = json.loads(summary_path.read_text())

    stats_path = artifacts_dir / "sequence_length_stats.json"
    seq_length_stats = json.loads(stats_path.read_text())

    return {
        "vocab": vocab,
        "vectorizer": vectorizer,
        "summary": summary,
        "seq_length_stats": seq_length_stats,
    }


# ---------------------------------------------------------------------------
# R. Console summary printing
# ---------------------------------------------------------------------------
def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def print_preprocessing_summary(result: PreprocessingResult, artifact_paths: dict[str, Path]) -> None:
    """Print the full console summary required by spec section R."""
    cfg = result.config

    _print_section("Text Cleaning Policy")
    print(f"  Lowercase: {cfg.lowercase}")
    print("  Strategy: strip + normalize whitespace + optional lowercase")
    print("  Preserves: punctuation, contractions, digits")

    _print_section("Tokenizer")
    print("  Type: whitespace split")

    _print_section("Vocabulary")
    print(f"  Size: {result.vocab.size}")
    print(f"  PAD token: '{PAD_TOKEN}' (id={PAD_ID})")
    print(f"  UNK token: '{UNK_TOKEN}' (id={UNK_ID})")

    _print_section("Sequence Length Stats (training)")
    for k, v in result.seq_length_stats.items():
        print(f"  {k}: {v}")
    print(f"  Chosen max_seq_length: {cfg.max_seq_length}")

    _print_section("OOV / UNK Rates")
    for split, stats in result.oov_stats.items():
        print(f"  {split}: {stats['unknown_tokens']}/{stats['total_tokens']} ({stats['oov_rate']:.4%})")

    _print_section("Truncation Rates")
    for split, stats in result.truncation_stats.items():
        print(f"  {split}: {stats['truncated']}/{stats['total_sequences']} ({stats['truncation_rate']:.4%})")

    _print_section("Padded Tensor Shapes")
    for split in _SPLIT_NAMES:
        print(f"  {split}: sequences={list(result.sequences[split].shape)}, labels={list(result.labels[split].shape)}")

    _print_section("TF-IDF Feature Shapes")
    print(f"  Fitted vocabulary size: {result.tfidf.vocab_size}")
    print(f"  Fitted feature dimension: {result.tfidf.feature_dim}")
    print(f"  train:      {result.tfidf.train.shape}")
    print(f"  validation: {result.tfidf.validation.shape}")
    print(f"  test:       {result.tfidf.test.shape}")

    _print_section("Artifact Locations")
    for name, path in artifact_paths.items():
        print(f"  {name}: {path}")


# ---------------------------------------------------------------------------
# Markdown summary builder
# ---------------------------------------------------------------------------
def _build_summary_markdown(summary: dict) -> str:
    lines = ["# Preprocessing Summary\n"]
    for key, value in summary.items():
        if isinstance(value, dict):
            lines.append(f"## {key}\n")
            for k, v in value.items():
                lines.append(f"- **{k}**: {v}")
        else:
            lines.append(f"- **{key}**: {value}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sample example builder (spec section R — before/after examples)
# ---------------------------------------------------------------------------
def _build_sample_examples(
    raw_texts: dict[str, list[str]],
    cleaned_texts: dict[str, list[str]],
    tokenized: dict[str, list[list[str]]],
    numericalized: dict[str, list[list[int]]],
    vocab: Vocabulary,
    n: int = 3,
) -> list[dict]:
    """Build sample before/after transformations including OOV demonstrations."""
    examples: list[dict] = []
    for split in _SPLIT_NAMES:
        for i in range(min(n, len(raw_texts[split]))):
            ids = numericalized[split][i]
            has_unk = any(tok_id == UNK_ID for tok_id in ids)
            examples.append(
                {
                    "split": split,
                    "index": i,
                    "raw_text": raw_texts[split][i],
                    "cleaned_text": cleaned_texts[split][i],
                    "tokens": tokenized[split][i],
                    "token_ids": ids,
                    "contains_unk": has_unk,
                }
            )
    return examples


# ---------------------------------------------------------------------------
# Main orchestrator (spec: run_preprocessing)
# ---------------------------------------------------------------------------
def run_preprocessing(
    texts_by_split: dict[str, list[str]],
    labels_by_split: dict[str, list[int]],
    config: PreprocessingConfig,
) -> PreprocessingResult:
    """Execute the full preprocessing pipeline.

    Args:
        texts_by_split: raw text lists keyed by split name
        labels_by_split: integer label lists keyed by split name
        config: preprocessing settings
    """
    # -- Validate inputs ------------------------------------------------
    for split in _SPLIT_NAMES:
        assert split in texts_by_split, f"Missing split: {split}"
        assert split in labels_by_split, f"Missing labels for split: {split}"
        assert len(texts_by_split[split]) > 0, f"Empty split: {split}"
        assert len(texts_by_split[split]) == len(labels_by_split[split]), (
            f"Text/label count mismatch in {split}: "
            f"{len(texts_by_split[split])} texts vs {len(labels_by_split[split])} labels"
        )

    # -- A. Clean -------------------------------------------------------
    cleaned: dict[str, list[str]] = {}
    for split in _SPLIT_NAMES:
        cleaned[split] = [clean_text(t, lowercase=config.lowercase) for t in texts_by_split[split]]

    # -- C. Tokenize ----------------------------------------------------
    tokenized: dict[str, list[list[str]]] = {}
    for split in _SPLIT_NAMES:
        tokenized[split] = [tokenize_text(t) for t in cleaned[split]]

    # -- D. Build vocabulary (train only) --------------------------------
    vocab = build_vocab(tokenized["train"], min_freq=config.min_token_freq)
    assert PAD_TOKEN in vocab.token_to_id and vocab.token_to_id[PAD_TOKEN] == PAD_ID
    assert UNK_TOKEN in vocab.token_to_id and vocab.token_to_id[UNK_TOKEN] == UNK_ID

    # -- E. Numericalize ------------------------------------------------
    numericalized: dict[str, list[list[int]]] = {}
    for split in _SPLIT_NAMES:
        numericalized[split] = [numericalize(toks, vocab) for toks in tokenized[split]]

    # Validate: no invalid token ids
    max_id = vocab.size - 1
    for split in _SPLIT_NAMES:
        for seq in numericalized[split]:
            assert all(0 <= tok_id <= max_id for tok_id in seq), f"Invalid token id in {split}"

    # -- F. Sequence length stats (train only) --------------------------
    train_lengths = [len(seq) for seq in numericalized["train"]]
    seq_length_stats = compute_sequence_length_stats(train_lengths)

    # -- OOV stats ------------------------------------------------------
    oov_stats: dict[str, dict] = {}
    for split in _SPLIT_NAMES:
        oov_stats[split] = compute_oov_stats(tokenized[split], vocab)

    # -- G. Pad / truncate ----------------------------------------------
    all_lengths: dict[str, list[int]] = {}
    for split in _SPLIT_NAMES:
        all_lengths[split] = [len(seq) for seq in numericalized[split]]

    truncation_stats: dict[str, dict] = {}
    for split in _SPLIT_NAMES:
        truncation_stats[split] = compute_truncation_stats(all_lengths[split], config.max_seq_length)

    sequences: dict[str, torch.Tensor] = {}
    for split in _SPLIT_NAMES:
        padded = [
            pad_or_truncate(
                seq,
                config.max_seq_length,
                PAD_ID,
                padding_side=config.padding_side,
                truncation_side=config.truncation_side,
            )
            for seq in numericalized[split]
        ]
        t = torch.tensor(padded, dtype=torch.long)
        assert t.shape[1] == config.max_seq_length
        assert not torch.isnan(t.float()).any()
        assert not torch.isinf(t.float()).any()
        sequences[split] = t

    # -- H. Labels (reuse Step 2 mapping) --------------------------------
    labels: dict[str, torch.Tensor] = {}
    for split in _SPLIT_NAMES:
        labels[split] = torch.tensor(labels_by_split[split], dtype=torch.long)
        assert labels[split].shape[0] == sequences[split].shape[0], f"Label/sequence count mismatch in {split}"

    # -- K. TF-IDF -------------------------------------------------------
    tfidf = build_tfidf_features(cleaned["train"], cleaned["validation"], cleaned["test"], config)
    assert tfidf.train.shape[1] == tfidf.validation.shape[1] == tfidf.test.shape[1]
    for mat, split in zip([tfidf.train, tfidf.validation, tfidf.test], _SPLIT_NAMES, strict=True):
        arr = mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)
        assert not np.isnan(arr).any(), f"NaN in TF-IDF {split}"
        assert not np.isinf(arr).any(), f"Inf in TF-IDF {split}"

    # -- Build summary dict (spec section O) ----------------------------
    summary = _build_summary_dict(
        config=config,
        vocab=vocab,
        seq_length_stats=seq_length_stats,
        oov_stats=oov_stats,
        truncation_stats=truncation_stats,
        sequences=sequences,
        labels=labels,
        tfidf=tfidf,
    )

    return PreprocessingResult(
        sequences=sequences,
        labels=labels,
        vocab=vocab,
        tfidf=tfidf,
        seq_length_stats=seq_length_stats,
        oov_stats=oov_stats,
        truncation_stats=truncation_stats,
        config=config,
        summary=summary,
    )


def _build_summary_dict(
    *,
    config: PreprocessingConfig,
    vocab: Vocabulary,
    seq_length_stats: dict,
    oov_stats: dict[str, dict],
    truncation_stats: dict[str, dict],
    sequences: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    tfidf: TFIDFResult,
) -> dict:
    return {
        "text_cleaning_policy": {
            "lowercase": config.lowercase,
            "strategy": "strip + normalize whitespace + optional lowercase",
            "preserves": "punctuation, contractions, digits",
        },
        "tokenizer": "whitespace split",
        "vocabulary_size": vocab.size,
        "special_tokens": {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
        },
        "max_seq_length": config.max_seq_length,
        "sequence_length_stats": seq_length_stats,
        "oov_stats": oov_stats,
        "truncation_stats": truncation_stats,
        "tfidf_config": {
            "max_features": config.tfidf_max_features,
            "ngram_range": list(config.tfidf_ngram_range),
            "lowercase": False,
            "stop_words": None,
            "note": "Lowercase handled by shared cleaning, not duplicated by vectorizer",
        },
        "tfidf_fitted_vocab_size": tfidf.vocab_size,
        "tfidf_fitted_feature_dim": tfidf.feature_dim,
        "output_shapes": {
            split: {
                "sequences": list(sequences[split].shape),
                "labels": list(labels[split].shape),
                "tfidf": list(getattr(tfidf, split).shape),
            }
            for split in _SPLIT_NAMES
        },
        "vocab_fitted_on": "training data only",
        "tfidf_fitted_on": "training data only",
        "step2_label_mapping_artifacts": {
            "label_to_id": str(ARTIFACTS_DIR / "label_to_id.json"),
            "id_to_label": str(ARTIFACTS_DIR / "id_to_label.json"),
        },
        "artifact_output_directory": str(config.artifacts_dir),
        "inference_reuse_note": "Inference must reuse saved vocab and TF-IDF vectorizer; never re-fit.",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
