"""Offline provenance regressions and production data-loading identity checks."""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DATASET_CONFIG, BiLSTMConfig, DatasetConfig, MLPBaselineConfig, TextCNNConfig
from src.constants import NUM_CLASSES, OOS_LABEL_ID, OOS_LABEL_NAME
from src.dataset import CLINCDataset
from src.enums import ModelID
from src.preprocessing import build_vocab
from src.provenance import (
    HardwareIdentity,
    hash_examples,
    hash_file,
    hash_labels,
    hash_tensor,
    hash_texts,
    hash_tfidf_inputs,
    identify_dataset,
    identify_hardware,
    identify_source,
    provenance_dict,
    validate_generation_identity,
)
from src.train import load_neural_data, load_tfidf_data

_SPLITS = ("train", "validation", "test")
_REVISION = "155b9c710419136e17307b80d0a13e68cd46b4ec"


@pytest.fixture
def dataset() -> CLINCDataset:
    names = [OOS_LABEL_NAME if i == OOS_LABEL_ID else f"label_{i}" for i in range(NUM_CLASSES)]
    features = Features({"text": Value("string"), "intent": ClassLabel(names=names)})
    splits = DatasetDict(
        {
            split: Dataset.from_dict(
                {"text": ["  FIRST common  ", "Second common"], "intent": [0, NUM_CLASSES - 1]}, features=features
            )
            for split in _SPLITS
        }
    )
    return CLINCDataset(
        DATASET_CONFIG, splits, names, {name: i for i, name in enumerate(names)}, dict(enumerate(names))
    )


def test_dataset_revision_forwarded(dataset: CLINCDataset, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = replace(DATASET_CONFIG, cache_dir=tmp_path, revision=_REVISION)
    loader = Mock(return_value=DatasetDict({split: dataset[split] for split in _SPLITS}))
    monkeypatch.setattr("src.dataset.load_dataset", loader)
    loaded = CLINCDataset.load(config)
    assert DatasetConfig().revision == _REVISION
    loader.assert_called_once_with(config.name, config.subset, revision=_REVISION, cache_dir=str(tmp_path))
    assert loaded.label_names == dataset.label_names


def test_raw_content_hashes_preserve_order_case_whitespace_and_labels() -> None:
    texts, labels = ["FIRST", "second"], [0, 1]
    original = hash_examples(texts, labels)
    assert original == hash_examples(list(texts), list(labels))
    assert original != hash_examples(texts[::-1], labels[::-1])
    assert original != hash_examples(["first", "second"], labels)
    assert original != hash_examples(["FIRST ", "second"], labels)
    assert original != hash_examples(texts, labels[::-1])
    assert hash_texts(texts) != hash_texts(texts[::-1])
    assert hash_labels(labels) != hash_labels(labels[::-1])
    with pytest.raises(ValueError, match="Text/label counts differ"):
        hash_examples(texts, [0])


def test_dataset_identity_uses_example_count_and_label_order(dataset: CLINCDataset) -> None:
    identity = identify_dataset(dataset, DATASET_CONFIG)
    assert identity.revision == _REVISION
    assert identity.label_order_sha256 == hash_texts(dataset.label_names)
    for split in _SPLITS:
        raw = dataset[split]
        actual = identity.splits[split]
        assert actual.count == 2
        assert actual.texts_sha256 == hash_texts(raw["text"])
        assert actual.labels_sha256 == hash_labels(raw["intent"])
        assert actual.examples_sha256 == hash_examples(raw["text"], raw["intent"])


def test_tensor_hash_covers_values_order_dtype_shape_and_contiguity() -> None:
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    original = hash_tensor(tensor)
    assert original == hash_tensor(tensor.clone())
    assert original == hash_tensor(tensor.T.contiguous().T)
    assert original != hash_tensor(tensor.flip(0))
    assert original != hash_tensor(tensor.to(torch.int32))
    assert original != hash_tensor(tensor.reshape(4))
    assert original != hash_tensor(tensor + 1)
    assert hash_tensor(torch.ones(2, dtype=torch.bfloat16)) != hash_tensor(torch.zeros(2, dtype=torch.bfloat16))
    assert hash_tensor(torch.tensor(1)) != hash_tensor(torch.tensor(2))


def test_tfidf_hash_identifies_encoded_float32_values_without_mutating_input() -> None:
    matrix = csr_matrix(np.array([[0.1, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float64))
    original_data = matrix.data.copy()
    original = hash_tfidf_inputs(matrix)
    assert np.array_equal(matrix.data, original_data)
    assert original == hash_tfidf_inputs(matrix.astype(np.float32))
    wider_indices = matrix.copy()
    wider_indices.indices = wider_indices.indices.astype(np.int64)
    wider_indices.indptr = wider_indices.indptr.astype(np.int64)
    assert original == hash_tfidf_inputs(wider_indices)
    assert original != hash_tfidf_inputs(matrix[::-1])
    assert original != hash_tfidf_inputs(matrix + csr_matrix(np.ones(matrix.shape)))
    assert original != hash_tfidf_inputs(csr_matrix(np.zeros((3, 3))))
    duplicates = csr_matrix((np.array([0.1, 0.2, 0.0]), np.array([0, 0, 1]), np.array([0, 3])), shape=(1, 2))
    encoded = torch.tensor(duplicates.toarray(), dtype=torch.float32).numpy()
    assert hash_tfidf_inputs(duplicates) == hash_tfidf_inputs(csr_matrix(encoded))


def _source_tree(root: Path) -> None:
    for name in ("main.py", "pyproject.toml", "uv.lock", "requirements.txt"):
        (root / name).write_text(name + "\n")
    (root / "src").mkdir()
    (root / "scripts").mkdir()
    (root / "src/model.py").write_text("value = 1\n")
    (root / "scripts/bootstrap.sh").write_text("#!/bin/sh\n")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def test_source_archive_explicitly_lacks_git_identity(tmp_path: Path) -> None:
    _source_tree(tmp_path)
    identity = identify_source(tmp_path)
    assert identity.git_commit is None
    assert identity.source_dirty is None
    assert identity.files_sha256["src/model.py"] == hashlib.sha256(b"value = 1\n").hexdigest()
    assert identity.files_sha256["scripts/bootstrap.sh"] == hash_file(tmp_path / "scripts/bootstrap.sh")


def test_source_identity_detects_modified_and_untracked_runtime_code(tmp_path: Path) -> None:
    _source_tree(tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(
        tmp_path, "-c", "user.name=Provenance Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"
    )
    clean = identify_source(tmp_path)
    assert clean.git_commit == _git(tmp_path, "rev-parse", "HEAD")
    assert clean.source_dirty is False
    (tmp_path / "report.txt").write_text("unrelated output\n")
    assert identify_source(tmp_path) == clean
    (tmp_path / "src/model.py").write_text("value = 2\n")
    (tmp_path / "src/new.py").write_text("added = True\n")
    dirty = identify_source(tmp_path)
    assert dirty.git_commit == clean.git_commit
    assert dirty.source_dirty is True
    assert dirty.files_sha256["src/model.py"] != clean.files_sha256["src/model.py"]
    assert "src/new.py" in dirty.files_sha256


def test_broken_git_metadata_is_not_silently_ignored(tmp_path: Path) -> None:
    _source_tree(tmp_path)
    (tmp_path / ".git").write_text("gitdir: missing-repository\n")
    with pytest.raises(subprocess.CalledProcessError):
        identify_source(tmp_path)


def test_macos_hardware_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.provenance.platform.system", lambda: "Darwin")
    monkeypatch.setattr("src.provenance.platform.release", lambda: "25.0.0")
    monkeypatch.setattr("src.provenance.platform.machine", lambda: "arm64")
    monkeypatch.setattr("src.provenance.os.cpu_count", lambda: 10)
    values = {"hw.model": "Mac14,9", "machdep.cpu.brand_string": "Apple M2 Pro", "hw.memsize": "17179869184"}
    monkeypatch.setattr("src.provenance._read_sysctl", values.__getitem__)
    actual = identify_hardware()
    assert actual == HardwareIdentity("Darwin", "25.0.0", "arm64", "Mac14,9", "Apple M2 Pro", 17179869184, 10)


def test_unavailable_hardware_fields_are_logged_and_null(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("src.provenance.platform.system", lambda: "Darwin")
    monkeypatch.setattr("src.provenance.platform.processor", lambda: "")
    monkeypatch.setattr(
        "src.provenance.subprocess.run", lambda args, **kwargs: subprocess.CompletedProcess(args, 1, "", "unknown oid")
    )
    actual = identify_hardware()
    assert actual.model is actual.processor is actual.memory_bytes is None
    assert "Hardware field hw.model unavailable" in caplog.text
    assert "Hardware field hw.memsize unavailable" in caplog.text


def test_linux_hardware_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model, cpu = tmp_path / "model", tmp_path / "cpu"
    model.write_text("Test Server\n")
    cpu.write_text("processor : 0\nmodel name : Exact CPU Name\n")
    monkeypatch.setattr("src.provenance.platform.system", lambda: "Linux")
    monkeypatch.setattr("src.provenance._LINUX_MODEL_PATH", model)
    monkeypatch.setattr("src.provenance._LINUX_PROCESSOR_PATH", cpu)
    monkeypatch.setattr("src.provenance.os.sysconf", {"SC_PHYS_PAGES": 1024, "SC_PAGE_SIZE": 4096}.__getitem__)
    actual = identify_hardware()
    assert actual.model == "Test Server"
    assert actual.processor == "Exact CPU Name"
    assert actual.memory_bytes == 4194304


@pytest.mark.parametrize("error", [OSError("unsupported counter"), ValueError("unknown sysconf key")])
def test_linux_memory_probe_failure_is_logged_and_null(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    monkeypatch.setattr("src.provenance.platform.system", lambda: "Linux")
    monkeypatch.setattr("src.provenance._read_linux_hardware", lambda _path: None)
    monkeypatch.setattr("src.provenance.os.sysconf", Mock(side_effect=error))
    assert identify_hardware().memory_bytes is None
    assert "Physical memory unavailable" in caplog.text


@pytest.mark.parametrize("pages,page_size", [(-1, 4096), (1024, -1), (0, 4096), (1024, 0)])
def test_linux_unknown_memory_counters_are_logged_and_null(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, pages: int, page_size: int
) -> None:
    monkeypatch.setattr("src.provenance.platform.system", lambda: "Linux")
    monkeypatch.setattr("src.provenance._read_linux_hardware", lambda _path: None)
    monkeypatch.setattr("src.provenance.os.sysconf", {"SC_PHYS_PAGES": pages, "SC_PAGE_SIZE": page_size}.__getitem__)
    assert identify_hardware().memory_bytes is None
    assert "Physical memory unavailable" in caplog.text


def test_provenance_serialization_and_missing_artifact_failure(
    dataset: CLINCDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source_tree(tmp_path)
    artifact = tmp_path / "manifest.json"
    artifact.write_text('{"tokenizer":"whitespace"}')
    hardware = HardwareIdentity("Unknown", "", "unknown", None, None, None, None)
    monkeypatch.setattr("src.provenance.identify_hardware", lambda: hardware)
    result = provenance_dict(
        identify_dataset(dataset, DATASET_CONFIG),
        {"manifest": "manifest.json"},
        {"train": "encoded-input-hash"},
        project_root=tmp_path,
        effective_settings={"training_seed": 1337, "batch_size": 64, "device": "cpu"},
    )
    serialized = json.loads(json.dumps(result))
    assert serialized["source"]["git_commit"] is None
    assert serialized["dataset"]["splits"]["test"]["count"] == 2
    assert serialized["preprocessing_artifact_refs"] == {"manifest": "manifest.json"}
    assert serialized["preprocessing_artifact_hashes"]["manifest"] == hash_file(artifact)
    assert serialized["model_input_hashes"] == {"train": "encoded-input-hash"}
    assert serialized["effective_settings"]["training_seed"] == 1337
    assert serialized["timer_boundaries"]["evaluation_clock"] == "time.perf_counter"
    assert "CPU result collection" in serialized["timer_boundaries"]["evaluation_includes"]
    with pytest.raises(FileNotFoundError):
        provenance_dict(identify_dataset(dataset, DATASET_CONFIG), {"bad": "missing"}, {}, project_root=tmp_path)


@pytest.mark.parametrize("model_id", list(ModelID))
def test_production_metadata_identifies_raw_and_encoded_inputs(
    dataset: CLINCDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_id: ModelID
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    names = dataset.label_names
    vocab = build_vocab([["first", "second", "common"]])
    summary = {
        "vocabulary_size": vocab.size,
        "special_tokens": {"<PAD>": 0, "<UNK>": 1},
        "max_seq_length": 4,
        "tokenizer": "whitespace split",
        "text_cleaning_policy": {"lowercase": True},
        "timestamp": "2026-09-30T00:00:00+00:00",
        "tfidf_fitted_feature_dim": 3,
    }
    for filename, content in (
        ("label_to_id.json", {name: i for i, name in enumerate(names)}),
        ("id_to_label.json", dict(enumerate(names))),
        ("vocab.json", vocab.to_dict()),
        ("preprocessing_summary.json", summary),
        ("sequence_length_stats.json", {}),
    ):
        (artifacts / filename).write_text(json.dumps(content))
    with (artifacts / "tfidf_vectorizer.pkl").open("wb") as stream:
        pickle.dump(TfidfVectorizer().fit(["first common", "second common"]), stream)
    monkeypatch.setattr("src.train.CLINCDataset.load", lambda _config: dataset)
    monkeypatch.setattr("src.train.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.train.ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr("src.train.CHECKPOINTS_DIR", tmp_path / "checkpoints")
    if model_id is ModelID.MLP:
        loaders, _, _, metadata = load_tfidf_data(MLPBaselineConfig(batch_size=2))
    else:
        config = TextCNNConfig(max_seq_length=4) if model_id is ModelID.TEXT_CNN else BiLSTMConfig(max_seq_length=4)
        loaders, metadata = load_neural_data(config)
    assert metadata["dataset_identity"] == identify_dataset(dataset, DATASET_CONFIG)
    assert metadata["texts_by_split"] == {split: ["first common", "second common"] for split in _SPLITS}
    for split in _SPLITS:
        # Read the already encoded dataset rows; hashing never needs a test-loader traversal.
        actual_inputs = torch.stack([loaders[split].dataset[i][0] for i in range(2)])
        actual_hash = (
            hash_tfidf_inputs(csr_matrix(actual_inputs.numpy()))
            if model_id is ModelID.MLP
            else hash_tensor(actual_inputs)
        )
        assert metadata["model_input_hashes"][split] == actual_hash


@pytest.fixture
def records(dataset: CLINCDataset) -> list[dict[str, Any]]:
    identity = {
        "source": {"git_commit": None, "source_dirty": None, "files_sha256": {"src/model.py": "source-hash"}},
        "dataset": asdict(identify_dataset(dataset, DATASET_CONFIG)),
        "preprocessing_artifact_hashes": {"vocab": "vocab-hash", "label_to_id": "labels-hash"},
        "model_input_hashes": {split: f"{split}-input-hash" for split in _SPLITS},
        "frozen_config_hash": "frozen-hash",
        "effective_settings": {"training_seed": 42},
    }
    first = {"run_id": "first", "provenance": identity}
    second = deepcopy(first)
    second["run_id"] = "second"
    second["provenance"]["effective_settings"]["training_seed"] = 1337
    return [first, second]


def test_historical_identity_remains_explicitly_unknown() -> None:
    validate_generation_identity([])
    validate_generation_identity([{"run_id": "old-first"}, {"run_id": "old-second", "provenance": None}])


def test_mixed_historical_and_current_records_rejected(records: list[dict[str, Any]]) -> None:
    with pytest.raises(ValueError, match="Cannot mix historical"):
        validate_generation_identity([records[0], {"run_id": "historical"}])


def test_same_model_identity_allows_distinct_effective_seeds(records: list[dict[str, Any]]) -> None:
    validate_generation_identity(records)


@pytest.mark.parametrize("field", ["source", "dataset", "preprocessing_artifact_hashes", "model_input_hashes"])
def test_generation_identity_missing_fields_rejected(records: list[dict[str, Any]], field: str) -> None:
    del records[1]["provenance"][field]
    with pytest.raises(ValueError, match=f"missing fields: {field}"):
        validate_generation_identity(records)


@pytest.mark.parametrize(
    "field", ["source", "dataset", "preprocessing_artifact_hashes", "model_input_hashes", "frozen_config_hash"]
)
def test_same_model_identity_mismatch_rejected(records: list[dict[str, Any]], field: str) -> None:
    value = records[1]["provenance"][field]
    if field == "frozen_config_hash":
        records[1]["provenance"][field] = "different-frozen-hash"
    elif field == "source":
        value["files_sha256"]["src/model.py"] = "modified-source-hash"
    elif field == "dataset":
        value["revision"] = "changed-revision"
    else:
        value[next(iter(value))] = "different-content-hash"
    with pytest.raises(ValueError, match=f"identity mismatch for {field}"):
        validate_generation_identity(records)


@pytest.mark.parametrize("bad_value", [{}, [], "unknown", None])
def test_incomplete_or_malformed_current_identity_rejected(records: list[dict[str, Any]], bad_value: object) -> None:
    records[1]["provenance"]["source"] = bad_value
    with pytest.raises(ValueError, match="Provenance record 1 source"):
        validate_generation_identity(records)


def test_cross_model_identity_allows_model_specific_inputs_and_hyperparameters(records: list[dict[str, Any]]) -> None:
    records[1]["provenance"]["model_input_hashes"] = {"train": "neural-inputs"}
    records[1]["provenance"]["frozen_config_hash"] = "neural-config"
    records[0]["provenance"]["preprocessing_artifact_hashes"]["tfidf_vectorizer"] = "mlp-only-hash"
    validate_generation_identity(records, same_model=False)


def test_cross_model_identity_rejects_shared_artifact_mismatch(records: list[dict[str, Any]]) -> None:
    records[1]["provenance"]["preprocessing_artifact_hashes"]["vocab"] = "changed-vocab"
    with pytest.raises(ValueError, match="shared preprocessing artifact vocab"):
        validate_generation_identity(records, same_model=False)


def test_cross_model_shared_artifacts_compared_pairwise(records: list[dict[str, Any]]) -> None:
    # The first model does not use this artifact; the other two still must agree.
    third = deepcopy(records[1])
    records[1]["provenance"]["preprocessing_artifact_hashes"]["neural_only"] = "neural-hash"
    third["provenance"]["preprocessing_artifact_hashes"]["neural_only"] = "changed-neural-hash"
    with pytest.raises(ValueError, match="shared preprocessing artifact neural_only"):
        validate_generation_identity([*records, third], same_model=False)
