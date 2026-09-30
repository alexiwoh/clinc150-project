"""Content identity for future runs, without retroactive claims about old artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix

from src.config import DatasetConfig
from src.constants import PROJECT_ROOT
from src.dataset import CLINCDataset

_SPLITS = ("train", "validation", "test")
_SOURCE_FILES = ("main.py", "pyproject.toml", "uv.lock", "requirements.txt")
_SOURCE_SUFFIXES = {".py", ".sh"}
_LINUX_MODEL_PATH = Path("/sys/devices/virtual/dmi/id/product_name")
_LINUX_PROCESSOR_PATH = Path("/proc/cpuinfo")
logger = logging.getLogger(__name__)

TIMER_BOUNDARIES: dict[str, str] = {
    "training_clock": "time.perf_counter",
    "training_includes": "epoch training, epoch validation, scheduler steps and checkpoint writes",
    "training_excludes": (
        "preprocessing, model/optimizer setup, neural smoke checks, tuning and final log serialization"
    ),
    "evaluation_clock": "time.perf_counter",
    "evaluation_includes": "loader traversal, device transfer, forward pass, softmax and CPU result collection",
    "evaluation_excludes": (
        "preprocessing, model/checkpoint loading, concatenation, metric computation and artifact writes"
    ),
    "evaluation_interpretation": "batched evaluation throughput; no controlled warmup or repeated latency trials",
}


@dataclass(frozen=True)
class SplitIdentity:
    """SHA-256 of ordered raw texts, labels and paired examples, with the split size."""

    count: int
    texts_sha256: str
    labels_sha256: str
    examples_sha256: str


@dataclass(frozen=True)
class DatasetIdentity:
    """Configured public dataset revision and the actual loaded contents."""

    name: str
    subset: str
    revision: str
    label_order_sha256: str
    splits: dict[str, SplitIdentity]


@dataclass(frozen=True)
class SourceIdentity:
    """Git state plus hashes of the actual runtime source and dependency lock."""

    git_commit: str | None
    source_dirty: bool | None
    files_sha256: dict[str, str]


@dataclass(frozen=True)
class HardwareIdentity:
    """Machine and memory identifiers; unavailable measurements are explicit nulls."""

    system: str
    system_release: str
    machine: str
    model: str | None
    processor: str | None
    memory_bytes: int | None
    logical_cpu_count: int | None


def hash_file(path: Path) -> str:
    """Hash a file without loading its complete contents into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _hash_records(records: Iterable[str | int | tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def hash_texts(texts: Sequence[str]) -> str:
    """Hash exact ordered strings, including raw whitespace and case."""
    return _hash_records(texts)


def hash_labels(labels: Sequence[int]) -> str:
    """Hash ordered class identifiers independently of the text contents."""
    return _hash_records(labels)


def hash_examples(texts: Sequence[str], labels: Sequence[int]) -> str:
    """Hash exact ordered raw examples; preserve case, whitespace and labels."""
    if len(texts) != len(labels):
        raise ValueError("Text/label counts differ while recording dataset identity")
    return _hash_records(zip(texts, labels, strict=True))


def identify_dataset(dataset: CLINCDataset, config: DatasetConfig) -> DatasetIdentity:
    """Identify every official split and the loaded label order."""
    return DatasetIdentity(
        name=config.name,
        subset=config.subset,
        revision=config.revision,
        label_order_sha256=hash_texts(dataset.label_names),
        splits={
            split: SplitIdentity(
                count=len(dataset[split]["text"]),
                texts_sha256=hash_texts(dataset[split]["text"]),
                labels_sha256=hash_labels(dataset[split]["intent"]),
                examples_sha256=hash_examples(dataset[split]["text"], dataset[split]["intent"]),
            )
            for split in _SPLITS
        },
    )


def hash_tensor(tensor: torch.Tensor) -> str:
    """Identify an ordered tensor by dtype, shape, and exact CPU contents."""
    cpu_tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{cpu_tensor.dtype}:{tuple(cpu_tensor.shape)}".encode())
    digest.update(cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def hash_tfidf_inputs(matrix: csr_matrix) -> str:
    """Identify canonical CSR float32 values consumed by the MLP dataset wrapper."""
    canonical = matrix.copy()
    canonical.sum_duplicates()
    canonical = canonical.astype(np.float32, copy=False)
    canonical.sort_indices()
    canonical.eliminate_zeros()
    digest = hashlib.sha256(f"csr_float32:{canonical.shape}".encode())
    for array in (canonical.data, canonical.indices.astype(np.int64), canonical.indptr.astype(np.int64)):
        digest.update(f"{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def identify_source(project_root: Path = PROJECT_ROOT) -> SourceIdentity:
    """Record the code actually on disk; source archives explicitly lack Git state."""
    paths = [project_root / name for name in _SOURCE_FILES]
    paths.extend(
        path
        for folder in ("src", "scripts")
        for path in (project_root / folder).rglob("*")
        if path.is_file() and path.suffix in _SOURCE_SUFFIXES
    )
    files = {path.relative_to(project_root).as_posix(): hash_file(path) for path in sorted(paths)}
    commit: str | None = None
    dirty: bool | None = None
    if (project_root / ".git").exists():
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", "src", "scripts", *_SOURCE_FILES],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        dirty = bool(status)
    return SourceIdentity(commit, dirty, files)


def _read_sysctl(key: str) -> str | None:
    try:
        result = subprocess.run(["sysctl", "-n", key], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        logger.warning("Hardware field %s unavailable: sysctl was not found", key)
        return None
    if result.returncode:
        logger.warning("Hardware field %s unavailable: %s", key, result.stderr.strip())
        return None
    return result.stdout.strip() or None


def _read_linux_hardware(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text().strip() or None
    except PermissionError as error:
        logger.warning("Hardware field %s unavailable: %s", path, error)
        return None


def _linux_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError) as error:
        logger.warning("Physical memory unavailable: %s", error)
        return None
    if pages <= 0 or page_size <= 0:
        logger.warning("Physical memory unavailable: sysconf returned %s pages of %s bytes", pages, page_size)
        return None
    return pages * page_size


def identify_hardware() -> HardwareIdentity:
    """Record macOS/Linux model, processor and physical RAM; unknown fields stay null."""
    system = platform.system()
    model: str | None = None
    processor: str | None = platform.processor() or None
    memory: int | None = None
    if system == "Darwin":
        model = _read_sysctl("hw.model")
        processor = _read_sysctl("machdep.cpu.brand_string") or processor
        memory_text = _read_sysctl("hw.memsize")
        memory = int(memory_text) if memory_text is not None else None
    elif system == "Linux":
        model = _read_linux_hardware(_LINUX_MODEL_PATH)
        cpu_info = _read_linux_hardware(_LINUX_PROCESSOR_PATH)
        if cpu_info is not None:
            processor = next(
                (line.partition(":")[2].strip() for line in cpu_info.splitlines() if line.startswith("model name")),
                processor,
            )
        memory = _linux_memory_bytes()
    return HardwareIdentity(system, platform.release(), platform.machine(), model, processor, memory, os.cpu_count())


def provenance_dict(
    dataset: DatasetIdentity,
    artifact_refs: dict[str, str],
    input_hashes: dict[str, str],
    *,
    project_root: Path = PROJECT_ROOT,
    effective_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture source, raw splits, preprocessing artifacts and transformed inputs."""
    return {
        "source": asdict(identify_source(project_root)),
        "dataset": asdict(dataset),
        "preprocessing_artifact_refs": dict(artifact_refs),
        "preprocessing_artifact_hashes": {key: hash_file(project_root / ref) for key, ref in artifact_refs.items()},
        "model_input_hashes": dict(input_hashes),
        "hardware": asdict(identify_hardware()),
        "timer_boundaries": dict(TIMER_BOUNDARIES),
        "effective_settings": dict(effective_settings) if effective_settings is not None else None,
    }


def _identity_mapping(value: Any, label: str, required: Sequence[str]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"Provenance {label} must be an object")
    missing = set(required) - value.keys()
    if missing:
        raise ValueError(f"Provenance {label} missing fields: {', '.join(sorted(missing))}")
    return value


def _validate_hashes(value: Any, label: str, required: Sequence[str] = ()) -> dict[str, str]:
    hashes = _identity_mapping(value, label, required)
    if not hashes or any(type(digest) is not str or not digest for digest in hashes.values()):
        raise ValueError(f"Provenance {label} must contain nonempty content hashes")
    return hashes


def validate_generation_identity(records: Sequence[dict[str, Any]], *, same_model: bool = True) -> None:
    """Reject mixed or disagreeing current identities; wholly historical records stay unknown.

    Effective seeds intentionally differ across repeated runs. Cross-model comparisons
    share raw data/source and overlapping preprocessing artifacts, while model-specific
    encodings and frozen hyperparameters may differ.
    """
    if not records:
        return
    raw = [record.get("provenance") for record in records]
    if all(value is None for value in raw):
        return
    if any(value is None for value in raw):
        raise ValueError("Cannot mix historical records with unknown provenance and current generation identities")

    required = ("source", "dataset", "preprocessing_artifact_hashes")
    if same_model:
        required += ("model_input_hashes", "frozen_config_hash")
    identities = [_identity_mapping(value, f"record {index}", required) for index, value in enumerate(raw)]
    for index, identity in enumerate(identities):
        label = f"record {index}"
        source = _identity_mapping(
            identity["source"], f"{label} source", ("git_commit", "source_dirty", "files_sha256")
        )
        _validate_hashes(source["files_sha256"], f"{label} source files_sha256")
        dataset = _identity_mapping(
            identity["dataset"],
            f"{label} dataset",
            ("name", "subset", "revision", "label_order_sha256", "splits"),
        )
        splits = _identity_mapping(dataset["splits"], f"{label} dataset splits", _SPLITS)
        for split in _SPLITS:
            _identity_mapping(
                splits[split], f"{label} dataset {split}", ("count", "texts_sha256", "labels_sha256", "examples_sha256")
            )
        _validate_hashes(identity["preprocessing_artifact_hashes"], f"{label} preprocessing_artifact_hashes")
        if same_model:
            _validate_hashes(identity["model_input_hashes"], f"{label} model_input_hashes", _SPLITS)
            if type(identity["frozen_config_hash"]) is not str or not identity["frozen_config_hash"]:
                raise ValueError(f"Provenance {label} frozen_config_hash must be a nonempty content hash")

    for index, identity in enumerate(identities[1:], start=1):
        fields = required if same_model else ("source", "dataset")
        for field in fields:
            if identity[field] != identities[0][field]:
                raise ValueError(f"Generation identity mismatch for {field} in record {index}")
        if not same_model:
            for previous_index, previous in enumerate(identities[:index]):
                current_hashes = identity["preprocessing_artifact_hashes"]
                previous_hashes = previous["preprocessing_artifact_hashes"]
                for key in current_hashes.keys() & previous_hashes.keys():
                    if current_hashes[key] != previous_hashes[key]:
                        raise ValueError(
                            f"Generation identity mismatch for shared preprocessing artifact {key} "
                            f"in records {previous_index} and {index}"
                        )
