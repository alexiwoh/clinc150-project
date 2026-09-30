"""Current-generation identities for cross-model artifact regression fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.enums import ModelID


def write_current_provenance(root: Path, model_id: ModelID, *, dataset_digest: str = "a" * 64) -> None:
    """Give every completed fixture run a matching, explicitly recorded identity."""
    digest = "a" * 64
    provenance: dict[str, Any] = {
        "source": {"git_commit": "fixture", "source_dirty": False, "files_sha256": {"src/train.py": digest}},
        "dataset": {
            "name": "synthetic",
            "subset": "fixture",
            "revision": "pinned",
            "label_order_sha256": digest,
            "splits": {
                split: {
                    "count": 200,
                    "texts_sha256": digest,
                    "labels_sha256": digest,
                    "examples_sha256": dataset_digest,
                }
                for split in ("train", "validation", "test")
            },
        },
        "preprocessing_artifact_hashes": {"label_order": digest},
        "model_input_hashes": {split: digest for split in ("train", "validation", "test")},
        "frozen_config_hash": digest,
    }
    directory = root / "outputs" / model_id
    ledger = json.loads((directory / "aggregate/run_ledger.json").read_text())
    for run_id in ledger["completed_run_ids"]:
        metadata_path = directory / "final_runs" / run_id / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["provenance"] = provenance
        metadata_path.write_text(json.dumps(metadata))
