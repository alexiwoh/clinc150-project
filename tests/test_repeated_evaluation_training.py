"""Tiny offline CPU regressions across the real repeated-evaluation training paths."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from src.config import (
    BaseModelConfig,
    BiLSTMConfig,
    FrozenModelConfig,
    MLPBaselineConfig,
    RepeatedRunProtocol,
    TextCNNConfig,
)
from src.constants import (
    LABEL_ORDER_REF,
    NUM_CLASSES,
    OOS_LABEL_ID,
    PREPROCESSING_MANIFEST_REF,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
)
from src.dataset import create_dataloaders
from src.enums import ModelID
from src.repeated_evaluation import execute_single_run, run_repeated_evaluation, save_evaluation_protocol
from src.trainers.trainer import Trainer


@dataclass(frozen=True)
class InitializationSignature:
    weights_sha256: str
    torch_seed: int
    config_seed: int
    config_loader_seed: int
    loader_seed: int


def _tiny_frozen_config(model_id: ModelID) -> FrozenModelConfig:
    common: dict[str, Any] = {"max_epochs": 1, "batch_size": 4, "dropout_rate": 0.2}
    config: BaseModelConfig
    if model_id is ModelID.MLP:
        config = MLPBaselineConfig(hidden_dim=8, **common)
    elif model_id is ModelID.TEXT_CNN:
        config = TextCNNConfig(
            vocab_size=20, embedding_dim=4, num_filters=3, kernel_sizes=(2, 3), max_seq_length=6, **common
        )
    else:
        config = BiLSTMConfig(vocab_size=20, embedding_dim=4, hidden_dim=4, max_seq_length=6, **common)
    return FrozenModelConfig(
        schema_version=SCHEMA_VERSION,
        protocol_version=PROTOCOL_VERSION,
        model_id=model_id,
        model_name=model_id.display_name,
        hyperparameters=config.to_dict(),
        source_tuning_artifact="synthetic-test-fixture",
        winning_row_id="tiny-cpu-regression",
        selection_metric="best_val_metric",
        selection_value=0.5,
        vocab_size=None if model_id is ModelID.MLP else 20,
        max_seq_length=None if model_id is ModelID.MLP else 6,
        monitor_metric="val_macro_f1",
        preprocessing_manifest_ref=PREPROCESSING_MANIFEST_REF,
        label_order_ref=LABEL_ORDER_REF,
        oos_strategy="explicit_class",
        oos_class_id=OOS_LABEL_ID,
    )


@pytest.mark.parametrize("model_id", list(ModelID))
def test_real_training_uses_run_seeds_and_repeats_on_cpu(model_id: ModelID, tmp_path: Path) -> None:
    """Seeds 42/1337/42 change initialization, then exactly repeat each architecture."""
    frozen = _tiny_frozen_config(model_id)
    frozen_before = json.dumps(frozen.to_dict(), sort_keys=True)
    generator = torch.Generator().manual_seed(5)
    inputs = (
        torch.randn(12, 6, generator=generator)
        if model_id is ModelID.MLP
        else torch.randint(1, 20, (12, 6), generator=generator)
    )
    targets = torch.tensor([0, 1, 2, OOS_LABEL_ID] * 3)
    dataset = TensorDataset(inputs, targets)
    label_names = [f"intent_{index}" for index in range(NUM_CLASSES)]
    label_names[OOS_LABEL_ID] = "oos"
    data_bundle: dict[str, Any] = {
        "loaders": create_dataloaders(
            {split: dataset for split in ("train", "validation", "test")}, batch_size=4, num_workers=0, pin_memory=False
        ),
        "num_classes": len(label_names),
        "input_dim": 6,
        "metadata": {"artifact_refs": {}, "label_names": label_names},
    }
    signatures: list[InitializationSignature] = []
    probabilities: list[np.ndarray] = []
    original_fit = Trainer.fit

    def capture_fit(trainer: Trainer, *args: Any, **kwargs: Any) -> dict[str, Any]:
        loader_generator = kwargs["train_loader"].generator
        assert loader_generator is not None
        weights = b"".join(parameter.detach().cpu().numpy().tobytes() for parameter in trainer.model.parameters())
        signatures.append(
            InitializationSignature(
                weights_sha256=hashlib.sha256(weights).hexdigest(),
                torch_seed=torch.initial_seed(),
                config_seed=kwargs["config"]["random_seed"],
                config_loader_seed=kwargs["config"]["dataloader_seed"],
                loader_seed=loader_generator.initial_seed(),
            )
        )
        return original_fit(trainer, *args, **kwargs)

    original_thread_count = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            patch("src.train.get_device", return_value=torch.device("cpu")),
            patch("src.repeated_evaluation.get_device", return_value=torch.device("cpu")),
            patch.object(Trainer, "fit", capture_fit),
        ):
            for index, seed in enumerate((42, 1337, 42), start=1):
                run_dir = tmp_path / f"run_{index}"
                result = execute_single_run(
                    model_id,
                    frozen,
                    seed,
                    index,
                    RepeatedRunProtocol(),
                    run_dir,
                    data_bundle,
                    {"test": [f"example {row}" for row in range(len(targets))]},
                )
                assert result.status == "completed", result.failure_reason
                assert result.best_epoch == 1
                assert result.training_seed == seed
                assert result.dataloader_seed == seed + 1
                metadata = json.loads((run_dir / "run_metadata.json").read_text())
                assert metadata["training_seed"] == seed
                assert metadata["dataloader_seed"] == seed + 1
                assert metadata["probability_saving_policy"] == "all_runs"
                assert (run_dir / "confusion_matrix.csv").exists()
                with np.load(run_dir / "confidences.npz", allow_pickle=False) as saved:
                    probabilities.append(saved["probabilities"].copy())
                checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
                assert checkpoint["config"]["random_seed"] == seed
                assert checkpoint["config"]["dataloader_seed"] == seed + 1
    finally:
        torch.set_num_threads(original_thread_count)

    assert len(signatures) == 3
    assert signatures[0] == signatures[2]
    assert signatures[0].weights_sha256 != signatures[1].weights_sha256
    assert [signature.torch_seed for signature in signatures] == [42, 1337, 42]
    assert [signature.config_seed for signature in signatures] == [42, 1337, 42]
    assert [signature.config_loader_seed for signature in signatures] == [43, 1338, 43]
    assert [signature.loader_seed for signature in signatures] == [43, 1338, 43]
    np.testing.assert_array_equal(probabilities[0], probabilities[2])
    assert not np.array_equal(probabilities[0], probabilities[1])
    assert json.dumps(frozen.to_dict(), sort_keys=True) == frozen_before


def test_invalid_frozen_settings_fail_before_output_creation(tmp_path: Path) -> None:
    frozen = _tiny_frozen_config(ModelID.MLP)
    frozen.hyperparameters["optimizer"] = "sgd"
    with patch("src.repeated_evaluation.model_output_dir", return_value=tmp_path / "mlp"):
        with pytest.raises(ValueError, match="optimizer"):
            run_repeated_evaluation(ModelID.MLP, frozen, RepeatedRunProtocol())
    assert not (tmp_path / "mlp").exists()


def test_saved_protocol_describes_effective_seed_and_artifact_contract(tmp_path: Path) -> None:
    with patch("src.repeated_evaluation.SHARED_DIR", tmp_path):
        protocol_path = save_evaluation_protocol(RepeatedRunProtocol(), list(ModelID))
    protocol = json.loads(protocol_path.read_text())
    assert protocol["protocol_config"]["probability_saving_policy"] == "all_runs"
    assert protocol["seed_policy"]["derivation_rule"] == "training_seed = seed, dataloader_seed = seed + 1"
    assert "random_seed = seed" in protocol["seed_policy"]["report_note"]
    assert "representative run selected separately" in protocol["final_model_rule"]
