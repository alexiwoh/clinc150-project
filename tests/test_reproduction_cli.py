"""Offline frozen-config and CLI regressions with all writes confined to tmp_path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from scripts import run_model_pipeline, run_repeated_evaluation
from src.config import FrozenModelConfig, RepeatedRunProtocol
from src.constants import PROJECT_ROOT, PROTOCOL_MANIFEST_REF
from src.enums import ModelID
from src.repeated_evaluation import (
    TuningSource,
    _find_tuning_csv,
    extract_and_freeze_configs,
    extract_frozen_config,
    normalize_tuning_artifacts,
    save_frozen_config,
)


def _committed_config(model_id: ModelID) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / "outputs" / model_id / "frozen_final_config.json").read_text())


def _snapshot_files(root: Path) -> dict[Path, bytes]:
    return {file.relative_to(root): file.read_bytes() for file in root.rglob("*") if file.is_file()}


@pytest.mark.parametrize("model_id", list(ModelID))
def test_saved_frozen_config_is_readable_by_cli(model_id: ModelID, tmp_path: Path) -> None:
    """All three real committed files, including protocol_manifest_ref, round-trip."""
    stored = _committed_config(model_id)
    frozen = FrozenModelConfig.from_dict(stored)
    assert frozen.protocol_manifest_ref == PROTOCOL_MANIFEST_REF
    with (
        patch("src.repeated_evaluation.model_output_dir", side_effect=lambda model: tmp_path / model),
        patch("src.constants.model_output_dir", side_effect=lambda model: tmp_path / model),
    ):
        saved = save_frozen_config(frozen)
        loaded = run_repeated_evaluation._load_frozen_config(model_id)
    assert json.loads(saved.read_text()) == stored
    assert loaded.to_dict() == frozen.to_dict()
    assert loaded.to_model_config() == frozen.to_model_config()


def test_older_frozen_payload_without_protocol_ref_is_supported() -> None:
    stored = _committed_config(ModelID.MLP)
    stored.pop("protocol_manifest_ref")
    assert FrozenModelConfig.from_dict(stored).protocol_manifest_ref == PROTOCOL_MANIFEST_REF


def test_real_tuning_extraction_outputs_all_three_readable_configs(tmp_path: Path) -> None:
    for model_id in ModelID:
        tuning_dir = tmp_path / model_id / "tuning"
        tuning_dir.mkdir(parents=True)
        tuning_path = PROJECT_ROOT / "outputs" / model_id / "tuning" / "tuning_results.csv"
        (tuning_dir / "tuning_results.csv").write_bytes(tuning_path.read_bytes())
    with (
        patch("src.repeated_evaluation.model_output_dir", side_effect=lambda model: tmp_path / model),
        patch("src.constants.model_output_dir", side_effect=lambda model: tmp_path / model),
        patch("src.repeated_evaluation.REPORTS_DIR", tmp_path / "reports"),
        patch("src.repeated_evaluation.SHARED_DIR", tmp_path / "shared"),
    ):
        extracted = extract_and_freeze_configs(list(ModelID), RepeatedRunProtocol())
        for model_id in ModelID:
            loaded = run_repeated_evaluation._load_frozen_config(model_id)
            assert loaded.to_dict() == extracted[model_id].to_dict()
            expected = _committed_config(model_id)
            assert loaded.winning_row_id == expected["winning_row_id"]
            assert loaded.selection_value == expected["selection_value"]
            assert loaded.to_model_config().to_dict() == expected["hyperparameters"]


@pytest.mark.parametrize("entrypoint", [run_model_pipeline, run_repeated_evaluation])
@pytest.mark.parametrize("run_count", ["-1", "0", "4", "5", "1.5", "invalid"])
def test_cli_invalid_run_count_is_argparse_error(entrypoint: Any, run_count: str) -> None:
    with pytest.raises(SystemExit) as error:
        entrypoint._parse_args(["--run-count", run_count])
    assert error.value.code == 2


@pytest.mark.parametrize("entrypoint", [run_model_pipeline, run_repeated_evaluation])
@pytest.mark.parametrize("run_count", [1, 2, 3])
def test_cli_supported_run_count(entrypoint: Any, run_count: int) -> None:
    assert entrypoint._parse_args(["--run-count", str(run_count)]).run_count == run_count


def test_pipeline_does_not_advertise_or_accept_retune(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        run_model_pipeline._parse_args(["--retune"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as help_exit:
        run_model_pipeline._parse_args(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "--retune" not in help_text
    assert "--tuning-source" in help_text


@pytest.mark.parametrize("source", list(TuningSource))
def test_cli_explicit_tuning_source(source: TuningSource) -> None:
    assert run_model_pipeline._parse_args(["--tuning-source", source]).tuning_source is source


def test_cli_unknown_tuning_source_is_argparse_error() -> None:
    with pytest.raises(SystemExit) as error:
        run_model_pipeline._parse_args(["--tuning-source", "latest"])
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "2.0"),
        ("protocol_version", "unknown"),
        ("model_id", "other"),
        ("model_name", "CNN"),
        ("selection_metric", "test_macro_f1"),
        ("selection_value", float("nan")),
        ("selection_value", float("inf")),
        ("selection_value", -0.1),
        ("selection_value", 1.1),
        ("selection_value", True),
        ("source_tuning_artifact", ""),
        ("winning_row_id", ""),
        ("protocol_manifest_ref", "elsewhere.json"),
        ("preprocessing_manifest_ref", "elsewhere.json"),
        ("label_order_ref", "elsewhere.json"),
        ("oos_class_id", 42.0),
        ("vocab_size", 1),
        ("unexpected", "value"),
    ],
)
def test_malformed_frozen_metadata_rejected(field: str, value: object) -> None:
    stored = _committed_config(ModelID.MLP)
    stored[field] = value
    with pytest.raises(ValueError):
        FrozenModelConfig.from_dict(stored)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("batch_size", True),
        ("batch_size", "64"),
        ("max_epochs", 0),
        ("early_stopping_patience", 0),
        ("learning_rate", 0.0),
        ("learning_rate", float("nan")),
        ("weight_decay", -0.1),
        ("dropout_rate", 1.1),
        ("random_seed", -1),
        ("random_seed", 2**32),
        ("dataloader_seed", -1),
        ("dataloader_seed", 2**64),
        ("hidden_dim", 0),
        ("second_hidden_dim", 0),
        ("activation", "unknown"),
        ("use_class_weights", "False"),
        ("use_lr_scheduler", 1),
        ("unknown_parameter", 1),
    ],
)
def test_malformed_frozen_hyperparameters_rejected(field: str, value: object) -> None:
    stored = _committed_config(ModelID.MLP)
    stored["hyperparameters"][field] = value
    with pytest.raises(ValueError):
        FrozenModelConfig.from_dict(stored)


@pytest.mark.parametrize("field", ["hidden_dim", "dropout_rate", "monitor_metric"])
def test_missing_frozen_hyperparameters_rejected(field: str) -> None:
    stored = _committed_config(ModelID.MLP)
    stored["hyperparameters"].pop(field)
    with pytest.raises(ValueError, match="all and only"):
        FrozenModelConfig.from_dict(stored)


@pytest.mark.parametrize("kernels", [[], [0, 3], [2.5, 3], [3, 21], "[3, 4, 5]"])
def test_bad_frozen_cnn_kernel_sizes_rejected(kernels: object) -> None:
    stored = _committed_config(ModelID.TEXT_CNN)
    stored["hyperparameters"]["kernel_sizes"] = kernels
    with pytest.raises(ValueError):
        FrozenModelConfig.from_dict(stored)


@pytest.mark.parametrize("model_id", [ModelID.TEXT_CNN, ModelID.BILSTM])
@pytest.mark.parametrize("field", ["vocab_size", "max_seq_length"])
def test_neural_frozen_resolved_metadata_must_agree(model_id: ModelID, field: str) -> None:
    stored = _committed_config(model_id)
    stored[field] += 1
    with pytest.raises(ValueError, match="must match"):
        FrozenModelConfig.from_dict(stored)


def test_invalid_later_frozen_config_preserves_protocol_and_outputs(tmp_path: Path) -> None:
    for model_id in ModelID:
        config_dir = tmp_path / model_id
        config_dir.mkdir()
        stored = _committed_config(model_id)
        if model_id is ModelID.BILSTM:
            stored["hyperparameters"]["optimizer"] = "sgd"
        (config_dir / "frozen_final_config.json").write_text(json.dumps(stored))
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    (shared_dir / "evaluation_protocol.json").write_text("existing protocol must survive")
    before = _snapshot_files(tmp_path)
    with (
        patch("src.constants.model_output_dir", side_effect=lambda model: tmp_path / model),
        patch("src.repeated_evaluation.SHARED_DIR", shared_dir),
        patch.object(run_repeated_evaluation, "run_repeated_evaluation") as run,
    ):
        with pytest.raises(ValueError, match="optimizer"):
            run_repeated_evaluation.main(["--model", "all"])
    run.assert_not_called()
    assert _snapshot_files(tmp_path) == before


def _write_mlp_tuning(path: Path, score: float, hidden_dim: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run_name": f"score_{score}",
                "best_val_metric": score,
                "best_epoch": 1,
                "hidden_dim": hidden_dim,
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
            }
        ]
    ).to_csv(path, index=False)


@pytest.fixture
def tuning_locations(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "mlp" / "tuning" / "tuning_results.csv", tmp_path / "reports" / "mlp_tuning_results.csv"


def test_ambiguous_tuning_sources_fail_without_changes(tuning_locations: tuple[Path, Path], tmp_path: Path) -> None:
    canonical, legacy = tuning_locations
    _write_mlp_tuning(canonical, 0.5)
    _write_mlp_tuning(legacy, 0.9)
    before = _snapshot_files(tmp_path)
    with (
        patch("src.repeated_evaluation.model_output_dir", side_effect=lambda model: tmp_path / model),
        patch("src.repeated_evaluation.REPORTS_DIR", legacy.parent),
        patch("src.repeated_evaluation.SHARED_DIR", tmp_path / "shared"),
    ):
        with pytest.raises(ValueError, match="--tuning-source"):
            extract_and_freeze_configs([ModelID.MLP], RepeatedRunProtocol())
    assert _snapshot_files(tmp_path) == before


@pytest.mark.parametrize("source", [TuningSource.CANONICAL, TuningSource.LEGACY])
def test_explicit_tuning_source_selects_requested_winner_and_preserves_csvs(
    source: TuningSource, tuning_locations: tuple[Path, Path], tmp_path: Path
) -> None:
    canonical, legacy = tuning_locations
    _write_mlp_tuning(canonical, 0.5)
    _write_mlp_tuning(legacy, 0.9)
    csvs_before = canonical.read_bytes(), legacy.read_bytes()
    with (
        patch("src.repeated_evaluation.model_output_dir", side_effect=lambda model: tmp_path / model),
        patch("src.repeated_evaluation.REPORTS_DIR", legacy.parent),
    ):
        selected = normalize_tuning_artifacts(ModelID.MLP, source)
        frozen = extract_frozen_config(ModelID.MLP, selected)
    expected_path, expected_score = (canonical, 0.5) if source is TuningSource.CANONICAL else (legacy, 0.9)
    assert selected == expected_path
    assert frozen.selection_value == expected_score
    assert canonical.read_bytes() == csvs_before[0]
    assert legacy.read_bytes() == csvs_before[1]
    summary = json.loads((canonical.parent / "selection_summary.json").read_text())
    assert summary["source_tuning_artifact"] == str(expected_path)
    assert summary["winning_best_val_metric"] == expected_score


def test_identical_tuning_sources_auto_select_canonical(tuning_locations: tuple[Path, Path], tmp_path: Path) -> None:
    canonical, legacy = tuning_locations
    _write_mlp_tuning(canonical, 0.5)
    legacy.parent.mkdir()
    legacy.write_bytes(canonical.read_bytes())
    with (
        patch("src.repeated_evaluation.model_output_dir", side_effect=lambda model: tmp_path / model),
        patch("src.repeated_evaluation.REPORTS_DIR", legacy.parent),
    ):
        assert _find_tuning_csv(ModelID.MLP) == canonical


def test_legacy_only_tuning_copies_without_removing_source(tuning_locations: tuple[Path, Path], tmp_path: Path) -> None:
    canonical, legacy = tuning_locations
    _write_mlp_tuning(legacy, 0.9)
    with (
        patch("src.repeated_evaluation.model_output_dir", side_effect=lambda model: tmp_path / model),
        patch("src.repeated_evaluation.REPORTS_DIR", legacy.parent),
    ):
        assert normalize_tuning_artifacts(ModelID.MLP) == canonical
    assert canonical.read_bytes() == legacy.read_bytes()


def test_invalid_second_tuning_model_does_not_create_or_replace_outputs(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write_mlp_tuning(reports / "mlp_tuning_results.csv", 0.9)
    pd.DataFrame(
        [
            {
                "run_name": "bad_cnn",
                "best_val_metric": 0.8,
                "best_epoch": 1,
                "embedding_dim": -1,
                "num_filters": 3,
                "kernel_sizes": "[2, 3]",
                "dropout_rate": 0.3,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
            }
        ]
    ).to_csv(reports / "text_cnn_tuning_results.csv", index=False)
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "evaluation_protocol.json").write_text("existing protocol must survive")
    before = _snapshot_files(tmp_path)
    with (
        patch("src.repeated_evaluation.model_output_dir", side_effect=lambda model: tmp_path / model),
        patch("src.repeated_evaluation.REPORTS_DIR", reports),
        patch("src.repeated_evaluation.SHARED_DIR", shared),
    ):
        with pytest.raises(ValueError, match="embedding_dim"):
            extract_and_freeze_configs([ModelID.MLP, ModelID.TEXT_CNN], RepeatedRunProtocol())
    assert _snapshot_files(tmp_path) == before
    assert not (tmp_path / ModelID.MLP).exists()


@pytest.mark.parametrize("run_count", [-1, 0, 4])
def test_protocol_invalid_run_count_rejected(run_count: int) -> None:
    with pytest.raises(ValueError, match="run_count"):
        RepeatedRunProtocol(run_count=run_count)


@pytest.mark.parametrize("seed_list", [(42, 42, 2024), (-1, 1337, 2024), (2**32, 1337, 2024)])
def test_protocol_invalid_seed_list_rejected(seed_list: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="seed_list"):
        RepeatedRunProtocol(seed_list=seed_list)
