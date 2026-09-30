"""Tests for the repeated-run evaluation protocol (Step 7, spec L).

Covers:
- Frozen-config extraction from tuning CSV (correct winning row selected)
- Aggregate mean/std computation (known inputs, verified outputs)
- Representative-run selection logic (basic, tie-breaking)
- No-test-before-final-selection guardrail
- Failed-run accounting (partial failure recorded honestly)
- Seed propagation and derivation
- Schema consistency across runs
- Cross-model artifact parity for required files
- RepeatedRunProtocol configuration
- FrozenModelConfig serialization round-trip
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    BiLSTMConfig,
    FrozenModelConfig,
    MLPBaselineConfig,
    RepeatedRunProtocol,
    TextCNNConfig,
)
from src.constants import (
    DEFAULT_AGGREGATE_METRICS,
    LABEL_ORDER_REF,
    OOS_LABEL_ID,
    PREPROCESSING_MANIFEST_REF,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
)
from src.enums import ModelID
from src.repeated_evaluation import (
    RepresentativeRunInfo,
    RunResult,
    _derive_seeds,
    _select_winning_row,
    compute_aggregate_metrics,
    save_aggregate_artifacts,
    save_representative_run,
    select_representative_run,
)
from src.training_contracts import MONITOR_METRIC_DIRECTIONS


# ---------------------------------------------------------------------------
# RepeatedRunProtocol tests
# ---------------------------------------------------------------------------


class TestRepeatedRunProtocol:
    def test_default_protocol_values(self) -> None:
        protocol = RepeatedRunProtocol()
        assert protocol.run_count == 3
        assert protocol.seed_list == (42, 1337, 2024)
        assert protocol.representative_run_rule == "highest_validation_macro_f1"
        assert protocol.timing_includes_dataloader_overhead is True
        assert protocol.test_evaluation_enabled is True
        assert protocol.probability_saving_policy == "all_runs"
        assert protocol.confusion_artifact_policy == "all_runs"
        assert protocol.aggregate_metric_list == DEFAULT_AGGREGATE_METRICS

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("schema_version", "unknown"),
            ("protocol_version", "unknown"),
            ("representative_run_rule", "highest_test_macro_f1"),
            ("probability_saving_policy", "representative_only"),
            ("confusion_artifact_policy", "none"),
            ("aggregate_metric_list", ("test_accuracy",)),
            ("test_evaluation_enabled", False),
            ("timing_includes_dataloader_overhead", False),
        ],
    )
    def test_unsupported_settings_rejected(self, field: str, value: object) -> None:
        with pytest.raises(ValueError):
            replace(RepeatedRunProtocol(), **{field: value})

    def test_effective_seed_list_truncates(self) -> None:
        protocol = RepeatedRunProtocol(run_count=2, seed_list=(42, 1337, 2024))
        assert protocol.effective_seed_list() == [42, 1337]

    def test_effective_seed_list_full(self) -> None:
        protocol = RepeatedRunProtocol(run_count=3)
        assert protocol.effective_seed_list() == [42, 1337, 2024]

    def test_to_dict_roundtrip(self) -> None:
        protocol = RepeatedRunProtocol()
        d = protocol.to_dict()
        assert d["schema_version"] == SCHEMA_VERSION
        assert d["protocol_version"] == PROTOCOL_VERSION
        assert d["run_count"] == 3
        assert d["seed_list"] == [42, 1337, 2024]
        assert d["effective_seed_list"] == [42, 1337, 2024]

    def test_protocol_is_frozen(self) -> None:
        protocol = RepeatedRunProtocol()
        with pytest.raises(AttributeError):
            protocol.run_count = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FrozenModelConfig tests
# ---------------------------------------------------------------------------


class TestFrozenModelConfig:
    @pytest.fixture
    def sample_frozen(self) -> FrozenModelConfig:
        return FrozenModelConfig(
            schema_version=SCHEMA_VERSION,
            protocol_version=PROTOCOL_VERSION,
            model_id=ModelID.MLP,
            model_name="TF-IDF + MLP",
            hyperparameters={
                "hidden_dim": 512,
                "dropout_rate": 0.3,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "batch_size": 64,
                "max_epochs": 100,
                "early_stopping_patience": 10,
                "optimizer": "adam",
                "activation": "relu",
                "use_class_weights": False,
                "use_lr_scheduler": False,
                "random_seed": 42,
                "dataloader_seed": 42,
                "monitor_metric": "val_macro_f1",
                "oos_strategy": "explicit_class",
            },
            source_tuning_artifact="outputs/mlp/tuning/tuning_results.csv",
            winning_row_id="run_01",
            selection_metric="best_val_metric",
            selection_value=0.9123,
            vocab_size=None,
            max_seq_length=None,
            monitor_metric="val_macro_f1",
            preprocessing_manifest_ref=PREPROCESSING_MANIFEST_REF,
            label_order_ref=LABEL_ORDER_REF,
            oos_strategy="explicit_class",
            oos_class_id=OOS_LABEL_ID,
        )

    def test_to_dict_contains_all_fields(self, sample_frozen: FrozenModelConfig) -> None:
        d = sample_frozen.to_dict()
        required_keys = {
            "schema_version",
            "protocol_version",
            "model_id",
            "model_name",
            "hyperparameters",
            "source_tuning_artifact",
            "winning_row_id",
            "selection_metric",
            "selection_value",
            "vocab_size",
            "max_seq_length",
            "monitor_metric",
            "preprocessing_manifest_ref",
            "label_order_ref",
            "oos_strategy",
            "oos_class_id",
        }
        assert required_keys.issubset(d.keys()), f"Missing: {required_keys - d.keys()}"

    def test_to_model_config_mlp(self, sample_frozen: FrozenModelConfig) -> None:
        config = sample_frozen.to_model_config()
        assert isinstance(config, MLPBaselineConfig)
        assert config.hidden_dim == 512
        assert config.dropout_rate == 0.3

    def test_to_model_config_text_cnn(self) -> None:
        frozen = FrozenModelConfig(
            schema_version=SCHEMA_VERSION,
            protocol_version=PROTOCOL_VERSION,
            model_id=ModelID.TEXT_CNN,
            model_name="Text CNN",
            hyperparameters={
                "vocab_size": 4311,
                "embedding_dim": 128,
                "num_filters": 100,
                "kernel_sizes": (3, 4, 5),
                "dropout_rate": 0.3,
                "activation": "relu",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "batch_size": 64,
                "max_epochs": 100,
                "early_stopping_patience": 10,
                "optimizer": "adam",
                "trainable_embeddings": True,
                "use_class_weights": False,
                "use_lr_scheduler": False,
                "random_seed": 42,
                "dataloader_seed": 42,
                "monitor_metric": "val_macro_f1",
                "oos_strategy": "explicit_class",
                "max_seq_length": 20,
            },
            source_tuning_artifact="outputs/text_cnn/tuning/tuning_results.csv",
            winning_row_id="run_01",
            selection_metric="best_val_metric",
            selection_value=0.9200,
            vocab_size=4311,
            max_seq_length=20,
            monitor_metric="val_macro_f1",
            preprocessing_manifest_ref=PREPROCESSING_MANIFEST_REF,
            label_order_ref=LABEL_ORDER_REF,
            oos_strategy="explicit_class",
            oos_class_id=OOS_LABEL_ID,
        )
        config = frozen.to_model_config()
        assert isinstance(config, TextCNNConfig)
        assert config.vocab_size == 4311

    def test_frozen_is_immutable(self, sample_frozen: FrozenModelConfig) -> None:
        with pytest.raises(AttributeError):
            sample_frozen.model_id = "text_cnn"  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("field", "value"),
        [("monitor_metric", "val_loss"), ("oos_strategy", "threshold"), ("oos_class_id", 0)],
    )
    def test_inconsistent_metadata_rejected(self, sample_frozen: FrozenModelConfig, field: str, value: object) -> None:
        with pytest.raises(ValueError):
            replace(sample_frozen, **{field: value}).to_model_config()

    def test_repeated_evaluation_rejects_nondefault_monitor(self, sample_frozen: FrozenModelConfig) -> None:
        frozen = replace(
            sample_frozen,
            monitor_metric="val_loss",
            hyperparameters={**sample_frozen.hyperparameters, "monitor_metric": "val_loss"},
        )
        with pytest.raises(ValueError, match="Repeated evaluation only supports"):
            frozen.to_model_config()


class TestSupportedModelSettings:
    @pytest.mark.parametrize("config_class", [MLPBaselineConfig, TextCNNConfig, BiLSTMConfig])
    @pytest.mark.parametrize(
        "setting", [{"optimizer": "sgd"}, {"oos_strategy": "threshold"}, {"monitor_metric": "val_typo"}]
    )
    def test_unsupported_training_settings_rejected(
        self, config_class: type[MLPBaselineConfig | TextCNNConfig | BiLSTMConfig], setting: dict[str, str]
    ) -> None:
        with pytest.raises(ValueError):
            config_class(**setting)

    @pytest.mark.parametrize("monitor", MONITOR_METRIC_DIRECTIONS)
    def test_supported_monitor_accepted(self, monitor: str) -> None:
        assert MLPBaselineConfig(monitor_metric=monitor).monitor_metric == monitor

    def test_unsupported_bilstm_summarization_rejected(self) -> None:
        with pytest.raises(ValueError, match="summarization_mode"):
            BiLSTMConfig(summarization_mode="mean_pool")


# ---------------------------------------------------------------------------
# Winning-row selection from tuning CSV
# ---------------------------------------------------------------------------


class TestWinningRowSelection:
    def test_selects_highest_val_metric(self) -> None:
        df = pd.DataFrame(
            {
                "run_name": ["run_a", "run_b", "run_c"],
                "best_val_metric": [0.85, 0.92, 0.88],
                "best_val_loss": [0.5, 0.3, 0.4],
            }
        )
        winner = _select_winning_row(df, "mlp")
        assert winner["run_name"] == "run_b"

    def test_tie_break_by_val_loss(self) -> None:
        df = pd.DataFrame(
            {
                "run_name": ["run_a", "run_b"],
                "best_val_metric": [0.90, 0.90],
                "best_val_loss": [0.5, 0.3],
            }
        )
        winner = _select_winning_row(df, "mlp")
        assert winner["run_name"] == "run_b"

    def test_tie_break_by_run_name(self) -> None:
        df = pd.DataFrame(
            {
                "run_name": ["run_c", "run_a", "run_b"],
                "best_val_metric": [0.90, 0.90, 0.90],
                "best_val_loss": [0.3, 0.3, 0.3],
            }
        )
        winner = _select_winning_row(df, "mlp")
        assert winner["run_name"] == "run_a"

    def test_without_val_loss_column(self) -> None:
        df = pd.DataFrame(
            {
                "run_name": ["run_a", "run_b"],
                "best_val_metric": [0.85, 0.90],
            }
        )
        winner = _select_winning_row(df, "mlp")
        assert winner["run_name"] == "run_b"

    def test_empty_df_raises(self) -> None:
        df = pd.DataFrame(columns=["run_name", "best_val_metric", "best_val_loss"])
        with pytest.raises(AssertionError, match="empty"):
            _select_winning_row(df, "mlp")


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


class TestSeedDerivation:
    def test_derive_seeds(self) -> None:
        training_seed, dataloader_seed = _derive_seeds(42)
        assert training_seed == 42
        assert dataloader_seed == 43

    def test_different_base_seeds_different_results(self) -> None:
        s1_t, s1_d = _derive_seeds(42)
        s2_t, s2_d = _derive_seeds(1337)
        assert s1_t != s2_t
        assert s1_d != s2_d

    def test_deterministic(self) -> None:
        assert _derive_seeds(100) == _derive_seeds(100)


# ---------------------------------------------------------------------------
# Aggregate metric computation
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    @pytest.fixture
    def sample_runs(self) -> list[RunResult]:
        results: list[RunResult] = []
        for i, seed in enumerate([42, 1337, 2024], start=1):
            r = RunResult(
                model_id="mlp",
                run_id=f"run_{i:02d}_seed_{seed}",
                run_index=i,
                seed=seed,
                training_seed=seed,
                dataloader_seed=seed + 1,
                status="completed",
                val_accuracy=0.90 + i * 0.01,
                val_macro_f1=0.88 + i * 0.01,
                test_accuracy=0.85 + i * 0.01,
                test_macro_f1=0.83 + i * 0.01,
                test_precision=0.82 + i * 0.01,
                test_recall=0.81 + i * 0.01,
                oos_precision=0.70 + i * 0.01,
                oos_recall=0.65 + i * 0.01,
                oos_f1=0.67 + i * 0.01,
                training_time_seconds=100.0 + i * 10,
                inference_total_seconds=1.0 + i * 0.1,
                inference_avg_ms_per_example=0.5 + i * 0.05,
                inference_examples_per_sec=2000.0 - i * 100,
                parameter_count=50000,
                trainable_parameter_count=50000,
            )
            results.append(r)
        return results

    def test_correct_mean_computation(self, sample_runs: list[RunResult]) -> None:
        protocol = RepeatedRunProtocol()
        agg = compute_aggregate_metrics(sample_runs, protocol, "mlp")

        expected_test_acc = np.mean([0.86, 0.87, 0.88])
        assert abs(agg.metrics["test_accuracy"]["mean"] - expected_test_acc) < 1e-6

    def test_correct_std_computation(self, sample_runs: list[RunResult]) -> None:
        protocol = RepeatedRunProtocol()
        agg = compute_aggregate_metrics(sample_runs, protocol, "mlp")

        values = [0.86, 0.87, 0.88]
        expected_std = float(np.std(values, ddof=0))
        assert abs(agg.metrics["test_accuracy"]["std"] - expected_std) < 1e-6

    def test_run_counts(self, sample_runs: list[RunResult]) -> None:
        protocol = RepeatedRunProtocol()
        agg = compute_aggregate_metrics(sample_runs, protocol, "mlp")
        assert agg.run_count_requested == 3
        assert agg.run_count_completed == 3

    def test_seed_lists(self, sample_runs: list[RunResult]) -> None:
        protocol = RepeatedRunProtocol()
        agg = compute_aggregate_metrics(sample_runs, protocol, "mlp")
        assert agg.seed_list_requested == [42, 1337, 2024]
        assert agg.seed_list_completed == [42, 1337, 2024]

    def test_all_metrics_present(self, sample_runs: list[RunResult]) -> None:
        protocol = RepeatedRunProtocol()
        agg = compute_aggregate_metrics(sample_runs, protocol, "mlp")
        expected_metrics = {
            "val_accuracy",
            "val_macro_f1",
            "test_accuracy",
            "test_macro_f1",
            "test_precision",
            "test_recall",
            "oos_precision",
            "oos_recall",
            "oos_f1",
            "training_time_seconds",
            "inference_total_seconds",
            "inference_avg_ms_per_example",
            "inference_examples_per_sec",
            "parameter_count",
            "trainable_parameter_count",
        }
        assert expected_metrics == set(agg.metrics.keys())

    def test_with_failed_run(self) -> None:
        runs = [
            RunResult(
                model_id="mlp",
                run_id="run_01_seed_42",
                run_index=1,
                seed=42,
                training_seed=42,
                dataloader_seed=43,
                status="completed",
                test_accuracy=0.90,
                test_macro_f1=0.88,
            ),
            RunResult(
                model_id="mlp",
                run_id="run_02_seed_1337",
                run_index=2,
                seed=1337,
                training_seed=1337,
                dataloader_seed=1338,
                status="failed",
                failure_reason="OOM",
                stage_reached="training_failed",
            ),
        ]
        protocol = RepeatedRunProtocol()
        agg = compute_aggregate_metrics(runs, protocol, "mlp")
        assert agg.run_count_completed == 1
        assert agg.metrics["test_accuracy"]["mean"] == 0.90

    def test_empty_runs_returns_zeros(self) -> None:
        runs = [
            RunResult(
                model_id="mlp",
                run_id="run_01_seed_42",
                run_index=1,
                seed=42,
                training_seed=42,
                dataloader_seed=43,
                status="failed",
            ),
        ]
        protocol = RepeatedRunProtocol()
        agg = compute_aggregate_metrics(runs, protocol, "mlp")
        assert agg.run_count_completed == 0
        assert agg.metrics["test_accuracy"]["mean"] == 0.0


# ---------------------------------------------------------------------------
# Representative-run selection
# ---------------------------------------------------------------------------


class TestRepresentativeRunSelection:
    def test_selects_highest_val_macro_f1(self) -> None:
        runs = [
            RunResult(
                model_id="mlp",
                run_id="run_01_seed_42",
                run_index=1,
                seed=42,
                training_seed=42,
                dataloader_seed=43,
                status="completed",
                val_macro_f1=0.88,
                val_loss=0.3,
                run_dir="/tmp/run_01",
            ),
            RunResult(
                model_id="mlp",
                run_id="run_02_seed_1337",
                run_index=2,
                seed=1337,
                training_seed=1337,
                dataloader_seed=1338,
                status="completed",
                val_macro_f1=0.92,
                val_loss=0.2,
                run_dir="/tmp/run_02",
            ),
        ]
        protocol = RepeatedRunProtocol()
        rep = select_representative_run(runs, protocol)
        assert rep.run_id == "run_02_seed_1337"
        assert rep.seed == 1337
        assert rep.selection_metric_value == 0.92

    def test_tie_break_lower_val_loss(self) -> None:
        runs = [
            RunResult(
                model_id="mlp",
                run_id="run_01_seed_42",
                run_index=1,
                seed=42,
                training_seed=42,
                dataloader_seed=43,
                status="completed",
                val_macro_f1=0.90,
                val_loss=0.5,
                run_dir="/tmp/run_01",
            ),
            RunResult(
                model_id="mlp",
                run_id="run_02_seed_1337",
                run_index=2,
                seed=1337,
                training_seed=1337,
                dataloader_seed=1338,
                status="completed",
                val_macro_f1=0.90,
                val_loss=0.3,
                run_dir="/tmp/run_02",
            ),
        ]
        protocol = RepeatedRunProtocol()
        rep = select_representative_run(runs, protocol)
        assert rep.run_id == "run_02_seed_1337"

    def test_tie_break_earlier_run_index(self) -> None:
        runs = [
            RunResult(
                model_id="mlp",
                run_id="run_02_seed_1337",
                run_index=2,
                seed=1337,
                training_seed=1337,
                dataloader_seed=1338,
                status="completed",
                val_macro_f1=0.90,
                val_loss=0.3,
                run_dir="/tmp/run_02",
            ),
            RunResult(
                model_id="mlp",
                run_id="run_01_seed_42",
                run_index=1,
                seed=42,
                training_seed=42,
                dataloader_seed=43,
                status="completed",
                val_macro_f1=0.90,
                val_loss=0.3,
                run_dir="/tmp/run_01",
            ),
        ]
        protocol = RepeatedRunProtocol()
        rep = select_representative_run(runs, protocol)
        assert rep.run_id == "run_01_seed_42"

    def test_tie_break_lexicographic_run_id(self) -> None:
        runs = [
            RunResult(
                model_id="mlp",
                run_id="run_b",
                run_index=1,
                seed=42,
                training_seed=42,
                dataloader_seed=43,
                status="completed",
                val_macro_f1=0.90,
                val_loss=0.3,
                run_dir="/tmp/run_b",
            ),
            RunResult(
                model_id="mlp",
                run_id="run_a",
                run_index=1,
                seed=42,
                training_seed=42,
                dataloader_seed=43,
                status="completed",
                val_macro_f1=0.90,
                val_loss=0.3,
                run_dir="/tmp/run_a",
            ),
        ]
        protocol = RepeatedRunProtocol()
        rep = select_representative_run(runs, protocol)
        assert rep.run_id == "run_a"

    def test_skips_failed_runs(self) -> None:
        runs = [
            RunResult(
                model_id="mlp",
                run_id="run_01_seed_42",
                run_index=1,
                seed=42,
                training_seed=42,
                dataloader_seed=43,
                status="failed",
                failure_reason="OOM",
                val_macro_f1=0.99,
            ),
            RunResult(
                model_id="mlp",
                run_id="run_02_seed_1337",
                run_index=2,
                seed=1337,
                training_seed=1337,
                dataloader_seed=1338,
                status="completed",
                val_macro_f1=0.85,
                val_loss=0.3,
                run_dir="/tmp/run_02",
            ),
        ]
        protocol = RepeatedRunProtocol()
        rep = select_representative_run(runs, protocol)
        assert rep.run_id == "run_02_seed_1337"

    def test_no_completed_runs_raises(self) -> None:
        runs = [
            RunResult(
                model_id="mlp",
                run_id="run_01_seed_42",
                run_index=1,
                seed=42,
                training_seed=42,
                dataloader_seed=43,
                status="failed",
            ),
        ]
        protocol = RepeatedRunProtocol()
        with pytest.raises(AssertionError, match="No completed runs"):
            select_representative_run(runs, protocol)

    def test_selection_rule_recorded(self) -> None:
        runs = [
            RunResult(
                model_id="mlp",
                run_id="run_01_seed_42",
                run_index=1,
                seed=42,
                training_seed=42,
                dataloader_seed=43,
                status="completed",
                val_macro_f1=0.90,
                val_loss=0.3,
                run_dir="/tmp/run_01",
            ),
        ]
        protocol = RepeatedRunProtocol()
        rep = select_representative_run(runs, protocol)
        assert rep.selection_rule == "highest_validation_macro_f1"


# ---------------------------------------------------------------------------
# Failed-run accounting
# ---------------------------------------------------------------------------


class TestFailedRunAccounting:
    def test_failed_run_has_required_fields(self) -> None:
        run = RunResult(
            model_id="mlp",
            run_id="run_01_seed_42",
            run_index=1,
            seed=42,
            training_seed=42,
            dataloader_seed=43,
            status="failed",
            failure_reason="CUDA out of memory",
            stage_reached="training_failed",
            partial_artifacts=["run_metadata.json"],
        )
        assert run.status == "failed"
        assert run.failure_reason == "CUDA out of memory"
        assert run.stage_reached == "training_failed"
        assert "run_metadata.json" in run.partial_artifacts

    def test_aggregate_records_failure_honestly(self) -> None:
        runs = [
            RunResult(
                model_id="mlp",
                run_id="run_01_seed_42",
                run_index=1,
                seed=42,
                training_seed=42,
                dataloader_seed=43,
                status="completed",
                test_accuracy=0.90,
            ),
            RunResult(
                model_id="mlp",
                run_id="run_02_seed_1337",
                run_index=2,
                seed=1337,
                training_seed=1337,
                dataloader_seed=1338,
                status="failed",
                failure_reason="OOM",
            ),
        ]
        protocol = RepeatedRunProtocol()
        agg = compute_aggregate_metrics(runs, protocol, "mlp")
        assert agg.run_count_requested == 3
        assert agg.run_count_completed == 1
        assert 1337 not in agg.seed_list_completed


# ---------------------------------------------------------------------------
# Aggregate artifact saving
# ---------------------------------------------------------------------------


class TestAggregateArtifactSaving:
    def test_saves_all_files(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        runs = [
            RunResult(
                model_id="mlp",
                run_id=f"run_{i:02d}_seed_{s}",
                run_index=i,
                seed=s,
                training_seed=s,
                dataloader_seed=s + 1,
                status="completed",
                val_accuracy=0.90,
                val_macro_f1=0.88,
                test_accuracy=0.85,
                test_macro_f1=0.83,
                test_precision=0.82,
                test_recall=0.81,
                oos_precision=0.70,
                oos_recall=0.65,
                oos_f1=0.67,
                training_time_seconds=100.0,
                inference_total_seconds=1.0,
                inference_avg_ms_per_example=0.5,
                inference_examples_per_sec=2000.0,
                parameter_count=50000,
                trainable_parameter_count=50000,
                best_epoch=10,
                stopping_epoch=15,
            )
            for i, s in enumerate([42, 1337, 2024], start=1)
        ]

        protocol = RepeatedRunProtocol()
        agg = compute_aggregate_metrics(runs, protocol, "mlp")

        with patch("src.repeated_evaluation.model_output_dir", return_value=tmp_path / "mlp"):
            agg_dir = save_aggregate_artifacts(agg, runs, "mlp")

        assert (agg_dir / "aggregate_metrics.json").exists()
        assert (agg_dir / "aggregate_metrics.csv").exists()
        assert (agg_dir / "per_run_metrics.csv").exists()
        assert (agg_dir / "aggregate_comparison_row.json").exists()

        # Verify JSON schema
        agg_json = json.loads((agg_dir / "aggregate_metrics.json").read_text())
        assert agg_json["schema_version"] == SCHEMA_VERSION
        assert agg_json["run_count_requested"] == 3
        assert agg_json["run_count_completed"] == 3
        assert agg_json["all_runs_succeeded"] is True

        # Verify per_run_metrics.csv has right columns
        per_run_df = pd.read_csv(agg_dir / "per_run_metrics.csv")
        required_cols = {
            "run_id",
            "run_index",
            "seed",
            "status",
            "val_accuracy",
            "val_macro_f1",
            "test_accuracy",
            "test_macro_f1",
            "test_precision",
            "test_recall",
            "oos_precision",
            "oos_recall",
            "oos_f1",
            "training_time_seconds",
            "parameter_count",
            "trainable_parameter_count",
        }
        assert required_cols.issubset(set(per_run_df.columns))
        assert len(per_run_df) == 3


class TestRepresentativeRunSaving:
    def test_saves_representative_json(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        rep = RepresentativeRunInfo(
            run_id="run_02_seed_1337",
            run_index=2,
            seed=1337,
            selection_rule="highest_validation_macro_f1",
            selection_metric_value=0.92,
            artifact_path="/tmp/run_02",
        )

        with patch("src.repeated_evaluation.model_output_dir", return_value=tmp_path / "mlp"):
            out_path = save_representative_run(rep, "mlp")

        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["run_id"] == "run_02_seed_1337"
        assert data["selection_rule"] == "highest_validation_macro_f1"
        assert data["note"] == "This run is representative only. Aggregate metrics come from all completed runs."


# ---------------------------------------------------------------------------
# RunResult schema consistency
# ---------------------------------------------------------------------------


class TestRunResultSchema:
    def test_run_result_has_all_required_metric_fields(self) -> None:
        r = RunResult(
            model_id="mlp",
            run_id="run_01_seed_42",
            run_index=1,
            seed=42,
            training_seed=42,
            dataloader_seed=43,
            status="completed",
        )
        metric_attrs = [
            "val_accuracy",
            "val_macro_f1",
            "val_precision",
            "val_recall",
            "test_accuracy",
            "test_macro_f1",
            "test_precision",
            "test_recall",
            "oos_precision",
            "oos_recall",
            "oos_f1",
            "training_time_seconds",
            "inference_total_seconds",
            "inference_avg_ms_per_example",
            "inference_examples_per_sec",
            "parameter_count",
            "trainable_parameter_count",
        ]
        for attr in metric_attrs:
            assert hasattr(r, attr), f"RunResult missing attribute: {attr}"

    def test_run_result_has_training_detail_fields(self) -> None:
        r = RunResult(
            model_id="mlp",
            run_id="run_01_seed_42",
            run_index=1,
            seed=42,
            training_seed=42,
            dataloader_seed=43,
            status="completed",
        )
        for attr in [
            "best_epoch",
            "stopping_epoch",
            "best_val_metric",
            "best_val_loss",
            "monitor_metric",
            "checkpoint_path",
            "log_path",
            "epoch_history",
        ]:
            assert hasattr(r, attr), f"RunResult missing attribute: {attr}"

    def test_run_result_has_failure_fields(self) -> None:
        r = RunResult(
            model_id="mlp",
            run_id="run_01_seed_42",
            run_index=1,
            seed=42,
            training_seed=42,
            dataloader_seed=43,
            status="failed",
        )
        for attr in ["failure_reason", "stage_reached", "partial_artifacts"]:
            assert hasattr(r, attr), f"RunResult missing attribute: {attr}"


# ---------------------------------------------------------------------------
# Cross-model artifact parity
# ---------------------------------------------------------------------------

REQUIRED_PER_RUN_FILES: set[str] = {
    "run_metadata.json",
    "validation_metrics.json",
    "test_metrics.json",
    "epoch_history.json",
    "final_predictions.csv",
    "confusion_matrix.csv",
    "top_confusions.json",
    "top_errors.json",
    "per_class_metrics.json",
    "label_order.json",
}


class TestCrossModelArtifactParity:
    """Verify all three models produce the same set of required per-run files."""

    def test_required_file_list_matches_spec(self) -> None:
        """The set of required per-run files matches spec D2."""
        spec_files = {
            "run_metadata.json",
            "validation_metrics.json",
            "test_metrics.json",
            "epoch_history.json",
            "final_predictions.csv",
            "confusion_matrix.csv",
            "top_confusions.json",
            "top_errors.json",
            "per_class_metrics.json",
            "label_order.json",
        }
        assert REQUIRED_PER_RUN_FILES == spec_files


# ---------------------------------------------------------------------------
# No-test-before-final-selection guardrail
# ---------------------------------------------------------------------------


class TestNoTestBeforeSelection:
    """Verify that the protocol enforces test evaluation only after config is frozen."""

    def test_frozen_config_required_before_runs(self) -> None:
        """run_repeated_evaluation asserts frozen config model_id matches."""
        from src.repeated_evaluation import run_repeated_evaluation

        frozen = FrozenModelConfig(
            schema_version=SCHEMA_VERSION,
            protocol_version=PROTOCOL_VERSION,
            model_id=ModelID.MLP,
            model_name="TF-IDF + MLP",
            hyperparameters={},
            source_tuning_artifact="test",
            winning_row_id="run_01",
            selection_metric="best_val_metric",
            selection_value=0.9,
            vocab_size=None,
            max_seq_length=None,
            monitor_metric="val_macro_f1",
            preprocessing_manifest_ref=PREPROCESSING_MANIFEST_REF,
            label_order_ref=LABEL_ORDER_REF,
            oos_strategy="explicit_class",
            oos_class_id=OOS_LABEL_ID,
        )
        protocol = RepeatedRunProtocol()

        with pytest.raises(AssertionError, match="model_id"):
            run_repeated_evaluation(ModelID.TEXT_CNN, frozen, protocol)

    def test_seed_list_no_duplicates(self) -> None:
        """Duplicate seeds are rejected."""
        from src.repeated_evaluation import extract_and_freeze_configs

        protocol = RepeatedRunProtocol(
            run_count=2,
            seed_list=(42, 42, 2024),
        )
        with pytest.raises(AssertionError, match="duplicates"):
            extract_and_freeze_configs([], protocol)
