"""Tests for src.report.sections and src.report.figures — section generators.

Each generator is tested against small synthetic artifacts.  Verifications:
  - JSON schema compliance (required contract fields present)
  - source_artifact / source_artifacts populated and non-empty
  - aggregate vs representative scope tagging
  - heading structure (## / ### hierarchy)
  - figure catalogue: captions, scope tags, section assignments, total count
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.enums import ModelID

# ---------------------------------------------------------------------------
# Synthetic artifact builder
# ---------------------------------------------------------------------------


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def _build_all_artifacts(root: Path) -> None:
    """Create every upstream artifact consumed by section generators."""
    _write(
        root / "data/artifacts/dataset_summary.json",
        {
            "dataset_source": "clinc/clinc_oos",
            "subset": "plus",
            "split_sizes": {"train": 15250, "validation": 3100, "test": 5500},
            "num_classes": 151,
            "oos_label_name": "oos",
            "oos_label_id": 42,
            "oos_counts": {"train": 250, "validation": 100, "test": 1000},
            "in_scope_counts": {"train": 15000, "validation": 3000, "test": 4500},
            "train_class_count_min": 100,
            "train_class_count_max": 100,
            "quirks_or_caveats": ["Test has ~18% OOS vs ~1.6% in train"],
        },
    )
    _write(
        root / "data/artifacts/preprocessing_summary.json",
        {
            "text_cleaning_policy": {"strategy": "lowercase+strip", "preserves": "punctuation, contractions, digits"},
            "tokenizer": "whitespace_split",
            "vocabulary_size": 6161,
            "special_tokens": {"<PAD>": 0, "<UNK>": 1},
            "max_seq_length": 20,
            "sequence_length_stats": {"mean": 8.5, "median": 8.0, "p90": 13, "p95": 15, "max": 28, "min": 1},
            "oov_stats": {
                s: {"total_tokens": 10000, "unknown_tokens": 50, "oov_rate": 0.005}
                for s in ("train", "validation", "test")
            },
            "truncation_stats": {
                s: {"total_sequences": 5000, "truncated": 30, "truncation_rate": 0.006}
                for s in ("train", "validation", "test")
            },
            "tfidf_config": {"max_features": 10000, "ngram_range": [1, 2]},
        },
    )

    for mid in ModelID:
        hp: dict[str, Any] = {
            "hidden_dim": 512 if mid == ModelID.MLP else 256,
            "dropout_rate": 0.2 if mid == ModelID.MLP else (0.5 if mid == ModelID.TEXT_CNN else 0.3),
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 64,
            "max_epochs": 100,
            "early_stopping_patience": 10,
            "monitor_metric": "val_macro_f1",
        }
        if mid != ModelID.MLP:
            hp["embedding_dim"] = 256
        if mid == ModelID.TEXT_CNN:
            hp.update(kernel_sizes=[3, 4, 5], num_filters=100)
        if mid == ModelID.BILSTM:
            hp.update(num_layers=2, summarization_mode="concat_final_hidden", max_grad_norm=1.0)
        _write(
            root / f"outputs/{mid}/frozen_final_config.json",
            {"model_name": mid.display_name, "hyperparameters": hp},
        )

    shared = root / "outputs/shared"

    _write(
        shared / "evaluation_protocol.json",
        {
            "protocol_config": {
                "run_count": 3,
                "seed_list": [42, 1337, 2024],
                "representative_run_rule": "highest_validation_macro_f1",
                "timing_includes_dataloader_overhead": True,
            },
            "metric_definitions": {"accuracy": "correct / total", "macro_f1": "macro-averaged F1"},
            "oos_evaluation_policy": {
                "oos_class_name": "oos",
                "oos_class_id": 42,
                "evaluation_method": "one-vs-rest",
                "one_vs_rest_rule": "OOS positive, all in-scope negative",
            },
            "seed_policy": {
                "derivation_rule": "training_seed=seed, dataloader_seed=seed+1",
                "report_note": (
                    "The repeated-evaluation pipeline derives `training_seed = seed` and "
                    "`dataloader_seed = seed + 1`. `dataloader_seed` controls train-batch "
                    "shuffling, but model initialization currently depends on "
                    "`config.random_seed`, not necessarily the nominal seed."
                ),
            },
        },
    )

    def _model_row(mid: ModelID, **extra: Any) -> dict[str, Any]:
        return {"model_id": str(mid), "model_name": mid.display_name, "display_name": mid.display_name, **extra}

    _write(
        shared / "model_comparison_aggregate.json",
        {
            "rows": [
                _model_row(
                    m,
                    run_count=3,
                    test_accuracy_mean=0.85,
                    test_accuracy_std=0.01,
                    test_macro_f1_mean=0.83,
                    test_macro_f1_std=0.02,
                    test_precision_mean=0.82,
                    test_precision_std=0.015,
                    test_recall_mean=0.81,
                    test_recall_std=0.015,
                )
                for m in ModelID
            ]
        },
    )

    _write(
        shared / "oos_summary_table.json",
        {
            "rows": [
                _model_row(
                    m,
                    run_count=3,
                    oos_precision_mean=0.70,
                    oos_precision_std=0.05,
                    oos_recall_mean=0.60,
                    oos_recall_std=0.04,
                    oos_f1_mean=0.65,
                    oos_f1_std=0.03,
                )
                for m in ModelID
            ]
        },
    )

    _write(
        shared / "efficiency_summary_table.json",
        {
            "rows": [
                _model_row(
                    m,
                    run_count=3,
                    parameter_count=500000,
                    trainable_parameter_count=500000,
                    training_time_seconds_mean=120.0,
                    training_time_seconds_std=10.0,
                    inference_avg_ms_per_example_mean=0.05,
                    inference_avg_ms_per_example_std=0.01,
                    inference_examples_per_sec_mean=20000.0,
                    inference_examples_per_sec_std=2000.0,
                )
                for m in ModelID
            ]
        },
    )

    figures = [
        {
            "figure_type": "model_comparison_test_macro_f1",
            "figure_path": "outputs/shared/figures/model_comparison_test_macro_f1.png",
            "scope": "aggregate",
            "source_artifact_paths": ["outputs/shared/model_comparison_aggregate.json"],
            "caption_context": {
                "analysis_basis": "aggregate comparison across repeated evaluation runs",
                "dataset_name": "CLINC150",
                "dataset_split": "test",
                "run_count": 3,
            },
        },
        {
            "figure_type": "val_macro_f1_curve",
            "figure_path": "outputs/mlp/figures/representative_val_macro_f1_curve.png",
            "scope": "representative",
            "model_name": "mlp",
            "representative_run_id": "run_01_seed_42",
            "source_artifact_paths": ["outputs/mlp/final_runs/run_01_seed_42/epoch_history.json"],
            "caption_context": {
                "analysis_basis": "representative run",
                "dataset_name": "CLINC150",
                "dataset_split": "validation",
                "selection_rule": "highest_validation_macro_f1",
            },
        },
        {
            "figure_type": "confusion_matrix",
            "figure_path": "outputs/mlp/figures/representative_confusion_matrix.png",
            "scope": "representative",
            "model_name": "mlp",
            "representative_run_id": "run_01_seed_42",
            "source_artifact_paths": ["outputs/mlp/final_runs/run_01_seed_42/confusion_matrix.csv"],
            "caption_context": {
                "analysis_basis": "representative run",
                "class_count": 151,
                "dataset_name": "CLINC150",
                "dataset_split": "test",
            },
        },
        {
            "figure_type": "oos_roc_comparison",
            "figure_path": "outputs/shared/analysis/oos_roc_comparison.png",
            "scope": "analysis",
            "source_artifact_paths": ["outputs/shared/analysis/oos_threshold_comparison.json"],
            "caption_context": {
                "analysis_basis": "cross-model OOS ROC comparison using one representative run per model",
                "dataset_name": "CLINC150",
                "dataset_split": "test",
            },
        },
    ]
    _write(shared / "figure_manifest.json", {"figures": figures})
    for fig in figures:
        fp = root / fig["figure_path"]
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(b"PNG")

    analysis = shared / "analysis"

    _write(
        analysis / "extended_metrics_comparison.json",
        {
            "rows": [
                _model_row(m, macro_f1=0.83, micro_f1=0.85, weighted_f1=0.84, macro_precision=0.82, macro_recall=0.81)
                for m in ModelID
            ]
        },
    )

    _write(
        analysis / "calibration_summary.json",
        {
            "rows": [
                _model_row(m, ece=0.05 + i * 0.01, mce=0.15, brier_score=0.10, nll=0.50) for i, m in enumerate(ModelID)
            ]
        },
    )

    _write(
        analysis / "error_taxonomy_summary.json",
        {
            "rows": [
                {
                    "model_id": str(m),
                    **{
                        f"{cat}_count": 100 if cat == "oos_as_inscope" else 30
                        for cat in (
                            "oos_as_inscope",
                            "inscope_as_oos",
                            "near_semantic_confusion",
                            "cross_domain_confusion",
                            "short_query_ambiguity",
                        )
                    },
                    **{
                        f"{cat}_fraction": 0.4 if cat == "oos_as_inscope" else 0.12
                        for cat in (
                            "oos_as_inscope",
                            "inscope_as_oos",
                            "near_semantic_confusion",
                            "cross_domain_confusion",
                            "short_query_ambiguity",
                        )
                    },
                }
                for m in ModelID
            ]
        },
    )

    _write(
        analysis / "oos_threshold_comparison.json",
        {
            "rows": [
                _model_row(
                    m,
                    auroc=0.90 + i * 0.02,
                    aupr=0.80,
                    fpr_at_95tpr=0.20,
                    fpr_at_90tpr=0.15,
                    msp_auroc=0.70,
                    msp_aupr=0.60,
                )
                for i, m in enumerate(ModelID)
            ]
        },
    )

    _write(
        analysis / "oos_false_accept_comparison.json",
        {
            "models": {
                str(m): {
                    "false_accept_count": 50,
                    "top_capturing_intents": [
                        {"intent": "translate", "count": 10},
                        {"intent": "change_language", "count": 8},
                        {"intent": "next_song", "count": 5},
                    ],
                }
                for m in ModelID
            }
        },
    )

    _write(
        analysis / "cross_model_error_comparison.json",
        {
            "total_test_examples": 5500,
            "categories": {
                "all_correct": {"count": 4500, "fraction": 0.818},
                "all_wrong": {"count": 200, "fraction": 0.036},
                "model_specific_error": {"count": 500, "fraction": 0.091},
                "partial_error": {"count": 300, "fraction": 0.055},
            },
            "all_wrong_agreement": 0.45,
        },
    )

    _write(
        analysis / "worst_classes_comparison.json",
        {
            "shared_worst_classes": ["translate", "change_language"],
            "unique_per_model": {"mlp": ["next_song"], "text_cnn": ["play_music"], "bilstm": ["book_flight"]},
        },
    )

    _write(
        shared / "most_confused_pairs_table.json",
        {
            "rows": [
                {"model_id": "mlp", "predicted_label_name": "translate", "true_label_name": "oos", "count": 10},
                {"model_id": "text_cnn", "predicted_label_name": "translate", "true_label_name": "oos", "count": 8},
                {"model_id": "bilstm", "predicted_label_name": "play_music", "true_label_name": "oos", "count": 7},
            ]
        },
    )

    cats = ["oos_as_inscope"] * 4 + ["near_semantic_confusion"] * 4 + ["cross_domain_confusion"] * 4
    cats += ["short_query_ambiguity"] * 4 + ["inscope_as_oos"] * 2
    model_ids = [str(m) for m in ModelID]
    _write(
        analysis / "curated_report_examples.json",
        {
            "total_curated": len(cats),
            "examples": [
                {
                    "text": f"example text {i}",
                    "true_label_name": "oos" if "oos" in cat else f"label_{i}",
                    "predicted_label_name": f"predicted_{i}",
                    "model_id": model_ids[i % 3],
                    "max_confidence": 0.95 - i * 0.02,
                    "primary_category": cat,
                    "annotation_tag": f"tag_{cat}",
                }
                for i, cat in enumerate(cats)
            ],
        },
    )

    _write(
        analysis / "error_analysis_summary.json",
        {
            "top_recommendation": "Collect more OOS training data.",
            "headline": {
                "best_model": "bilstm",
                "test_macro_f1_mean": 0.83,
                "test_macro_f1_std": 0.02,
                "source_artifact": "outputs/shared/model_comparison_aggregate.json",
            },
            "calibration": {
                "best_calibrated": "TF-IDF + MLP",
                "best_ece": 0.05,
                "worst_calibrated": "BiLSTM",
                "worst_ece": 0.07,
                "source_artifact": "outputs/shared/analysis/calibration_summary.json",
            },
            "oos_detection": {
                "best_model": "bilstm",
                "best_auroc": 0.94,
                "best_aupr": 0.86,
                "source_artifact": "outputs/shared/analysis/oos_threshold_comparison.json",
            },
            "confusion": {
                "dominant_error_categories": {
                    "mlp": "oos_as_inscope",
                    "text_cnn": "oos_as_inscope",
                    "bilstm": "oos_as_inscope",
                },
                "source_artifact": "outputs/shared/analysis/error_taxonomy_summary.json",
            },
            "architecture_comparison": {
                "all_wrong_count": 200,
                "all_wrong_fraction": 0.036,
                "model_specific_count": 500,
                "model_specific_fraction": 0.091,
                "all_wrong_agreement": 0.45,
                "source_artifact": "outputs/shared/analysis/cross_model_error_comparison.json",
            },
            "length": {
                "mlp": {"short_accuracy": 0.80},
                "text_cnn": {"short_accuracy": 0.82},
                "bilstm": {"short_accuracy": 0.81},
                "source_artifact": "outputs/shared/analysis/error_analysis_summary.json",
            },
        },
    )

    (analysis / "error_analysis_notes.md").parent.mkdir(parents=True, exist_ok=True)
    (analysis / "error_analysis_notes.md").write_text("# Error Analysis Notes\n")

    (root / "uv.lock").write_text("")
    (root / "LICENSE").write_text("GNU GENERAL PUBLIC LICENSE\nVersion 3\n")


# ---------------------------------------------------------------------------
# Patching helper
# ---------------------------------------------------------------------------


@pytest.fixture()
def synth(tmp_path: Path):
    """Build synthetic artifacts and patch path constants."""
    _build_all_artifacts(tmp_path)
    with ExitStack() as stack:
        stack.enter_context(patch("src.report.artifact_loader.PROJECT_ROOT", tmp_path))
        stack.enter_context(patch("src.report.artifact_loader.SHARED_DIR", tmp_path / "outputs" / "shared"))
        stack.enter_context(
            patch("src.report.artifact_loader.ANALYSIS_DIR", tmp_path / "outputs" / "shared" / "analysis")
        )
        stack.enter_context(patch("src.report.artifact_loader.REPORT_DIR", tmp_path / "outputs" / "shared" / "report"))
        stack.enter_context(patch("src.report.sections.PROJECT_ROOT", tmp_path))
        yield tmp_path


# ---------------------------------------------------------------------------
# Section B: Abstract
# ---------------------------------------------------------------------------


class TestAbstract:
    def test_json_contract(self, synth: Path) -> None:
        from src.report.sections import generate_abstract

        md, meta = generate_abstract()
        required = [
            "project_goal",
            "project_framing",
            "models",
            "headline_result",
            "oos_headline",
            "key_finding",
            "source_artifacts",
        ]
        for key in required:
            assert key in meta, f"Missing key: {key}"
        assert isinstance(meta["source_artifacts"], list) and len(meta["source_artifacts"]) > 0
        assert meta["headline_result"]["mean"] == 0.83

    def test_heading(self, synth: Path) -> None:
        from src.report.sections import generate_abstract

        md, _ = generate_abstract()
        assert md.startswith("## Abstract")

    def test_supervised_and_descriptive_scope(self, synth: Path) -> None:
        from src.report.sections import generate_abstract

        markdown, _ = generate_abstract()
        assert "supervised 151st class" in markdown
        assert "aggregate argmax OOS F1" in markdown
        assert "population standard deviations" in markdown
        assert "not confidence intervals or significance tests" in markdown


# ---------------------------------------------------------------------------
# Section C: Dataset description
# ---------------------------------------------------------------------------


class TestDatasetDescription:
    def test_json_contract(self, synth: Path) -> None:
        from src.report.sections import generate_dataset_description

        _, meta = generate_dataset_description()
        required = [
            "dataset_source",
            "subset",
            "split_sizes",
            "num_classes",
            "oos_label_name",
            "oos_label_id",
            "oos_counts",
            "in_scope_counts",
            "class_balance_note",
            "distribution_quirks",
            "source_artifact",
        ]
        for key in required:
            assert key in meta, f"Missing key: {key}"
        assert meta["num_classes"] == 151
        assert meta["oos_label_id"] == 42

    def test_balance_and_preserved_benchmark_overlaps(self, synth: Path) -> None:
        from src.report.sections import generate_dataset_description

        markdown, metadata = generate_dataset_description()
        assert "In-scope classes are balanced" in metadata["class_balance_note"]
        assert "100 training examples per class" in markdown
        assert "OOS support is reported separately" in markdown
        assert "3 normalized texts shared between train and validation" in markdown
        assert "2 shared between train and test" in markdown
        assert (
            "Two of the three train/validation overlaps and both train/test overlaps have conflicting labels"
            in markdown
        )
        assert "official splits are preserved" in markdown

    def test_missing_balance_statistics_do_not_invent_counts(self, synth: Path) -> None:
        from src.report.sections import generate_dataset_description

        path = synth / "data/artifacts/dataset_summary.json"
        summary = json.loads(path.read_text())
        del summary["train_class_count_min"]
        del summary["train_class_count_max"]
        _write(path, summary)
        markdown, metadata = generate_dataset_description()
        assert "not recorded" in metadata["class_balance_note"]
        assert "None training examples" not in markdown


# ---------------------------------------------------------------------------
# Section D: Preprocessing summary
# ---------------------------------------------------------------------------


class TestPreprocessingSummary:
    def test_json_contract(self, synth: Path) -> None:
        from src.report.sections import generate_preprocessing_summary

        _, meta = generate_preprocessing_summary()
        required = [
            "cleaning_policy",
            "tokenizer",
            "vocab_size",
            "special_tokens",
            "max_seq_length",
            "sequence_length_stats",
            "oov_rates",
            "truncation_rates",
            "tfidf_config",
            "source_artifact",
        ]
        for key in required:
            assert key in meta, f"Missing key: {key}"
        assert meta["vocab_size"] == 6161


# ---------------------------------------------------------------------------
# Section E: Model architectures
# ---------------------------------------------------------------------------


class TestModelArchitectures:
    def test_json_contract(self, synth: Path) -> None:
        from src.report.sections import generate_model_architectures

        _, meta = generate_model_architectures()
        assert "models" in meta
        assert len(meta["models"]) == 3
        for m in meta["models"]:
            model_keys = [
                "model_id",
                "model_name",
                "display_name",
                "input_type",
                "architecture_summary",
                "hyperparameters",
                "parameter_count",
                "trainable_parameter_count",
            ]
            for key in model_keys:
                assert key in m, f"Missing key in model: {key}"
        assert "comparison_table" in meta
        assert "source_artifacts" in meta

    def test_mlp_input_dim_uses_tfidf(self, synth: Path) -> None:
        from src.report.sections import generate_model_architectures

        md, _ = generate_model_architectures()
        assert "10,000 TF-IDF features" in md

    def test_fixed_padding_limitation_and_cnn_attribution(self, synth: Path) -> None:
        from src.report.sections import generate_model_architectures

        markdown, _ = generate_model_architectures()
        assert "https://aclanthology.org/D14-1181/" in markdown
        assert "concatenates final forward/backward hidden states" in markdown
        assert "zero PAD embedding does not mask recurrent transitions" in markdown
        assert "changing padding length can change logits" in markdown


# ---------------------------------------------------------------------------
# Section F: Experimental setup
# ---------------------------------------------------------------------------


class TestExperimentalSetup:
    def test_json_contract(self, synth: Path) -> None:
        from src.report.sections import generate_experimental_setup

        _, meta = generate_experimental_setup()
        required = [
            "run_count",
            "seed_list",
            "representative_run_rule",
            "optimizer",
            "per_model_lr",
            "per_model_weight_decay",
            "batch_size",
            "max_epochs",
            "early_stopping_patience",
            "monitor_metric",
            "metric_definitions",
            "oos_evaluation_policy",
            "timing_policy",
            "source_artifacts",
        ]
        for key in required:
            assert key in meta, f"Missing key: {key}"
        assert meta["seed_list"] == [42, 1337, 2024]

    def test_markdown_reports_actual_seed_behavior(self, synth: Path) -> None:
        from src.report.sections import generate_experimental_setup

        md, _ = generate_experimental_setup()

        assert "**Seed behavior**" in md
        assert "`dataloader_seed` controls train-batch shuffling" in md
        assert "`config.random_seed`" in md

    def test_current_seed_note_replaces_historical_disclosure(self, synth: Path) -> None:
        from src.report.sections import generate_experimental_setup

        path = synth / "outputs/shared/evaluation_protocol.json"
        protocol = json.loads(path.read_text())
        note = "Effective training seed controls initialization; dataloader seed controls train-batch shuffling."
        protocol["seed_policy"]["report_note"] = note
        _write(path, protocol)
        markdown, _ = generate_experimental_setup()
        assert note in markdown
        assert "model initialization currently depends" not in markdown

    def test_missing_seed_note_does_not_invent_historical_behavior(self, synth: Path) -> None:
        from src.report.sections import generate_experimental_setup

        path = synth / "outputs/shared/evaluation_protocol.json"
        protocol = json.loads(path.read_text())
        del protocol["seed_policy"]["report_note"]
        _write(path, protocol)
        markdown, _ = generate_experimental_setup()
        assert "nominal seed labels alone do not verify historical seed behavior" in markdown
        assert "model initialization currently depends" not in markdown

    def test_pipeline_comparison_and_original_tuning_order_caveat(self, synth: Path) -> None:
        from src.report.sections import generate_experimental_setup

        markdown, _ = generate_experimental_setup()
        assert "limited model-specific searches" in markdown
        assert "original tuning reused advancing loader RNG state" in markdown
        assert "neural smoke checks also consumed a train shuffle" in markdown
        assert "compares complete pipelines rather than isolating architecture" in markdown
        assert "MPS, then CUDA, then CPU" in markdown
        assert "hardware model, processor and RAM" in markdown


# ---------------------------------------------------------------------------
# Section G: Main results
# ---------------------------------------------------------------------------


class TestMainResults:
    def test_json_contract(self, synth: Path) -> None:
        from src.report.sections import generate_main_results

        _, meta = generate_main_results()
        assert "tables" in meta and len(meta["tables"]) == 5
        assert "claims" in meta and len(meta["claims"]) == 5
        for t in meta["tables"]:
            for key in ("table_id", "title", "data_scope", "columns", "rows", "source_artifact"):
                assert key in t, f"Missing key in table: {key}"
        for claim in meta["claims"]:
            for key in ("claim_id", "claim_text", "source_artifacts", "related_figure_paths", "related_figure_types"):
                assert key in claim, f"Missing key in claim: {key}"
        assert "source_artifacts" in meta

    def test_scope_annotations(self, synth: Path) -> None:
        from src.report.sections import generate_main_results

        md, _ = generate_main_results()
        assert "Aggregate over 3 runs" in md
        assert "Representative run only" in md

    def test_figure_references(self, synth: Path) -> None:
        from src.report.sections import generate_main_results

        md, _ = generate_main_results()
        assert "![" in md

    def test_efficiency_and_argmax_scope_are_local_to_tables(self, synth: Path) -> None:
        from src.report.sections import generate_main_results

        markdown, _ = generate_main_results()
        assert "score argmax predictions in the supervised 151-class classifier" in markdown
        assert "Batched evaluation timing" in markdown
        assert "single-query deployment latency was not measured" in markdown


# ---------------------------------------------------------------------------
# Section H: OOS detection
# ---------------------------------------------------------------------------


class TestOosDetection:
    def test_json_contract(self, synth: Path) -> None:
        from src.report.sections import generate_oos_detection

        _, meta = generate_oos_detection()
        required = [
            "aggregate_oos_metrics",
            "threshold_metrics",
            "msp_baseline",
            "best_oos_model",
            "false_accept_summary",
            "figure_references",
            "claims",
            "source_artifacts",
        ]
        for key in required:
            assert key in meta, f"Missing key: {key}"
        assert len(meta["claims"]) == 3

    def test_roc_figure_reference(self, synth: Path) -> None:
        from src.report.sections import generate_oos_detection

        md, _ = generate_oos_detection()
        assert "oos_roc_comparison" in md

    def test_different_argmax_and_probability_leaders_keep_metric_scope(self, synth: Path) -> None:
        from src.report.sections import generate_oos_detection

        oos_path = synth / "outputs/shared/oos_summary_table.json"
        oos = json.loads(oos_path.read_text())
        for row in oos["rows"]:
            row["oos_f1_mean"] = 0.95 if row["model_id"] == ModelID.MLP else 0.50
        _write(oos_path, oos)
        threshold_path = synth / "outputs/shared/analysis/oos_threshold_comparison.json"
        threshold = json.loads(threshold_path.read_text())
        for row in threshold["rows"]:
            row["auroc"] = 0.98 if row["model_id"] == ModelID.TEXT_CNN else 0.80
            row["msp_auroc"] = 0.99
        _write(threshold_path, threshold)
        markdown, metadata = generate_oos_detection()
        assert "TF-IDF + MLP achieved the highest aggregate argmax OOS F1" in markdown
        assert "Text CNN has the highest representative-run OOS-probability AUROC" in markdown
        assert metadata["best_oos_model"]["model_id"] == ModelID.TEXT_CNN
        assert "substantially outperforms" not in markdown
        assert "probability ranking measures different behavior" in markdown

    def test_msp_and_test_roc_points_are_diagnostics(self, synth: Path) -> None:
        from src.report.sections import generate_oos_detection

        markdown, _ = generate_oos_detection()
        assert "`1 - max(p)` over all 151 softmax classes, including OOS" in markdown
        assert "confidently correct OOS prediction can therefore receive a low MSP OOS score" in markdown
        assert "representative test ROC curve" in markdown
        assert "thresholds and calibration must be selected on validation data" in markdown
        assert "do not establish a deployable operating point" in markdown


# ---------------------------------------------------------------------------
# Section I: Figure catalogue
# ---------------------------------------------------------------------------


class TestFigureCatalogue:
    def test_json_contract(self, synth: Path) -> None:
        from src.report.figures import generate_figure_catalogue

        _, meta = generate_figure_catalogue()
        for key in ("total_figure_count", "sections", "source_artifact"):
            assert key in meta, f"Missing key: {key}"
        assert meta["total_figure_count"] == 4

    def test_captions_non_empty(self, synth: Path) -> None:
        from src.report.figures import generate_figure_catalogue

        _, meta = generate_figure_catalogue()
        for sec in meta["sections"]:
            for fig in sec["figures"]:
                assert fig["caption"], f"Empty caption for {fig['figure_path']}"
                assert "source_artifact_paths" in fig

    def test_scope_tags_valid(self, synth: Path) -> None:
        from src.report.figures import generate_figure_catalogue

        _, meta = generate_figure_catalogue()
        valid = {"aggregate", "representative", "analysis"}
        for sec in meta["sections"]:
            for fig in sec["figures"]:
                assert fig["scope"] in valid, f"Invalid scope: {fig['scope']}"


# ---------------------------------------------------------------------------
# Section J: Error analysis
# ---------------------------------------------------------------------------


class TestErrorAnalysis:
    def test_json_contract(self, synth: Path) -> None:
        from src.report.sections import generate_error_analysis

        _, meta = generate_error_analysis()
        required = [
            "taxonomy_summary",
            "calibration_finding",
            "oos_finding",
            "cross_model_overlap",
            "worst_classes_finding",
            "confused_pairs_finding",
            "length_finding",
            "claims",
            "source_artifacts",
        ]
        for key in required:
            assert key in meta, f"Missing key: {key}"
        assert len(meta["claims"]) >= 6

    def test_every_claim_cites_artifact(self, synth: Path) -> None:
        from src.report.sections import generate_error_analysis

        md, _ = generate_error_analysis()
        assert "outputs/shared/analysis/error_taxonomy_summary.json" in md

    def test_confusion_matrix_figure(self, synth: Path) -> None:
        from src.report.sections import generate_error_analysis

        md, _ = generate_error_analysis()
        assert "confusion_matrix" in md

    def test_taxonomy_is_heuristic_and_dominant_claim_tracks_actual_counts(self, synth: Path) -> None:
        from src.report.sections import generate_error_analysis

        path = synth / "outputs/shared/analysis/error_taxonomy_summary.json"
        taxonomy = json.loads(path.read_text())
        for row in taxonomy["rows"]:
            row["near_semantic_confusion_count"] = 10000
        _write(path, taxonomy)
        markdown, metadata = generate_error_analysis()
        assert "same-domain misclassification" in markdown
        assert "at most 5 whitespace tokens" in markdown
        assert "do not establish semantic similarity or query ambiguity" in markdown
        assert "not three-run aggregates" in markdown
        assert "All models share `oos_as_inscope` as the dominant" not in markdown
        taxonomy_claim = next(claim for claim in metadata["claims"] if claim["claim_id"] == "error_taxonomy")
        assert "`near_semantic_confusion`" in taxonomy_claim["claim_text"]
        assert metadata["oos_finding"]["main_failure_mode"] == "near_semantic_confusion"


# ---------------------------------------------------------------------------
# Section K: Representative examples
# ---------------------------------------------------------------------------


class TestRepresentativeExamples:
    def test_json_contract(self, synth: Path) -> None:
        from src.report.sections import generate_representative_examples

        _, meta = generate_representative_examples()
        required = [
            "selection_criteria",
            "total_curated_count",
            "selected_count",
            "examples",
            "data_scope",
            "claims",
            "source_artifact",
        ]
        for key in required:
            assert key in meta, f"Missing key: {key}"
        assert meta["data_scope"] == "representative"
        assert meta["selected_count"] >= 12
        assert len(meta["claims"]) == 4

    def test_scope_label(self, synth: Path) -> None:
        from src.report.sections import generate_representative_examples

        md, _ = generate_representative_examples()
        assert "Representative run only" in md

    def test_example_labels_do_not_prove_semantics_or_ambiguity(self, synth: Path) -> None:
        from src.report.sections import generate_representative_examples

        markdown, _ = generate_representative_examples()
        assert "same-domain cases tagged `near_semantic_confusion`" in markdown
        assert "short-query tagged errors" in markdown
        assert "do not establish semantic similarity or query ambiguity" in markdown


# ---------------------------------------------------------------------------
# Section L: Key findings
# ---------------------------------------------------------------------------


class TestKeyFindings:
    def test_json_contract(self, synth: Path) -> None:
        from src.report.sections import generate_key_findings

        _, meta = generate_key_findings()
        assert "findings" in meta and len(meta["findings"]) == 8
        categories = {f["category"] for f in meta["findings"]}
        expected = {
            "headline",
            "calibration",
            "oos_detection",
            "confusion",
            "architecture",
            "length",
            "efficiency",
            "recommendation",
        }
        assert categories == expected
        for f in meta["findings"]:
            assert f["source_artifact"], f"No source for finding {f['finding_id']}"
            for key in ("claim_id", "claim_text", "source_artifacts", "related_figure_paths", "related_figure_types"):
                assert key in f, f"Missing key in finding: {key}"

    def test_scopes_and_dispersion_do_not_imply_significance(self, synth: Path) -> None:
        from src.report.sections import generate_key_findings

        markdown, metadata = generate_key_findings()
        assert "not confidence intervals or significance tests" in markdown
        assert "architecture-only causal interpretation" in markdown
        calibration_claim = next(f["claim"] for f in metadata["findings"] if f["category"] == "calibration")
        assert "Lowest representative-run ECE" in calibration_claim
        assert "Best calibrated" not in calibration_claim
        oos = next(f["claim"] for f in metadata["findings"] if f["category"] == "oos_detection")
        assert "highest representative-run OOS-probability AUROC" in oos
        assert "best OOS detection" not in oos


# ---------------------------------------------------------------------------
# Section M: Limitations
# ---------------------------------------------------------------------------


class TestLimitations:
    def test_json_contract(self, synth: Path) -> None:
        from src.report.sections import generate_limitations

        _, meta = generate_limitations()
        assert len(meta["analysis_limitations"]) == 5
        assert len(meta["project_limitations"]) == 8
        assert "source_artifacts" in meta

    def test_two_subsections(self, synth: Path) -> None:
        from src.report.sections import generate_limitations

        md, _ = generate_limitations()
        assert "### Analysis-Level Limitations" in md
        assert "### Project-Level Limitations" in md

    def test_preserves_scientific_caveats(self, synth: Path) -> None:
        from src.report.sections import generate_limitations

        markdown, _ = generate_limitations()
        assert "Only the 150 in-scope classes are balanced" in markdown
        assert "CLINC150 is balanced" not in markdown
        assert "supervised 151st class" in markdown
        assert "fixed right-PAD positions" in markdown
        assert "sensitive to padding length" in markdown
        assert "3 train/validation text overlaps (2 with conflicting labels)" in markdown
        assert "2 train/test overlaps (both with conflicting labels)" in markdown
        assert "original trial order part of the search" in markdown
        assert "without measuring their variability across seeds" in markdown


# ---------------------------------------------------------------------------
# Section N: Future improvements
# ---------------------------------------------------------------------------


class TestFutureImprovements:
    def test_json_contract(self, synth: Path) -> None:
        from src.report.sections import generate_future_improvements

        _, meta = generate_future_improvements()
        assert len(meta["high_priority"]) == 4
        assert len(meta["medium_priority"]) == 4
        assert len(meta["lower_priority"]) == 5
        for item in meta["high_priority"]:
            assert "improvement" in item and "rationale" in item

    def test_validation_selected_followups_preserve_benchmark(self, synth: Path) -> None:
        from src.report.sections import generate_future_improvements

        markdown, _ = generate_future_improvements()
        assert "Merge or relabel" not in markdown
        assert "preserve official benchmark labels" in markdown
        assert "Select OOS or abstention thresholds on validation data" in markdown
        assert "Fit temperature scaling on validation data" in markdown
        assert "requires new tuning and final evaluations" in markdown
        assert "modeling change requires new training, tuning and comparisons" in markdown
        assert "service-latency claims from batch throughput" in markdown


# ---------------------------------------------------------------------------
# Section O: Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_json_contract(self, synth: Path) -> None:
        from src.report.sections import generate_reproducibility

        _, meta = generate_reproducibility()
        required = [
            "python_version",
            "install_command",
            "pinned_versions_available",
            "dataset_source",
            "pipeline_command",
            "seed_list",
            "run_count",
            "approximate_runtime_minutes",
            "hardware_context",
            "nondeterminism_notes",
            "reproduction_steps",
            "source_artifacts",
        ]
        for key in required:
            assert key in meta, f"Missing key: {key}"
        assert meta["seed_list"] == [42, 1337, 2024]

    def test_numbered_checklist(self, synth: Path) -> None:
        from src.report.sections import generate_reproducibility

        md, _ = generate_reproducibility()
        assert "1. Clone" in md
        assert "2. Install" in md

    def test_exact_timing_boundaries_and_hardware_identity(self, synth: Path) -> None:
        from src.report.sections import generate_reproducibility

        markdown, _ = generate_reproducibility()
        assert "final training-log serialization" in markdown
        assert "time.perf_counter" in markdown
        assert "hardware model, processor and RAM" in markdown
        assert "Batched evaluation timing covers loader traversal, device transfer" in markdown
        assert "softmax and CPU result collection" in markdown
        assert (
            "excludes preprocessing, checkpoint loading, array concatenation, metrics and artifact writes" in markdown
        )
        assert "no controlled warmup or repeated timing trials" in markdown
        assert "single-query deployment latency was not measured" in markdown

    def test_primary_citations_and_separate_licenses(self, synth: Path) -> None:
        from src.report.sections import generate_reproducibility

        markdown, metadata = generate_reproducibility()
        assert "https://aclanthology.org/D19-1131/" in markdown
        assert "https://aclanthology.org/D14-1181/" in markdown
        assert "https://huggingface.co/datasets/clinc/clinc_oos" in markdown
        assert "dataset card metadata lists CC BY 3.0" in markdown
        assert "Dataset licensing is separate" in markdown
        assert "[GPLv3 source license](../../../LICENSE)" in markdown
        license_ref = next(ref for ref in metadata["references"] if ref["title"] == "GPLv3 source license")
        report_dir = synth / "outputs/shared/report"
        assert (report_dir / license_ref["url"]).resolve() == synth / "LICENSE"
        assert (report_dir / license_ref["url"]).resolve().is_file()


def test_reproduction_commands_include_environment_and_all_stages(synth: Path) -> None:
    from src.report.sections import generate_reproducibility

    markdown, metadata = generate_reproducibility()
    for script in (
        "run_repeated_evaluation",
        "run_experiment_tracking",
        "run_report_figures",
        "run_error_analysis",
        "run_report_generation",
    ):
        assert f"uv run --frozen python scripts/{script}.py" in markdown
    assert "run_preprocessing.py" in markdown
    assert "network access" in markdown
    assert "trained checkpoints" in markdown
    assert "do not retune" in markdown
    assert metadata["install_command"] == "uv sync --frozen"
