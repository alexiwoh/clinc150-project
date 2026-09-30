"""Tests for analysis modules: slicing, cross-model, class analysis, curation, handoff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.constants import (
    CALIBRATION_SUMMARY_FILENAME,
    CROSS_MODEL_ERROR_COMPARISON_FILENAME,
    ERROR_ANALYSIS_NOTES_FILENAME,
    ERROR_ANALYSIS_SUMMARY_FILENAME,
    ERROR_TAXONOMY_SUMMARY_FILENAME,
    EXTENDED_METRICS_COMPARISON_FILENAME,
    OOS_THRESHOLD_COMPARISON_FILENAME,
    STEP11_HANDOFF_FILENAME,
)
from src.analysis.cross_model import align_predictions_across_models
from src.analysis.class_analysis import compute_and_save_confusion_stability
from src.analysis.enums import ErrorCategory, LengthBucket
from src.analysis.slicing import bucket_by_length
from src.analysis.utils import discover_all_run_dirs, resolve_handoff
from src.constants import (
    AGGREGATE_SUBDIR,
    FINAL_RUNS_SUBDIR,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
)
from src.enums import ModelID
from src.run_ledger import load_current_run_ledger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LABEL_NAMES = ["balance", "transfer", "recipe", "calories", "oos"]
_N_CLASSES = len(_LABEL_NAMES)
_OOS_IDX = _LABEL_NAMES.index("oos")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def _make_confidences(run_dir: Path, n: int = 200, seed: int = 42) -> tuple:
    rng = np.random.RandomState(seed)
    targets = rng.randint(0, _N_CLASSES, size=n).astype(np.intp)
    targets[:20] = _OOS_IDX
    probs = rng.dirichlet(np.ones(_N_CLASSES) * 2, size=n).astype(np.float64)
    # Boost only 60% to correct class — leave 40% as random (many will be wrong)
    boost_mask = rng.random(n) < 0.6
    boost_indices = np.where(boost_mask)[0]
    probs[boost_indices, :] *= 0.1
    probs[boost_indices, targets[boost_indices]] += 2.0
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1).astype(np.intp)
    np.savez(
        run_dir / "confidences.npz",
        probabilities=probs,
        predictions=preds,
        targets=targets,
        label_names=np.array(_LABEL_NAMES),
    )
    return probs, preds, targets


def _make_predictions_csv(
    run_dir: Path, preds: np.ndarray, targets: np.ndarray, probs: np.ndarray, run_id: str = "run_01_seed_42"
) -> None:
    rows = []
    short_texts = ["hi", "ok", "yes", "no"]
    medium_texts = ["i want to check my balance", "how is the weather today"]
    for i in range(len(targets)):
        if i < 4:
            text = short_texts[i % len(short_texts)]
        elif i < 6:
            text = medium_texts[i % len(medium_texts)]
        else:
            text = f"sample text number {i} with some words in it for this query"
        rows.append(
            {
                "example_index": i,
                "text": text,
                "true_label_id": int(targets[i]),
                "true_label_name": _LABEL_NAMES[targets[i]],
                "predicted_label_id": int(preds[i]),
                "predicted_label_name": _LABEL_NAMES[preds[i]],
                "run_id": run_id,
                "max_confidence": float(np.max(probs[i])),
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / "final_predictions.csv", index=False)


def _make_per_class_metrics(run_dir: Path) -> None:
    classes = []
    for i, name in enumerate(_LABEL_NAMES):
        classes.append(
            {
                "label_name": name,
                "label_id": i,
                "precision": 0.7 + i * 0.05,
                "recall": 0.65 + i * 0.05,
                "f1": 0.67 + i * 0.05,
                "support": 10,
                "is_oos": name == "oos",
            }
        )
    _write_json(
        run_dir / "per_class_metrics.json",
        {"schema_version": SCHEMA_VERSION, "protocol_version": PROTOCOL_VERSION, "classes": classes},
    )


def _make_confusion_matrix(run_dir: Path) -> None:
    cm = pd.DataFrame(
        np.array([[8, 1, 0, 1, 0], [2, 7, 1, 0, 0], [0, 0, 9, 1, 0], [1, 0, 1, 7, 1], [0, 1, 0, 0, 9]]),
        index=_LABEL_NAMES,
        columns=_LABEL_NAMES,
    )
    cm.to_csv(run_dir / "confusion_matrix.csv")


def _make_label_order(run_dir: Path) -> None:
    _write_json(
        run_dir / "label_order.json",
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "label_names": _LABEL_NAMES,
        },
    )


def _make_test_metrics(run_dir: Path) -> None:
    _write_json(
        run_dir / "test_metrics.json",
        {
            "schema_version": SCHEMA_VERSION,
            "macro_f1": 0.80,
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78,
        },
    )


def _build_mock_model(tmp_path: Path, model_id: str) -> None:
    """Build minimal artifacts for one model (three runs)."""
    model_dir = tmp_path / "outputs" / model_id
    for seed_idx, seed in enumerate([42, 1337, 2024]):
        run_id = f"run_{seed_idx + 1:02d}_seed_{seed}"
        run_dir = model_dir / FINAL_RUNS_SUBDIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        probs, preds, targets = _make_confidences(run_dir, seed=seed + hash(model_id) % 100)
        _make_predictions_csv(run_dir, preds, targets, probs, run_id=run_id)
        _make_per_class_metrics(run_dir)
        _make_confusion_matrix(run_dir)
        _make_label_order(run_dir)
        _make_test_metrics(run_dir)
        _write_json(
            run_dir / "run_metadata.json",
            {
                "model_id": model_id,
                "run_id": run_id,
                "run_index": seed_idx + 1,
                "seed": seed,
                "status": "completed",
            },
        )
        _write_json(run_dir / "top_confusions.json", {"schema_version": SCHEMA_VERSION, "confusions": []})
        _write_json(
            run_dir / "top_errors.json",
            {"schema_version": SCHEMA_VERSION, "errors": []},
        )

    agg_dir = model_dir / AGGREGATE_SUBDIR
    agg_dir.mkdir(parents=True, exist_ok=True)
    rep_run = "run_01_seed_42"
    run_ids = [f"run_{index:02d}_seed_{seed}" for index, seed in enumerate([42, 1337, 2024], 1)]
    _write_json(
        agg_dir / "run_ledger.json",
        {
            "model_id": model_id,
            "run_count_requested": 3,
            "run_count_completed": 3,
            "run_count_failed": 0,
            "run_count_skipped": 0,
            "requested_run_ids": run_ids,
            "completed_run_ids": run_ids,
            "failed_run_ids": [],
            "skipped_run_ids": [],
            "seed_list_requested": [42, 1337, 2024],
            "seed_list_completed": [42, 1337, 2024],
            "per_run_metadata_refs": {
                run_id: f"outputs/{model_id}/{FINAL_RUNS_SUBDIR}/{run_id}/run_metadata.json" for run_id in run_ids
            },
        },
    )
    _write_json(
        agg_dir / "error_analysis_handoff.json",
        {
            "schema_version": SCHEMA_VERSION,
            "model_id": model_id,
            "representative_run_id": rep_run,
            "artifacts": {
                "final_predictions": f"outputs/{model_id}/{FINAL_RUNS_SUBDIR}/{rep_run}/final_predictions.csv",
                "per_class_metrics": f"outputs/{model_id}/{FINAL_RUNS_SUBDIR}/{rep_run}/per_class_metrics.json",
                "confusion_matrix": f"outputs/{model_id}/{FINAL_RUNS_SUBDIR}/{rep_run}/confusion_matrix.csv",
                "top_confusions": f"outputs/{model_id}/{FINAL_RUNS_SUBDIR}/{rep_run}/top_confusions.json",
                "top_errors": f"outputs/{model_id}/{FINAL_RUNS_SUBDIR}/{rep_run}/top_errors.json",
                "label_order": f"outputs/{model_id}/{FINAL_RUNS_SUBDIR}/{rep_run}/label_order.json",
            },
        },
    )


def _build_mock_shared(tmp_path: Path) -> None:
    shared_dir = tmp_path / "outputs" / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mid in ModelID:
        rows.append(
            {
                "model_id": str(mid),
                "display_name": mid.display_name,
                "run_count": 3,
                "test_accuracy_mean": 0.85,
                "test_accuracy_std": 0.01,
                "test_macro_f1_mean": 0.83,
                "test_macro_f1_std": 0.02,
                "oos_f1_mean": 0.67,
                "oos_f1_std": 0.01,
            }
        )
    _write_json(
        shared_dir / "model_comparison_aggregate.json",
        {"schema_version": SCHEMA_VERSION, "protocol_version": PROTOCOL_VERSION, "rows": rows},
    )
    _write_json(
        shared_dir / "oos_summary_table.json",
        {"schema_version": SCHEMA_VERSION, "protocol_version": PROTOCOL_VERSION, "rows": rows},
    )
    _write_json(
        shared_dir / "most_confused_pairs_table.json",
        {"schema_version": SCHEMA_VERSION, "rows": []},
    )
    _write_json(
        shared_dir / "representative_examples_index.json",
        {"schema_version": SCHEMA_VERSION, "examples": []},
    )
    _write_json(
        shared_dir / "figure_manifest.json",
        {"schema_version": SCHEMA_VERSION, "protocol_version": PROTOCOL_VERSION, "figures": []},
    )


@pytest.fixture()
def mock_analysis_env(tmp_path: Path) -> Path:
    """Build mock outputs for all three models."""
    for mid in ModelID:
        _build_mock_model(tmp_path, str(mid))
    _build_mock_shared(tmp_path)
    return tmp_path


def _patch_for_analysis(tmp_path: Path):
    """Return combined patches needed for analysis modules."""
    _model_dir = lambda mid: tmp_path / "outputs" / str(mid)  # noqa: E731
    return [
        patch("src.constants.PROJECT_ROOT", tmp_path),
        patch("src.constants.OUTPUTS_DIR", tmp_path / "outputs"),
        patch("src.constants.SHARED_DIR", tmp_path / "outputs" / "shared"),
        patch("src.constants.model_output_dir", side_effect=_model_dir),
        patch("src.analysis.utils.PROJECT_ROOT", tmp_path),
        patch("src.analysis.utils.SHARED_DIR", tmp_path / "outputs" / "shared"),
        patch("src.analysis.utils.model_output_dir", side_effect=_model_dir),
        patch("src.analysis.pipeline.SHARED_DIR", tmp_path / "outputs" / "shared"),
        patch("src.analysis.pipeline.PROJECT_ROOT", tmp_path),
        patch("src.analysis.pipeline.model_output_dir", side_effect=_model_dir),
    ]


class TestCurrentRunDiscovery:
    def test_three_to_one_ledger_excludes_preserved_stale_runs(self, mock_analysis_env: Path) -> None:
        model_dir = mock_analysis_env / "outputs" / "mlp"
        ledger_path = model_dir / AGGREGATE_SUBDIR / "run_ledger.json"
        ledger = json.loads(ledger_path.read_text())
        first_run = ledger["completed_run_ids"][0]
        for key in ("requested_run_ids", "completed_run_ids", "seed_list_requested", "seed_list_completed"):
            ledger[key] = ledger[key][:1]
        ledger["run_count_requested"] = ledger["run_count_completed"] = 1
        ledger["per_run_metadata_refs"] = {first_run: ledger["per_run_metadata_refs"][first_run]}
        _write_json(ledger_path, ledger)
        stale_metadata = model_dir / FINAL_RUNS_SUBDIR / "run_02_seed_1337" / "run_metadata.json"
        stale_metadata.write_text("historical data deliberately outside current validation")
        stale_before = stale_metadata.read_bytes()

        patches = _patch_for_analysis(mock_analysis_env)
        for context in patches:
            context.start()
        try:
            run_dirs = discover_all_run_dirs(ModelID.MLP)
            ctx = resolve_handoff(ModelID.MLP)
            stability, records = compute_and_save_confusion_stability(ctx)
        finally:
            for context in reversed(patches):
                context.stop()

        assert [directory.name for directory in run_dirs] == [first_run]
        assert "Fewer than 2 runs" in stability["note"]
        assert records == []
        assert stale_metadata.read_bytes() == stale_before
        assert (model_dir / FINAL_RUNS_SUBDIR / "run_03_seed_2024").is_dir()

    def test_missing_ledger_has_no_directory_fallback(self, mock_analysis_env: Path) -> None:
        model_dir = mock_analysis_env / "outputs" / "mlp"
        (model_dir / AGGREGATE_SUBDIR / "run_ledger.json").unlink()
        with pytest.raises(FileNotFoundError, match="Current run ledger not found"):
            load_current_run_ledger(ModelID.MLP, model_dir=model_dir, project_root=mock_analysis_env)

    def test_handoff_cannot_point_to_inactive_run(self, mock_analysis_env: Path) -> None:
        model_dir = mock_analysis_env / "outputs" / "mlp"
        handoff_path = model_dir / AGGREGATE_SUBDIR / "error_analysis_handoff.json"
        handoff = json.loads(handoff_path.read_text())
        handoff["representative_run_id"] = "historical_run"
        _write_json(handoff_path, handoff)
        with (
            patch("src.analysis.utils.PROJECT_ROOT", mock_analysis_env),
            patch("src.analysis.utils.model_output_dir", return_value=model_dir),
            pytest.raises(ValueError, match="not a completed member"),
        ):
            resolve_handoff(ModelID.MLP)

    @pytest.mark.parametrize("artifact_key", ["final_predictions", "label_order"])
    def test_handoff_artifact_from_different_active_run_rejected(
        self, mock_analysis_env: Path, artifact_key: str
    ) -> None:
        model_dir = mock_analysis_env / "outputs" / "mlp"
        handoff_path = model_dir / AGGREGATE_SUBDIR / "error_analysis_handoff.json"
        handoff = json.loads(handoff_path.read_text())
        handoff["artifacts"][artifact_key] = handoff["artifacts"][artifact_key].replace(
            "run_01_seed_42", "run_02_seed_1337"
        )
        _write_json(handoff_path, handoff)
        with (
            patch("src.analysis.utils.PROJECT_ROOT", mock_analysis_env),
            patch("src.analysis.utils.model_output_dir", return_value=model_dir),
            pytest.raises(ValueError, match="references a different run"),
        ):
            resolve_handoff(ModelID.MLP)


# ---------------------------------------------------------------------------
# TestTaxonomySummaryFigure
# ---------------------------------------------------------------------------


class TestTaxonomySummaryFigure:
    def test_figure_omits_short_query_ambiguity_bucket(self, tmp_path: Path) -> None:
        """Keep the taxonomy summary intact while hiding the short-query bar."""
        from src.analysis.taxonomy import generate_taxonomy_summary

        captured_plot: dict[str, Any] = {}
        total_count = sum(range(1, len(ErrorCategory) + 1))

        def _model_dir(model_id: ModelID) -> Path:
            return tmp_path / "outputs" / str(model_id)

        def _capture_plot(
            group_labels: list[str],
            series: dict[str, list[float]],
            ylabel: str,
            title: str,
            out_path: Path,
            **_: Any,
        ) -> None:
            captured_plot["group_labels"] = list(group_labels)
            captured_plot["series"] = {label: list(values) for label, values in series.items()}
            captured_plot["ylabel"] = ylabel
            captured_plot["title"] = title
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"test-figure")

        per_model_taxonomy: dict[str, dict[str, Any]] = {}
        for model_id in ModelID:
            categories: dict[ErrorCategory, dict[str, float | int]] = {}
            for category_index, category in enumerate(ErrorCategory, start=1):
                categories[category] = {
                    "count": category_index,
                    "fraction_of_errors": category_index / total_count,
                    "fraction_of_test_set": category_index / 100.0,
                }
            per_model_taxonomy[str(model_id)] = {"categories": categories}

        patches = [
            patch("src.analysis.utils.PROJECT_ROOT", tmp_path),
            patch("src.analysis.utils.SHARED_DIR", tmp_path / "outputs" / "shared"),
            patch("src.analysis.utils.model_output_dir", side_effect=_model_dir),
            patch("src.analysis.taxonomy.plot_grouped_bar", side_effect=_capture_plot),
        ]
        for active_patch in patches:
            active_patch.start()
        try:
            generate_taxonomy_summary(list(ModelID), per_model_taxonomy)
        finally:
            for active_patch in patches:
                active_patch.stop()

        expected_group_labels = [
            category.value for category in ErrorCategory if category != ErrorCategory.SHORT_QUERY_AMBIGUITY
        ]
        assert captured_plot["group_labels"] == expected_group_labels
        assert captured_plot["ylabel"] == "Fraction of Errors"
        assert captured_plot["title"] == "Error Taxonomy Comparison"
        for values in captured_plot["series"].values():
            assert len(values) == len(expected_group_labels)

        summary_path = tmp_path / "outputs" / "shared" / "analysis" / ERROR_TAXONOMY_SUMMARY_FILENAME
        summary = json.loads(summary_path.read_text())
        for row in summary["rows"]:
            assert "short_query_ambiguity_count" in row
            assert "short_query_ambiguity_fraction" in row


# ---------------------------------------------------------------------------
# TestLengthBucketing
# ---------------------------------------------------------------------------


class TestLengthBucketing:
    def test_short_query(self) -> None:
        texts = pd.Series(["hi there"])
        buckets = bucket_by_length(texts)
        assert buckets.iloc[0] == LengthBucket.SHORT

    def test_medium_query(self) -> None:
        texts = pd.Series(["i want to check my bank balance"])
        buckets = bucket_by_length(texts)
        assert buckets.iloc[0] == LengthBucket.MEDIUM

    def test_long_query(self) -> None:
        texts = pd.Series(["this is a very long query with many tokens in it for testing"])
        buckets = bucket_by_length(texts)
        assert buckets.iloc[0] == LengthBucket.LONG

    def test_four_tokens_is_short(self) -> None:
        texts = pd.Series(["one two three four"])
        buckets = bucket_by_length(texts)
        assert buckets.iloc[0] == LengthBucket.SHORT

    def test_five_tokens_is_medium(self) -> None:
        texts = pd.Series(["one two three four five"])
        buckets = bucket_by_length(texts)
        assert buckets.iloc[0] == LengthBucket.MEDIUM

    def test_nine_tokens_is_long(self) -> None:
        texts = pd.Series(["one two three four five six seven eight nine"])
        buckets = bucket_by_length(texts)
        assert buckets.iloc[0] == LengthBucket.LONG


# ---------------------------------------------------------------------------
# TestCrossModelAlignment
# ---------------------------------------------------------------------------


class TestCrossModelAlignment:
    def test_alignment_full_coverage(self) -> None:
        texts = [f"text_{i}" for i in range(100)]
        dfs: dict[str, pd.DataFrame] = {}
        for mid in ["mlp", "text_cnn", "bilstm"]:
            dfs[mid] = pd.DataFrame(
                {
                    "text": texts,
                    "true_label_name": ["balance"] * 100,
                    "predicted_label_name": ["balance"] * 100,
                }
            )
        aligned = align_predictions_across_models(dfs)
        assert len(aligned) == 100

    def test_alignment_with_different_predictions(self) -> None:
        texts = ["hello", "world", "test"]
        dfs = {
            "mlp": pd.DataFrame(
                {"text": texts, "true_label_name": ["a", "b", "c"], "predicted_label_name": ["a", "b", "c"]}
            ),
            "cnn": pd.DataFrame(
                {"text": texts, "true_label_name": ["a", "b", "c"], "predicted_label_name": ["a", "x", "c"]}
            ),
        }
        aligned = align_predictions_across_models(dfs)
        assert len(aligned) == 3
        assert "mlp_correct" in aligned.columns
        assert "cnn_correct" in aligned.columns
        assert aligned.loc[aligned["text"] == "world", "cnn_correct"].iloc[0] is np.bool_(False)

    def test_partial_overlap_inner_join(self) -> None:
        dfs = {
            "a": pd.DataFrame({"text": ["x", "y"], "true_label_name": ["t", "t"], "predicted_label_name": ["t", "t"]}),
            "b": pd.DataFrame({"text": ["y", "z"], "true_label_name": ["t", "t"], "predicted_label_name": ["t", "t"]}),
        }
        aligned = align_predictions_across_models(dfs)
        assert len(aligned) == 1
        assert aligned.iloc[0]["text"] == "y"


# ---------------------------------------------------------------------------
# TestPreflightValidation
# ---------------------------------------------------------------------------


class TestPreflightValidation:
    def test_missing_artifact_raises(self, mock_analysis_env: Path) -> None:
        from src.analysis.pipeline import run_preflight_validation

        patches = _patch_for_analysis(mock_analysis_env)
        for p in patches:
            p.start()
        try:
            shared = mock_analysis_env / "outputs" / "shared"
            (shared / "model_comparison_aggregate.json").unlink()
            with pytest.raises(FileNotFoundError, match="preflight failed"):
                run_preflight_validation(list(ModelID))
        finally:
            for p in patches:
                p.stop()

    def test_passes_with_all_artifacts(self, mock_analysis_env: Path) -> None:
        from src.analysis.pipeline import run_preflight_validation

        patches = _patch_for_analysis(mock_analysis_env)
        for p in patches:
            p.start()
        try:
            run_preflight_validation(list(ModelID))
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# TestErrorTaxonomy (Section E categorization logic)
# ---------------------------------------------------------------------------


class TestErrorTaxonomy:
    _domain_map = {
        "balance": "banking",
        "transfer": "banking",
        "recipe": "kitchen_and_dining",
        "calories": "kitchen_and_dining",
    }

    def test_oos_false_accept(self) -> None:
        from src.analysis.taxonomy import classify_single_error

        primary, _ = classify_single_error("test", "oos", "balance", domain_map=self._domain_map)
        assert primary == ErrorCategory.OOS_AS_INSCOPE

    def test_inscope_false_reject(self) -> None:
        from src.analysis.taxonomy import classify_single_error

        primary, _ = classify_single_error("check balance", "balance", "oos", domain_map=self._domain_map)
        assert primary == ErrorCategory.INSCOPE_AS_OOS

    def test_near_semantic_same_domain(self) -> None:
        from src.analysis.taxonomy import classify_single_error

        primary, _ = classify_single_error("send money", "balance", "transfer", domain_map=self._domain_map)
        assert primary == ErrorCategory.NEAR_SEMANTIC_CONFUSION

    def test_cross_domain_different_domain(self) -> None:
        from src.analysis.taxonomy import classify_single_error

        primary, _ = classify_single_error("cook food", "recipe", "balance", domain_map=self._domain_map)
        assert primary == ErrorCategory.CROSS_DOMAIN_CONFUSION

    def test_short_query_secondary(self) -> None:
        from src.analysis.taxonomy import classify_single_error

        _, secondary = classify_single_error("hi", "balance", "transfer", domain_map=self._domain_map)
        assert ErrorCategory.SHORT_QUERY_AMBIGUITY in secondary


# ---------------------------------------------------------------------------
# TestOOSThresholdMetrics
# ---------------------------------------------------------------------------


class TestOOSThresholdMetrics:
    def test_synthetic_auroc_in_range(self) -> None:
        from src.analysis.metrics import compute_oos_detection_metrics

        rng = np.random.RandomState(42)
        n, nc, oos_idx = 100, 5, 2
        targets = rng.randint(0, nc, size=n).astype(np.intp)
        targets[:20] = oos_idx
        probs = rng.dirichlet(np.ones(nc), size=n).astype(np.float64)
        result = compute_oos_detection_metrics(probs, targets, oos_idx)
        assert 0.0 <= result["auroc"] <= 1.0
        assert 0.0 <= result["aupr"] <= 1.0

    def test_perfect_separation(self) -> None:
        from src.analysis.metrics import compute_oos_detection_metrics

        n, nc, oos_idx = 100, 5, 2
        targets = np.zeros(n, dtype=np.intp)
        targets[:30] = oos_idx
        probs = np.full((n, nc), 0.01)
        probs[targets == oos_idx, oos_idx] = 0.99
        probs = probs / probs.sum(axis=1, keepdims=True)
        result = compute_oos_detection_metrics(probs, targets, oos_idx)
        assert result["auroc"] > 0.95


# ---------------------------------------------------------------------------
# TestConfidenceStratification
# ---------------------------------------------------------------------------


class TestConfidenceStratification:
    def test_known_bin_assignment(self) -> None:
        from src.analysis.metrics import stratify_by_confidence

        confs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        correct = np.array([True, True, False, True, True])
        strata = stratify_by_confidence(confs, correct)
        assert len(strata) == 5
        assert strata[0]["count"] == 1
        assert strata[-1]["count"] == 1
        assert strata[-1]["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# TestHandoffValidation
# ---------------------------------------------------------------------------


class TestHandoffValidation:
    def test_missing_path_raises(self, tmp_path: Path) -> None:
        from src.analysis.handoff import validate_handoff_paths

        handoff = {
            "per_model_artifacts": {"mlp": {"test.json": "MISSING:test.json"}},
            "shared_artifacts": {},
        }
        handoff_path = tmp_path / "handoff.json"
        handoff_path.write_text(json.dumps(handoff))

        patches = [patch("src.analysis.handoff.PROJECT_ROOT", tmp_path)]
        for p in patches:
            p.start()
        try:
            with pytest.raises(FileNotFoundError, match="handoff validation failed"):
                validate_handoff_paths(handoff_path)
        finally:
            for p in patches:
                p.stop()

    def test_valid_paths_pass(self, tmp_path: Path) -> None:
        from src.analysis.handoff import validate_handoff_paths

        (tmp_path / "outputs" / "mlp" / "analysis").mkdir(parents=True)
        (tmp_path / "outputs" / "mlp" / "analysis" / "test.json").write_text("{}")
        handoff = {
            "per_model_artifacts": {"mlp": {"test.json": "outputs/mlp/analysis/test.json"}},
            "shared_artifacts": {},
        }
        handoff_path = tmp_path / "handoff.json"
        handoff_path.write_text(json.dumps(handoff))

        patches = [patch("src.analysis.handoff.PROJECT_ROOT", tmp_path)]
        for p in patches:
            p.start()
        try:
            validate_handoff_paths(handoff_path)
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# TestFullPipelineIntegration
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    def test_pipeline_produces_key_artifacts(self, mock_analysis_env: Path) -> None:
        """Run the full pipeline on mock data and check key outputs exist."""
        from src.analysis.pipeline import run_error_analysis

        patches = _patch_for_analysis(mock_analysis_env)
        patches.extend(
            [
                patch("src.analysis.handoff.PROJECT_ROOT", mock_analysis_env),
                patch("src.analysis.handoff.SHARED_DIR", mock_analysis_env / "outputs" / "shared"),
                patch("src.analysis.summary.SHARED_DIR", mock_analysis_env / "outputs" / "shared"),
            ]
        )
        for p in patches:
            p.start()
        try:
            success = run_error_analysis(list(ModelID))
            assert success, "Pipeline should return True on success"

            shared_analysis = mock_analysis_env / "outputs" / "shared" / "analysis"
            assert (shared_analysis / STEP11_HANDOFF_FILENAME).exists()
            assert (shared_analysis / ERROR_ANALYSIS_SUMMARY_FILENAME).exists()
            assert (shared_analysis / ERROR_ANALYSIS_NOTES_FILENAME).exists()
            assert (shared_analysis / CALIBRATION_SUMMARY_FILENAME).exists()
            assert (shared_analysis / OOS_THRESHOLD_COMPARISON_FILENAME).exists()
            assert (shared_analysis / ERROR_TAXONOMY_SUMMARY_FILENAME).exists()
            assert (shared_analysis / CROSS_MODEL_ERROR_COMPARISON_FILENAME).exists()
            assert (shared_analysis / EXTENDED_METRICS_COMPARISON_FILENAME).exists()

            for mid in ModelID:
                model_analysis = mock_analysis_env / "outputs" / str(mid) / "analysis"
                assert model_analysis.exists()
                assert (model_analysis / "extended_test_metrics.json").exists()
                assert (model_analysis / "calibration_metrics.json").exists()
                assert (model_analysis / "oos_threshold_metrics.json").exists()
                assert (model_analysis / "error_taxonomy.json").exists()
                assert (model_analysis / "confidence_stratification.json").exists()
                assert (model_analysis / "length_slice_analysis.json").exists()
                assert (model_analysis / "frequency_slice_analysis.json").exists()
                assert (model_analysis / "worst_classes_deep_dive.json").exists()
                assert (model_analysis / "confusion_stability.json").exists()
        finally:
            for p in patches:
                p.stop()


def test_scope_slice_is_independent_of_test_support(mock_analysis_env: Path) -> None:
    from contextlib import ExitStack

    from src.analysis.slicing import compute_and_save_frequency_analysis
    from src.analysis.utils import resolve_handoff

    with ExitStack() as stack:
        for context_patch in _patch_for_analysis(mock_analysis_env):
            stack.enter_context(context_patch)
        ctx = resolve_handoff(ModelID.MLP)
        # Give every class equal test support. Quartiles would collapse them
        # into one tier, while class scope still has the two correct groups.
        artifact = compute_and_save_frequency_analysis(ctx)
        predictions = pd.read_csv(ctx.final_predictions_path)
        expected_oos = int((predictions.true_label_name == "oos").sum())
        slices = {row["slice"]: row for row in artifact["slices"]}
        assert list(slices) == ["in_scope", "oos"]
        assert slices["oos"]["count"] == expected_oos
        assert slices["in_scope"]["count"] == len(predictions) - expected_oos
        assert artifact["analysis_basis"] == "in_scope_vs_oos"
        assert artifact["source_artifact"].endswith("final_predictions.csv")
        assert "not a training-frequency analysis" in artifact["note"]
        assert "quartile_thresholds" not in artifact
        scope = predictions.true_label_name == "oos"
        correct = predictions.true_label_id == predictions.predicted_label_id
        assert slices["oos"]["accuracy"] == pytest.approx(correct[scope].mean())
        assert slices["in_scope"]["accuracy"] == pytest.approx(correct[~scope].mean())
