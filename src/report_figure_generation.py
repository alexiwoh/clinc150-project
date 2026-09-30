"""Metadata-driven figure generation and error-analysis handoff orchestrator.

Reads saved tracking artifacts and per-run evaluation outputs, generates
report-ready figures (representative-run diagnostics and aggregate cross-model
comparisons), and produces the metadata bundles needed for downstream
error analysis and report writing.

All figure generation is artifact-driven — no model objects, live
checkpoints, or re-evaluation are required.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.constants import (
    AGGREGATE_SUBDIR,
    MODEL_FIGURES_SUBDIR,
    PROJECT_ROOT,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    SHARED_DIR,
    SHARED_FIGURES_DIR,
    model_output_dir,
)
from src.enums import ModelID
from src.report.figure_metadata import FigureCaptionContext
from src.run_ledger import load_current_generation, load_current_run_ledger, resolve_current_run
from src.utils import ensure_dir
from src.visualizers import ResultsVisualizer, TrainingVisualizer

logger = logging.getLogger(__name__)

_DEFAULT_DATASET_NAME = "CLINC150"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo_relative(path: Path) -> str:
    """Return the repo-relative string for *path*."""
    return str(path.relative_to(PROJECT_ROOT))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


# ---------------------------------------------------------------------------
# Figure record (collected during generation, used for the manifest)
# ---------------------------------------------------------------------------


@dataclass
class FigureRecord:
    """Metadata for a single generated figure."""

    figure_path: str
    figure_type: str
    scope: str  # "representative", "aggregate", or "analysis"
    source_artifact_paths: list[str]
    model_name: str | None = None
    representative_run_id: str | None = None
    seed: int | None = None
    best_epoch: int | None = None
    stopping_epoch: int | None = None
    caption_context: FigureCaptionContext = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Resolved representative run context
# ---------------------------------------------------------------------------


@dataclass
class RepresentativeContext:
    """All paths and metadata needed to generate figures for one model's representative run."""

    model_id: ModelID
    run_id: str
    seed: int
    best_epoch: int
    stopping_epoch: int
    selection_rule: str
    artifact_dir: Path
    figures_dir: Path
    epoch_history: list[dict[str, Any]] = field(default_factory=list)
    confusion_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    class_metrics: list[dict[str, Any]] = field(default_factory=list)
    top_confusions: list[dict[str, Any]] = field(default_factory=list)
    top_errors: list[dict[str, Any]] = field(default_factory=list)
    oos_metrics: dict[str, float] = field(default_factory=dict)
    label_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Preflight validation
# ---------------------------------------------------------------------------

_PER_MODEL_REQUIRED = (f"{AGGREGATE_SUBDIR}/representative_run.json", f"{AGGREGATE_SUBDIR}/run_ledger.json")

_PER_RUN_REQUIRED = (
    "run_metadata.json",
    "epoch_history.json",
    "confusion_matrix.csv",
    "per_class_metrics.json",
    "top_confusions.json",
    "top_errors.json",
    "test_metrics.json",
    "label_order.json",
    "final_predictions.csv",
)

_SHARED_REQUIRED = (
    "model_comparison_aggregate.json",
    "oos_summary_table.json",
    "efficiency_summary_table.json",
)


def run_preflight_validation(model_ids: list[ModelID]) -> None:
    """Verify all required source artifacts exist before generating any figure.

    Raises ``FileNotFoundError`` with a summary of all missing paths.
    """
    missing: list[str] = []

    for shared_name in _SHARED_REQUIRED:
        p = SHARED_DIR / shared_name
        if not p.exists():
            missing.append(str(p))

    for mid in model_ids:
        model_dir = model_output_dir(mid)
        for rel in _PER_MODEL_REQUIRED:
            p = model_dir / rel
            if not p.exists():
                missing.append(str(p))

        rep_path = model_dir / AGGREGATE_SUBDIR / "representative_run.json"
        if rep_path.exists() and (model_dir / AGGREGATE_SUBDIR / "run_ledger.json").exists():
            _, run_dir = _load_current_representative(mid)
            for run_rel in _PER_RUN_REQUIRED:
                p = run_dir / run_rel
                if not p.exists():
                    missing.append(str(p))

    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(
            f"Preflight validation failed — {len(missing)} required artifact(s) missing:\n  {formatted}"
        )

    load_current_generation(
        list(ModelID), model_dirs={mid: model_output_dir(mid) for mid in ModelID}, project_root=PROJECT_ROOT
    )
    logger.info("Preflight validation passed for %d model(s).", len(model_ids))


# ---------------------------------------------------------------------------
# Resolve representative run context
# ---------------------------------------------------------------------------


def _load_current_representative(model_id: ModelID) -> tuple[dict[str, Any], Path]:
    """Resolve representative metadata against the authoritative current ledger."""
    model_dir = model_output_dir(model_id)
    rep_meta = _read_json(model_dir / AGGREGATE_SUBDIR / "representative_run.json")
    if rep_meta.get("model_id") != model_id:
        raise ValueError(f"Representative model_id does not match {model_id}")
    artifact_dir = resolve_current_run(
        model_id,
        rep_meta["run_id"],
        artifact_path=rep_meta["artifact_path"],
        model_dir=model_dir,
        project_root=PROJECT_ROOT,
    )
    run_meta = _read_json(artifact_dir / "run_metadata.json")
    if rep_meta.get("seed") != run_meta["seed"] or rep_meta.get("run_index") != run_meta["run_index"]:
        raise ValueError(f"Representative run_index/seed does not match current metadata for {model_id}")
    return rep_meta, artifact_dir


def _resolve_representative_context(model_id: ModelID) -> RepresentativeContext:
    """Load all artifacts for one model's current representative run."""
    model_dir = model_output_dir(model_id)
    rep_meta, artifact_dir = _load_current_representative(model_id)
    run_meta = _read_json(artifact_dir / "run_metadata.json")

    epoch_raw = _read_json(artifact_dir / "epoch_history.json")
    epoch_history: list[dict[str, Any]] = epoch_raw["epochs"]

    confusion_df = pd.read_csv(artifact_dir / "confusion_matrix.csv", index_col=0)

    class_raw = _read_json(artifact_dir / "per_class_metrics.json")
    class_metrics: list[dict[str, Any]] = class_raw["classes"]

    top_conf_raw = _read_json(artifact_dir / "top_confusions.json")
    top_confusions: list[dict[str, Any]] = top_conf_raw["confusions"]

    top_err_raw = _read_json(artifact_dir / "top_errors.json")
    top_errors: list[dict[str, Any]] = top_err_raw["errors"]

    test_metrics = _read_json(artifact_dir / "test_metrics.json")
    oos_metrics = {
        "oos_precision": test_metrics["oos_precision"],
        "oos_recall": test_metrics["oos_recall"],
        "oos_f1": test_metrics["oos_f1"],
    }

    label_raw = _read_json(artifact_dir / "label_order.json")
    label_names: list[str] = label_raw["label_names"]

    figures_dir = ensure_dir(model_dir / MODEL_FIGURES_SUBDIR)

    return RepresentativeContext(
        model_id=model_id,
        run_id=rep_meta["run_id"],
        seed=rep_meta["seed"],
        best_epoch=run_meta["best_epoch"],
        stopping_epoch=run_meta["stopping_epoch"],
        selection_rule=rep_meta["selection_rule"],
        artifact_dir=artifact_dir,
        figures_dir=figures_dir,
        epoch_history=epoch_history,
        confusion_df=confusion_df,
        class_metrics=class_metrics,
        top_confusions=top_confusions,
        top_errors=top_errors,
        oos_metrics=oos_metrics,
        label_names=label_names,
    )


# ---------------------------------------------------------------------------
# Figure-manifest helpers
# ---------------------------------------------------------------------------


def _caption_context(
    *,
    analysis_basis: str,
    dataset_split: str | None = None,
    metric_names: list[str] | None = None,
    run_count: int | None = None,
    class_count: int | None = None,
    top_k: int | None = None,
    selection_rule: str | None = None,
) -> FigureCaptionContext:
    """Build a compact producer-side caption context."""

    context: FigureCaptionContext = {
        "analysis_basis": analysis_basis,
        "dataset_name": _DEFAULT_DATASET_NAME,
    }
    if dataset_split is not None:
        context["dataset_split"] = dataset_split
    if metric_names:
        context["metric_names"] = metric_names
    if run_count is not None:
        context["run_count"] = run_count
    if class_count is not None:
        context["class_count"] = class_count
    if top_k is not None:
        context["top_k"] = top_k
    if selection_rule is not None:
        context["selection_rule"] = selection_rule
    return context


def figure_record_to_manifest_entry(rec: FigureRecord) -> dict[str, Any]:
    """Serialize a ``FigureRecord`` into the manifest entry format."""

    entry: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "figure_path": rec.figure_path,
        "figure_type": rec.figure_type,
        "scope": rec.scope,
        "source_artifact_paths": rec.source_artifact_paths,
    }
    if rec.model_name is not None:
        entry["model_name"] = rec.model_name
    if rec.representative_run_id is not None:
        entry["representative_run_id"] = rec.representative_run_id
    if rec.seed is not None:
        entry["seed"] = rec.seed
    if rec.best_epoch is not None:
        entry["best_epoch"] = rec.best_epoch
    if rec.stopping_epoch is not None:
        entry["stopping_epoch"] = rec.stopping_epoch
    if rec.caption_context:
        entry["caption_context"] = rec.caption_context
    return entry


# ---------------------------------------------------------------------------
# Representative-run figure generation
# ---------------------------------------------------------------------------


def generate_representative_figures(model_id: ModelID) -> list[FigureRecord]:
    """Generate all representative-run figures for one model.

    Returns a list of :class:`FigureRecord` for the figure manifest.
    """
    ctx = _resolve_representative_context(model_id)
    records: list[FigureRecord] = []

    def _src(*names: str) -> list[str]:
        return [_repo_relative(ctx.artifact_dir / n) for n in names]

    training_record_kwargs = {
        "seed": ctx.seed,
        "best_epoch": ctx.best_epoch,
        "stopping_epoch": ctx.stopping_epoch,
        "caption_context": _caption_context(
            analysis_basis="representative run",
            dataset_split="validation",
            selection_rule=ctx.selection_rule,
        ),
    }

    # --- Training curves ---
    loss_path = ctx.figures_dir / "representative_train_val_loss_curve.png"
    TrainingVisualizer.plot_loss_curves(ctx.epoch_history, loss_path, best_epoch=ctx.best_epoch)
    records.append(
        FigureRecord(
            figure_path=_repo_relative(loss_path),
            figure_type="train_val_loss_curve",
            scope="representative",
            source_artifact_paths=_src("epoch_history.json"),
            model_name=str(model_id),
            representative_run_id=ctx.run_id,
            **training_record_kwargs,
        )
    )
    logger.info("Saved: %s", loss_path)

    f1_path = ctx.figures_dir / "representative_val_macro_f1_curve.png"
    TrainingVisualizer.plot_val_metric_curve(
        ctx.epoch_history,
        f1_path,
        metric_key="val_macro_f1",
        best_epoch=ctx.best_epoch,
    )
    records.append(
        FigureRecord(
            figure_path=_repo_relative(f1_path),
            figure_type="val_macro_f1_curve",
            scope="representative",
            source_artifact_paths=_src("epoch_history.json"),
            model_name=str(model_id),
            representative_run_id=ctx.run_id,
            **training_record_kwargs,
        )
    )
    logger.info("Saved: %s", f1_path)

    acc_path = ctx.figures_dir / "representative_val_accuracy_curve.png"
    TrainingVisualizer.plot_val_metric_curve(
        ctx.epoch_history,
        acc_path,
        metric_key="val_accuracy",
        best_epoch=ctx.best_epoch,
    )
    records.append(
        FigureRecord(
            figure_path=_repo_relative(acc_path),
            figure_type="val_accuracy_curve",
            scope="representative",
            source_artifact_paths=_src("epoch_history.json"),
            model_name=str(model_id),
            representative_run_id=ctx.run_id,
            **training_record_kwargs,
        )
    )
    logger.info("Saved: %s", acc_path)

    # --- Confusion matrix ---
    cm_path = ctx.figures_dir / "representative_confusion_matrix.png"
    ResultsVisualizer.plot_confusion_matrix(ctx.confusion_df, cm_path)
    records.append(
        FigureRecord(
            figure_path=_repo_relative(cm_path),
            figure_type="confusion_matrix",
            scope="representative",
            source_artifact_paths=_src("confusion_matrix.csv"),
            model_name=str(model_id),
            representative_run_id=ctx.run_id,
            caption_context=_caption_context(
                analysis_basis="representative run",
                dataset_split="test",
                class_count=len(ctx.label_names),
                selection_rule=ctx.selection_rule,
            ),
        )
    )
    logger.info("Saved: %s", cm_path)

    # --- Top confused pairs ---
    tcp_path = ctx.figures_dir / "representative_top_confused_pairs.png"
    ResultsVisualizer.plot_top_confused_pairs(ctx.top_confusions, tcp_path)
    records.append(
        FigureRecord(
            figure_path=_repo_relative(tcp_path),
            figure_type="top_confused_pairs",
            scope="representative",
            source_artifact_paths=_src("top_confusions.json"),
            model_name=str(model_id),
            representative_run_id=ctx.run_id,
            caption_context=_caption_context(
                analysis_basis="representative run",
                dataset_split="test",
                top_k=len(ctx.top_confusions),
                selection_rule=ctx.selection_rule,
            ),
        )
    )
    logger.info("Saved: %s", tcp_path)

    # --- Bottom classes by F1 ---
    bf1_path = ctx.figures_dir / "representative_bottom_classes_f1.png"
    ResultsVisualizer.plot_bottom_classes_f1_from_metrics(ctx.class_metrics, bf1_path)
    records.append(
        FigureRecord(
            figure_path=_repo_relative(bf1_path),
            figure_type="bottom_classes_f1",
            scope="representative",
            source_artifact_paths=_src("per_class_metrics.json"),
            model_name=str(model_id),
            representative_run_id=ctx.run_id,
            caption_context=_caption_context(
                analysis_basis="representative run",
                dataset_split="test",
                selection_rule=ctx.selection_rule,
            ),
        )
    )
    logger.info("Saved: %s", bf1_path)

    # --- OOS metrics ---
    oos_path = ctx.figures_dir / "representative_oos_metrics.png"
    ResultsVisualizer.plot_oos_metrics(ctx.oos_metrics, oos_path)
    records.append(
        FigureRecord(
            figure_path=_repo_relative(oos_path),
            figure_type="oos_metrics",
            scope="representative",
            source_artifact_paths=_src("test_metrics.json"),
            model_name=str(model_id),
            representative_run_id=ctx.run_id,
            caption_context=_caption_context(
                analysis_basis="representative run",
                dataset_split="test",
                metric_names=["OOS precision", "OOS recall", "OOS F1"],
                selection_rule=ctx.selection_rule,
            ),
        )
    )
    logger.info("Saved: %s", oos_path)

    # --- Error summary ---
    err_path = ctx.figures_dir / "representative_error_summary.png"
    ResultsVisualizer.plot_error_summary(ctx.top_errors, err_path)
    records.append(
        FigureRecord(
            figure_path=_repo_relative(err_path),
            figure_type="error_summary",
            scope="representative",
            source_artifact_paths=_src("top_errors.json"),
            model_name=str(model_id),
            representative_run_id=ctx.run_id,
            caption_context=_caption_context(
                analysis_basis="representative run",
                dataset_split="test",
                top_k=len(ctx.top_errors),
                selection_rule=ctx.selection_rule,
            ),
        )
    )
    logger.info("Saved: %s", err_path)

    return records


# ---------------------------------------------------------------------------
# Aggregate cross-model figure generation
# ---------------------------------------------------------------------------


def generate_aggregate_figures() -> list[FigureRecord]:
    """Generate shared aggregate comparison figures from the tracking tables."""
    load_current_generation(
        list(ModelID), model_dirs={mid: model_output_dir(mid) for mid in ModelID}, project_root=PROJECT_ROOT
    )
    figures_dir = ensure_dir(SHARED_FIGURES_DIR)
    records: list[FigureRecord] = []

    comp_path = SHARED_DIR / "model_comparison_aggregate.json"
    comp_data = _read_json(comp_path)
    comp_rows: list[dict[str, Any]] = comp_data["rows"]
    comp_src = [_repo_relative(comp_path)]
    run_count = int(comp_rows[0]["run_count"]) if comp_rows else None

    # Test accuracy comparison
    acc_stats = [(r["display_name"], r["test_accuracy_mean"], r["test_accuracy_std"]) for r in comp_rows]
    acc_fig = figures_dir / "model_comparison_test_accuracy.png"
    ResultsVisualizer.plot_model_comparison_bar(acc_stats, "Test Accuracy", acc_fig)
    records.append(
        FigureRecord(
            figure_path=_repo_relative(acc_fig),
            figure_type="model_comparison_test_accuracy",
            scope="aggregate",
            source_artifact_paths=comp_src,
            caption_context=_caption_context(
                analysis_basis="aggregate comparison across repeated evaluation runs",
                dataset_split="test",
                metric_names=["test accuracy"],
                run_count=run_count,
            ),
        )
    )
    logger.info("Saved: %s", acc_fig)

    # Test macro F1 comparison
    f1_stats = [(r["display_name"], r["test_macro_f1_mean"], r["test_macro_f1_std"]) for r in comp_rows]
    f1_fig = figures_dir / "model_comparison_test_macro_f1.png"
    ResultsVisualizer.plot_model_comparison_bar(f1_stats, "Test Macro F1", f1_fig)
    records.append(
        FigureRecord(
            figure_path=_repo_relative(f1_fig),
            figure_type="model_comparison_test_macro_f1",
            scope="aggregate",
            source_artifact_paths=comp_src,
            caption_context=_caption_context(
                analysis_basis="aggregate comparison across repeated evaluation runs",
                dataset_split="test",
                metric_names=["test macro F1"],
                run_count=run_count,
            ),
        )
    )
    logger.info("Saved: %s", f1_fig)

    # OOS F1 comparison
    oos_f1_stats = [(r["display_name"], r["oos_f1_mean"], r["oos_f1_std"]) for r in comp_rows]
    oos_f1_fig = figures_dir / "model_comparison_oos_f1.png"
    ResultsVisualizer.plot_model_comparison_bar(oos_f1_stats, "OOS F1", oos_f1_fig)
    records.append(
        FigureRecord(
            figure_path=_repo_relative(oos_f1_fig),
            figure_type="model_comparison_oos_f1",
            scope="aggregate",
            source_artifact_paths=comp_src,
            caption_context=_caption_context(
                analysis_basis="aggregate comparison across repeated evaluation runs",
                dataset_split="test",
                metric_names=["OOS F1"],
                run_count=run_count,
            ),
        )
    )
    logger.info("Saved: %s", oos_f1_fig)

    # OOS metrics comparison (precision, recall, F1)
    oos_path = SHARED_DIR / "oos_summary_table.json"
    oos_data = _read_json(oos_path)
    oos_rows: list[dict[str, Any]] = oos_data["rows"]
    oos_fig = figures_dir / "oos_metrics_comparison.png"
    ResultsVisualizer.plot_oos_metrics_comparison(oos_rows, oos_fig)
    records.append(
        FigureRecord(
            figure_path=_repo_relative(oos_fig),
            figure_type="oos_metrics_comparison",
            scope="aggregate",
            source_artifact_paths=[_repo_relative(oos_path)],
            caption_context=_caption_context(
                analysis_basis="aggregate comparison across repeated evaluation runs",
                dataset_split="test",
                metric_names=["OOS precision", "OOS recall", "OOS F1"],
                run_count=run_count,
            ),
        )
    )
    logger.info("Saved: %s", oos_fig)

    # Efficiency comparison
    eff_path = SHARED_DIR / "efficiency_summary_table.json"
    eff_data = _read_json(eff_path)
    eff_rows: list[dict[str, Any]] = eff_data["rows"]
    eff_fig = figures_dir / "model_efficiency_comparison.png"
    ResultsVisualizer.plot_efficiency_comparison(eff_rows, eff_fig)
    records.append(
        FigureRecord(
            figure_path=_repo_relative(eff_fig),
            figure_type="model_efficiency_comparison",
            scope="aggregate",
            source_artifact_paths=[_repo_relative(eff_path)],
            caption_context=_caption_context(
                analysis_basis="aggregate comparison across repeated evaluation runs",
                metric_names=["parameter count", "training time", "inference throughput"],
                run_count=run_count,
            ),
        )
    )
    logger.info("Saved: %s", eff_fig)

    return records


# ---------------------------------------------------------------------------
# Representative misclassifications
# ---------------------------------------------------------------------------


def generate_representative_misclassifications(model_id: ModelID) -> Path:
    """Extract misclassified examples from the representative run's predictions.

    Returns the path to the saved CSV.
    """
    model_dir = model_output_dir(model_id)
    _, artifact_dir = _load_current_representative(model_id)

    preds_df = pd.read_csv(artifact_dir / "final_predictions.csv")
    misclassified = preds_df[preds_df["true_label_id"] != preds_df["predicted_label_id"]].copy()

    keep_cols = ["text", "true_label_name", "predicted_label_name", "run_id"]
    if "max_confidence" in misclassified.columns:
        keep_cols.append("max_confidence")
    misclassified = misclassified[keep_cols]

    out_path = model_dir / AGGREGATE_SUBDIR / "representative_misclassifications.csv"
    misclassified.to_csv(out_path, index=False)
    logger.info("Saved: %s (%d rows)", out_path, len(misclassified))
    return out_path


# ---------------------------------------------------------------------------
# Most confused pairs table (shared, across models)
# ---------------------------------------------------------------------------


def generate_most_confused_pairs_table(model_ids: list[ModelID]) -> tuple[Path, Path]:
    """Combine representative-run top-confusion data across models into shared tables."""
    rows: list[dict[str, Any]] = []

    for mid in model_ids:
        rep_meta, artifact_dir = _load_current_representative(mid)
        top_conf = _read_json(artifact_dir / "top_confusions.json")

        for rank, conf in enumerate(top_conf["confusions"], start=1):
            rows.append(
                {
                    "model_name": mid.display_name,
                    "model_id": str(mid),
                    "representative_run_id": rep_meta["run_id"],
                    "rank": rank,
                    "true_label_name": conf["true_label"],
                    "predicted_label_name": conf["predicted_label"],
                    "count": conf["count"],
                }
            )

    csv_path = SHARED_DIR / "most_confused_pairs_table.csv"
    json_path = SHARED_DIR / "most_confused_pairs_table.json"

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    table_json: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "rows": rows,
    }
    _write_json(json_path, table_json)
    logger.info("Saved: %s, %s (%d rows)", csv_path, json_path, len(rows))
    return csv_path, json_path


# ---------------------------------------------------------------------------
# Representative examples index
# ---------------------------------------------------------------------------


def generate_representative_examples_index(model_ids: list[ModelID]) -> Path:
    """Build the shared index mapping each model to its representative-run artifacts."""
    entries: dict[str, dict[str, Any]] = {}

    for mid in model_ids:
        model_dir = model_output_dir(mid)
        agg_dir = model_dir / AGGREGATE_SUBDIR
        rep_meta, artifact_dir = _load_current_representative(mid)

        entry: dict[str, Any] = {
            "representative_run_id": rep_meta["run_id"],
            "representative_predictions_path": _repo_relative(artifact_dir / "final_predictions.csv"),
            "representative_misclassifications_path": _repo_relative(agg_dir / "representative_misclassifications.csv"),
            "representative_confusion_path": _repo_relative(artifact_dir / "confusion_matrix.csv"),
        }

        handoff_path = agg_dir / "step10_handoff.json"
        if handoff_path.exists():
            entry["step10_handoff_ref"] = _repo_relative(handoff_path)

        entries[str(mid)] = entry

    index: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "models": entries,
    }
    out_path = SHARED_DIR / "representative_examples_index.json"
    _write_json(out_path, index)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Figure manifest
# ---------------------------------------------------------------------------


def generate_figure_manifest(figure_records: list[FigureRecord]) -> Path:
    """Write the shared figure manifest from collected figure records."""
    entries = [figure_record_to_manifest_entry(rec) for rec in figure_records]

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "figures": entries,
    }
    out_path = SHARED_DIR / "figure_manifest.json"
    _write_json(out_path, manifest)
    logger.info("Saved: %s (%d figure entries)", out_path, len(entries))
    return out_path


# ---------------------------------------------------------------------------
# Step 10 handoff (per model)
# ---------------------------------------------------------------------------


def generate_step10_handoff(model_id: ModelID) -> Path:
    """Write the Step 10 handoff bundle for one model."""
    model_dir = model_output_dir(model_id)
    agg_dir = model_dir / AGGREGATE_SUBDIR
    fig_dir = model_dir / MODEL_FIGURES_SUBDIR
    rep_meta, artifact_dir = _load_current_representative(model_id)

    handoff: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_id": str(model_id),
        "model_name": model_id.display_name,
        "representative_run_id": rep_meta["run_id"],
        "representative_seed": rep_meta["seed"],
        "total_runs_aggregated": len(
            load_current_run_ledger(model_id, model_dir=model_dir, project_root=PROJECT_ROOT).completed_run_ids
        ),
        "selection_rule": rep_meta["selection_rule"],
        "note": (
            "Aggregate metrics come from all completed runs. "
            "Representative-run artifacts are for qualitative analysis only."
        ),
        "artifacts": {
            "aggregate_metrics": _repo_relative(agg_dir / "aggregate_metrics.json"),
            "representative_run_metadata": _repo_relative(agg_dir / "representative_run.json"),
            "final_predictions": _repo_relative(artifact_dir / "final_predictions.csv"),
            "representative_misclassifications": _repo_relative(agg_dir / "representative_misclassifications.csv"),
            "top_errors": _repo_relative(artifact_dir / "top_errors.json"),
            "confusion_matrix": _repo_relative(artifact_dir / "confusion_matrix.csv"),
            "top_confusions": _repo_relative(artifact_dir / "top_confusions.json"),
            "per_class_metrics": _repo_relative(artifact_dir / "per_class_metrics.json"),
            "label_order": _repo_relative(artifact_dir / "label_order.json"),
        },
        "figures": {
            "train_val_loss_curve": _repo_relative(fig_dir / "representative_train_val_loss_curve.png"),
            "val_macro_f1_curve": _repo_relative(fig_dir / "representative_val_macro_f1_curve.png"),
            "val_accuracy_curve": _repo_relative(fig_dir / "representative_val_accuracy_curve.png"),
            "confusion_matrix": _repo_relative(fig_dir / "representative_confusion_matrix.png"),
            "top_confused_pairs": _repo_relative(fig_dir / "representative_top_confused_pairs.png"),
            "bottom_classes_f1": _repo_relative(fig_dir / "representative_bottom_classes_f1.png"),
            "oos_metrics": _repo_relative(fig_dir / "representative_oos_metrics.png"),
            "error_summary": _repo_relative(fig_dir / "representative_error_summary.png"),
        },
    }

    out_path = agg_dir / "step10_handoff.json"
    _write_json(out_path, handoff)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Post-generation validation
# ---------------------------------------------------------------------------


def _validate_manifest_paths(manifest_path: Path) -> None:
    """Assert every figure path recorded in the manifest exists on disk."""
    manifest = _read_json(manifest_path)
    missing: list[str] = []
    for entry in manifest["figures"]:
        p = PROJECT_ROOT / entry["figure_path"]
        if not p.exists():
            missing.append(entry["figure_path"])
    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(f"Figure manifest references {len(missing)} missing file(s):\n  {formatted}")


def _validate_handoff_paths(handoff_path: Path) -> None:
    """Assert every path recorded in a Step 10 handoff exists on disk."""
    handoff = _read_json(handoff_path)
    missing: list[str] = []
    for section_key in ("artifacts", "figures"):
        for _name, rel_path in handoff.get(section_key, {}).items():
            p = PROJECT_ROOT / rel_path
            if not p.exists():
                missing.append(rel_path)
    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(f"Step 10 handoff references {len(missing)} missing file(s):\n  {formatted}")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_report_figure_generation(model_ids: list[ModelID]) -> bool:
    """Run the full figure-generation and handoff pipeline.

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        run_preflight_validation(model_ids)

        all_records: list[FigureRecord] = []

        # Representative-run figures and misclassifications per model
        for mid in model_ids:
            records = generate_representative_figures(mid)
            all_records.extend(records)
            generate_representative_misclassifications(mid)

        # Aggregate cross-model figures
        agg_records = generate_aggregate_figures()
        all_records.extend(agg_records)

        # Shared metadata tables
        generate_most_confused_pairs_table(model_ids)

        # Step 10 handoff per model
        for mid in model_ids:
            generate_step10_handoff(mid)

        # Representative examples index (after handoffs exist)
        generate_representative_examples_index(model_ids)

        # Figure manifest
        manifest_path = generate_figure_manifest(all_records)

        # Post-generation validation
        _validate_manifest_paths(manifest_path)
        for mid in model_ids:
            handoff_path = model_output_dir(mid) / AGGREGATE_SUBDIR / "step10_handoff.json"
            _validate_handoff_paths(handoff_path)

        logger.info("Report figure generation complete — all validations passed.")
        return True

    except Exception:
        logger.exception("Report figure generation failed.")
        return False
