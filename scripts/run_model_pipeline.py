"""Canonical pipeline runner for tuning reuse, repeated evaluation, tracking, and visualization.

This is the primary user-facing entry point for Steps 7-9 of the CLINC150 project.
It orchestrates frozen-config extraction, repeated final runs, artifact tracking, and
figure generation across all three models.

Usage:
    python scripts/run_model_pipeline.py --model all --run-count 3
    python scripts/run_model_pipeline.py --model mlp --run-count 5
    python scripts/run_model_pipeline.py --model text_cnn --retune
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import RepeatedRunProtocol
from src.constants import DEFAULT_SEED_LIST
from src.enums import ModelID
from src.experiment_tracking import run_experiment_tracking
from src.repeated_evaluation import (
    ModelEvaluationResult,
    extract_and_freeze_configs,
    run_all_repeated_evaluations,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLINC150 model pipeline: tuning reuse, repeated evaluation, tracking, and visualization.",
    )
    parser.add_argument(
        "--model",
        choices=[*ModelID, "all"],
        default="all",
        help="Model(s) to run. 'all' runs mlp, text_cnn, bilstm in order.",
    )
    parser.add_argument(
        "--run-count",
        type=int,
        default=3,
        help="Number of repeated final runs per model (default: 3).",
    )
    parser.add_argument(
        "--retune",
        action="store_true",
        help="Force re-run hyperparameter tuning even if valid tuning artifacts exist.",
    )
    return parser.parse_args(argv)


def _resolve_models(model_arg: str) -> list[ModelID]:
    if model_arg == "all":
        return list(ModelID)
    return [ModelID(model_arg)]


def _print_final_summary(results: dict[ModelID, ModelEvaluationResult]) -> None:
    """Print a cross-model summary after all evaluations complete."""
    print("\n" + "=" * 60)
    print("  Final Cross-Model Summary")
    print("=" * 60)

    header = f"  {'Model':<15} {'Test Acc':>10} {'Test F1':>10} {'OOS F1':>10} {'Time (s)':>10}"
    print(header)
    print("  " + "-" * 55)

    for model_id, result in results.items():
        agg = result.aggregate.metrics
        print(
            f"  {result.frozen_config.model_name:<15}"
            f" {agg['test_accuracy']['mean']:>9.4f}"
            f" {agg['test_macro_f1']['mean']:>9.4f}"
            f" {agg['oos_f1']['mean']:>9.4f}"
            f" {agg['training_time_seconds']['mean']:>9.1f}"
        )

    print()
    seed_lists = [r.aggregate.seed_list_completed for r in results.values()]
    if seed_lists:
        print(f"  Shared seed list: {seed_lists[0]}")
        all_same = all(sl == seed_lists[0] for sl in seed_lists)
        print(f"  Same seeds across all models: {all_same}")

    for model_id, result in results.items():
        rep = result.representative
        print(f"  {result.frozen_config.model_name} representative: {rep.run_id} (seed={rep.seed})")

    print("=" * 60)
    print()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = _parse_args(argv)
    models = _resolve_models(args.model)
    run_count: int = args.run_count

    assert run_count <= len(DEFAULT_SEED_LIST), (
        f"run_count={run_count} exceeds available seeds ({len(DEFAULT_SEED_LIST)}). "
        f"Add more seeds to DEFAULT_SEED_LIST in src/constants.py."
    )

    protocol = RepeatedRunProtocol(run_count=run_count)

    print("=" * 60)
    print("  CLINC150 Model Pipeline")
    print("=" * 60)
    print(f"  Models:    {[str(m) for m in models]}")
    print(f"  Run count: {run_count}")
    print(f"  Seeds:     {protocol.effective_seed_list()}")
    print(f"  Retune:    {args.retune}")
    print()

    if args.retune:
        print("  [--retune] Retuning is not yet supported in this pipeline.")
        print("  Use the per-model scripts (run_mlp_baseline.py, etc.) to retune,")
        print("  then rerun this pipeline without --retune.")
        sys.exit(1)

    # Frozen-config extraction (tuning normalization, protocol, winning-row selection)
    frozen_configs = extract_and_freeze_configs(models, protocol)

    # Repeated-run evaluation across shared seed list
    results = run_all_repeated_evaluations(models, frozen_configs, protocol)

    # Print final cross-model summary
    _print_final_summary(results)

    # Experiment tracking — same validation/materialization logic
    # as the standalone scripts/run_experiment_tracking.py entry point.
    tracking_success = run_experiment_tracking(models)

    print("=" * 60)
    print("  Pipeline Summary")
    print("=" * 60)
    print("  Repeated evaluation: complete")
    print(f"  Experiment tracking: {'PASS' if tracking_success else 'FAIL'}")
    print("=" * 60)

    if not tracking_success:
        print("Experiment tracking validation failed. Review errors above.")
        sys.exit(1)

    print("Model pipeline complete. Artifacts ready for visualization.")


if __name__ == "__main__":
    main()
