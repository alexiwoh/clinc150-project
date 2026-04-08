"""Dedicated Step 7 entry point: repeated-run evaluation only (no tuning).

Assumes frozen configs already exist (produced by Phase 1 of the pipeline or
a previous run). Runs the repeated-evaluation phase for selected models.

Usage:
    python scripts/run_repeated_evaluation.py --model all --run-count 3
    python scripts/run_repeated_evaluation.py --model mlp
    python scripts/run_repeated_evaluation.py --model bilstm --run-count 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import FrozenModelConfig, RepeatedRunProtocol
from src.constants import DEFAULT_SEED_LIST
from src.enums import ModelID
from src.repeated_evaluation import (
    ModelEvaluationResult,
    run_repeated_evaluation,
    save_evaluation_protocol,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 7: Run repeated evaluation using existing frozen configs.",
    )
    parser.add_argument(
        "--model",
        choices=[*ModelID, "all"],
        default="all",
        help="Model(s) to evaluate. 'all' runs mlp, text_cnn, bilstm in order.",
    )
    parser.add_argument(
        "--run-count",
        type=int,
        default=3,
        help="Number of repeated final runs per model (default: 3).",
    )
    return parser.parse_args(argv)


def _resolve_models(model_arg: str) -> list[ModelID]:
    if model_arg == "all":
        return list(ModelID)
    return [ModelID(model_arg)]


def _load_frozen_config(model_id: ModelID) -> FrozenModelConfig:
    """Load a frozen config from disk."""
    from src.constants import model_output_dir

    config_path: Path = model_output_dir(model_id) / "frozen_final_config.json"
    assert config_path.exists(), (
        f"Frozen config not found for '{model_id}' at {config_path}. "
        f"Run the full pipeline first (scripts/run_model_pipeline.py) to generate it."
    )
    data: dict = json.loads(config_path.read_text())
    return FrozenModelConfig(**data)


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
    print("  Step 7: Repeated-Run Evaluation (frozen configs only)")
    print("=" * 60)
    print(f"  Models:    {[str(m) for m in models]}")
    print(f"  Run count: {run_count}")
    print(f"  Seeds:     {protocol.effective_seed_list()}")
    print()

    save_evaluation_protocol(protocol, models)

    results: dict[ModelID, ModelEvaluationResult] = {}
    for model_id in models:
        frozen = _load_frozen_config(model_id)
        result = run_repeated_evaluation(model_id, frozen, protocol)
        results[model_id] = result

    # Cross-model seed validation
    seed_lists = [r.aggregate.seed_list_completed for r in results.values()]
    if len(seed_lists) > 1:
        for sl in seed_lists[1:]:
            assert sl == seed_lists[0], f"Seed lists differ across models: {seed_lists}"

    print("\n" + "=" * 60)
    print("  Repeated evaluation complete.")
    print("=" * 60)
    for model_id, result in results.items():
        agg = result.aggregate.metrics
        rep = result.representative
        print(f"  {result.frozen_config.model_name}:")
        print(f"    Test accuracy:  {agg['test_accuracy']['mean']:.4f} ± {agg['test_accuracy']['std']:.4f}")
        print(f"    Test macro F1:  {agg['test_macro_f1']['mean']:.4f} ± {agg['test_macro_f1']['std']:.4f}")
        print(f"    OOS F1:         {agg['oos_f1']['mean']:.4f} ± {agg['oos_f1']['std']:.4f}")
        print(f"    Representative: {rep.run_id}")
    print()


if __name__ == "__main__":
    main()
