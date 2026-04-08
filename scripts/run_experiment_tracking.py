"""Experiment tracking entry point: validation, enrichment, and cross-model comparison.

Usage:
    python scripts/run_experiment_tracking.py --model all
    python scripts/run_experiment_tracking.py --model mlp
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.enums import ModelID
from src.experiment_tracking import run_experiment_tracking


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment tracking — validate, enrich, and build cross-model tables.",
    )
    parser.add_argument(
        "--model",
        choices=[*ModelID, "all"],
        default="all",
        help="Model(s) to track. 'all' runs mlp, text_cnn, bilstm in order.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = _parse_args(argv)

    if args.model == "all":
        model_ids: list[ModelID] = list(ModelID)
    else:
        model_ids = [ModelID(args.model)]

    success = run_experiment_tracking(model_ids)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
