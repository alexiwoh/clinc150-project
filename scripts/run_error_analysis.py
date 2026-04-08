"""CLI entry point for Step 10 error analysis.

Usage::

    python scripts/run_error_analysis.py --models all
    python scripts/run_error_analysis.py --models mlp text_cnn
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.analysis.pipeline import run_error_analysis
from src.enums import ModelID


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 10 error analysis")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", "mlp", "text_cnn", "bilstm"],
        help="Models to analyze (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    if "all" in args.models:
        model_ids = list(ModelID)
    else:
        model_ids = [ModelID(m) for m in args.models]

    print(f"Running Step 10 error analysis for: {[m.display_name for m in model_ids]}")

    success = run_error_analysis(model_ids)
    if not success:
        print("Step 10 error analysis FAILED.", file=sys.stderr)
        sys.exit(1)

    print("Step 10 error analysis completed successfully.")


if __name__ == "__main__":
    main()
