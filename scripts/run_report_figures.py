"""Generate report-ready figures and error-analysis handoff bundles.

Reads saved tracking artifacts and per-run evaluation outputs, produces
representative-run diagnostic figures, aggregate cross-model comparison
figures, and the metadata bundles needed for error analysis and report
writing.

Usage:
    python scripts/run_report_figures.py --model all
    python scripts/run_report_figures.py --model mlp
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.enums import ModelID
from src.report_figure_generation import run_report_figure_generation


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report-ready figures and error-analysis handoff bundles.",
    )
    parser.add_argument(
        "--model",
        choices=[*ModelID, "all"],
        default="all",
        help="Model(s) to generate figures for. 'all' runs mlp, text_cnn, bilstm in order.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = _parse_args(argv)
    models = list(ModelID) if args.model == "all" else [ModelID(args.model)]

    success = run_report_figure_generation(models)
    if not success:
        print("Report figure generation failed. Review errors above.")
        sys.exit(1)

    print("Report figure generation complete.")


if __name__ == "__main__":
    main()
