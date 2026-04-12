"""CLI entry point for Step 11: report generation.

Usage::

    python scripts/run_report_generation.py
    python scripts/run_report_generation.py --sections abstract main_results key_findings
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from typing import Any

from src.report.artifact_loader import save_section
from src.report.assembler import assemble_full_report, generate_report_manifest
from src.report.figures import generate_figure_catalogue
from src.report.preflight import validate_artifacts
from src.report.sections import (
    generate_abstract,
    generate_dataset_description,
    generate_error_analysis,
    generate_experimental_setup,
    generate_future_improvements,
    generate_key_findings,
    generate_limitations,
    generate_main_results,
    generate_model_architectures,
    generate_oos_detection,
    generate_preprocessing_summary,
    generate_representative_examples,
    generate_reproducibility,
)
from src.report.validation import validate_report_outputs

logger = logging.getLogger(__name__)

SectionOutput = tuple[str, dict[str, Any]]

SECTION_GENERATORS: dict[str, Callable[[], SectionOutput]] = {
    "abstract": generate_abstract,
    "dataset_description": generate_dataset_description,
    "preprocessing_summary_report": generate_preprocessing_summary,
    "model_architectures": generate_model_architectures,
    "experimental_setup": generate_experimental_setup,
    "main_results": generate_main_results,
    "oos_detection_results": generate_oos_detection,
    "error_analysis_discussion": generate_error_analysis,
    "representative_examples": generate_representative_examples,
    "key_findings": generate_key_findings,
    "limitations": generate_limitations,
    "future_improvements": generate_future_improvements,
    "reproducibility": generate_reproducibility,
    "figure_catalogue": generate_figure_catalogue,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 11: generate report section drafts and assembled report.")
    parser.add_argument(
        "--sections",
        nargs="*",
        default=None,
        choices=list(SECTION_GENERATORS),
        help="Generate only the specified sections. Default: all.",
    )
    return parser.parse_args(argv)


def run_report_generation(sections: list[str] | None = None) -> bool:
    """Generate report sections, manifest, assembled report, and validate.

    Returns ``True`` on success.
    """
    logger.info("Step 11: Report generation starting.")

    logger.info("Running preflight validation...")
    validate_artifacts()
    logger.info("Preflight passed.")

    targets = sections or list(SECTION_GENERATORS)

    for section_id in targets:
        gen = SECTION_GENERATORS[section_id]
        logger.info("Generating section: %s", section_id)
        md, metadata = gen()
        save_section(section_id, md, metadata)

    logger.info("Building report structure manifest...")
    generate_report_manifest()

    logger.info("Assembling full report draft...")
    assemble_full_report()

    logger.info("Running output validation...")
    failures = validate_report_outputs()

    if failures:
        logger.error("Validation found %d issue(s):", len(failures))
        for msg in failures:
            logger.error("  %s", msg)
        return False

    logger.info("Step 11 complete: all validation checks passed.")
    return True


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = _parse_args(argv)

    print("=" * 60)
    print("  CLINC150 Report Generation (Step 11)")
    print("=" * 60)
    print(f"  Sections: {args.sections or 'all'}")
    print()

    try:
        success = run_report_generation(args.sections)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    if not success:
        print("\nReport generation completed with validation failures.")
        sys.exit(1)

    print("\nReport generation complete.")


if __name__ == "__main__":
    main()
