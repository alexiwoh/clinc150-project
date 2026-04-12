"""Report generation package for Step 11 of the CLINC150 project.

Reads upstream artifacts from Steps 2-10 and produces structured markdown
section drafts, JSON metadata, a figure catalogue, and one assembled report
under ``outputs/shared/report/``.  No retraining, no new analysis.
"""

from src.report.artifact_loader import (
    REPORT_DIR,
    load_handoff,
    load_json,
    repo_relative,
    resolve_repo_path,
    save_section,
    write_json,
)
from src.report.assembler import (
    SECTION_ORDER,
    assemble_full_report,
    generate_report_manifest,
)
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
from src.report.tables import (
    build_markdown_table,
    format_mean_std,
    format_ms_per_example,
    format_param_count,
    format_ratio,
    format_time_seconds,
)
from src.report.validation import validate_report_outputs

__all__ = [
    "REPORT_DIR",
    "SECTION_ORDER",
    "assemble_full_report",
    "build_markdown_table",
    "format_mean_std",
    "format_ms_per_example",
    "format_param_count",
    "format_ratio",
    "format_time_seconds",
    "generate_abstract",
    "generate_dataset_description",
    "generate_error_analysis",
    "generate_experimental_setup",
    "generate_figure_catalogue",
    "generate_future_improvements",
    "generate_key_findings",
    "generate_limitations",
    "generate_main_results",
    "generate_model_architectures",
    "generate_oos_detection",
    "generate_preprocessing_summary",
    "generate_representative_examples",
    "generate_report_manifest",
    "generate_reproducibility",
    "load_handoff",
    "load_json",
    "repo_relative",
    "resolve_repo_path",
    "save_section",
    "validate_artifacts",
    "validate_report_outputs",
    "write_json",
]
