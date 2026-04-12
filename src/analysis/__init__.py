"""Error analysis package for Step 10 of the CLINC150 project.

Provides modular error analysis: calibration, OOS threshold analysis,
error taxonomy, confidence stratification, slicing, cross-model comparison,
per-class deep dives, and curated report examples.

All analysis is artifact-driven -- no model objects or re-inference required.
"""

from src.analysis.pipeline import run_error_analysis

__all__ = ["run_error_analysis"]
