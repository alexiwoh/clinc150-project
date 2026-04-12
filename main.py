"""Top-level project entry point for the CLINC150 intent classification pipeline.

Delegates to the canonical pipeline runner at ``scripts/run_model_pipeline.py``.
For per-model experiments without the pipeline, use the individual scripts::

    python scripts/run_mlp_baseline.py
    python scripts/run_text_cnn.py
    python scripts/run_bilstm.py
"""

from __future__ import annotations

import sys

from scripts.run_model_pipeline import main as pipeline_main


def main() -> None:
    pipeline_main(sys.argv[1:])


if __name__ == "__main__":
    main()
