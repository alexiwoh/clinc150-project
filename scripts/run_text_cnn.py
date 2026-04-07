"""Step 5 entry point: full Text CNN experiment pipeline.

Runs default training, staged hyperparameter tuning, selects the best model,
performs final test evaluation, and saves all artifacts. Includes a fresh-process
checkpoint reload verification.
"""

from __future__ import annotations

import logging
import sys

from src.train import run_text_cnn_experiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)


def main() -> None:
    print("=" * 60)
    print("  Step 5: Text CNN — Full Experiment Pipeline")
    print("=" * 60)

    experiment = run_text_cnn_experiment()

    test = experiment["test_results"]

    print(f"\n{'=' * 60}")
    print("  Step 5 COMPLETE — Final Summary")
    print("=" * 60)
    print(f"  Best run:            {experiment['best_run_name']}")
    print(f"  Test accuracy:       {test['test_accuracy']:.4f}")
    print(f"  Test macro F1:       {test['test_macro_f1']:.4f}")
    print(f"  Test precision:      {test['test_precision']:.4f}")
    print(f"  Test recall:         {test['test_recall']:.4f}")
    print(f"  OOS F1:              {test['oos_f1']:.4f}")
    print(f"  Inference:           {test['inference_latency']['avg_ms_per_example']:.2f} ms/example")
    print(f"  Total params:        {test['total_parameters']:,}")
    print(f"  Trainable params:    {test['trainable_parameters']:,}")
    print(f"  Checkpoint:          {test['checkpoint_path']}")


if __name__ == "__main__":
    main()
