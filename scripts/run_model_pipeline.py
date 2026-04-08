"""Canonical pipeline runner for tuning reuse, repeated evaluation, tracking, and visualization.

This is the primary user-facing entry point for Steps 7-9 of the CLINC150 project.
It orchestrates frozen-config extraction, repeated final runs, artifact tracking, and
figure generation across all three models.

Usage (Step 7+):
    python scripts/run_model_pipeline.py --model all --run-count 3
    python scripts/run_model_pipeline.py --model mlp --run-count 5
    python scripts/run_model_pipeline.py --model text_cnn --retune

Canonical model IDs:
    mlp, text_cnn, bilstm

Canonical model display names:
    TF-IDF + MLP, Text CNN, BiLSTM

Canonical model order for shared tables and plots:
    mlp, text_cnn, bilstm

Step 7 directory structure (target):
    outputs/
      reports/
        shared/
          evaluation_protocol.json
          model_comparison_aggregate.csv
          model_comparison_aggregate.json
          oos_summary_table.csv
          oos_summary_table.json
          efficiency_summary_table.csv
          efficiency_summary_table.json
          most_confused_pairs_table.csv
          most_confused_pairs_table.json
          representative_examples_index.json
          figure_manifest.json
        <model_name>/
          tuning/
            tuning_results.csv        (normalized from legacy <model>_tuning_results.csv)
            selection_summary.json
          frozen_final_config.json
          final_runs/
            run_01_seed_42/
              run_metadata.json
              validation_metrics.json
              test_metrics.json
              epoch_history.json
              final_predictions.csv
              confusion_matrix.csv
              top_confusions.json
              top_errors.json
              per_class_metrics.json
              label_order.json
              confidences.npz         (optional for non-representative runs)
            run_02_seed_1337/...
            run_03_seed_2024/...
          aggregate/
            per_run_metrics.csv
            aggregate_metrics.json
            aggregate_metrics.csv
            aggregate_comparison_row.json
            representative_run.json
      checkpoints/
        <model_name>/
          final_runs/
            run_01_seed_42/best_run_01_seed_42.pt
            ...
      logs/
        <model_name>/
          final_runs/
            run_01_seed_42/training_log.json
            ...
      figures/
        shared/
          model_comparison_test_accuracy.png
          model_comparison_test_macro_f1.png
          model_comparison_oos_f1.png
          oos_metrics_comparison.png
          model_efficiency_comparison.png
        <model_name>/
          train_val_loss_curve.png
          val_macro_f1_curve.png
          confusion_matrix.png
          tuning_summary.png
          ...

Default seed list:
    [42, 1337, 2024]

Protocol:
    For each selected model:
      1. Validate preprocessing artifacts (Step 3)
      2. Load or rerun tuning (Steps 4-6 via --retune)
      3. Extract frozen config from winning tuning row  [Step 7]
      4. Run repeated final training+evaluation         [Step 7]
      5. Compute per-model aggregate metrics             [Step 7]
      6. Select representative run                       [Step 7]
      7. Validate and organize artifacts                 [Step 8]
      8. Generate figures and handoff bundles             [Step 9]
"""

from __future__ import annotations

import argparse

CANONICAL_MODEL_IDS = ("mlp", "text_cnn", "bilstm")
DEFAULT_SEED_LIST = [42, 1337, 2024]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLINC150 model pipeline: tuning reuse, repeated evaluation, tracking, and visualization.",
    )
    parser.add_argument(
        "--model",
        choices=[*CANONICAL_MODEL_IDS, "all"],
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


def _resolve_models(model_arg: str) -> list[str]:
    if model_arg == "all":
        return list(CANONICAL_MODEL_IDS)
    return [model_arg]


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    models = _resolve_models(args.model)
    run_count = args.run_count
    seed_list = DEFAULT_SEED_LIST[:run_count]

    print("=" * 60)
    print("  CLINC150 Model Pipeline")
    print("=" * 60)
    print(f"  Models:    {models}")
    print(f"  Run count: {run_count}")
    print(f"  Seeds:     {seed_list}")
    print(f"  Retune:    {args.retune}")
    print()

    for model_id in models:
        print(f"--- {model_id} ---")

        # Step 7: Frozen-config extraction and repeated final runs
        raise NotImplementedError(
            f"Step 7 repeated-run evaluation for '{model_id}' is not yet implemented. "
            f"This will be added in Step 7. For now, use the per-model scripts:\n"
            f"  python scripts/run_mlp_baseline.py\n"
            f"  python scripts/run_text_cnn.py\n"
            f"  python scripts/run_bilstm.py"
        )


if __name__ == "__main__":
    main()
