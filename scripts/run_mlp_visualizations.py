"""Generate all Step 4 visualizations from saved MLP baseline artifacts.

Can be run independently of training — only reads from outputs/ directory.
"""

from __future__ import annotations

import json
import logging
import sys

import pandas as pd

from src.constants import FIGURES_DIR, LOGS_DIR, REPORTS_DIR
from src.utils import ensure_dir
from src.visualizers import ResultsVisualizer, TrainingVisualizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def _find_best_run_name() -> tuple[str, int]:
    """Read tuning CSV and run summary to determine the best run name and epoch."""
    tuning_df = pd.read_csv(REPORTS_DIR / "mlp_tuning_results.csv")
    best_idx = int(tuning_df["best_val_metric"].idxmax())
    best_run_name: str = tuning_df.iloc[best_idx]["run_name"]

    summary_path = REPORTS_DIR / f"mlp_run_summary_{best_run_name}.json"
    summary = json.loads(summary_path.read_text())
    best_epoch: int = summary["best_epoch"]

    return best_run_name, best_epoch


def main() -> None:
    figures_dir = ensure_dir(FIGURES_DIR)
    best_run_name, best_epoch = _find_best_run_name()

    logger.info("Best run: %s (best epoch: %d)", best_run_name, best_epoch)

    # --- Training curves (from best run's log) ---
    log_path = LOGS_DIR / f"mlp_training_log_{best_run_name}.json"
    epoch_history: list[dict] = json.loads(log_path.read_text())

    loss_path = figures_dir / "mlp_train_val_loss_curve.png"
    TrainingVisualizer.plot_loss_curves(epoch_history, loss_path, best_epoch=best_epoch)
    logger.info("Saved: %s", loss_path)

    f1_path = figures_dir / "mlp_val_macro_f1_curve.png"
    TrainingVisualizer.plot_val_metric_curve(epoch_history, f1_path, metric_key="val_macro_f1", best_epoch=best_epoch)
    logger.info("Saved: %s", f1_path)

    # --- Tuning summary ---
    tuning_df = pd.read_csv(REPORTS_DIR / "mlp_tuning_results.csv")
    tuning_path = figures_dir / "mlp_tuning_summary.png"
    ResultsVisualizer.plot_tuning_summary(tuning_df, tuning_path)
    logger.info("Saved: %s", tuning_path)

    # --- Confusion matrix ---
    confusion_df = pd.read_csv(REPORTS_DIR / "mlp_confusion_matrix.csv", index_col=0)
    cm_path = figures_dir / "mlp_confusion_matrix.png"
    ResultsVisualizer.plot_confusion_matrix(confusion_df, cm_path)
    logger.info("Saved: %s", cm_path)

    # --- Top confused pairs ---
    top_confusions: list[dict] = json.loads((REPORTS_DIR / "mlp_top_confusions.json").read_text())
    confused_path = figures_dir / "mlp_top_confused_pairs.png"
    ResultsVisualizer.plot_top_confused_pairs(top_confusions, confused_path)
    logger.info("Saved: %s", confused_path)

    # --- Bottom classes by F1 ---
    test_metrics_raw = json.loads((REPORTS_DIR / "mlp_test_metrics.json").read_text())
    summary = json.loads((REPORTS_DIR / f"mlp_run_summary_{best_run_name}.json").read_text())
    label_names: list[str] = summary["label_name_ordering"]

    top_errors_raw = json.loads((REPORTS_DIR / "mlp_top_errors.json").read_text())
    errors: list[dict] = top_errors_raw["errors"]

    # Reconstruct predictions/targets from the confusion matrix for per-class F1
    targets: list[int] = []
    predictions: list[int] = []
    cm_values = confusion_df.values
    for true_idx in range(cm_values.shape[0]):
        for pred_idx in range(cm_values.shape[1]):
            count = int(cm_values[true_idx, pred_idx])
            targets.extend([true_idx] * count)
            predictions.extend([pred_idx] * count)

    bottom_path = figures_dir / "mlp_bottom_classes_f1.png"
    ResultsVisualizer.plot_bottom_classes_f1(targets, predictions, label_names, bottom_path)
    logger.info("Saved: %s", bottom_path)

    # --- OOS metrics ---
    oos_metrics = {
        "oos_precision": test_metrics_raw["oos_precision"],
        "oos_recall": test_metrics_raw["oos_recall"],
        "oos_f1": test_metrics_raw["oos_f1"],
    }
    oos_path = figures_dir / "mlp_oos_metrics.png"
    ResultsVisualizer.plot_oos_metrics(oos_metrics, oos_path)
    logger.info("Saved: %s", oos_path)

    # --- Error summary ---
    error_path = figures_dir / "mlp_error_summary.png"
    ResultsVisualizer.plot_error_summary(errors, error_path)
    logger.info("Saved: %s", error_path)

    logger.info("All MLP visualizations saved to %s", figures_dir)


if __name__ == "__main__":
    main()
