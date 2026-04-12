"""Generate all Step 6 BiLSTM visualizations from saved artifacts.

Can be run independently of training — only reads from outputs/ directory.
Produces 7 figures per spec AA requirements.
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


def _find_best_run() -> tuple[str, int, dict]:
    """Read tuning CSV and run summary to determine the best BiLSTM run."""
    tuning_df = pd.read_csv(REPORTS_DIR / "bilstm_tuning_results.csv")
    valid = tuning_df.dropna(subset=["best_val_metric"])
    valid = valid.sort_values(
        by=["best_val_metric", "best_val_loss", "trainable_parameters", "training_duration", "run_name"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)
    best_run_name: str = valid.iloc[0]["run_name"]

    summary_path = REPORTS_DIR / f"bilstm_run_summary_{best_run_name}.json"
    summary: dict = json.loads(summary_path.read_text())
    best_epoch: int = summary["best_epoch"]

    return best_run_name, best_epoch, summary


def main() -> None:
    figures_dir = ensure_dir(FIGURES_DIR)
    best_run_name, best_epoch, summary = _find_best_run()

    logger.info("Best run: %s (best epoch: %d)", best_run_name, best_epoch)

    # 1. Training loss curves
    log_path = LOGS_DIR / f"bilstm_training_log_{best_run_name}.json"
    epoch_history: list[dict] = json.loads(log_path.read_text())

    loss_path = figures_dir / "bilstm_train_val_loss_curve.png"
    TrainingVisualizer.plot_loss_curves(epoch_history, loss_path, best_epoch=best_epoch)
    logger.info("Saved: %s", loss_path)

    # 2. Validation macro F1 curve (with best epoch AND early stop markers)
    f1_path = figures_dir / "bilstm_val_macro_f1_curve.png"
    TrainingVisualizer.plot_val_metric_curve(epoch_history, f1_path, metric_key="val_macro_f1", best_epoch=best_epoch)
    logger.info("Saved: %s", f1_path)

    # 3. Tuning summary
    tuning_df = pd.read_csv(REPORTS_DIR / "bilstm_tuning_results.csv")
    tuning_path = figures_dir / "bilstm_tuning_summary.png"
    ResultsVisualizer.plot_tuning_summary(tuning_df, tuning_path)
    logger.info("Saved: %s", tuning_path)

    # 4. Confusion matrix
    confusion_df = pd.read_csv(REPORTS_DIR / "bilstm_confusion_matrix.csv", index_col=0)
    cm_path = figures_dir / "bilstm_confusion_matrix.png"
    ResultsVisualizer.plot_confusion_matrix(confusion_df, cm_path)
    logger.info("Saved: %s", cm_path)

    # 5. Top confused pairs
    top_confusions: list[dict] = json.loads((REPORTS_DIR / "bilstm_top_confusions.json").read_text())
    confused_path = figures_dir / "bilstm_top_confused_pairs.png"
    ResultsVisualizer.plot_top_confused_pairs(top_confusions, confused_path)
    logger.info("Saved: %s", confused_path)

    # 6. Bottom classes by F1
    label_names: list[str] = json.loads((REPORTS_DIR / "bilstm_label_order.json").read_text())
    cm_values = confusion_df.values
    targets: list[int] = []
    predictions: list[int] = []
    for true_idx in range(cm_values.shape[0]):
        for pred_idx in range(cm_values.shape[1]):
            count = int(cm_values[true_idx, pred_idx])
            targets.extend([true_idx] * count)
            predictions.extend([pred_idx] * count)

    bottom_path = figures_dir / "bilstm_bottom_classes_f1.png"
    ResultsVisualizer.plot_bottom_classes_f1(targets, predictions, label_names, bottom_path)
    logger.info("Saved: %s", bottom_path)

    # 7. OOS metrics
    test_metrics = json.loads((REPORTS_DIR / "bilstm_test_metrics.json").read_text())
    oos_metrics = {
        "oos_precision": test_metrics["oos_precision"],
        "oos_recall": test_metrics["oos_recall"],
        "oos_f1": test_metrics["oos_f1"],
    }
    oos_path = figures_dir / "bilstm_oos_metrics.png"
    ResultsVisualizer.plot_oos_metrics(oos_metrics, oos_path)
    logger.info("Saved: %s", oos_path)

    # Verification assertions
    saved_best_epoch_in_history = max(epoch_history, key=lambda r: r["val_macro_f1"])["epoch"]
    assert saved_best_epoch_in_history == best_epoch, (
        f"Plot best epoch {saved_best_epoch_in_history} != summary best epoch {best_epoch}"
    )

    assert confusion_df.shape[0] == len(label_names), (
        f"Confusion matrix rows {confusion_df.shape[0]} != label count {len(label_names)}"
    )

    logger.info("All BiLSTM visualizations saved to %s", figures_dir)


if __name__ == "__main__":
    main()
