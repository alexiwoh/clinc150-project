"""Supported checkpoint metrics shared by configuration and training."""

from typing import Literal


# Validation loss is minimized; classification scores are maximized.
MONITOR_METRIC_DIRECTIONS: dict[str, Literal["min", "max"]] = {
    "val_loss": "min",
    "val_accuracy": "max",
    "val_macro_f1": "max",
    "val_precision": "max",
    "val_recall": "max",
    "val_oos_precision": "max",
    "val_oos_recall": "max",
    "val_oos_f1": "max",
}
