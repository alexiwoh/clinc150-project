"""Training-level visualizations for monitoring experiments (Steps 4-6 and 9).

This module will contain plotting methods for training artifacts such as loss
curves and validation accuracy over epochs.  All methods will follow the same
conventions as ``DatasetVisualizer``: accept pre-extracted data, save to a
path, and use the shared style from ``Visualizer``.
"""

from __future__ import annotations

from src.visualizers.base import Visualizer


class TrainingVisualizer(Visualizer):
    """Plotting methods for training monitoring.

    Planned methods (to be implemented in Steps 4-6 and 9):
    """

    # TODO (Step 4): plot_loss_curves — line chart of training and validation
    #   loss per epoch for a single model run.

    # TODO (Step 4): plot_accuracy_curves — line chart of validation accuracy
    #   per epoch for a single model run.

    # TODO (Step 9): plot_all_loss_curves — overlay loss curves from all three
    #   models on one chart for comparison.
