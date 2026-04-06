"""Results-level visualizations for model evaluation (Steps 7 and 9).

This module will contain plotting methods for evaluation artifacts produced
after training completes.  All methods will follow the same conventions as
``DatasetVisualizer``: accept pre-extracted data, save to a path, and use
the shared style from ``Visualizer``.
"""

from __future__ import annotations

from src.visualizers.base import Visualizer


class ResultsVisualizer(Visualizer):
    """Plotting methods for model evaluation results.

    Planned methods (to be implemented in Steps 7 and 9):
    """

    # TODO (Step 7): plot_confusion_matrix — heatmap of predicted vs true labels
    #   for the best model.  Accepts a confusion matrix (np.ndarray) and label
    #   names.

    # TODO (Step 7): plot_per_class_accuracy — horizontal bar chart of per-class
    #   accuracy, sorted ascending to highlight weak classes.

    # TODO (Step 9): plot_model_comparison — grouped bar chart comparing test
    #   accuracy, macro F1, and OOS F1 across all three models.

    # TODO (Step 9): plot_oos_metrics_summary — bar chart focused on OOS
    #   precision, recall, and F1 for each model.
