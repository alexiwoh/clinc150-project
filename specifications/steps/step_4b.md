Add Step 4 visualization requirements for the TF-IDF + MLP baseline.

Goal:
Produce a small set of practical, report-ready visualizations that help diagnose training behavior, summarize tuning results, and support later comparison with CNN and BiLSTM models.

Important scope constraints:
- Keep visualizations lightweight and directly useful
- Do not create dozens of redundant plots
- Prefer plots that help with debugging, model selection, final report writing, and cross-model comparison
- Save all plots to a stable outputs/figures/ directory structure
- Use the same label ordering and metric conventions as the saved artifacts

Implementation requirements:

A. Training-curve visualizations
- Save a training loss curve across epochs
- Save a validation loss curve across epochs
- Save a validation macro F1 curve across epochs
- If useful, also save validation accuracy across epochs, but macro F1 should remain the main plot for model selection
- If plotted together, make sure the chart is still readable
- Prefer either:
  - one figure for loss curves
  - one figure for validation metrics
  or
  - separate figures if clarity is better

Recommended outputs:
- outputs/figures/mlp_train_val_loss_curve.png
- outputs/figures/mlp_val_macro_f1_curve.png
- optionally:
  - outputs/figures/mlp_val_accuracy_curve.png

B. Early-stopping / best-epoch visualization support
- Mark the best validation epoch on the validation macro F1 plot
- If easy, also mark the early stopping point
- Ensure the best epoch shown in the plot matches the saved checkpoint metadata exactly
- Make the primary validation metric visually obvious

C. Hyperparameter tuning summary visualization
- Create at least one compact visualization summarizing the tried baseline runs
- Good options:
  - bar chart of best validation macro F1 by run/config
  - scatter plot of validation macro F1 vs dropout / hidden_dim / learning_rate
  - heatmap if the search grid is small and structured
- Keep it readable and small-scope
- The goal is not pretty benchmarking, just an interpretable tuning summary

Recommended outputs:
- outputs/figures/mlp_tuning_summary.png
- optionally:
  - outputs/figures/mlp_tuning_heatmap.png

D. Confusion visualization
- Save a confusion matrix for final test predictions
- Use the exact saved label ordering from artifacts
- If a full 151-class confusion matrix is too unreadable, still save it, but also produce a more digestible summary
- Add a top-confused-pairs visualization or summary artifact
- If the full confusion matrix is visually messy, generate:
  - a top-N confused intent pairs bar chart
  - or a filtered confusion matrix for the most problematic classes

Recommended outputs:
- outputs/figures/mlp_confusion_matrix.png
- outputs/figures/mlp_top_confused_pairs.png

E. Class-level performance visualization
- Compute per-class precision / recall / F1 if practical
- Save a bar chart of the worst-performing classes by F1
- This is especially useful for report discussion and later comparison against CNN / BiLSTM
- Keep the chart focused, for example bottom 10 or bottom 15 classes

Recommended outputs:
- outputs/figures/mlp_bottom_classes_f1.png

F. OOS-specific visualization
- If OOS is treated as an explicit class and OOS metrics are computed, include a simple OOS-focused visualization
- Good options:
  - OOS precision / recall / F1 bar chart
  - counts of OOS true positives / false positives / false negatives
- Only do this if it is consistent with the chosen project OOS strategy

Recommended outputs:
- outputs/figures/mlp_oos_metrics.png

G. Inference / efficiency visualization
- Since Step 4 now tracks inference latency, consider saving a tiny efficiency summary visualization
- For Step 4 alone, this can be optional
- It becomes much more useful once CNN and BiLSTM are added
- If created now, keep it simple:
  - one bar for training time
  - one bar for inference latency
- More realistically, structure the code so this can later plug into cross-model comparison plots

Recommended outputs:
- optional:
  - outputs/figures/mlp_efficiency_summary.png

H. Error-analysis visualization support
- Save or generate a plot-friendly artifact for top errors
- If practical, create a simple chart showing the most frequent wrong predicted labels or most confused label pairs
- Keep representative misclassified text samples in JSON/CSV even if they are not plotted directly

Recommended outputs:
- outputs/figures/mlp_error_summary.png
- outputs/reports/mlp_top_errors.json

I. Comparison-readiness visualization schema
- Design visualization code so the same plot patterns can later be reused for CNN and BiLSTM
- At minimum, keep naming and metric conventions consistent for:
  - training curves
  - validation macro F1 plots
  - confusion matrix outputs
  - bottom-class F1 plots
  - tuning summary plots
- This will make final cross-model comparisons much easier

J. Console / summary expectations
- When visualizations are generated, log exactly which files were saved
- Save visualization file references in the run summary artifact if practical
- Make sure the saved plot filenames correspond to the actual run/config

K. Validation checks / assertions
- Ensure plotting does not silently fail if directories do not exist
- Ensure the best epoch shown in plots matches the tracked best checkpoint epoch
- Ensure confusion matrix label ordering matches the saved label-name ordering artifact
- Ensure plots can still be generated in non-interactive environments
- Ensure training-curve plots are based on the saved epoch history rather than recomputed ad hoc values

L. Definition of done for Step 4 visualizations
Step 4 visualizations are complete if all of the following are true:
- train/validation loss curves are saved
- validation macro F1 curve is saved
- best epoch is visually marked on the primary validation metric plot
- at least one tuning-summary visualization is saved
- confusion matrix visualization is saved
- at least one confusion-summary or top-confused-pairs visualization is saved
- at least one class-level error analysis visualization is saved or the data for it is saved cleanly
- visualization artifacts are saved in stable output paths
- label ordering is consistent across confusion / class-level plots
- the plots are good enough to drop into the final report or presentation with minimal cleanup
