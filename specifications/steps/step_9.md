Implement and verify Step 9: visualization and Step 10 handoff for the CLINC150 project.

Goal:
Produce a clean, report-ready visualization layer on top of the Step 7 repeated-run protocol and the Step 8 tracking schema. This step should clearly separate representative-run diagnostic figures from aggregate cross-model comparison figures, reuse the saved artifacts rather than re-running models or recomputing logic ad hoc, and package the right confusion, per-class, and misclassification outputs for Step 10 error analysis and later report writing.

Primary outcome:
At the end of this step, I want to have:
1. representative-run training and error-analysis figures for each model
2. aggregate cross-model comparison figures based on repeated-run mean/std metrics
3. explicit OOS-focused comparison figures and tables
4. class-level error-analysis figures that support Step 10 qualitative analysis
5. a stable figure-generation workflow that reads saved artifacts rather than live model objects
6. consistent figure naming and output structure across all models
7. a clean Step 10 handoff bundle for representative-run error analysis
8. report-ready visual outputs that Step 11 can consume directly

Important scope constraints:
- Do not use aggregate mean/std metrics to fabricate confusion matrices or class-level examples
- Do not use one lucky seed’s metrics as the headline comparison result after Step 7
- Do not require retraining to generate figures
- Do not recompute label mappings or metric definitions inside plotting code
- Do not let plotting scripts infer run identity from fragile filename guesses alone
- Keep visualizations practical and directly useful for debugging, comparison, and report writing
- Do not generate dozens of redundant figures with no reporting value

Out of scope for Step 9:
- interactive dashboards
- web-based visualization tools
- calibration plots unless explicitly added later
- publication-grade styling beyond what is needed for a strong course project report
- full error interpretation write-up, which belongs to Step 10

Cross-cutting invariants:
- aggregate figures must come from aggregate repeated-run artifacts
- representative-run figures must come from the documented representative run only
- all plots must respect the saved label ordering
- plotting code should consume saved artifacts, not raw training code internals
- figure filenames and metadata must make it obvious whether a figure is:
  - representative-run
  - aggregate
  - or shared cross-model

High-level design requirement:
Step 9 should establish two complementary figure families:
- representative-run figures for within-model diagnostics and qualitative error analysis
- aggregate cross-model figures for final comparison and reporting

The figure-generation path should be artifact-driven:
- read protocol and aggregate artifacts from Step 8
- read representative-run artifacts for the nominated run
- generate figures without re-running training or evaluation
- save figure paths in machine-readable summary metadata where practical

Recommended default implementation path:
- keep the existing visualizer classes in:
  - `src/visualizers/training.py`
  - `src/visualizers/results.py`
- keep the existing per-model scripts as the starting point:
  - `scripts/run_mlp_visualizations.py`
  - `scripts/run_text_cnn_visualizations.py`
  - `scripts/run_bilstm_visualizations.py`
- refactor them to read Step 8 representative-run and aggregate artifacts
- add a new shared cross-model visualization entry point for aggregate comparison plots
- extend the existing TODOs in the visualizer modules rather than creating a parallel plotting stack

Implementation requirements:

A. Keep visualization generation artifact-driven
- Figure generation must read saved artifacts only
- Do not require model objects, live checkpoints, or re-executing evaluation code to draw standard figures
- At minimum, plotting code should be able to consume:
  - aggregate metrics artifacts
  - representative-run metadata
  - per-run epoch history
  - confusion matrices
  - top-confusions artifacts
  - per-class metrics artifacts
  - final predictions artifacts
- If a figure requires an artifact that is not currently saved, Step 8 should be updated first rather than recomputing it ad hoc in plotting code

A2. Reuse and extend the current visualizer modules
- Extend `src/visualizers/training.py` for training-curve and aggregate-training comparisons
- Extend `src/visualizers/results.py` for:
  - aggregate model-comparison plots
  - aggregate OOS comparison plots
  - representative-run confusion and class-level plots
- Keep figure helpers model-agnostic wherever possible
- Do not hardcode Text CNN- or BiLSTM-specific assumptions into shared plotting helpers

B. Generate representative-run training diagnostics per model
- For each model, generate training-monitoring plots from the representative run only
- Required representative-run training figures:
  - train/validation loss curve
  - validation macro F1 curve
- Secondary representative-run training figures if practical:
  - validation accuracy curve
  - learning-rate curve if a scheduler is ever enabled
- Mark the best validation epoch clearly
- Mark the stopping epoch if it differs from the best epoch
- Ensure the plotted best epoch matches the representative-run metadata exactly

B2. Representative-run training figure expectations
Recommended outputs:
- `outputs/figures/<model_name>/representative_train_val_loss_curve.png`
- `outputs/figures/<model_name>/representative_val_macro_f1_curve.png`
- optionally:
  - `outputs/figures/<model_name>/representative_val_accuracy_curve.png`

- Do not save these under ambiguous names once repeated runs exist
- The figure metadata or sidecar summary should record:
  - representative run ID
  - seed
  - best epoch
  - source epoch-history artifact

C. Generate representative-run confusion and error-analysis figures per model
- For each model, generate the following from the representative run:
  - confusion matrix figure
  - top confused intent pairs figure
  - bottom-N classes by F1 figure
  - OOS metrics bar chart
  - error-summary figure if the saved top-errors artifact supports it
- Use the saved representative-run label ordering artifact
- Do not reconstruct label order from ad hoc sorting or confusion-matrix row names alone
- If the full confusion matrix is visually dense, still save it, but also save a more digestible summary

C2. Representative-run qualitative-analysis support
- Save or point to a compact, plot-friendly artifact for representative misclassified examples
- At minimum, the representative run should support Step 10 by exposing:
  - representative misclassified examples
  - true vs predicted label names
  - confidence if available
  - the text itself if available
- This artifact may be tabular rather than graphical, but it must be part of the Step 9 output bundle

D. Generate aggregate cross-model comparison figures
- Use aggregate repeated-run metrics for all headline model-comparison figures
- Required aggregate cross-model figures:
  - grouped bar chart comparing test macro F1 mean across models
  - grouped bar chart comparing test accuracy mean across models
  - grouped bar chart comparing OOS F1 mean across models
- These aggregate plots must include error bars or another explicit variability encoding based on standard deviation
- Keep the same model order across all comparison figures
- Make sure the values used in the plots match the saved aggregate comparison tables exactly

D2. Recommended shared aggregate outputs
Recommended outputs:
- `outputs/figures/shared/model_comparison_test_macro_f1.png`
- `outputs/figures/shared/model_comparison_test_accuracy.png`
- `outputs/figures/shared/model_comparison_oos_f1.png`
- `outputs/figures/shared/model_comparison_efficiency.png`

- If multiple metrics are combined into one grouped chart, keep it readable
- Prefer a few strong summary figures over a crowded wall of small plots

E. Add OOS-focused comparison outputs
- Because OOS is a project-specific concern, Step 9 should include explicit OOS comparison outputs
- Required OOS outputs:
  - cross-model OOS precision/recall/F1 figure
  - OOS summary table artifact ready for the report
- If the report will discuss OOS trade-offs, the figure should make precision vs recall trade-offs visible
- Keep OOS plots aggregate-based for model comparison and representative-run-based for within-model qualitative diagnosis

E2. Recommended OOS outputs
Recommended outputs:
- `outputs/figures/shared/oos_metrics_comparison.png`
- `outputs/reports/shared/oos_summary_table.csv`
- `outputs/reports/shared/oos_summary_table.json`

- The OOS summary table must align with the aggregate comparison-table schema from Step 8

F. Add class-level comparison and report-strengthening outputs
- The implementation spec explicitly asks for report-strengthening outputs beyond the core metrics
- Required outputs for this step:
  - per-class accuracy or F1 bar chart for the representative run
  - table of the most frequently confused intent pairs
  - summary table focused on OOS precision, recall, and F1
  - representative misclassified examples for qualitative discussion
- The class-level chart should focus on the worst-performing classes rather than trying to display all 151 classes clearly in one unreadable figure
- Recommended default:
  - bottom 10 or bottom 15 classes by F1

F2. Aggregate vs representative class-level policy
- Use the representative run for class-level plots and qualitative examples
- Do not attempt to create an “aggregate confusion matrix” or “aggregate class-level example set”
- If aggregate class-level statistics are ever added later, label them clearly as aggregate and keep them separate from representative-run plots

G. Add efficiency and practicality comparison figures
- Since the project now tracks training time, parameter count, and inference latency, Step 9 should expose at least one compact efficiency figure
- Recommended aggregate efficiency comparison figure:
  - training time mean by model
  - inference latency mean by model
  - parameter count by model
- If combining these on one chart becomes cluttered, split them into:
  - one efficiency figure
  - one size/latency figure
- Keep the timing protocol consistent with Step 7 and Step 8 definitions

G2. Recommended efficiency outputs
Recommended outputs:
- `outputs/figures/shared/model_efficiency_comparison.png`
- optionally:
  - `outputs/figures/shared/model_parameter_count_comparison.png`
  - `outputs/figures/shared/model_latency_comparison.png`

H. Refactor visualization scripts around representative-run and aggregate artifacts
- The current per-model visualization scripts should be updated so they:
  - resolve the representative run from Step 8 metadata
  - read the representative-run artifacts
  - read aggregate per-model and shared comparison artifacts
- Add one shared script such as:
  - `scripts/run_comparison_visualizations.py`
  - or `scripts/run_step9_visualizations.py`
- This shared script should generate the cross-model aggregate figures and report-ready summary tables
- Keep script responsibilities explicit rather than hiding everything in one giant all-purpose script

H2. Save figure metadata or a figure manifest
- Save one machine-readable figure manifest that records:
  - figure path
  - figure type
  - source artifact paths
  - model name if applicable
  - representative run ID if applicable
  - aggregate vs representative flag
- A figure manifest is especially useful for Step 11 report support

I. Make Step 10 handoff explicit
- At the end of Step 9, there should be one Step 10 handoff bundle per model
- The Step 10 handoff bundle should include or reference:
  - representative run ID
  - final predictions artifact
  - top-errors artifact
  - confusion matrix artifact
  - top-confusions artifact
  - per-class metrics artifact
  - bottom-classes figure
  - OOS figure
- This handoff bundle should be enough for Step 10 to start qualitative analysis without hunting for files

I2. Prepare notes-ready artifacts for Step 10
- Save a small representative misclassification table in a convenient format such as CSV or JSONL
- Save the most-frequently confused pairs in a table artifact, not just a plot
- Save a short metadata artifact that explains:
  - why this run is the representative run
  - how many total runs were aggregated
  - which metrics define the aggregate result
- This makes Step 10 interpretation easier and reduces the chance of mixing aggregate claims with representative-run examples

J. Keep plots non-interactive and reproducible
- Continue using non-interactive matplotlib output
- Ensure all figure generation works in headless environments
- Do not require manual notebook execution to produce the final figure set
- If notebooks are used for exploration, the authoritative Step 9 artifacts must still come from scripts or reusable modules

K. Figure and table expectations
Recommended representative-run figure outputs per model:
- `outputs/figures/<model_name>/representative_train_val_loss_curve.png`
- `outputs/figures/<model_name>/representative_val_macro_f1_curve.png`
- `outputs/figures/<model_name>/representative_confusion_matrix.png`
- `outputs/figures/<model_name>/representative_top_confused_pairs.png`
- `outputs/figures/<model_name>/representative_bottom_classes_f1.png`
- `outputs/figures/<model_name>/representative_oos_metrics.png`
- optionally:
  - `outputs/figures/<model_name>/representative_error_summary.png`

Recommended shared aggregate outputs:
- `outputs/figures/shared/model_comparison_test_accuracy.png`
- `outputs/figures/shared/model_comparison_test_macro_f1.png`
- `outputs/figures/shared/model_comparison_oos_f1.png`
- `outputs/figures/shared/oos_metrics_comparison.png`
- `outputs/figures/shared/model_efficiency_comparison.png`
- `outputs/reports/shared/most_confused_pairs_table.csv`
- `outputs/reports/shared/oos_summary_table.csv`
- `outputs/reports/shared/representative_examples_index.json`
- `outputs/reports/shared/figure_manifest.json`

L. Validation checks / assertions
- Assert that representative-run figure scripts resolve the same representative run recorded in Step 8 metadata
- Assert that aggregate figures read aggregate artifacts rather than representative-run metric files
- Assert that figure labels and model ordering are consistent across shared comparison plots
- Assert that the plotted values match the saved aggregate tables exactly
- Assert that confusion-matrix label ordering matches the saved label-order artifact
- Assert that class-level figures read the correct representative-run predictions
- Assert that every path saved in the figure manifest exists
- Assert that figure generation works in a non-interactive environment

M. Test requirements
- Add or update tests covering:
  - aggregate model-comparison plotting helpers
  - OOS comparison plotting helpers
  - representative-run resolution from metadata
  - figure manifest generation
  - label-order correctness in representative-run confusion and class-level plots
- Extend the existing visualizer tests where possible rather than creating detached plotting tests with little behavioral value

N. Definition of done
Step 9 is only complete if all of the following are true:
- each model has representative-run training curves
- each model has representative-run confusion and class-level figures
- aggregate cross-model comparison figures exist and use repeated-run mean/std metrics
- OOS-focused comparison outputs exist
- efficiency comparison outputs exist
- representative-run qualitative-analysis artifacts are packaged for Step 10
- figure naming makes aggregate vs representative status obvious
- plotting code reads saved artifacts rather than re-running models
- a figure manifest or equivalent metadata index exists
- the outputs are ready for Step 10 qualitative analysis and Step 11 report support

N2. Final Step 9 exit gate
- Before Step 9 is considered closed, confirm all of the following are true:
  - representative-run figures exist for MLP, Text CNN, and BiLSTM
  - aggregate cross-model figures exist
  - the OOS summary table exists
  - the most-confused-pairs table exists
  - representative misclassified examples are saved for later discussion
  - the figure manifest exists and points to real files
  - Step 10 can begin without additional artifact discovery work

O. Deliverable quality bar
Step 9 should make the project visually legible. A reader should be able to see both the stable aggregate comparison across models and the concrete representative-run behavior behind the qualitative discussion. The figures should not be decorative; they should clarify training behavior, comparison results, OOS behavior, and the errors that matter most for Step 10.

P. Suggested self-check after implementation
After coding, self-check against this exact checklist and confirm:
1. which figures are representative-run figures and which are aggregate figures
2. how the representative run is resolved for each model
3. where the representative-run loss and macro F1 curves are saved
4. where the representative-run confusion and top-confusion figures are saved
5. where the representative-run bottom-classes figure is saved
6. where the aggregate cross-model comparison figures are saved
7. where the OOS comparison figure and table are saved
8. where the efficiency comparison figure is saved
9. where the most-confused-pairs table is saved
10. where representative misclassified examples are saved for Step 10
11. where the figure manifest is saved
12. that the plots come from saved artifacts rather than retraining or ad hoc recomputation
13. that figure labels and model ordering are consistent across shared comparison plots
14. that Step 10 can start directly from the Step 9 handoff bundle
15. that Step 11 can consume the saved figure and table outputs directly
