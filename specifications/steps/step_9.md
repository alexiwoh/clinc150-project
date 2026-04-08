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

Cross-step ownership and data flow:
- Step 7 owns:
  - frozen-config selection handoff
  - repeated-run execution
  - per-run artifacts
  - per-model aggregate metrics
  - representative-run selection
- Step 8 owns:
  - schema validation
  - provenance metadata
  - run ledgers
  - artifact parity audits
  - cross-model comparison tables
  - downstream metadata bundles
- Step 9 owns:
  - figure generation
  - figure manifest generation
  - final Step 10 handoff bundles with figure references
- Data flow summary:
  - Step 7 -> frozen configs, per-run artifacts, per-model aggregates, representative-run metadata
  - Step 8 -> validated schemas, provenance links, cross-model tables, downstream metadata bundles
  - Step 9 -> representative figures, aggregate figures, figure manifest, Step 10 handoff bundles

Shared protocol metadata and canonical naming:
- Canonical model IDs:
  - `mlp`
  - `text_cnn`
  - `bilstm`
- Canonical display names:
  - `TF-IDF + MLP`
  - `Text CNN`
  - `BiLSTM`
- Canonical model order for shared tables and plots:
  - `mlp`
  - `text_cnn`
  - `bilstm`
- Every machine-readable artifact must include:
  - `schema_version`
  - `protocol_version`
- Every machine-readable path reference must be repo-relative and must resolve under the repository root
- Shared report filenames are locked to:
  - `outputs/reports/shared/evaluation_protocol.json`
  - `outputs/reports/shared/model_comparison_aggregate.csv`
  - `outputs/reports/shared/model_comparison_aggregate.json`
  - `outputs/reports/shared/oos_summary_table.csv`
  - `outputs/reports/shared/oos_summary_table.json`
  - `outputs/reports/shared/efficiency_summary_table.csv`
  - `outputs/reports/shared/efficiency_summary_table.json`
  - `outputs/reports/shared/most_confused_pairs_table.csv`
  - `outputs/reports/shared/most_confused_pairs_table.json`
  - `outputs/reports/shared/representative_examples_index.json`
  - `outputs/reports/shared/figure_manifest.json`
- Shared figure filenames are locked to:
  - `outputs/figures/shared/model_comparison_test_accuracy.png`
  - `outputs/figures/shared/model_comparison_test_macro_f1.png`
  - `outputs/figures/shared/model_comparison_oos_f1.png`
  - `outputs/figures/shared/oos_metrics_comparison.png`
  - `outputs/figures/shared/model_efficiency_comparison.png`
- Do not keep alternate filenames active once these canonical names exist

Shared metric and efficiency conventions:
- Save all metrics as raw ratios in `[0.0, 1.0]`
- Unqualified `precision` and `recall` mean macro precision and macro recall unless labeled otherwise
- Headline `macro_f1` includes the full multiclass label set including the OOS class unless explicitly labeled otherwise
- Save the OOS evaluation policy explicitly, including:
  - OOS class identity and index
  - one-vs-rest rule
  - explicit-class vs alternate policy
- Keep zero-division behavior identical across all models
- Save the following efficiency fields with shared names:
  - `training_time_seconds`
  - `inference_total_seconds`
  - `inference_avg_ms_per_example`
  - `inference_examples_per_sec`
  - `parameter_count`
  - `trainable_parameter_count`
- State explicitly whether timing includes DataLoader overhead

High-level design requirement:
Step 9 should establish two complementary figure families:
- representative-run figures for within-model diagnostics and qualitative error analysis
- aggregate cross-model figures for final comparison and reporting

The figure-generation path must be metadata-driven:
- read protocol and aggregate artifacts from Step 8
- resolve the representative run only from `outputs/reports/<model_name>/aggregate/representative_run.json`
- read representative-run artifacts through the metadata references recorded there
- generate figures without re-running training or evaluation
- save figure paths in machine-readable summary metadata

Recommended execution story:
- Use a dedicated Step 9 entry point such as `scripts/run_step9_visualizations.py`
- The canonical pipeline runner `scripts/run_model_pipeline.py` may invoke the Step 9 visualization entry point automatically after Step 8 tracking and validation complete for the selected model or models
- If the existing model-specific scripts are preserved, they should become thin wrappers around the shared Step 9 plotting logic
- Automatic invocation through the pipeline runner must execute the same Step 9 plotting, manifest-generation, and handoff-generation logic as the standalone visualization script
- Shared comparison plotting should read the canonical Step 8 shared tables directly rather than rediscovering data from per-model files or tuning outputs

Recommended default implementation path:
- keep the existing visualizer classes in:
  - `src/visualizers/training.py`
  - `src/visualizers/results.py`
- refactor them to read Step 8 representative-run and aggregate artifacts
- use `src/visualizers/training.py` for representative training curves
- use `src/visualizers/results.py` for:
  - aggregate model-comparison plots
  - aggregate OOS comparison plots
  - representative-run confusion and class-level plots
  - figure-manifest and handoff metadata generation
- extend the current plotting modules rather than creating a parallel plotting stack

Implementation requirements:

A. Keep visualization generation fully metadata-driven
- Figure generation must read saved artifacts only
- Do not require model objects, live checkpoints, or re-executing evaluation code to draw standard figures
- Representative-run resolution must come only from:
  - `outputs/reports/<model_name>/aggregate/representative_run.json`
- Shared comparison plots must read the canonical shared Step 8 tables:
  - `outputs/reports/shared/model_comparison_aggregate.csv` or `.json`
  - `outputs/reports/shared/oos_summary_table.csv` or `.json`
  - `outputs/reports/shared/efficiency_summary_table.csv` or `.json`
- Do not rediscover runs from filenames, tuning CSVs, or ad hoc directory scans

A2. Run explicit preflight validation before plotting
- Before generating any figure, validate that the required metadata and source artifacts exist
- At minimum verify:
  - `representative_run.json` exists for each model being plotted
  - required shared aggregate tables exist for shared plots
  - representative-run `epoch_history.json` exists
  - representative-run `label_order.json` exists
  - representative-run `confusion_matrix.csv` exists
  - representative-run `top_confusions.json` exists
  - all referenced paths are repo-relative and resolve correctly
- If a required artifact is missing, fail clearly instead of guessing or silently skipping the figure

B. Generate representative-run training diagnostics per model
- For each model, generate training-monitoring plots from the representative run only
- Step 7 and Step 8 guarantee `epoch_history.json` for all three models, so Step 9 should require representative training curves for all three models
- Required representative-run training figures:
  - train and validation loss curve
  - validation macro F1 curve
  - validation accuracy curve
- Secondary representative-run training figures may be added later, but they are not part of the core Step 9 deliverable set
- Mark the best validation epoch clearly
- Mark the stopping epoch if it differs from the best epoch
- Ensure the plotted best epoch and stopping epoch match the representative-run metadata exactly

B2. Representative-run training figure expectations
Recommended outputs:
- `outputs/figures/<model_name>/representative_train_val_loss_curve.png`
- `outputs/figures/<model_name>/representative_val_macro_f1_curve.png`
- `outputs/figures/<model_name>/representative_val_accuracy_curve.png`

- Do not save these under ambiguous names once repeated runs exist
- The figure metadata should record:
  - representative run ID
  - seed
  - best epoch
  - stopping epoch
  - source `epoch_history.json`

C. Generate representative-run confusion and diagnostic outputs per model
- For each model, generate the following from the representative run:
  - confusion matrix figure
  - top confused intent pairs figure
  - bottom classes by F1 figure
  - OOS metrics bar chart
  - error-summary figure if the saved top-errors artifact supports it
- Use the saved representative-run label ordering artifact
- Do not reconstruct label order from ad hoc sorting or confusion-matrix row names alone
- Use `confusion_matrix.csv` only for the confusion matrix itself
- Use `per_class_metrics.json` or `final_predictions.csv` for class-level rankings and misclassification support

C2. Canonical class-level diagnostic policy
- The canonical representative-run class-level diagnostic is:
  - `bottom_classes_f1`
- Focus this figure on the worst-performing classes rather than trying to display all classes in one unreadable plot
- Recommended default:
  - bottom 10 or bottom 15 classes by F1
- Do not leave the spec vague as “per-class accuracy” once `bottom_classes_f1` is the chosen deliverable

C3. Representative-run qualitative-analysis support
- Save a compact representative misclassification artifact under:
  - `outputs/reports/<model_name>/aggregate/representative_misclassifications.csv`
- This artifact must include at minimum:
  - `text` if available
  - `true_label_name`
  - `predicted_label_name`
  - optional `max_confidence`
  - `run_id`
- This artifact must be referenced in both:
  - the model-level `step10_handoff.json`
  - `outputs/reports/shared/representative_examples_index.json`

D. Generate aggregate cross-model comparison figures
- Use aggregate repeated-run metrics for all headline model-comparison figures
- Required aggregate cross-model figures:
  - grouped bar chart comparing test accuracy mean across models
  - grouped bar chart comparing test macro F1 mean across models
  - grouped bar chart comparing OOS F1 mean across models
- These aggregate plots must include error bars or another explicit variability encoding based on standard deviation
- Keep the same canonical model order across all comparison figures
- Make sure the values used in the plots match the saved aggregate comparison tables exactly

D2. Recommended shared aggregate outputs
Recommended outputs:
- `outputs/figures/shared/model_comparison_test_accuracy.png`
- `outputs/figures/shared/model_comparison_test_macro_f1.png`
- `outputs/figures/shared/model_comparison_oos_f1.png`

- Prefer a few strong summary figures over a crowded wall of small plots
- Do not keep alternate names such as `model_comparison_efficiency.png`

E. Add OOS-focused comparison outputs and fix the extra-plot policy
- Because OOS is a project-specific concern, Step 9 should include explicit OOS comparison outputs
- Required OOS outputs:
  - cross-model OOS precision, recall, and F1 figure
  - OOS summary table artifact ready for the report
- Keep OOS plots aggregate-based for model comparison and representative-run-based for within-model qualitative diagnosis
- Do not require extra shared precision or recall headline plots beyond the OOS-focused comparison unless a later step asks for them explicitly

E2. Recommended OOS outputs
Recommended outputs:
- `outputs/figures/shared/oos_metrics_comparison.png`
- `outputs/reports/shared/oos_summary_table.csv`
- `outputs/reports/shared/oos_summary_table.json`

- The OOS summary table must align with the aggregate comparison-table schema from Step 8

F. Add efficiency comparison outputs
- Since the project now tracks training time, parameter count, and inference latency, Step 9 should expose one compact efficiency figure
- Build this figure from `outputs/reports/shared/efficiency_summary_table.csv` or `.json`
- Keep the timing protocol consistent with Step 7 and Step 8 definitions
- Additional size-only or latency-only figures are optional later, but are not part of the required Step 9 deliverable set

F2. Recommended efficiency outputs
Recommended outputs:
- `outputs/figures/shared/model_efficiency_comparison.png`

G. Build shared tables and metadata needed for Step 10 and Step 11
- Save a shared table of the most confused representative-run label pairs across models
- This shared table must be built from the representative-run `top_confusions.json` artifacts, not from tuning files or hand-edited notes
- Save one shared representative-examples index
- Save one shared figure manifest

G2. `most_confused_pairs_table` contract
- Save:
  - `outputs/reports/shared/most_confused_pairs_table.csv`
  - `outputs/reports/shared/most_confused_pairs_table.json`
- Each row must include at minimum:
  - `model_name`
  - `representative_run_id`
  - `rank`
  - `true_label_name`
  - `predicted_label_name`
  - `count`

G3. `representative_examples_index.json` contract
- Save:
  - `outputs/reports/shared/representative_examples_index.json`
- This index should map each model to:
  - representative run ID
  - representative predictions artifact path
  - representative misclassification table path
  - representative confusion artifact path
  - optional `step10_handoff_ref`

G4. `figure_manifest.json` contract
- Save:
  - `outputs/reports/shared/figure_manifest.json`
- Each figure-manifest entry must include:
  - `schema_version`
  - `protocol_version`
  - `figure_path`
  - `figure_type`
  - `scope`
  - `model_name` if applicable
  - `representative_run_id` if applicable
  - `source_artifact_paths`

H. Make Step 10 handoff explicit
- At the end of Step 9, there must be one `outputs/reports/<model_name>/aggregate/step10_handoff.json` per model
- The Step 10 handoff bundle must include or reference:
  - aggregate metrics artifact
  - representative-run metadata artifact
  - final predictions artifact
  - representative misclassification table
  - top-errors artifact
  - confusion matrix artifact
  - top-confusions artifact
  - per-class metrics artifact
  - label-order artifact
  - representative figure paths
- This handoff bundle should be enough for Step 10 to start qualitative analysis without additional artifact discovery

H2. Prepare notes-ready metadata for Step 10
- Save metadata that explains:
  - why this run is the representative run
  - how many total runs were aggregated
  - which metrics define the aggregate result
- This reduces the chance of mixing aggregate claims with representative-run examples

I. Refactor visualization scripts around representative-run and aggregate artifacts
- The official Step 9 entry point should generate both per-model representative figures and shared aggregate figures
- The official Step 9 entry point should be callable both directly and from `scripts/run_model_pipeline.py`
- Existing model-specific visualization scripts, if retained, must resolve all paths through `representative_run.json`
- Shared comparison plotting helpers must read the canonical Step 8 shared tables directly
- Keep script responsibilities explicit rather than hiding everything in one giant all-purpose script

I2. Keep plots non-interactive and reproducible
- Continue using non-interactive matplotlib output
- Ensure all figure generation works in headless environments
- Do not require manual notebook execution to produce the final figure set
- If notebooks are used for exploration, the authoritative Step 9 artifacts must still come from scripts or reusable modules

J. Clarify legacy and tuning figure policy
- Old single-run figure names may remain only as legacy outputs and must not be treated as Step 9 deliverables
- Tuning-summary figures may remain under tuning or legacy locations for reference, but they are excluded from:
  - `figure_manifest.json`
  - `step10_handoff.json`
  - the core Step 9 deliverable checklist
- Only figures regenerated under the canonical Step 9 filenames count as current deliverables

K. Figure and table expectations
Recommended representative-run figure outputs per model:
- `outputs/figures/<model_name>/representative_train_val_loss_curve.png`
- `outputs/figures/<model_name>/representative_val_macro_f1_curve.png`
- `outputs/figures/<model_name>/representative_val_accuracy_curve.png`
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
- `outputs/reports/shared/most_confused_pairs_table.json`
- `outputs/reports/shared/oos_summary_table.csv`
- `outputs/reports/shared/oos_summary_table.json`
- `outputs/reports/shared/representative_examples_index.json`
- `outputs/reports/shared/figure_manifest.json`

L. Validation checks and assertions
- Assert that representative-run figure scripts resolve the same representative run recorded in `representative_run.json`
- Assert that aggregate figures read aggregate artifacts rather than representative-run metric files
- Assert that figure labels and model ordering are consistent across shared comparison plots
- Assert that the plotted values match the saved aggregate tables exactly
- Assert that confusion-matrix label ordering matches the saved label-order artifact
- Assert that class-level figures read `per_class_metrics.json` or `final_predictions.csv`, not only `confusion_matrix.csv`
- Assert that every path saved in the figure manifest exists
- Assert that every path saved in `step10_handoff.json` exists
- Assert that figure generation works in a non-interactive environment

M. Test requirements
- Add or update tests covering:
  - preflight validation failures for missing artifacts
  - aggregate model-comparison plotting helpers
  - OOS comparison plotting helpers
  - representative-run resolution from metadata
  - figure-manifest generation
  - most-confused-pairs table generation
  - label-order correctness in representative-run confusion and class-level plots
  - `step10_handoff.json` generation
- Extend the existing visualizer tests where possible rather than creating detached plotting tests with little behavioral value

N. Definition of done
Step 9 is only complete if all of the following are true:
- each model has representative-run training curves
- each model has representative-run confusion and class-level figures
- aggregate cross-model comparison figures exist and use repeated-run mean and standard deviation metrics
- OOS-focused comparison outputs exist
- efficiency comparison outputs exist
- representative-run qualitative-analysis artifacts are packaged for Step 10
- figure naming makes aggregate versus representative status obvious
- plotting code reads saved artifacts rather than re-running models
- `figure_manifest.json` exists
- each model has a `step10_handoff.json`
- the outputs are ready for Step 10 qualitative analysis and Step 11 report support

N2. Final Step 9 exit gate
- Before Step 9 is considered closed, confirm all of the following are true:
  - representative-run figures exist for MLP, Text CNN, and BiLSTM
  - aggregate cross-model figures exist
  - the OOS summary table exists
  - the most-confused-pairs table exists
  - representative misclassified examples are saved for later discussion
  - the figure manifest exists and points to real files
  - `step10_handoff.json` exists for every model
  - Step 10 can begin without additional artifact discovery work

O. Deliverable quality bar
Step 9 should make the project visually legible. A reader should be able to see both the stable aggregate comparison across models and the concrete representative-run behavior behind the qualitative discussion. The figures should not be decorative; they should clarify training behavior, comparison results, OOS behavior, and the errors that matter most for Step 10.

P. Suggested self-check after implementation
After coding, self-check against this exact checklist and confirm:
1. which figures are representative-run figures and which are aggregate figures
2. that the representative run is resolved only from `representative_run.json` for each model
3. where the representative-run loss, macro F1, and accuracy curves are saved
4. where the representative-run confusion and top-confusion figures are saved
5. where the representative-run bottom-classes figure is saved
6. where the aggregate cross-model comparison figures are saved
7. where the OOS comparison figure and table are saved
8. where the efficiency comparison figure is saved
9. where the most-confused-pairs table is saved
10. where representative misclassified examples are saved for Step 10
11. where the figure manifest is saved
12. where each `step10_handoff.json` is saved
13. that the plots come from saved artifacts rather than retraining or ad hoc recomputation
14. that figure labels and model ordering are consistent across shared comparison plots
15. that Step 10 can start directly from the Step 9 handoff bundle
