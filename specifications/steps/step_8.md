Implement and verify Step 8: experiment tracking and report-ready artifact schemas for the CLINC150 project.

Goal:
Build a clean, explicit, multi-run experiment-tracking layer on top of the Step 7 repeated-run protocol so that every tuning result, every repeated final run, every aggregate summary, and every comparison table is saved in a stable, machine-readable structure. This step should make it easy to answer questions like “which config was frozen?”, “which seeds were used?”, “which run produced the confusion matrix?”, “what are the mean and standard deviation metrics for this model?”, and “which exact artifacts should the final report consume?” without relying on memory or manual file inspection.

Primary outcome:
At the end of this step, I want to have:
1. one stable directory and file schema for repeated-run experiment outputs
2. one explicit protocol manifest describing the repeated-run setup
3. one consistent per-run artifact schema shared by all models
4. one consistent aggregate artifact schema shared by all models
5. one clean comparison-table contract for cross-model reporting
6. parity across MLP, Text CNN, and BiLSTM tracking artifacts
7. one explicit mapping from representative run to Step 9 and Step 10 downstream consumers
8. one report-ready bundle that Step 11 can consume directly

Important scope constraints:
- Do not let artifact naming remain ambiguous after repeated runs are introduced
- Do not silently overwrite one seed’s outputs with another seed’s outputs
- Do not mix tuning artifacts and repeated final-run artifacts in the same flat namespace without labeling them clearly
- Do not force later report-writing steps to infer protocol details from filenames alone
- Do not keep model-specific artifact schemas unnecessarily different once Step 8 is complete
- Do not remove provenance fields that would be needed to reproduce a run later
- Keep the tracking system machine-readable first; human readability is important, but secondary

Out of scope for Step 8:
- new model training logic unrelated to tracking
- dashboarding systems, databases, or external experiment trackers
- cloud logging infrastructure
- report writing itself
- full results interpretation

Cross-cutting invariants:
- Step 7 remains the source of truth for the repeated-run protocol
- one model’s aggregate artifact schema must be directly comparable to another model’s
- representative-run artifacts must always reference the aggregate protocol they came from
- tracking must remain consistent with the frozen preprocessing manifest and label ordering
- artifact paths saved in summaries must correspond to real files

High-level design requirement:
This step should create a layered artifact model with four distinct levels:
- protocol-level artifacts shared across the project
- model-level tuning and frozen-config artifacts
- per-final-run artifacts for each seed
- aggregate summaries and cross-model comparison artifacts

Recommended default storage design:
- shared protocol files under `outputs/reports/shared/`
- model-specific artifacts under:
  - `outputs/reports/mlp/`
  - `outputs/reports/text_cnn/`
  - `outputs/reports/bilstm/`
- matching structure under:
  - `outputs/logs/`
  - `outputs/checkpoints/`
  - `outputs/figures/`
- one representative-run reference artifact per model
- one shared cross-model comparison bundle for the final report

Implementation requirements:

A. Create a clear artifact directory structure
- Introduce a model-specific artifact layout rather than continuing to place everything in one flat root
- Recommended structure:
  - `outputs/reports/shared/`
  - `outputs/reports/<model_name>/tuning/`
  - `outputs/reports/<model_name>/final_runs/`
  - `outputs/reports/<model_name>/aggregate/`
  - `outputs/logs/<model_name>/tuning/`
  - `outputs/logs/<model_name>/final_runs/`
  - `outputs/checkpoints/<model_name>/tuning/`
  - `outputs/checkpoints/<model_name>/final_runs/`
  - `outputs/figures/<model_name>/`
  - `outputs/figures/shared/`
- The structure should make it obvious which files belong to:
  - tuning
  - repeated final runs
  - aggregate summaries
  - shared comparison outputs

A2. Avoid ambiguity in legacy flat files
- If legacy top-level files are preserved for backward compatibility, they must be labeled explicitly as:
  - representative
  - aggregate
  - or legacy single-run
- Do not keep ambiguous names like `text_cnn_test_metrics.json` once multiple repeated runs exist unless the file is clearly documented as:
  - the representative run
  - or the aggregate summary
- Prefer explicit names such as:
  - `aggregate_test_metrics.json`
  - `representative_run_test_metrics.json`
  - `per_run_metrics.csv`

B. Save a shared protocol manifest
- Save one protocol manifest under `outputs/reports/shared/`
- The protocol manifest should contain at minimum:
  - protocol version
  - run count
  - seed list
  - representative-run rule
  - macro F1 definition
  - OOS metric definition
  - final-model rule
  - whether logits / probabilities are saved for all runs or representative run only
  - whether confusion artifacts are saved for all runs or representative run only
- This manifest should be referenced by model-level summaries and representative-run metadata

B2. Save a frozen-config manifest per model
- For each model, save one frozen final-config artifact after tuning and before repeated runs
- The frozen-config artifact should contain:
  - model name
  - frozen config snapshot
  - source tuning artifact
  - selection metric
  - selection timestamp or run identifier
  - preprocessing artifact refs
  - label-order artifact ref
  - protocol manifest ref
- This file is the root provenance record for all repeated final runs of that model

C. Define the per-run artifact schema
- Every repeated final run must save a per-run artifact bundle
- Recommended per-run bundle contents:
  - `run_metadata.json`
  - `test_metrics.json`
  - `validation_metrics.json`
  - `epoch_history.csv` or `.json`
  - `run_summary.json`
  - `final_predictions.csv` or `.jsonl`
  - `confidences.npz` or equivalent if enabled
  - `top_errors.json`
  - `per_class_metrics.json`
  - `label_order.json`
  - `checkpoint_ref.json` or direct checkpoint path inside metadata
- Each per-run bundle must record:
  - run index
  - seed
  - training seed
  - DataLoader seed
  - model name
  - config snapshot
  - best epoch
  - best validation metric
  - checkpoint path
  - preprocessing artifact refs
  - protocol manifest ref

C2. Run identifier policy
- Every repeated run must have a stable run identifier
- Recommended pattern:
  - `run_<two_digit_index>_seed_<seed>`
- The run identifier must be used consistently across:
  - reports
  - logs
  - checkpoints
  - figures
  - representative-run references
- Do not allow run indices and seeds to drift apart in naming

D. Define the aggregate artifact schema per model
- Every model must save an aggregate bundle after all repeated runs complete
- Recommended aggregate bundle contents:
  - `per_run_metrics.csv`
  - `aggregate_metrics.json`
  - `aggregate_metrics.csv`
  - `representative_run.json`
  - `aggregate_comparison_row.json`
  - `aggregate_oos_summary.json`
  - `aggregate_efficiency_summary.json`
- The aggregate metrics artifact should include:
  - run_count_requested
  - run_count_completed
  - seed_list_requested
  - seed_list_completed
  - success/failure summary
  - mean and standard deviation for all required metrics
  - parameter count
  - average training time
  - average inference latency
  - representative run ID
  - frozen-config ref
  - protocol manifest ref

D2. Keep validation and test summaries separate inside aggregate outputs
- Save aggregate validation summaries separately from aggregate test summaries
- Do not collapse validation and test metrics into one mixed summary blob
- The aggregate bundle should make it easy to answer:
  - how stable was validation selection across seeds?
  - how stable was final test performance across seeds?
- If a combined summary file is added for convenience, it must still preserve separate field names for validation and test metrics

E. Define cross-model comparison-table contracts
- Save one aggregate comparison row per model using the aggregate repeated-run results, not representative-run results
- Save one combined cross-model table under `outputs/reports/shared/`
- Recommended shared outputs:
  - `model_comparison_aggregate.csv`
  - `model_comparison_aggregate.json`
  - `oos_comparison_summary.csv`
  - `efficiency_comparison_summary.csv`
- At minimum the cross-model comparison schema should include:
  - model_name
  - input_type
  - protocol_version
  - run_count
  - primary_val_metric
  - val_macro_f1_mean
  - val_macro_f1_std
  - test_accuracy_mean
  - test_accuracy_std
  - test_macro_f1_mean
  - test_macro_f1_std
  - test_precision_mean
  - test_precision_std
  - test_recall_mean
  - test_recall_std
  - oos_precision_mean
  - oos_precision_std
  - oos_recall_mean
  - oos_recall_std
  - oos_f1_mean
  - oos_f1_std
  - training_time_mean
  - training_time_std
  - inference_latency_mean
  - inference_latency_std
  - parameter_count
  - representative_run_id
  - frozen_config_ref
  - notes

E2. Keep representative-run comparison outputs separate
- If representative-run comparison rows are saved for convenience, they must not replace aggregate comparison rows
- Use explicit names such as:
  - `representative_run_comparison_row.json`
  - `aggregate_comparison_row.json`
- Do not let Step 11 or later report support accidentally ingest representative-run comparison rows as the headline model result

F. Enforce schema parity across all three models
- MLP, Text CNN, and BiLSTM must all export the same core tracking schema once Step 8 is complete
- Acceptable model-specific extras may exist, but the shared core fields must align
- Shared parity must include:
  - run metadata
  - test metrics
  - aggregate metrics
  - representative-run metadata
  - comparison rows
  - label-order artifacts
  - final predictions
  - per-class metrics
- Do not leave MLP permanently behind the neural models in artifact richness

F2. MLP parity catch-up requirements
- The current codebase already saves richer per-class and label-order artifacts for Text CNN and BiLSTM than for MLP
- Step 8 should explicitly close that gap
- MLP should also save:
  - label-order artifact
  - per-class metrics artifact
  - final predictions artifact
  - representative-run metadata
  - aggregate repeated-run summary artifacts
- Cross-model tracking should not require special-case logic because one model saved less metadata than the others

G. Track logs and checkpoints with the same clarity as reports
- Logs and checkpoints must mirror the repeated-run structure
- For each repeated run, save:
  - training log path
  - checkpoint path
  - checkpoint metadata
- Save checkpoint references in both per-run metadata and aggregate summaries
- Make it possible to reload the representative run and any individual repeated run without re-running the whole model

G2. Save provenance links between artifacts
- Per-run metadata should reference:
  - frozen-config artifact
  - protocol manifest
  - preprocessing manifest
  - label-order artifact
- Aggregate summaries should reference:
  - all per-run metadata paths
  - representative-run path
  - cross-model comparison outputs if already generated
- This should make downstream figure generation and report assembly straightforward and traceable

H. Save Step 10 and Step 11 handoff artifacts explicitly
- For Step 10, save one representative-run handoff bundle per model that points to:
  - representative run ID
  - final predictions artifact
  - top errors artifact
  - confusion artifact
  - top-confusions artifact
  - per-class metrics artifact
- For Step 11, save one report-support bundle per model that points to:
  - aggregate metrics
  - aggregate comparison row
  - representative-run figure paths
  - OOS summary artifact
  - efficiency summary artifact
- Keep these handoff bundles explicit so later steps do not need to rediscover paths manually

H2. Failure and partial-run tracking
- Save a model-level run ledger that records:
  - all requested runs
  - all completed runs
  - failed runs
  - skipped runs if any
  - failure reasons
- Aggregate artifacts must record if any run failed or was excluded
- Do not make it hard to tell whether a reported mean/std came from 3 runs, 2 runs, or 5 runs

I. Report-ready artifact expectations
Recommended shared artifacts:
- `outputs/reports/shared/evaluation_protocol.json`
- `outputs/reports/shared/model_comparison_aggregate.csv`
- `outputs/reports/shared/model_comparison_aggregate.json`
- `outputs/reports/shared/oos_comparison_summary.csv`
- `outputs/reports/shared/efficiency_comparison_summary.csv`

Recommended per-model artifacts:
- `outputs/reports/<model_name>/frozen_final_config.json`
- `outputs/reports/<model_name>/aggregate/per_run_metrics.csv`
- `outputs/reports/<model_name>/aggregate/aggregate_metrics.json`
- `outputs/reports/<model_name>/aggregate/aggregate_metrics.csv`
- `outputs/reports/<model_name>/aggregate/representative_run.json`
- `outputs/reports/<model_name>/aggregate/aggregate_comparison_row.json`
- `outputs/reports/<model_name>/final_runs/run_<index>_seed_<seed>/run_metadata.json`
- `outputs/reports/<model_name>/final_runs/run_<index>_seed_<seed>/test_metrics.json`
- `outputs/reports/<model_name>/final_runs/run_<index>_seed_<seed>/validation_metrics.json`
- `outputs/reports/<model_name>/final_runs/run_<index>_seed_<seed>/final_predictions.csv`
- `outputs/reports/<model_name>/final_runs/run_<index>_seed_<seed>/per_class_metrics.json`
- `outputs/reports/<model_name>/final_runs/run_<index>_seed_<seed>/label_order.json`

J. Validation checks / assertions
- Assert that every path saved in metadata exists
- Assert that every repeated run references the same frozen config for a given model
- Assert that the completed seed list matches the recorded per-run artifacts
- Assert that aggregate tables are built from the actual per-run metrics table
- Assert that representative-run metadata points to a real completed run
- Assert that label-order artifacts match across all repeated runs for a given model
- Assert that model comparison tables use aggregate results rather than representative-run results
- Assert that required MLP parity artifacts exist after Step 8
- Assert that all schemas validate cleanly before the step is considered complete

K. Test requirements
- Add or update tests covering:
  - per-run artifact schema validation
  - aggregate artifact schema validation
  - representative-run metadata correctness
  - cross-model comparison-table schema consistency
  - MLP parity with the neural models
  - provenance references pointing to real files
- Prefer tests that validate artifact content and linkage rather than just filename presence

L. Definition of done
Step 8 is only complete if all of the following are true:
- the repeated-run protocol is represented by an explicit shared manifest
- each model has one frozen final-config artifact
- each repeated run saves a complete per-run artifact bundle
- each model saves aggregate mean/std summaries
- each model saves representative-run metadata
- cross-model comparison tables are aggregate-based and report-ready
- logs and checkpoints follow the same repeated-run naming scheme as reports
- MLP artifact richness is brought into parity with Text CNN and BiLSTM for the shared core fields
- provenance links between protocol, config, per-run, aggregate, and representative-run artifacts are explicit
- Step 9 can generate figures directly from the saved artifacts
- Step 10 can locate representative-run qualitative-analysis inputs directly from saved handoff metadata
- Step 11 can consume the aggregate reporting bundle without ad hoc path discovery

L2. Final Step 8 exit gate
- Before Step 8 is considered closed, confirm all of the following are true:
  - there is no ambiguity about which files are per-run, aggregate, or representative-run
  - aggregate model comparison tables exist
  - OOS comparison outputs exist
  - efficiency comparison outputs exist
  - representative-run metadata exists for all models
  - MLP, Text CNN, and BiLSTM share the same core reporting schema
  - every saved artifact path referenced in metadata resolves to a real file

M. Deliverable quality bar
Step 8 should feel like a serious experiment-tracking layer, not a pile of convenient one-off exports. Someone reading the output directories later should be able to reconstruct what happened, how many runs were performed, which configuration was frozen, which run generated the confusion matrix, and which numbers belong in the final report without guesswork.

N. Suggested self-check after implementation
After coding, self-check against this exact checklist and confirm:
1. where the shared protocol manifest is saved
2. where the frozen final config is saved for each model
3. what the per-run directory structure is
4. what the aggregate directory structure is
5. where the representative-run metadata lives
6. how run IDs are formed
7. where the completed seed list is stored
8. that MLP exports the same core tracking artifacts as Text CNN and BiLSTM
9. where cross-model aggregate comparison tables are saved
10. where OOS-specific aggregate summaries are saved
11. where efficiency summaries are saved
12. how logs and checkpoints mirror the report artifact structure
13. how provenance links are stored between protocol, config, runs, and aggregates
14. how Step 10 finds the representative-run artifacts
15. how Step 11 finds the final report-ready tables and summaries
