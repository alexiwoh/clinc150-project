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
- Step 8 does not retrain, retune, or reevaluate models once Step 7 artifacts exist
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
  - `outputs/shared/evaluation_protocol.json`
  - `outputs/shared/model_comparison_aggregate.csv`
  - `outputs/shared/model_comparison_aggregate.json`
  - `outputs/shared/oos_summary_table.csv`
  - `outputs/shared/oos_summary_table.json`
  - `outputs/shared/efficiency_summary_table.csv`
  - `outputs/shared/efficiency_summary_table.json`
  - `outputs/shared/most_confused_pairs_table.csv`
  - `outputs/shared/most_confused_pairs_table.json`
  - `outputs/shared/representative_examples_index.json`
  - `outputs/shared/figure_manifest.json`
- Shared figure filenames are locked to:
  - `outputs/shared/figures/model_comparison_test_accuracy.png`
  - `outputs/shared/figures/model_comparison_test_macro_f1.png`
  - `outputs/shared/figures/model_comparison_oos_f1.png`
  - `outputs/shared/figures/oos_metrics_comparison.png`
  - `outputs/shared/figures/model_efficiency_comparison.png`
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
This step should create a layered artifact model with four distinct levels while treating Step 7 as the source of truth for model execution:
- protocol-level artifacts shared across the project
- model-level tuning and frozen-config artifacts
- per-final-run artifacts for each seed
- aggregate summaries and cross-model comparison artifacts
- Step 8 validates, enriches, and organizes these artifacts; it does not retrain, retune, or reevaluate models

Recommended default storage design (run-first layout):
- shared protocol and comparison files under `outputs/shared/`
- shared figures under `outputs/shared/figures/`
- model-specific artifacts under:
  - `outputs/mlp/`
  - `outputs/text_cnn/`
  - `outputs/bilstm/`
- each run directory contains `checkpoint/` and `logs/` subdirectories
- per-model figures under `outputs/<model_name>/figures/`
- one representative-run reference artifact per model
- one shared cross-model comparison bundle for the final report
- no new Step 8 artifact should require relocating Step 7 outputs into a different directory tree

Recommended execution story:
- Step 7 produces the repeated-run artifacts
- Step 8 runs through a dedicated entry point such as `scripts/run_step8_tracking.py`
- The canonical pipeline runner `scripts/run_model_pipeline.py` may invoke the Step 8 tracking entry point automatically after repeated evaluation completes for the selected model or models
- That entry point validates schemas, verifies provenance, materializes ledgers, and builds shared comparison tables from the saved Step 7 artifacts
- Automatic invocation through the pipeline runner must execute the same Step 8 validation and materialization logic as the standalone script rather than using a separate hidden code path
- If `main.py` or another existing entry point is used instead, the Step 8 mode must still remain explicit and isolated from training or evaluation logic

Implementation requirements:

A. Keep one canonical artifact directory structure
- Use the same model-specific layout introduced in Step 7
- Recommended run-first structure:
  - `outputs/shared/`
  - `outputs/shared/figures/`
  - `outputs/<model_name>/tuning/`
  - `outputs/<model_name>/final_runs/` (each run directory contains `checkpoint/` and `logs/` subdirectories)
  - `outputs/<model_name>/aggregate/`
  - `outputs/<model_name>/figures/`
- The structure should make it obvious which files belong to:
  - tuning
  - repeated final runs
  - aggregate summaries
  - shared comparison outputs

A2. Map legacy artifacts to the canonical schema
- If legacy flat files are preserved for backward compatibility, they must be labeled explicitly as legacy and must not remain part of the authoritative Step 8 contract
- Do not keep ambiguous names like `text_cnn_test_metrics.json` once repeated runs exist unless the file is documented as a legacy artifact
- Treat `run_metadata.json` as the canonical replacement for old `*_run_summary_*.json` or other model-specific summary names
- Treat `epoch_history.json` as the canonical replacement for ad hoc `epoch_history.csv` variants unless a legacy export is preserved separately for compatibility
- Legacy files may remain for historical reference, but new automation must resolve only through the canonical filenames and metadata references

B. Validate the shared protocol manifest produced by Step 7
- `outputs/shared/evaluation_protocol.json` is the canonical protocol manifest
- Step 8 must validate and, if needed, enrich this manifest rather than inventing a second protocol artifact
- The protocol manifest should contain at minimum:
  - `schema_version`
  - `protocol_version`
  - canonical model IDs, display names, and model order
  - `run_count`
  - `seed_list`
  - `representative_run_rule`
  - macro F1 definition
  - OOS metric definition
  - final-model rule
  - probability-saving policy
  - confusion-artifact policy
  - `timing_includes_dataloader_overhead`
- This manifest should be referenced by model-level summaries and representative-run metadata

B2. Clarify the frozen-config relationship to Step 7
- For each model, Step 7 creates one `outputs/<model_name>/frozen_final_config.json`
- Step 8 validates and may enrich the provenance fields in that artifact, but it must not create a second independent frozen-config decision
- The frozen-config artifact should contain at minimum:
  - `schema_version`
  - `protocol_version`
  - `model_name`
  - `model_id`
  - frozen config snapshot
  - source tuning artifact
  - selection metric
  - winning row identifier
  - preprocessing artifact refs
  - label-order artifact ref
  - protocol manifest ref
- This file is the root provenance record for all repeated final runs of that model

B3. Require a minimum tuning artifact bundle
- Each model’s `tuning/` directory must contain at minimum:
  - `tuning_results.csv`
  - `selection_summary.json`
  - enough provenance to identify the winning configuration unambiguously
- If the current codebase still emits legacy `*_tuning_results.csv` files at the root, Step 8 should record the mapping into the canonical tuning directory rather than leaving the relationship implicit

C. Define the canonical per-run artifact schema
- Every repeated final run must save a per-run artifact bundle under `outputs/<model_name>/final_runs/run_<index>_seed_<seed>/`
- The canonical per-run bundle contents are:
  - `run_metadata.json`
  - `test_metrics.json`
  - `validation_metrics.json`
  - `epoch_history.json`
  - `final_predictions.csv`
  - `confidences.npz` if enabled by policy
  - `top_errors.json`
  - `confusion_matrix.csv`
  - `top_confusions.json`
  - `per_class_metrics.json`
  - `label_order.json`
- Step 8 should validate this bundle; it should not reconstruct missing files from thinner artifacts

C2. `run_metadata.json` minimum fields
- Every `run_metadata.json` must include at minimum:
  - `schema_version`
  - `protocol_version`
  - `model_name`
  - `model_id`
  - `run_id`
  - `run_index`
  - `seed`
  - `training_seed`
  - `dataloader_seed`
  - `status`
  - `best_epoch`
  - `stopping_epoch`
  - `best_val_metric`
  - `best_val_loss`
  - `monitor_metric`
  - `checkpoint_path`
  - `log_path`
  - `frozen_config_ref`
  - `protocol_manifest_ref`
  - `preprocessing_manifest_ref`
  - `label_order_ref`

C3. Run identifier policy
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
- Every model must save an aggregate bundle under `outputs/<model_name>/aggregate/`
- Recommended aggregate bundle contents:
  - `per_run_metrics.csv`
  - `aggregate_metrics.json`
  - `aggregate_metrics.csv`
  - `representative_run.json`
  - `aggregate_comparison_row.json`
  - `run_ledger.json`
- Keep validation and test summaries separate within the aggregate structure even if a convenience summary is also emitted

D2. `aggregate_metrics.json` minimum fields
- `aggregate_metrics.json` must include at minimum:
  - `schema_version`
  - `protocol_version`
  - `model_name`
  - `model_id`
  - `run_count_requested`
  - `run_count_completed`
  - `seed_list_requested`
  - `seed_list_completed`
  - `all_runs_succeeded`
  - `successful_run_ids`
  - `failed_run_ids`
  - `validation_summary`
  - `test_summary`
  - `oos_summary`
  - `efficiency_summary`
  - `parameter_count_summary`
  - `representative_run_id`
  - `frozen_config_ref`
  - `protocol_manifest_ref`

D3. `run_ledger.json` minimum fields
- `run_ledger.json` is required and must record requested, completed, failed, and skipped runs with honest provenance
- At minimum include:
  - `schema_version`
  - `protocol_version`
  - `model_name`
  - `model_id`
  - requested run IDs
  - completed run IDs
  - failed run IDs
  - skipped run IDs
  - seed list requested
  - seed list completed
  - failure reasons keyed by run ID
  - per-run metadata refs
- Aggregate artifacts must record if any run failed or was excluded
- Do not make it hard to tell whether a reported mean and standard deviation came from 3 runs, 2 runs, or 5 runs

E. Define cross-model comparison-table contracts
- Each model produces one `aggregate_comparison_row.json` using aggregate repeated-run results, not representative-run results
- Step 8 builds the shared comparison outputs by merging those per-model rows in canonical model order
- The shared comparison outputs owned by Step 8 are:
  - `outputs/shared/model_comparison_aggregate.csv`
  - `outputs/shared/model_comparison_aggregate.json`
  - `outputs/shared/oos_summary_table.csv`
  - `outputs/shared/oos_summary_table.json`
  - `outputs/shared/efficiency_summary_table.csv`
  - `outputs/shared/efficiency_summary_table.json`
- Do not use alternate shared filenames such as `oos_comparison_summary.*` or `efficiency_comparison_summary.*`

E2. Shared comparison row requirements
- At minimum the cross-model comparison row schema should include:
  - `schema_version`
  - `protocol_version`
  - `model_name`
  - `model_id`
  - display name
  - input type
  - `run_count`
  - `primary_val_metric`
  - `val_accuracy_mean`
  - `val_accuracy_std`
  - `val_macro_f1_mean`
  - `val_macro_f1_std`
  - `test_accuracy_mean`
  - `test_accuracy_std`
  - `test_macro_f1_mean`
  - `test_macro_f1_std`
  - `test_precision_mean`
  - `test_precision_std`
  - `test_recall_mean`
  - `test_recall_std`
  - `oos_precision_mean`
  - `oos_precision_std`
  - `oos_recall_mean`
  - `oos_recall_std`
  - `oos_f1_mean`
  - `oos_f1_std`
  - `training_time_seconds_mean`
  - `training_time_seconds_std`
  - `inference_total_seconds_mean`
  - `inference_total_seconds_std`
  - `inference_avg_ms_per_example_mean`
  - `inference_avg_ms_per_example_std`
  - `inference_examples_per_sec_mean`
  - `inference_examples_per_sec_std`
  - `parameter_count`
  - `trainable_parameter_count`
  - `representative_run_id`
  - `frozen_config_ref`
  - optional notes

E3. Keep representative-run comparison outputs separate
- If representative-run comparison rows are saved for convenience, they must not replace aggregate comparison rows
- Use explicit names such as:
  - `representative_run_comparison_row.json`
  - `aggregate_comparison_row.json`
- Do not let Step 11 or later report support accidentally ingest representative-run comparison rows as the headline model result

F. Enforce schema parity across all three models
- MLP, Text CNN, and BiLSTM must all export the same core tracking schema once Step 8 is complete
- Acceptable model-specific extras may exist, but the shared core fields must align
- Shared parity must include:
  - tuning artifacts
  - run metadata
  - validation metrics
  - test metrics
  - aggregate metrics
  - run ledgers
  - representative-run metadata
  - comparison rows
  - label-order artifacts
  - final predictions
  - per-class metrics
- Replace model-specific parity wording with a full cross-model parity audit

F2. Close current parity gaps without special cases
- The current codebase already saves richer per-class and label-order artifacts for Text CNN and BiLSTM than for MLP
- Step 8 should explicitly close that gap, but the requirement is broader than an MLP-only cleanup
- Cross-model tracking should not require special-case logic because one model saved less metadata than the others

G. Track logs, checkpoints, and provenance with the same clarity as reports
- Logs and checkpoints live inside each run directory under `checkpoint/` and `logs/` subdirectories
- For each repeated run, save:
  - training log path
  - checkpoint path
  - checkpoint metadata if needed
- Save checkpoint references in both per-run metadata and aggregate summaries
- Make it possible to reload the representative run and any individual repeated run without re-running the whole model

G2. Enforce repo-relative path validation
- Every path saved in metadata must be repo-relative
- Every saved reference path must resolve to a real file before Step 8 is considered complete
- Reject paths that escape the repository root or depend on user-specific absolute locations

G3. No-reconstruction rule
- Do not reconstruct per-example predictions from confusion matrices
- Do not infer label order from sorted names, CSV headers, or confusion-matrix axes
- Do not infer representative-run identity from tuning files once `representative_run.json` exists
- If a required artifact is missing, fail validation clearly instead of silently rebuilding an approximation

H. Save downstream metadata bundles without stealing Step 9 ownership
- Step 8 may save provenance or input-reference bundles for later steps if that improves traceability
- Any such downstream bundles must be metadata-only and must not duplicate large data payloads
- Step 8 must not claim ownership of the final per-model `step10_handoff.json`
- Step 9 owns the final Step 10 handoff because it knows the final figure paths and representative diagnostic outputs

I. Report-ready artifact expectations
Recommended shared artifacts owned by Step 8:
- `outputs/shared/evaluation_protocol.json`
- `outputs/shared/model_comparison_aggregate.csv`
- `outputs/shared/model_comparison_aggregate.json`
- `outputs/shared/oos_summary_table.csv`
- `outputs/shared/oos_summary_table.json`
- `outputs/shared/efficiency_summary_table.csv`
- `outputs/shared/efficiency_summary_table.json`

Recommended per-model artifacts:
- `outputs/<model_name>/frozen_final_config.json`
- `outputs/<model_name>/tuning/tuning_results.csv`
- `outputs/<model_name>/tuning/selection_summary.json`
- `outputs/<model_name>/aggregate/per_run_metrics.csv`
- `outputs/<model_name>/aggregate/aggregate_metrics.json`
- `outputs/<model_name>/aggregate/aggregate_metrics.csv`
- `outputs/<model_name>/aggregate/run_ledger.json`
- `outputs/<model_name>/aggregate/representative_run.json`
- `outputs/<model_name>/aggregate/aggregate_comparison_row.json`
- `outputs/<model_name>/final_runs/run_<index>_seed_<seed>/run_metadata.json`
- `outputs/<model_name>/final_runs/run_<index>_seed_<seed>/test_metrics.json`
- `outputs/<model_name>/final_runs/run_<index>_seed_<seed>/validation_metrics.json`
- `outputs/<model_name>/final_runs/run_<index>_seed_<seed>/final_predictions.csv`
- `outputs/<model_name>/final_runs/run_<index>_seed_<seed>/per_class_metrics.json`
- `outputs/<model_name>/final_runs/run_<index>_seed_<seed>/label_order.json`

J. Validation checks and assertions
- Assert that every path saved in metadata is repo-relative and resolves to a real file
- Assert that every repeated run references the same frozen config for a given model
- Assert that the completed seed list matches the recorded per-run artifacts
- Assert that aggregate tables are built from the actual per-run metrics table
- Assert that representative-run metadata points to a real completed run
- Assert that label-order artifacts match across all repeated runs for a given model
- Assert that model comparison tables use aggregate results rather than representative-run results
- Assert that required parity artifacts exist for all three models
- Assert that all schemas validate cleanly before the step is considered complete

K. Test requirements
- Add or update tests covering:
  - protocol-manifest schema validation
  - per-run artifact schema validation
  - aggregate artifact schema validation
  - `run_ledger.json` correctness
  - representative-run metadata correctness
  - cross-model comparison-table schema consistency
  - parity across all three models
  - provenance references pointing to real files
- Prefer tests that validate artifact content and linkage rather than just filename presence

L. Definition of done
Step 8 is only complete if all of the following are true:
- the repeated-run protocol is represented by an explicit shared manifest
- each model has one frozen final-config artifact created by Step 7 and validated by Step 8
- each repeated run saves a complete per-run artifact bundle
- each model saves aggregate mean and standard deviation summaries
- each model saves representative-run metadata
- each model saves a `run_ledger.json`
- cross-model comparison tables are aggregate-based and report-ready
- logs and checkpoints are co-located inside each run directory under `checkpoint/` and `logs/` subdirectories
- the three models share the same core tracking schema
- provenance links between protocol, config, per-run, aggregate, and representative-run artifacts are explicit
- Step 9 can generate figures directly from the saved artifacts
- downstream steps do not need ad hoc path discovery

L2. Final Step 8 exit gate
- Before Step 8 is considered closed, confirm all of the following are true:
  - there is no ambiguity about which files are per-run, aggregate, representative-run, or legacy
  - aggregate model comparison tables exist
  - OOS summary tables exist
  - efficiency summary tables exist
  - representative-run metadata exists for all models
  - MLP, Text CNN, and BiLSTM share the same core reporting schema
  - every saved artifact path referenced in metadata is repo-relative and resolves to a real file

M. Deliverable quality bar
Step 8 should feel like a serious experiment-tracking layer, not a pile of convenient one-off exports. Someone reading the output directories later should be able to reconstruct what happened, how many runs were performed, which configuration was frozen, which run generated the confusion matrix, and which numbers belong in the final report without guesswork.

N. Suggested self-check after implementation
After coding, self-check against this exact checklist and confirm:
1. where the shared protocol manifest is saved
2. where the frozen final config is saved for each model
3. what the canonical tuning, per-run, and aggregate directory structure is
4. where the representative-run metadata lives
5. how run IDs are formed
6. where the completed seed list is stored
7. where `run_ledger.json` lives and what it records
8. that all three models export the same core tracking artifacts
9. where cross-model aggregate comparison tables are saved
10. where OOS-specific aggregate summary tables are saved
11. where efficiency summary tables are saved
12. how logs and checkpoints are co-located with run artifacts inside each run directory
13. how provenance links are stored between protocol, config, runs, and aggregates
14. that Step 8 does not retrain or reevaluate models
15. how Step 9 discovers the canonical artifacts it will plot
