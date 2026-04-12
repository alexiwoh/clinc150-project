Implement and verify Step 7: evaluation metrics and repeated-run protocol for the CLINC150 project.

Goal:
Turn the current single-run evaluation workflow into a fair, reproducible, repeated-run evaluation protocol that can be applied consistently to the TF-IDF + MLP baseline, Text CNN, and BiLSTM. This step should formalize exactly how final model configurations are frozen, how repeated runs are executed across a shared seed list, how per-run and aggregated metrics are computed, and how one representative run is chosen for confusion matrices and qualitative error analysis without contaminating the aggregated reporting protocol.

Primary outcome:
At the end of this step, I want to have:
1. one explicit repeated-run evaluation protocol shared by all models
2. one frozen final configuration per model selected using validation only
3. repeated final-train-and-evaluate runs across a shared seed list
4. per-run validation and test metrics saved separately for every seed
5. aggregated mean and standard deviation metrics saved for every model
6. one clear representative-run policy for confusion matrices and Step 10 error analysis
7. one shared definition of accuracy, macro F1, precision, recall, and OOS metrics across all models
8. a reusable orchestration path that cleanly separates tuning from repeated final evaluation
9. a clean handoff package to Step 8 artifact tracking and Step 9 visualization generation

Important scope constraints:
- Do not change preprocessing artifacts, label mappings, dataset splits, or OOS handling during this step
- Do not re-open Step 4-6 model architecture decisions unless a real bug is found
- Do not tune on the test split
- Do not choose the final frozen configuration separately for every seed
- Do not silently mix runs produced under different preprocessing artifacts, label orders, or metric definitions
- Do not let repeated runs become a second hidden tuning phase
- Keep the same seed list and evaluation protocol for MLP, Text CNN, and BiLSTM
- Keep test evaluation strictly downstream of validation-based config selection
- Keep the repeated-run protocol configurable, but use 3 runs by default
- Support 5 runs later without redesigning the pipeline

Out of scope for Step 7:
- new model architectures
- threshold-based OOS redesign if the project is already using the explicit OOS class approach
- hyperparameter-search redesign beyond what is needed to cleanly separate tuning from final repeated evaluation
- calibration experiments
- ablation studies
- report writing itself

Cross-cutting invariants:
- the preprocessing manifest from Step 3 remains frozen
- the label ordering used by Steps 4-6 remains frozen
- the same metric functions must be used across all models
- the same seed list must be reused across all models for final repeated evaluation
- aggregated reporting must stay separate from representative-run qualitative analysis
- any representative run used for confusion matrices or misclassified examples must be clearly labeled as representative rather than aggregate

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
This step should refactor the workflow into two distinct phases while adopting the final artifact layout immediately:
- phase 1: validation-only tuning and frozen-config extraction
- phase 2: repeated final runs using the frozen configuration and a shared seed list
- Step 7 should write artifacts directly into the canonical run-first structure that Step 8 validates later:
  - `outputs/shared/`
  - `outputs/<model_name>/tuning/`
  - `outputs/<model_name>/final_runs/` (each run directory contains `checkpoint/` and `logs/` subdirectories)
  - `outputs/<model_name>/aggregate/`
  - `outputs/<model_name>/figures/`
  - `outputs/shared/figures/`
- Do not create root-level aggregate artifacts that Step 8 would later need to relocate

The recommended evaluation contract is:
- pick one final config per model using validation only
- extract that config from the existing tuning outputs rather than retuning
- freeze the fully resolved runtime config in a machine-readable artifact
- rerun only the final train/validation/test workflow across a shared seed list
- save the same required per-run artifact bundle for every model and every completed run
- compute per-model aggregate mean and standard deviation metrics across runs
- choose one representative run for confusion matrices, top confusions, representative misclassified examples, and downstream handoff metadata

Recommended default implementation path:
- expose one canonical user-facing pipeline runner such as `scripts/run_model_pipeline.py`
- make that runner accept:
  - `--model {mlp,text_cnn,bilstm,all}`
  - `--run-count N`
  - optional `--retune`
- default to:
  - `--model all`
  - `--run-count 3`
- reuse the tuning outputs from Steps 4-6
- read each model’s winning row from the existing `*_tuning_results.csv` artifact
- normalize that tuning artifact under `outputs/<model_name>/tuning/tuning_results.csv` if the current file still lives in a legacy location
- serialize the resolved frozen config to `outputs/<model_name>/frozen_final_config.json`
- define one shared seed list such as `[42, 1337, 2024]`
- rerun the final training and test evaluation once per seed
- treat these repeated runs as the official reporting protocol
- keep the option to extend to 5 seeds later without changing filenames or schemas
- choose the representative run using validation-only information, not test metrics

Implementation requirements:

A. Freeze the final configuration by extracting it from existing tuning outputs
- Each model must have one explicit frozen final configuration before repeated-run evaluation starts
- The final configuration must be selected from the existing tuning outputs produced in Steps 4-6
- Read the model’s `*_tuning_results.csv` artifact and choose the winning row using validation-only criteria
- Do not rerun hyperparameter search unless a real bug invalidates the earlier tuning results
- Save the source tuning artifact path, winning row identifier, selection metric, and selection value in the frozen-config artifact
- The repeated-run loop may not mutate model architecture, optimizer policy, early-stopping policy, sequence length, label mapping, or OOS strategy
- If the canonical pipeline runner is used, it may call back into the Step 4-6 tuning logic only when the selected model’s tuning artifacts are missing, invalid, or `--retune` is passed

A2. Freeze the fully resolved runtime configuration
- The frozen config must include all training-relevant hyperparameters, preprocessing references, and monitoring settings
- The frozen config must record the runtime-resolved values that the final training path actually uses, including where relevant:
  - `vocab_size`
  - actual sequence length
  - actual monitor metric
  - preprocessing manifest reference
  - label-order reference
  - OOS strategy
- Save the frozen config in a machine-readable artifact before the repeated-run loop starts
- Do not save a partial config that still depends on implicit defaults or runtime guesses

A3. Explicitly separate tuning from repeated final evaluation
- Do not let the repeated-run loop call the hyperparameter search again
- Do not let the repeated-run loop pick a different config per seed
- Make the transition from tuning and selection to frozen-config repeated evaluation explicit in orchestration code, not just in comments
- If the current codebase still combines these concerns in a single function, refactor it before Step 7 is considered complete

B. Introduce a shared repeated-run evaluation configuration
- Add one shared repeated-run protocol config or dataclass in `src/config.py`
- The protocol configuration should be immutable for a completed run set
- At minimum make these configurable:
  - `schema_version`
  - `protocol_version`
  - `run_count`
  - `seed_list`
  - `representative_run_rule`
  - `probability_saving_policy`
  - `confusion_artifact_policy`
  - `aggregate_metric_list`
  - `test_evaluation_enabled`
  - `timing_includes_dataloader_overhead`
- Keep the default protocol simple and explicit
- Recommended default:
  - `run_count = 3`
  - `seed_list = [42, 1337, 2024]`
  - `representative_run_rule = highest_validation_macro_f1`
  - `timing_includes_dataloader_overhead = true`

B2. Seed-list and reproducibility policy
- The seed list must be explicit, saved, and reused across all models
- The same seed list must drive:
  - model initialization
  - DataLoader generator seeding
  - any other stochastic training behavior under project control
- If separate training and DataLoader seeds are used, derive them deterministically and save both
- Do not leave seed derivation implicit
- If `run_count` is smaller than the full saved seed list, document which prefix or subset was used
- Do not treat MLP as exempt from `dataloader_seed`; record it for all three models

C. Define one execution story and orchestration home
- Use a canonical user-facing pipeline entry point such as `scripts/run_model_pipeline.py`
- The canonical pipeline runner must accept:
  - `--model {mlp,text_cnn,bilstm,all}`
  - `--run-count N`
  - optional `--retune`
- Default user-facing behavior should be:
  - `--model all`
  - `--run-count 3`
- The pipeline runner should, for each selected model:
  - validate required preprocessing artifacts
  - reuse valid tuning outputs by default
  - retune only if tuning artifacts are missing, invalid, or `--retune` is passed
  - freeze the winning validation-selected config
  - run repeated final evaluation
  - emit per-model aggregate and representative-run metadata
  - hand off automatically to the Step 8 tracking stage and the Step 9 visualization stage unless the user explicitly requests a narrower stage-only execution path
- Use a dedicated Step 7 entry point such as `scripts/run_repeated_evaluation.py`
- Keep repeated-run orchestration in a dedicated module such as `src/repeated_evaluation.py`
- Keep the protocol dataclass in `src/config.py`
- `src/train.py` may delegate into the repeated-evaluation module, but it should not be the only place where the protocol contract exists
- The orchestration path should make it easy to run:
  - one model only
  - all models
  - one run count with a shared seed list

C2. Preserve model-specific training code while normalizing the repeated-run wrapper
- Keep model-specific training and evaluation logic in the appropriate modules
- Standardize the repeated-run outer loop rather than forcing all three models into one giant monolith
- Add adapters or refactor `evaluate_*` functions so the repeated-run wrapper can drive all models through one consistent contract
- Shared repeated-run logic should centralize:
  - seed handling
  - run indexing
  - aggregate metric computation
  - representative-run selection
  - artifact naming conventions
  - per-run output directory resolution
- Artifact-saving helpers must accept a per-run output directory instead of hardcoded flat output paths

C3. Close cross-model artifact parity before repeated runs are considered complete
- Step 7 must require the same core per-run outputs for `mlp`, `text_cnn`, and `bilstm`
- Do not defer core artifact parity to Step 8
- Explicit current gaps that must be closed before Step 7 is complete:
  - MLP lacks label-order parity
  - MLP lacks per-class-metrics parity
  - MLP lacks final-predictions parity
  - Text CNN lacks final-predictions parity
  - MLP lacks clear `dataloader_seed` parity

D. Define the final repeated-run contract per run
- Every repeated run must execute the full final workflow for that seed:
  - seed model initialization
  - build DataLoaders with the seeded policy
  - train with the frozen config
  - select the best checkpoint by the frozen validation metric
  - evaluate once on the validation split for the tracked validation metrics
  - evaluate once on the test split only after final-config selection is already complete
- Every repeated run must be reproducible as a standalone run
- Every completed run must have a stable run identifier using the canonical pattern:
  - `run_<two_digit_index>_seed_<seed>`

D2. Make the per-run artifact bundle mandatory and explicit
- Every completed run must save:
  - `run_metadata.json`
  - `validation_metrics.json`
  - `test_metrics.json`
  - `epoch_history.json`
  - `final_predictions.csv`
  - `confusion_matrix.csv`
  - `top_confusions.json`
  - `top_errors.json`
  - `per_class_metrics.json`
  - `label_order.json`
- `confidences.npz` may be optional for non-representative runs only if the probability-saving policy is saved explicitly in the protocol manifest and run metadata
- `final_predictions.csv` and `per_class_metrics.json` are required for all completed runs
- Do not assume downstream steps can reconstruct these artifacts later from thinner summaries

D3. `run_metadata.json` minimum schema
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

D4. `epoch_history.json` minimum schema
- `epoch_history.json` must be the canonical per-run training history artifact
- Save one record per epoch with at minimum:
  - `epoch`
  - `train_loss`
  - `val_loss`
  - `val_accuracy`
  - `val_macro_f1`
  - `val_macro_precision`
  - `val_macro_recall`
  - `val_oos_f1`
  - `learning_rate`
  - `checkpoint_updated`
  - optional `epoch_duration_seconds`

D5. `final_predictions.csv` minimum schema
- `final_predictions.csv` must include at minimum:
  - `example_index`
  - `text` if practical
  - `true_label_id`
  - `true_label_name`
  - `predicted_label_id`
  - `predicted_label_name`
  - `run_id`
  - optional `max_confidence`

D6. `per_class_metrics.json` minimum schema
- `per_class_metrics.json` must include one entry per class with at minimum:
  - `label_id`
  - `label_name`
  - `support`
  - `precision`
  - `recall`
  - `f1`
  - optional `accuracy`
  - optional `is_oos`

D7. Final-model and checkpoint policy
- Use the best checkpoint from that seed’s final training run directly
- Do not retrain after the seed-specific final run
- Every repeated run must write a unique checkpoint path under its run directory
- The checkpoint filename must encode the run index and seed to prevent collisions, for example:
  - `best_run_01_seed_42.pt`
- If retraining is ever introduced later, it must be treated as a new explicit protocol version rather than a silent change

D8. Confusion and error-analysis artifact policy
- Save `confusion_matrix.csv` and `top_confusions.json` for every completed run
- Save `top_errors.json` for every completed run
- Do not leave downstream steps guessing whether they must reconstruct confusion artifacts from predictions

E. Metric definition contract
- Use one shared metric implementation for all models
- At minimum compute for every repeated run:
  - `validation_accuracy`
  - `validation_macro_f1`
  - `validation_precision`
  - `validation_recall`
  - `test_accuracy`
  - `test_macro_f1`
  - `test_precision`
  - `test_recall`
- Keep zero-division behavior identical across all models
- Keep label ordering identical across all models
- Do not allow one model to use different averaging conventions than another

E2. OOS metric contract
- Keep OOS metrics explicit and shared across all models
- At minimum compute for every repeated run:
  - `oos_precision`
  - `oos_recall`
  - `oos_f1`
- Save whether OOS is being evaluated as:
  - an explicit class in the multiclass output
  - or another formally defined policy
- Save the OOS class identity and class index explicitly
- Do not change the OOS class index or definition relative to earlier steps
- If aggregate reporting includes OOS metrics, it must use the exact same one-vs-rest definition for every model

E3. Macro F1 and efficiency definition lock
- `macro_f1` means the full multiclass macro F1 including OOS unless a different metric is labeled explicitly
- If any secondary metric excludes OOS, label it clearly as secondary
- Save the following efficiency fields for every run:
  - `training_time_seconds`
  - `inference_total_seconds`
  - `inference_avg_ms_per_example`
  - `inference_examples_per_sec`
  - `parameter_count`
  - `trainable_parameter_count`
- Record whether timing includes DataLoader overhead in the shared protocol manifest and do not vary that rule by model

F. Save per-run validation and test metrics separately
- Per-run validation metrics and per-run test metrics must not be collapsed into one number too early
- Save one machine-readable per-run record per seed
- Save one per-model table containing all repeated-run metrics
- Include both validation and test metrics in that per-run table
- Include efficiency measurements in the same per-run table
- Save both `parameter_count` and `trainable_parameter_count` with each per-run record even if they are identical across seeds

F2. Aggregate repeated-run metrics correctly
- After all runs complete, compute aggregate mean and standard deviation across runs
- At minimum aggregate:
  - `validation_accuracy`
  - `validation_macro_f1`
  - `test_accuracy`
  - `test_macro_f1`
  - `test_precision`
  - `test_recall`
  - `oos_precision`
  - `oos_recall`
  - `oos_f1`
  - `training_time_seconds`
  - `inference_total_seconds`
  - `inference_avg_ms_per_example`
  - `inference_examples_per_sec`
  - `parameter_count`
  - `trainable_parameter_count`
- Save both mean and standard deviation
- If a later headline table omits validation accuracy for brevity, still keep `validation_accuracy` in the aggregate artifact
- If a metric is missing for a failed run, do not silently ignore it without recording the reason
- Save `run_count_requested`, `run_count_completed`, `seed_list_requested`, and `seed_list_completed` in the aggregate artifact

F3. Statistical reporting policy
- Use aggregated metrics for final model comparison tables
- Do not headline a single seed’s result as the main model result once Step 7 is implemented
- If a representative run is discussed qualitatively, label it as representative and keep it separate from aggregate performance claims
- If runtime only allows 3 runs, state that clearly
- If the protocol later expands to 5 runs, preserve the same artifact schema

G. Representative-run selection policy
- Choose one representative run per model for:
  - confusion matrix generation
  - top-confusion analysis
  - representative misclassified examples
  - class-level qualitative analysis
  - downstream Step 9 and Step 10 metadata
- The representative-run rule must be explicit, reproducible, and based on non-test information by default
- Recommended default:
  - choose the repeated final run with the highest validation macro F1
- If there is a tie, break it explicitly using:
  - lower validation loss
  - then earlier run index
  - then lexicographic run identifier
- If validation loss is unavailable, skip that tie-break and fall back to:
  - earlier run index
  - then lexicographic run identifier

G2. Keep representative-run analysis separate from aggregate reporting
- The representative run is not the aggregate result
- Confusion matrices, top-confusion charts, and representative errors should come from the representative run only unless there is a strong reason otherwise
- Aggregate tables may reference the representative run artifact path where useful, but must not replace mean and standard deviation reporting with representative-run metrics
- Every artifact derived from the representative run must record the run ID and seed it came from

H. Reproducibility and failure-handling requirements
- Set seeds before model initialization, DataLoader creation, and training
- Save the exact frozen config used by every run
- Save preprocessing artifact references and manifest references for every run
- Save environment information where practical:
  - device
  - Python version
  - PyTorch version
- Do not promise bit-perfect determinism on hardware that cannot guarantee it
- Do save enough metadata that the repeated-run evaluation can be reproduced faithfully

H2. Failed-run accounting
- If one repeated run fails, the failure must be logged honestly
- Save at minimum:
  - `seed`
  - `run_index`
  - `run_id`
  - failure reason
  - stage reached
  - any partial artifacts that were produced
- Failed runs must not vanish from the reporting trail
- Aggregate summaries must explicitly state whether all runs succeeded
- Do not silently reduce `run_count` after failures without recording it

I. Console and orchestration output expectations
- When repeated evaluation runs, print a concise but clear summary including:
  - model name
  - frozen config identifier
  - source tuning artifact
  - seed list
  - run count
  - current run index and seed
  - best validation metric per run
  - final test metrics per run
  - representative-run selection result
  - aggregate mean and standard deviation summary after all runs complete
- Make it obvious in console output when the pipeline transitions from:
  - tuning and selection
  - to repeated final evaluation

J. Recommended outputs
- Step 7 owns:
  - `outputs/shared/evaluation_protocol.json`
  - `outputs/<model_name>/tuning/tuning_results.csv`
  - `outputs/<model_name>/tuning/selection_summary.json`
  - `outputs/<model_name>/frozen_final_config.json`
  - `outputs/<model_name>/final_runs/run_<index>_seed_<seed>/...` (includes `checkpoint/` and `logs/` subdirectories)
  - `outputs/<model_name>/aggregate/per_run_metrics.csv`
  - `outputs/<model_name>/aggregate/aggregate_metrics.json`
  - `outputs/<model_name>/aggregate/aggregate_metrics.csv`
  - `outputs/<model_name>/aggregate/aggregate_comparison_row.json`
  - `outputs/<model_name>/aggregate/representative_run.json`
- Step 7 should not create shared cross-model comparison tables or figure manifests; those belong to Steps 8 and 9 under the same canonical directory structure

K. Validation checks and assertions
- Assert the frozen config exists before repeated runs begin
- Assert the seed list is non-empty and has no duplicates unless duplicates are explicitly justified
- Assert the same seed list is reused across all models
- Assert test evaluation does not run before frozen-config selection is complete
- Assert aggregate mean and standard deviation calculations use the actual saved per-run records
- Assert representative-run selection uses the documented rule
- Assert label ordering is identical across all runs and all models
- Assert preprocessing manifest references are identical across repeated runs for a given model
- Assert all repeated runs use the same monitoring metric
- Assert per-run artifact schemas validate before aggregate summaries are built
- Assert the canonical directory structure is used from Step 7 onward and that ambiguous root-level aggregate artifacts are not created

L. Test requirements
- Add or update tests covering:
  - repeated-run seed propagation
  - frozen-config extraction from tuning results
  - aggregate mean and standard deviation computation
  - representative-run selection logic
  - no-test-before-final-selection guardrails
  - failed-run accounting
  - schema consistency across runs
  - cross-model artifact parity for required files
- Extend nearby pipeline tests rather than creating disconnected low-value tests
- Prefer tests that validate behavior and saved content, not just filenames

M. Definition of done
Step 7 is only complete if all of the following are true:
- one frozen final configuration exists for each model
- repeated-run evaluation uses a shared seed list across all models
- the default protocol runs 3 seeds cleanly and can scale to 5 without redesign
- every completed run saves the full required per-run bundle
- aggregate mean and standard deviation metrics are saved under the canonical aggregate directory
- representative-run selection is explicit and reproducible
- confusion and error-analysis artifacts are tied to the representative run rather than mixed into aggregate reporting
- metric definitions are identical across all models
- OOS metrics are computed consistently across all models
- repeated-run failures, if any, are logged honestly
- the orchestration code clearly separates tuning from repeated final evaluation
- all three models satisfy the same core artifact parity requirements
- the outputs are ready for Step 8 tracking and Step 9 visualization work

M2. Final Step 7 exit gate
- Before Step 7 is considered closed, confirm all of the following are true:
  - aggregate metrics are derived from saved per-run artifacts
  - the same seeds were used across MLP, Text CNN, and BiLSTM
  - representative-run metadata exists for every model
  - `final_predictions.csv` and `per_class_metrics.json` exist for every completed run
  - test metrics are no longer presented as single-seed headline results
  - Step 8 can ingest the saved per-run and aggregate artifacts without schema ambiguity
  - Step 9 can generate both representative-run and aggregate figures without needing ad hoc recomputation

N. Deliverable quality bar
Step 7 should not be a thin wrapper around the current single-run scripts. It should establish a disciplined, project-wide evaluation protocol that makes model comparisons genuinely fair, reproducible, and defensible. When the report later says one model beat another, that claim should be grounded in aggregate repeated-run evidence rather than one lucky seed.

O. Suggested self-check after implementation
After coding, self-check against this exact checklist and confirm:
1. which existing tuning artifact was used to derive the frozen final config for each model
2. where the frozen final config is saved for each model
3. what the default seed list is and where it is saved
4. how tuning is separated from repeated final evaluation
5. which entry point launches repeated evaluation
6. which module owns the repeated-run orchestration logic
7. where per-run validation metrics are stored
8. where per-run test metrics are stored
9. where aggregate mean and standard deviation metrics are stored
10. how representative runs are selected
11. that representative-run selection does not use test metrics by default
12. how macro F1 is defined and whether it includes OOS
13. how OOS metrics are defined and saved
14. how failed repeated runs are recorded
15. that all models used the same seeds and evaluation protocol
16. that the repeated-run outputs are ready for Step 8 artifact tracking
17. that the repeated-run outputs are ready for Step 9 visualizations and Step 10 error analysis
