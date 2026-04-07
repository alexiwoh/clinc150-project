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

High-level design requirement:
This step should refactor the workflow into two distinct phases:
- phase 1: validation-only tuning and final-config selection
- phase 2: repeated final runs using the selected configuration and a shared seed list

The recommended evaluation contract is:
- pick one final config per model using validation only
- freeze that config
- rerun only the final train/validation/test workflow across a shared seed list
- save per-run artifacts for every repeated run
- compute aggregate mean and standard deviation metrics across runs
- choose one representative run for confusion matrices, top confusions, representative misclassified examples, and Step 10 handoff artifacts

Recommended default implementation path:
- keep the current validation-driven tuning logic from Step 4-6
- promote the selected final config into a frozen config artifact
- define one shared seed list such as `[42, 1337, 2024]`
- rerun the final training and test evaluation once per seed
- treat these repeated runs as the official reporting protocol
- keep the option to extend to 5 seeds later
- choose the representative run using validation-only information, not test metrics

Implementation requirements:

A. Freeze the final configuration before repeated runs begin
- Each model must have one explicit frozen final configuration before repeated-run evaluation starts
- The final configuration must be selected using validation only
- The frozen config must include all training-relevant hyperparameters, preprocessing references, and monitoring settings
- The frozen config must be saved in a machine-readable artifact before the repeated-run loop starts
- The repeated-run loop may not mutate model architecture, optimizer policy, early-stopping policy, sequence length, label mapping, or OOS strategy

A2. Explicitly separate tuning from repeated final evaluation
- Do not let the repeated-run loop call the hyperparameter search again
- Do not let the repeated-run loop pick a different config per seed
- The implementation should make it obvious where the pipeline transitions from:
  - tuning / selection
  - to frozen-config repeated evaluation
- If the current codebase still combines these concerns in a single function, refactor it before Step 7 is considered complete
- The separation should be visible in orchestration code, not just in comments

B. Introduce a shared repeated-run evaluation configuration
- Add one shared evaluation protocol configuration in `src/config.py`
- At minimum make these configurable:
  - run_count
  - seed_list
  - representative-run selection rule
  - whether to save logits / probabilities for every run or representative run only
  - whether to save confusion artifacts for every run or representative run only
  - aggregate metric list
  - test-evaluation enable/disable guard
- Keep the default protocol simple and explicit
- Recommended default:
  - run_count = 3
  - seed_list = `[42, 1337, 2024]`
  - representative-run rule = highest validation macro F1 among final repeated runs

B2. Seed-list policy
- The seed list must be explicit, saved, and reused across all models
- The same seed list must drive:
  - model initialization
  - DataLoader generator seeding
  - any other stochastic training behavior under project control
- If separate training and DataLoader seeds are used, derive them deterministically and save both
- Do not leave seed derivation implicit
- If `run_count` is smaller than the full saved seed list, document which prefix or subset was used

C. Refactor experiment orchestration for repeated runs
- Refactor the current experiment entry logic in `src/train.py` so tuning and repeated final runs are distinct phases
- Prefer explicit functions such as:
  - `select_final_mlp_config(...)`
  - `select_final_text_cnn_config(...)`
  - `select_final_bilstm_config(...)`
  - `run_repeated_mlp_evaluation(...)`
  - `run_repeated_text_cnn_evaluation(...)`
  - `run_repeated_bilstm_evaluation(...)`
  - `aggregate_repeated_run_metrics(...)`
- The orchestration path should make it easy to run:
  - one model only
  - all models
  - one run count with a shared seed list
- Keep the script entry points simple, but make the internal orchestration reusable

C2. Preserve model-specific training code while standardizing repeated evaluation
- Keep model-specific training and evaluation logic in the appropriate modules
- Standardize the repeated-run outer loop rather than forcing all three models into one giant monolith
- The repeated-run wrapper should be able to call:
  - the MLP final-train/evaluate path
  - the Text CNN final-train/evaluate path
  - the BiLSTM final-train/evaluate path
- Shared repeated-run logic should centralize:
  - seed handling
  - run indexing
  - aggregate metric computation
  - representative-run selection
  - artifact naming conventions

D. Define the final repeated-run contract per run
- Every repeated run must execute the full final workflow for that seed:
  - seed model initialization
  - build DataLoaders with the seeded policy
  - train with the frozen config
  - select the best checkpoint by the frozen validation metric
  - evaluate once on the test split
- Each repeated run must save:
  - run index
  - training seed
  - DataLoader seed
  - best validation metric
  - best epoch
  - test metrics
  - checkpoint path
  - epoch history path
  - preprocessing artifact refs
  - frozen config snapshot
- Every repeated run must be reproducible as a standalone run

D2. Final-model rule inside repeated runs
- Do not leave the repeated-run final-model rule ambiguous
- Choose one explicit policy and keep it identical across all models
- Recommended default:
  - for each seed, use the best checkpoint from that seed’s final training run directly
  - do not retrain after the seed-specific final run
- If retraining is ever introduced later, it must be treated as a new explicit protocol version rather than a silent change

E. Metric definition contract
- Use one shared metric implementation for all models
- At minimum compute for every repeated run:
  - validation accuracy
  - validation macro F1
  - validation precision
  - validation recall
  - test accuracy
  - test macro F1
  - test precision
  - test recall
- Keep zero-division policy identical across all models
- Keep label ordering identical across all models
- Do not allow one model to use different averaging conventions than another

E2. OOS metric contract
- Keep OOS metrics explicit and shared across all models
- At minimum compute for every repeated run:
  - OOS precision
  - OOS recall
  - OOS F1
- Save whether OOS is being evaluated as:
  - an explicit class in the multiclass output
  - or another formally defined policy
- Do not change the OOS class index or definition relative to earlier steps
- If aggregate reporting includes OOS metrics, it must use the exact same one-vs-rest definition for every model

E3. Macro F1 definition lock
- Explicitly document whether macro F1 includes the OOS class
- Recommended default:
  - keep the existing full multiclass macro F1 definition used by the current shared metric functions
- If any secondary metric excludes OOS, label it clearly as secondary
- Do not silently replace the main macro F1 definition used in Steps 4-6

F. Save per-run validation and test metrics separately
- Per-run validation metrics and per-run test metrics must not be collapsed into one number too early
- Save one machine-readable per-run record per seed
- Save one per-model table containing all repeated-run metrics
- Include both validation and test metrics in that per-run table
- Include training-time and inference-time measurements in the same per-run table
- Save parameter count with each per-run record even if the number is identical across seeds

F2. Aggregate repeated-run metrics correctly
- After all runs complete, compute aggregate mean and standard deviation across runs
- At minimum aggregate:
  - validation macro F1
  - test accuracy
  - test macro F1
  - test precision
  - test recall
  - OOS precision
  - OOS recall
  - OOS F1
  - training time
  - inference latency
- Save both:
  - mean
  - standard deviation
- If a metric is missing for a failed run, do not silently ignore it without recording the reason
- Save run_count_used and seed_list_used in the aggregate artifact

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
  - Step 10 handoff
- The representative-run rule must be explicit, reproducible, and based on non-test information by default
- Recommended default:
  - choose the repeated final run with the highest validation macro F1
- If there is a tie, break it explicitly using:
  - lower validation loss
  - then earlier run index
  - then lexicographic run identifier

G2. Keep representative-run analysis separate from aggregate reporting
- The representative run is not the aggregate result
- Confusion matrices, top-confusion charts, and representative errors should come from the representative run only unless there is a strong reason otherwise
- Aggregate tables should reference the representative run artifact path where useful, but must not replace mean/std reporting with representative-run metrics
- Every artifact derived from the representative run must say which run ID and seed it came from

H. Save per-run predictions and confidence outputs carefully
- For every repeated run, save final predictions in a machine-readable format if storage allows
- At minimum the representative run must save:
  - y_true
  - y_pred
  - label names
  - text if practical
  - confidence / probability if practical
- If logits or probabilities are too large to save for every run, require them for the representative run and document that policy
- The repeated-run checklist must not assume that confusion and qualitative analysis can be reconstructed later without saved predictions

I. Reproducibility requirements
- Set seeds before model initialization, DataLoader creation, and training
- Save the training seed and DataLoader seed for every run
- Save the exact frozen config used by every run
- Save preprocessing artifact references and manifest references for every run
- Save environment information where practical:
  - device
  - Python version
  - PyTorch version
- Do not promise bit-perfect determinism on hardware that cannot guarantee it
- Do save enough metadata that the repeated-run evaluation can be reproduced faithfully

I2. Failure handling in repeated runs
- If one repeated run fails, the failure must be logged honestly
- Save:
  - seed
  - run index
  - failure reason
  - stage reached
  - any partial artifacts that were produced
- Failed runs must not vanish from the final reporting trail
- Aggregate summaries must explicitly state whether all runs succeeded
- Do not silently reduce `run_count` after failures without recording it

J. Console and orchestration output expectations
- When repeated evaluation runs, print a concise but clear summary including:
  - model name
  - frozen config identifier
  - seed list
  - run count
  - current run index and seed
  - best validation metric per run
  - final test metrics per run
  - representative-run selection result
  - aggregate mean/std summary after all runs complete
- Make it obvious in console output when the pipeline transitions from:
  - tuning
  - to repeated final evaluation

K. Recommended outputs
- `outputs/reports/shared/evaluation_protocol.json`
- `outputs/reports/shared/seed_list.json`
- `outputs/reports/<model_name>/frozen_final_config.json`
- `outputs/reports/<model_name>/per_run_metrics.csv`
- `outputs/reports/<model_name>/aggregate_metrics.json`
- `outputs/reports/<model_name>/representative_run.json`
- `outputs/reports/<model_name>/final_runs/run_<index>_seed_<seed>/...`
- `outputs/logs/<model_name>/final_runs/run_<index>_seed_<seed>/...`
- `outputs/checkpoints/<model_name>/final_runs/run_<index>_seed_<seed>/...`

L. Validation checks / assertions
- Assert the frozen config exists before repeated runs begin
- Assert the seed list is non-empty and has no duplicates unless duplicates are explicitly justified
- Assert the same seed list is reused across all models
- Assert test evaluation does not run before frozen config selection is complete
- Assert aggregate mean/std calculations use the actual saved per-run records
- Assert representative-run selection uses the documented rule
- Assert label ordering is identical across all runs and all models
- Assert preprocessing manifest references are identical across repeated runs for a given model
- Assert all repeated runs use the same monitoring metric
- Assert per-run artifact schemas validate before aggregate summaries are built

M. Test requirements
- Add or update tests covering:
  - repeated-run seed propagation
  - aggregate mean/std computation
  - representative-run selection logic
  - no-test-before-final-selection guardrails
  - failed-run accounting
  - schema consistency across runs
- Extend nearby pipeline tests rather than creating disconnected low-value tests
- Prefer tests that validate behavior, not just filenames

N. Definition of done
Step 7 is only complete if all of the following are true:
- one frozen final configuration exists for each model
- repeated-run evaluation uses a shared seed list across all models
- the default protocol runs 3 seeds cleanly and can scale to 5 without redesign
- per-run validation metrics are saved
- per-run test metrics are saved
- aggregate mean/std metrics are saved
- representative-run selection is explicit and reproducible
- confusion/error-analysis artifacts are tied to the representative run rather than mixed into aggregate reporting
- metric definitions are identical across all models
- OOS metrics are computed consistently across all models
- repeated-run failures, if any, are logged honestly
- the orchestration code clearly separates tuning from repeated final evaluation
- the outputs are ready for Step 8 tracking and Step 9 visualization work

N2. Final Step 7 exit gate
- Before Step 7 is considered closed, confirm all of the following are true:
  - aggregate metrics are derived from saved per-run artifacts
  - the same seeds were used across MLP, Text CNN, and BiLSTM
  - representative-run metadata exists for every model
  - test metrics are no longer presented as single-seed headline results
  - Step 8 can ingest the saved per-run and aggregate artifacts without schema ambiguity
  - Step 9 can generate both representative-run and aggregate figures without needing ad hoc recomputation

O. Deliverable quality bar
Step 7 should not be a thin wrapper around the current single-run scripts. It should establish a disciplined, project-wide evaluation protocol that makes model comparisons genuinely fair, reproducible, and defensible. When the report later says one model beat another, that claim should be grounded in aggregate repeated-run evidence rather than one lucky seed.

P. Suggested self-check after implementation
After coding, self-check against this exact checklist and confirm:
1. where the frozen final config is saved for each model
2. what the default seed list is and where it is saved
3. how tuning is separated from repeated final evaluation
4. which functions launch repeated runs for each model
5. where per-run validation metrics are stored
6. where per-run test metrics are stored
7. where aggregate mean/std metrics are stored
8. how representative runs are selected
9. that representative-run selection does not use test metrics by default
10. how macro F1 is defined and whether it includes OOS
11. how OOS metrics are defined and saved
12. how failed repeated runs are recorded
13. that all models used the same seeds and evaluation protocol
14. that the repeated-run outputs are ready for Step 8 artifact tracking
15. that the repeated-run outputs are ready for Step 9 visualizations and Step 10 error analysis
