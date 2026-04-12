Implement and verify Step 4: TF-IDF + MLP baseline model for the CLINC150 project.

Goal:
Build a clean, reproducible, well-regularized TF-IDF + MLP baseline that serves as the first serious reference point for the project. This model should be trained carefully, with proper validation monitoring, early stopping, dropout, and sensible hyperparameter tuning, so that later comparisons against Text CNN and BiLSTM are fair and meaningful. Although multiple metrics should be computed and logged, validation macro F1 should be the primary model-selection and early-stopping metric.

Primary outcome:
At the end of this step, I want to have:
1. a working TF-IDF + MLP baseline implemented in PyTorch
2. a clear and reproducible baseline training pipeline
3. validation-driven model selection with early stopping
4. regularization handled explicitly, including dropout and optional weight decay
5. a small but disciplined hyperparameter tuning workflow
6. best checkpoint saved based on validation performance
7. final test-set metrics stored in report-ready artifacts
8. baseline results strong enough to serve as the first comparison point for later neural models
9. run artifacts aligned with the final cross-model comparison table schema
10. useful post-training error analysis artifacts for the final report

Important scope constraints:
- Do not implement Text CNN or BiLSTM here
- Do not change preprocessing or label mappings during this step
- Do not use test data for model selection, early stopping, threshold tuning, or hyperparameter tuning
- Do not over-engineer the baseline into a giant search project
- Keep tuning disciplined and limited enough to fit the project timeline
- Use Step 3 saved artifacts rather than rebuilding preprocessing from scratch
- Keep all metrics and training logic consistent with what later models will use
- Explicitly follow the OOS strategy chosen earlier in the project; do not improvise a new OOS policy inside Step 4
- Do not perform OOS threshold tuning in Step 4 unless threshold-based OOS was already explicitly approved as the chosen evaluation design and is tuned on validation only
- Never use test data for OOS threshold selection

High-level design requirement:
This step should consume the saved Step 3 preprocessing outputs for the TF-IDF branch and train a PyTorch MLP classifier on top of those fixed features. It should include:
- configurable architecture
- configurable dropout
- configurable optimizer / learning rate / weight decay
- validation monitoring
- early stopping
- checkpoint saving
- structured experiment logging

Implementation requirements:

A. Create a clean baseline training entry point
- Add baseline model training entry logic in src/train.py and/or main.py
- Keep the baseline workflow easy to run independently
- In addition to standalone use, this step must expose a pipeline-invocable entry path for `scripts/run_model_pipeline.py --model mlp`
- Prefer explicit functions such as:
  - build_mlp_baseline(...)
  - train_mlp_baseline(...)
  - evaluate_mlp_baseline(...)
  - run_mlp_experiment(...)
  - tune_mlp_baseline(...)
  - train_mlp_final_once(...)
- Make the training path reusable for later comparisons and report generation
- Ensure this step reads preprocessing artifacts from Step 3 rather than re-fitting TF-IDF
- The shared pipeline runner must be able to call this step to:
  - run or reuse tuning
  - write the model's tuning artifacts
  - launch one final-train/evaluate run from frozen hyperparameters
- By default, the pipeline runner should reuse valid existing tuning artifacts and only rerun tuning for MLP when artifacts are missing, invalid, or `--retune` is passed

B. Implement the MLP model cleanly
- Add the baseline model to src/models/mlp.py
- The model should be a straightforward feedforward classifier over TF-IDF features
- Support configurable architecture such as:
  - input layer
  - one or two hidden layers
  - nonlinearity
  - dropout
  - output classifier layer
- Keep the first implementation lightweight and justifiable
- Avoid turning the baseline into a huge deep network
- Make sure output dimension matches the number of classes from Step 2 label mapping
- Use a clean forward() definition returning logits for multiclass classification

Recommended bias:
- Start with a simple MLP:
  - input -> hidden -> dropout -> output
- Optionally allow a second hidden layer, but do not make it mandatory
- Keep architecture tunable but restrained

C. Define the baseline input contract explicitly
- Load the saved TF-IDF vectorizer / matrices or rebuilt transformed matrices from Step 3 artifacts only
- Confirm the input dimension used by the MLP matches the fitted TF-IDF feature dimension from Step 3
- Reuse Step 2 label mapping and Step 3 TF-IDF artifacts exactly
- Save the reference paths to preprocessing artifacts in training summary outputs
- Do not recompute label mappings or silently alter feature ordering
- Explicitly document whether TF-IDF inputs are consumed as dense arrays, sparse-origin batches converted to dense, or another clearly documented format
- Add a sanity check that the chosen TF-IDF input path is memory-safe for the dataset size and feature dimension
- Log whether the model ultimately consumes dense inputs or sparse-origin inputs
- Confirm the MLP output dimension and label mapping exactly match the chosen OOS strategy

C2. Commit to one OOS handling strategy
- Before implementing training logic, explicitly confirm the OOS strategy chosen earlier in the project
- Document whether OOS is handled as:
  - an explicit class in the multiclass output space
  - or a confidence-threshold decision rule applied after training
- For Step 4, prefer the explicit-class approach unless the project has already formally committed to threshold-based OOS detection
- If threshold-based OOS is the chosen strategy, document exactly when and how the threshold is tuned, and ensure it uses validation data only
- Save the chosen OOS strategy in run summaries so later models can match it exactly

D. Support disciplined hyperparameter configuration
- Centralize baseline hyperparameters in src/config.py
- At minimum make these configurable:
  - hidden_dim
  - optional second_hidden_dim
  - dropout_rate
  - learning_rate
  - weight_decay
  - batch_size
  - max_epochs
  - early_stopping_patience
  - optimizer choice if multiple are supported
  - activation function if configurable
  - decision to use one or two hidden layers
- Keep defaults simple and defensible
- Avoid scattering baseline-specific magic numbers across files

E. Hyperparameter tuning expectations
- Be diligent, but scoped
- Use a small validation-driven tuning workflow rather than random guessing
- Tune only on train/validation splits
- Do not touch test data until the final best configuration is selected
- Prefer a compact experiment grid or staged tuning approach rather than an enormous exhaustive search
- Save every tried configuration and its validation result
- Emit tuning outputs in a stable machine-readable form so the shared pipeline runner can detect and reuse them without guessing filenames or recomputing the selection result

Recommended bias:
Use a restrained search over a few plausible settings, for example:
- hidden_dim: a few options
- dropout_rate: a few options
- learning_rate: a few options
- weight_decay: a few options
- maybe 1-layer vs 2-layer MLP

Do not explode the search space. This is a course project, not a benchmark paper.

F. Explicit regularization requirements
- Include dropout in the MLP hidden layers
- Support weight decay through the optimizer
- Make regularization choices visible in logs and saved summaries
- Be explicit about why dropout and weight decay are included
- Keep these settings tunable and validation-driven
- Do not silently apply regularization without documenting it

G. Early stopping requirements
- Implement proper early stopping based on validation performance
- The monitored validation metric should be explicit and configurable
- Default and recommended setting:
  - monitor validation macro F1
- Validation loss may still be logged for diagnostics, but it should not be the default selection metric
- Save best checkpoint whenever the monitored metric improves
- Stop training when patience is exceeded
- Log:
  - best epoch
  - best validation metric
  - stopping epoch
  - reason for stopping
- Make sure early stopping is consistent and deterministic given the same seed/config/data

H. Checkpointing requirements
- Save the best baseline checkpoint to outputs/checkpoints/
- Include enough metadata to identify the run
- Save:
  - model_state_dict
  - optimizer_state_dict if useful
  - epoch
  - best validation metric
  - config snapshot
  - artifact references for preprocessing inputs
- Make checkpoint naming clean and machine-readable
- Ensure later evaluation loads the best checkpoint, not the last epoch by accident

I. Training loop requirements
- Use reusable trainer logic if available in src/trainers/trainer.py
- Training loop should log at minimum per epoch:
  - training loss
  - validation loss
  - validation accuracy
  - validation macro F1
- If OOS metrics are already wired consistently, also log validation OOS metrics
- Even though multiple validation metrics are logged, validation macro F1 should be treated as the primary validation metric for early stopping and baseline model selection
- Make the tracked primary validation metric explicit in logs and run summaries
- Make sure train/eval mode switches are handled correctly
- Make sure gradient steps, optimizer zeroing, and device placement are handled safely
- Confirm one full training run completes without shape/type/device errors

J. Baseline dataset / DataLoader expectations
- Use Step 3 TF-IDF outputs consistently
- If Step 3 created TF-IDF DataLoaders, reuse them
- If Step 3 kept TF-IDF as matrices only, create a minimal wrapper here or in Step 3-compatible utilities
- Ensure:
  - train loader has non-zero batches
  - validation loader has non-zero batches
  - test loader exists and is not used until final evaluation
- Batch shapes should be logged at least once for sanity checking
- Confirm input tensors are float tensors and targets are integer class ids

K. Loss function and optimization
- Use a standard multiclass classification loss such as cross-entropy loss
- Ensure logits and targets are shaped correctly for the chosen loss
- Support at least one sensible optimizer, usually Adam
- Weight decay should be configurable
- Learning rate should be configurable
- Optimizer choice should be logged and saved in run artifacts

K2. Class imbalance policy
- Explicitly decide whether class-weighted cross-entropy is used or not
- If class weights are supported, they must be computed from the training split only
- Keep class weighting configurable but simple
- Log the final class-weight policy in run summaries even if the decision is to use no class weights

K3. Learning-rate scheduling policy
- Explicitly document whether learning-rate scheduling is used
- It is acceptable for the default MLP baseline to use a fixed learning rate if that choice is justified clearly
- A reasonable default justification is that the baseline is shallow, the search space is intentionally compact, and early stopping already controls training length
- If learning-rate scheduling is tried, treat it as an optional tuning dimension and record it in tuning artifacts
- Do not leave the absence or presence of scheduling implicit

L. Metric consistency requirements
- Compute validation metrics using the same conventions intended for later models
- At minimum track:
  - accuracy
  - macro F1
  - precision
  - recall
- If available, also compute OOS precision, recall, and F1 consistently
- Make sure label handling and averaging settings match later experiments
- Document clearly which metric is used for early stopping / model selection
- For this baseline, validation macro F1 should be the default and preferred early-stopping / model-selection metric

M. Small, disciplined experiment tracking
- For every baseline run, save:
  - hyperparameters
  - training curves or epoch history
  - validation metrics
  - best checkpoint path
  - training duration
  - parameter count
- Save a summary table comparing tried baseline configurations
- Example artifact:
  - outputs/reports/mlp_tuning_results.csv
- Keep run names organized so the best baseline is easy to identify later
- Save results in a schema that can later merge cleanly with CNN and BiLSTM experiment outputs

N. Parameter count and model size tracking
- Compute parameter count for the baseline model
- Save it in the run summary
- Later comparisons will benefit from knowing:
  - parameter count
  - training time
  - final metrics
- Keep this consistent with what later models will report
- inference latency
- Track inference speed for the baseline, for example average milliseconds per example or examples per second
- Save inference-speed measurements in the run summary and final test metrics artifact

O. Console output expectations
When baseline training runs, it should print a concise but informative summary including:
- run name
- preprocessing artifact references
- input feature dimension
- number of classes
- model architecture summary
- parameter count
- optimizer / learning rate / weight decay / dropout
- monitored validation metric
- explicitly show that validation macro F1 is the primary early-stopping / model-selection metric unless deliberately overridden
- per-epoch training loss
- per-epoch validation loss
- per-epoch validation accuracy
- per-epoch validation macro F1
- early stopping events
- best checkpoint path
- total training time

P. Artifact expectations
At the end of Step 4, I expect artifacts similar to these:
- outputs/checkpoints/mlp_best_*.pt
- outputs/logs/mlp_training_log_*.json or .csv
- outputs/reports/mlp_run_summary_*.json
- outputs/reports/mlp_tuning_results.csv
- outputs/reports/mlp_test_metrics.json
- outputs/reports/mlp_comparison_row.json
- outputs/reports/mlp_confusion_matrix.csv or .json
- outputs/reports/mlp_top_confusions.json
- outputs/reports/mlp_top_errors.json
- optionally:
  - outputs/figures/mlp_loss_curve.png
  - outputs/figures/mlp_val_metric_curve.png

The run summary should contain at minimum:
1. run name
2. config snapshot
3. preprocessing artifact references
4. tfidf_input_dim
5. number of classes
6. model architecture details
7. parameter count
8. optimizer settings
9. dropout rate
10. weight decay
11. max epochs
12. early stopping patience
13. best epoch
14. best validation metric
15. primary model-selection metric used
16. training duration
17. checkpoint path
18. OOS strategy used
19. class-weight policy
20. TF-IDF input format
21. inference latency summary
22. whether learning-rate scheduling was used
23. shuffle policy for train / validation / test loaders
24. preprocessing artifact refs for final comparison-table export
25. label-name ordering reference

Q. Final test evaluation requirements
- After selecting the best baseline configuration using validation only, load the best checkpoint
- Run exactly one proper evaluation on the test split
- Save final test metrics in a report-ready artifact
- At minimum compute:
  - test accuracy
  - test macro F1
  - test precision
  - test recall
- If the OOS metric pipeline is ready, also compute:
  - OOS precision
  - OOS recall
  - OOS F1
- Save results in a format easy to merge into the final comparison table
- Compute and log inference latency on the test split, using a simple, clearly documented measurement procedure
- Save a confusion matrix or equivalent per-class confusion artifact
- Save a top-confusions artifact, for example the most confused intent pairs
- Save a top-errors artifact containing representative misclassified texts for later report writing
- Save the exact label-name ordering alongside these metric and confusion artifacts so later analysis cannot drift

Q2. Final model selection and retraining rule
- Do not leave the final-model rule ambiguous
- Choose and document one explicit policy:
  - either use the single best tuning run directly as the final baseline
  - or retrain exactly once using the selected configuration before final test evaluation
- Save the chosen policy in the run summary
- If retraining is used, document the seed policy clearly
- Final test evaluation must always be tied to this explicit final-model rule

R. Keep the baseline as the first reference point
- This model must be treated as a meaningful baseline, not a throwaway toy
- Later models should be compared against:
  - its final test metrics
  - its parameter count
  - its training time
  - its validation behavior
- This means Step 4 must be implemented carefully enough that beating it actually means something

S. Validation checks / assertions
Include explicit sanity checks so this step fails loudly if something is wrong:
- TF-IDF input dimension matches model input dimension
- number of output classes matches label mapping size
- train / validation / test loaders or matrices are non-empty
- label ids are within valid class range
- no NaN / inf values appear in model inputs
- no NaN loss values occur during training
- checkpoint directory exists or is created automatically
- best checkpoint is actually saved
- early stopping metric improves / tracks correctly
- loaded best checkpoint can be reloaded for evaluation
- test evaluation does not run before model selection is finished

T. Fresh-process reuse requirement
- Confirm the saved best checkpoint can be loaded in a fresh process
- Confirm the baseline can run evaluation from:
  - saved checkpoint
  - saved preprocessing artifacts
  - saved config summary
- Do not require re-training to reproduce final test evaluation

T2. Trainer generalization check
- If src/trainers/trainer.py is used, verify that it is model-agnostic rather than tightly coupled to the MLP baseline
- Confirm it can accept arbitrary PyTorch models and compatible DataLoaders without assuming TF-IDF-specific shapes or baseline-only device logic
- If the trainer ends Step 4 in an overly MLP-specific state, refactor it before moving to Step 5
- Run a minimal interface check showing the trainer can at least be called cleanly in a non-MLP setting or with a placeholder model

U. Determinism / reproducibility requirement
- Given the same data, seed, and config, training behavior should be as stable as reasonably possible
- Centralize seeding in src/utils.py or equivalent
- Set seed before model init, DataLoader creation with shuffle, and training loop execution
- Save the seed used for each run
- Do not promise bit-perfect determinism if the stack cannot guarantee it, but aim for strong reproducibility
- Save enough metadata that the run can be recreated faithfully later

U2. Step 5 / Step 6 readiness smoke tests
- Before closing Step 4, perform a lightweight smoke test that the token-sequence DataLoaders from Step 3 can still be loaded and iterated without error
- Confirm they produce batches of shape (batch_size, seq_len)
- Log the confirmed token-batch shapes in a Step 4 summary artifact or console output
- This is not model training for Steps 5 or 6; it is only a pipeline-readiness check

V. Suggested baseline tuning workflow
Use a staged, disciplined process such as:
1. run one default baseline to confirm pipeline correctness
2. tune dropout / hidden size / learning rate / weight decay on validation
3. compare a small set of candidate runs
4. select the best config using validation macro F1 by default, unless a different metric is explicitly justified and documented
5. retrain if necessary under the chosen config
6. evaluate once on test set
7. save final baseline metrics and summary

W. Recommended initial tuning dimensions
Start with a compact search, for example:
- hidden_dim: a few sizes
- dropout_rate: a few reasonable values
- learning_rate: a few values
- weight_decay: off vs a few small values
- optionally one hidden layer vs two hidden layers

Do not let the search become huge. Keep the number of runs realistic for the project timeline.

W2. Final comparison-table export requirements
- Export a clean baseline result record for later cross-model comparison
- At minimum include fields such as:
  - model_name
  - input_type
  - primary_val_metric
  - best_val_macro_f1
  - test_accuracy
  - test_macro_f1
  - test_precision
  - test_recall
  - OOS metrics if applicable
  - training_time
  - inference_latency
  - parameter_count
  - checkpoint_path
  - preprocessing_artifact_refs
  - notes
- Keep this schema stable so later CNN and BiLSTM results can be merged without rework

X. Definition of done
Step 4 is only complete if all of the following are true:
- TF-IDF + MLP model is implemented cleanly in PyTorch
- model input dimension matches Step 3 TF-IDF features
- output dimension matches Step 2 label mapping
- baseline training runs end-to-end without errors
- dropout and optional weight decay are implemented and configurable
- early stopping is implemented and used
- best checkpoint is saved based on validation performance
- a small, disciplined hyperparameter tuning process was run
- tried configurations and validation results are saved
- validation macro F1 was used as the default early-stopping / model-selection metric while other metrics were still computed and logged
- final best baseline is selected using validation only
- best checkpoint can be loaded in a fresh process
- final test metrics are computed and saved
- baseline parameter count and training time are saved
- artifacts are organized and report-ready
- this baseline is strong enough to serve as the first reference point for later comparisons
- the chosen OOS strategy is implemented consistently and documented in artifacts
- threshold-based OOS tuning was not performed unless explicitly approved and validation-only
- class-weight policy is documented
- learning-rate scheduling policy is documented
- inference latency is computed and saved
- confusion and top-error artifacts are saved
- the final-model selection / retraining rule is explicit and documented
- a comparison-table export artifact is saved
- token-sequence DataLoaders from Step 3 passed a basic readiness smoke test

Y. Deliverable quality bar
The baseline should be simple, strong, and reproducible. It should not be a lazy placeholder. It should reflect real care in model selection, regularization, and validation monitoring so that later claims like “CNN beat the baseline” or “BiLSTM underperformed the baseline” actually mean something.

Z. Suggested self-check after implementation
After coding, self-check against this exact checklist and confirm:
1. which files/functions implement the MLP baseline
2. how Step 3 TF-IDF artifacts are reused without re-fitting
3. which hyperparameters were tuned
4. that validation macro F1 selected the best model by default, and that any override was explicit and documented
5. how early stopping works
6. where checkpoints and tuning summaries are saved
7. what the final baseline test metrics are
8. that the best checkpoint can be loaded and evaluated in a fresh process
9. which OOS strategy was used and whether it matches the earlier project decision
10. how TF-IDF inputs are represented in PyTorch and whether the choice is memory-safe
11. whether class weights were used and how that decision was made
12. whether learning-rate scheduling was used or explicitly not used
13. what the baseline inference latency is
14. where confusion-matrix and top-error artifacts are saved
15. that the Step 3 token-sequence DataLoaders still load and iterate correctly for later steps