Implement and verify Step 5: Text CNN model for the CLINC150 project.

Goal:
Build a clean, reproducible, well-regularized Text CNN classifier for CLINC150 using the shared preprocessing pipeline from Step 3. This model should be trained carefully with validation-driven model selection, early stopping, dropout, sensible hyperparameter tuning, and strong artifact generation so that comparison against the TF-IDF + MLP baseline and the later BiLSTM is fair, meaningful, and easy to report. Although multiple metrics should be computed and logged, validation macro F1 should be the primary model-selection and early-stopping metric.

Primary outcome:
At the end of this step, I want to have:
1. a working Text CNN classifier implemented in PyTorch
2. a clean training pipeline that consumes Step 3 neural preprocessing artifacts
3. validation-driven model selection with early stopping
4. regularization handled explicitly, including dropout and optional weight decay
5. a small but disciplined hyperparameter tuning workflow
6. best checkpoint saved based on validation macro F1
7. final test-set metrics stored in report-ready artifacts
8. comparison-ready run artifacts aligned with the baseline and future BiLSTM outputs
9. useful post-training error analysis artifacts for the final report
10. useful visualization artifacts for debugging, tuning, and reporting

-Important scope constraints:
- Do not change preprocessing or label mappings during this step
- Do not use test data for model selection, early stopping, threshold tuning, or hyperparameter tuning
- Do not over-engineer the CNN into a giant architecture search project
- Keep tuning disciplined and limited enough to fit the project timeline
- Use Step 3 saved artifacts rather than rebuilding preprocessing from scratch
- Keep all metrics, label ordering, and evaluation logic consistent with Step 4
- Explicitly follow the OOS strategy chosen earlier in the project; do not improvise a new OOS policy inside Step 5
- Do not perform OOS threshold tuning in Step 5 unless threshold-based OOS was already explicitly approved as the chosen evaluation design and is tuned on validation only
- Never use test data for OOS threshold selection
- This step should focus on a strong, standard Text CNN rather than highly experimental variants
- Step 5 may not begin until the required Step 3 artifacts are already saved and verified: vocabulary, PAD id, UNK id, max sequence length, label_to_id, id_to_label, tokenizer/normalization settings, and train/validation/test split artifacts
- If vocabulary, max sequence length, tokenization, normalization, or label mapping changes, all prior Step 5 runs become invalid for fair comparison and should not be mixed with the new runs
- Do not peek at test results repeatedly; after the final config is selected, any further debugging or tuning must return to validation only

High-level design requirement:
This step should consume the saved Step 3 neural preprocessing outputs and train a Text CNN classifier using token-id sequences and sequence DataLoaders. It should include:
- embedding layer
- one or more convolution filter sizes
- pooling
- dropout
- configurable optimizer / learning rate / weight decay
- validation monitoring
- early stopping
- checkpoint saving
- structured experiment logging
- report-ready visualizations

Implementation requirements:

A. Create a clean Text CNN training entry point
- Add Text CNN training entry logic in src/train.py and/or main.py
- Keep the CNN workflow easy to run independently
- Prefer explicit functions such as:
  - build_text_cnn(...)
  - train_text_cnn(...)
  - evaluate_text_cnn(...)
  - run_text_cnn_experiment(...)
- Make the training path reusable for later comparisons and report generation
- Ensure this step reads preprocessing artifacts from Step 3 rather than re-fitting vocabulary or recreating sequence tensors

A2. Preprocessing manifest and frozen-artifact gate
- Require one saved preprocessing manifest from Step 3 that Step 5 loads before training begins
- The manifest should include at minimum:
  - vocabulary size
  - PAD id
  - UNK id
  - max sequence length
  - label ordering
  - tokenizer configuration
  - normalization policy
  - OOS class/index info if applicable
  - preprocessing config hash or version tag
- Step 5 must verify that loaded neural artifacts match this manifest before training starts
- Save the loaded manifest reference in run summaries
- Log the inherited text preprocessing policy clearly, including:
  - lowercase or not
  - punctuation kept or removed
  - numbers kept or normalized
  - whitespace normalization

 B. Implement the Text CNN model cleanly
- Add the model to src/models/text_cnn.py
- The model should be a straightforward sentence classification CNN over token sequences
- Use a standard architecture pattern such as:
  - embedding layer
  - parallel 1D convolutions with multiple kernel sizes
  - nonlinearity
  - max-over-time pooling
  - concatenation
  - dropout
  - final classifier layer
- Keep the first implementation lightweight and justifiable
- Avoid turning this into a huge experimental architecture
- Make sure output dimension matches the number of classes from Step 2 label mapping
- Use a clean forward() definition returning logits for multiclass classification
- Explicitly handle the PyTorch Conv1d tensor-shape requirement
- nn.Embedding outputs tensors shaped like (batch_size, seq_len, embedding_dim)
- Before applying Conv1d, explicitly transpose or permute to (batch_size, embedding_dim, seq_len)
- Add a logged sanity check or assertion for the tensor shape immediately before convolution
- Compute the classifier input dimension programmatically as len(kernel_sizes) * num_filters rather than hardcoding it
- Add an assertion in model initialization that the classifier input dimension matches the concatenated pooled feature size

Recommended bias:
- Start with a standard Text CNN:
  - embeddings -> conv filters of a few sizes -> max pool -> concat -> dropout -> linear classifier
- Keep the architecture tunable but restrained
- Prefer a small set of kernel sizes and a reasonable filter count over deep stacks unless clearly justified
- This should be a standard Kim-style sentence CNN baseline, not a hybrid or highly customized variant

C. Define the neural input contract explicitly
- Load the saved vocabulary, max sequence length, label mappings, and sequence-ready split data from Step 3 artifacts only
- Confirm the token-id inputs match the vocabulary and padding conventions from Step 3
- Reuse Step 2 label mapping and Step 3 neural preprocessing artifacts exactly
- Save the reference paths to preprocessing artifacts in training summary outputs
- Do not recompute label mappings or silently alter vocabulary ordering
- Confirm that embedding input ids align with saved PAD and UNK ids
- Confirm that model output dimension and label mapping exactly match the chosen OOS strategy
- Verify that preprocessing artifact versions or config hashes match what the run expects, so stale Step 3 artifacts are not silently mixed into Step 5
- Confirm fairness with the baseline by keeping label ordering, metric definitions, confusion-artifact format, and comparison-row schema aligned with Step 4 wherever possible

C2. Commit to one OOS handling strategy
- Before implementing training logic, explicitly confirm the OOS strategy chosen earlier in the project
- Document whether OOS is handled as:
  - an explicit class in the multiclass output space
  - or a confidence-threshold decision rule applied after training
- For Step 5, prefer the explicit-class approach unless the project has already formally committed to threshold-based OOS detection
- If threshold-based OOS is the chosen strategy, document exactly when and how the threshold is tuned, and ensure it uses validation data only
- Save the chosen OOS strategy in run summaries so later models can match it exactly

D. Support disciplined hyperparameter configuration
- Centralize CNN hyperparameters in src/config.py
- At minimum make these configurable:
  - vocab_size
  - embedding_dim
  - num_filters
  - kernel_sizes
  - dropout_rate
  - learning_rate
  - weight_decay
  - batch_size
  - max_epochs
  - early_stopping_patience
  - optimizer choice if multiple are supported
  - activation function if configurable
  - whether embeddings are trainable
  - optional use of pretrained embeddings if supported later
  - random seed and DataLoader generator seed policy
- Keep defaults simple and defensible
- Avoid scattering CNN-specific magic numbers across files

D2. Seed policy for tuning runs
- Each tuning run must log the training seed
- If a DataLoader generator seed is used, log it as well
- Use one fixed seed policy or one small preset seed list consistently across Step 5
- Do not leave seed choice implicit because it affects fair tuning comparisons

E. Hyperparameter tuning expectations
- Be diligent, but scoped
- Use a small validation-driven tuning workflow rather than random guessing
- Tune only on train/validation splits
- Do not touch test data until the final best configuration is selected
- Prefer a compact experiment grid or staged tuning approach rather than an enormous exhaustive search
- Save every tried configuration and its validation result
- Define and follow a rough cap on the number of Step 5 tuning runs so the CNN does not quietly become a mini research project
- Prefer a staged tuning order rather than a flat search

Recommended bias:
Use a restrained search over a few plausible settings, for example:
- embedding_dim: a few options
- num_filters: a few options
- kernel_sizes: a few combinations
- dropout_rate: a few options
- learning_rate: a few options
- weight_decay: off vs a few small values
- Recommended staged order:
  - first verify one default run
  - then tune learning rate and dropout
  - then tune embedding dimension and number of filters
  - then optionally try kernel-size combinations
- Failed or unfinished runs must still be logged with config and failure reason

Do not explode the search space. This is a course project, not a benchmark paper.

F. Explicit regularization requirements
- Include dropout after pooled/concatenated features and/or where appropriate in the classifier head
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
- Make sure early stopping is as reproducible as the hardware stack allows given the same seed/config/data
- Do not imply bit-exact determinism on hardware stacks such as MPS that may not guarantee it

H. Checkpointing requirements
- Save the best Text CNN checkpoint to outputs/checkpoints/
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
- Even though multiple validation metrics are logged, validation macro F1 should be treated as the primary validation metric for early stopping and CNN model selection
- Make the tracked primary validation metric explicit in logs and run summaries
- Make sure train/eval mode switches are handled correctly
- Make sure gradient steps, optimizer zeroing, and device placement are handled safely
- Confirm one full training run completes without shape/type/device errors

I2. One-batch smoke test before full training
- Before the first full training run, perform a one-batch forward/backward smoke test
- Confirm all of the following on one real batch:
  - input shape is correct
  - embedding output shape is correct
  - pre-convolution tensor shape is correct
  - logits shape is correct
  - loss computes successfully
  - backward pass succeeds
  - optimizer step succeeds
- Fail loudly before expensive tuning begins if this smoke test does not pass

J. Neural dataset / DataLoader expectations
- Use Step 3 token-sequence outputs consistently
- Reuse Step 3 Dataset and DataLoader builders if they already exist
- Ensure:
  - train loader has non-zero batches
  - validation loader has non-zero batches
  - test loader exists and is not used until final evaluation
- Batch shapes should be logged at least once for sanity checking
- Confirm:
  - input tensors are integer token-id tensors
  - targets are integer class ids
  - batch shape is consistent with (batch_size, seq_len)
- Confirm PAD handling is compatible with the model and embedding layer assumptions
- Log DataLoader reproducibility settings, including:
  - shuffle on or off for each split
  - num_workers
  - pin_memory if used
  - generator seed if used

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
- It is acceptable for the default CNN to use a fixed learning rate if that choice is justified clearly
- A reasonable default justification is that the search space is intentionally compact and early stopping already controls training length
- If learning-rate scheduling is tried, treat it as an optional tuning dimension and record it in tuning artifacts
- Do not leave the absence or presence of scheduling implicit
- Default recommendation for Step 5: do not use a scheduler unless a clear need appears

L. Embedding policy
- Explicitly document how embeddings are initialized
- Pass padding_idx explicitly to nn.Embedding using the saved PAD id from Step 3
- Verify padding behavior so PAD embeddings do not drift unexpectedly
- If UNK initialization needs to be defined explicitly, document that policy as well, especially if pretrained embeddings are ever used
- Start with randomly initialized trainable embeddings for the primary Step 5 result unless pretrained embeddings are already part of the approved design
- If pretrained embeddings are used, document:
  - source
  - dimensionality
  - whether embeddings are frozen or fine-tuned
- Save embedding policy in run summaries
- Keep the first version simple unless there is a strong reason to add pretrained embeddings now
- If pretrained embeddings are tried at all, treat them as a separate optional experiment rather than the primary Step 5 result

M. Metric consistency requirements
- Compute validation metrics using the same conventions intended for the baseline and later BiLSTM
- At minimum track:
  - accuracy
  - macro F1
  - precision
  - recall
- If available, also compute OOS precision, recall, and F1 consistently
- Make sure label handling and averaging settings match earlier experiments
- Document clearly which metric is used for early stopping / model selection
- For this model, validation macro F1 should be the default and preferred early-stopping / model-selection metric

M2. Sequence-coverage diagnostics
- Save the chosen max sequence length in the run summary
- Save the percentage of train, validation, and test examples that were truncated under the inherited Step 3 max sequence length
- Save UNK token coverage statistics for validation and test, and for train as well if available from Step 3 artifacts
- If CNN performance is weak, these diagnostics should help distinguish preprocessing limits from model limits

N. Small, disciplined experiment tracking
- For every Text CNN run, save:
  - hyperparameters
  - training curves or epoch history
  - validation metrics
  - best checkpoint path
  - training duration
  - parameter count
- Save a summary table comparing tried CNN configurations
- Example artifact:
  - outputs/reports/text_cnn_tuning_results.csv
- Keep run names organized so the best CNN is easy to identify later
- Save results in a schema that can later merge cleanly with MLP and BiLSTM experiment outputs
- Save raw epoch history in a machine-readable artifact, not just rendered plots
- Example epoch-history fields:
  - epoch
  - train_loss
  - val_loss
  - val_accuracy
  - val_macro_f1
- Failed or aborted runs must still save partial metadata including config, epochs completed if any, and failure reason

O. Parameter count and model size tracking
- Compute parameter count for the Text CNN
- Log parameter-count convention explicitly; if frozen embeddings are ever possible, prefer saving both total parameters and trainable parameters
- Save it in the run summary
- Later comparisons will benefit from knowing:
  - parameter count
  - training time
  - final metrics
- Keep this consistent with what other models will report
- Track inference speed for the CNN using one clearly documented procedure that can be reused for all models, for example average milliseconds per example or examples per second
- Save inference-speed measurements in the run summary and final test metrics artifact

O2. Device and environment logging
- Save the device used for the run, for example CPU, MPS, or CUDA
- Save Python version and PyTorch version in run summaries
- If Apple Silicon MPS is intended for use, document that reproducibility and speed may differ from CPU
- If practical, confirm at least one successful CPU path and one successful MPS path before relying on MPS for the rest of the project

P. Console output expectations
When CNN training runs, it should print a concise but informative summary including:
- run name
- preprocessing artifact references
- vocabulary size
- max sequence length
- number of classes
- model architecture summary
- embedding dimension
- kernel sizes
- number of filters
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

Q. Artifact expectations
At the end of Step 5, I expect artifacts similar to these:
- outputs/checkpoints/text_cnn_best_*.pt
- outputs/logs/text_cnn_training_log_*.json or .csv
- outputs/reports/text_cnn_run_summary_*.json
- outputs/reports/text_cnn_tuning_results.csv
- outputs/reports/text_cnn_test_metrics.json
- outputs/reports/text_cnn_comparison_row.json
- outputs/reports/text_cnn_confusion_matrix.csv or .json
- outputs/reports/text_cnn_top_confusions.json
- outputs/reports/text_cnn_top_errors.json
- outputs/reports/text_cnn_per_class_metrics.csv or .json
- outputs/reports/text_cnn_epoch_history.csv or .json
- optionally:
  - outputs/reports/text_cnn_confidences.npz or .jsonl
  - outputs/figures/text_cnn_loss_curve.png
  - outputs/figures/text_cnn_val_metric_curve.png

The run summary should contain at minimum:
1. run name
2. config snapshot
3. preprocessing artifact references
4. vocab_size
5. max_sequence_length
6. number_of_classes
7. model architecture details
8. embedding policy
9. kernel sizes
10. num filters
11. parameter count
12. optimizer settings
13. dropout rate
14. weight decay
15. max epochs
16. early stopping patience
17. best epoch
18. best validation metric
19. primary model-selection metric used
20. training duration
21. checkpoint path
22. OOS strategy used
23. class-weight policy
24. inference latency summary
25. whether learning-rate scheduling was used
26. shuffle policy for train / validation / test loaders
27. preprocessing artifact refs for final comparison-table export
28. label-name ordering reference
29. preprocessing manifest reference
30. preprocessing config hash or version
31. inherited text preprocessing policy summary
32. sequence truncation percentages by split
33. UNK coverage statistics by split
34. embedding policy field in machine-readable form
35. learning-rate scheduling policy field in machine-readable form
36. device and environment summary

R. Final test evaluation requirements
- After selecting the best CNN configuration using validation only, load the best checkpoint
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
- Save per-class precision, recall, and F1 in a machine-readable artifact
- If storage is manageable, save confidence outputs, probabilities, or logits for the best final run to support later OOS and calibration analysis
- All final test metrics, confusion artifacts, per-class artifacts, and error artifacts must come from the best saved checkpoint, not from the last in-memory epoch state

R2. Final model selection and retraining rule
- Do not leave the final-model rule ambiguous
- Choose and document one explicit policy:
  - either use the single best tuning run directly as the final CNN
  - or retrain exactly once using the selected configuration before final test evaluation
- Save the chosen policy in the run summary
- If retraining is used, document the seed policy clearly
- Final test evaluation must always be tied to this explicit final-model rule
- Default recommendation: use the single best validation-selected run directly unless there is a strong reason to retrain
- If two configs are effectively tied on validation macro F1, break ties explicitly using a documented rule such as lower validation loss, smaller model, or faster model

S. Keep the Text CNN as a serious comparison point
- This model must be treated as a meaningful model, not a throwaway midpoint
- Later comparisons should consider:
  - its final test metrics
  - its parameter count
  - its training time
  - its inference latency
  - its validation behavior
- This means Step 5 must be implemented carefully enough that saying it beat or lost to the MLP baseline or BiLSTM actually means something

T. Validation checks / assertions
Include explicit sanity checks so this step fails loudly if something is wrong:
- vocabulary size matches embedding expectations
- PAD id is valid
- number of output classes matches label mapping size
- train / validation / test loaders are non-empty
- label ids are within valid class range
- no NaN / inf values appear in model outputs or losses
- no NaN loss values occur during training
- checkpoint directory exists or is created automatically
- best checkpoint is actually saved
- early stopping metric improves / tracks correctly
- loaded best checkpoint can be reloaded for evaluation
- test evaluation does not run before model selection is finished
- kernel sizes are valid relative to sequence length assumptions
- reject configs where any kernel size exceeds the effective sequence length
- verify max(kernel_sizes) does not exceed the inherited sequence length used by the model
- batch input shape is compatible with convolution input handling
- preprocessing manifest and config hash match the expected Step 3 artifact versions

U. Fresh-process reuse requirement
- Confirm the saved best checkpoint can be loaded in a fresh process
- Confirm the CNN can run evaluation from:
  - saved checkpoint
  - saved preprocessing artifacts
  - saved config summary
- Do not require re-training to reproduce final test evaluation

U2. Trainer generalization check
- If src/trainers/trainer.py is used, verify that it remains model-agnostic rather than tightly coupled to the MLP baseline
- Confirm it can accept arbitrary PyTorch models and compatible DataLoaders without assuming TF-IDF-specific shapes or baseline-only device logic
- If the trainer ends Step 5 in an overly CNN-specific state, refactor it before moving to Step 6
- Run a concrete minimal interface check showing the trainer can be called cleanly with a dummy nn.Module and compatible DataLoaders without assuming CNN-specific inputs

U3. Fresh-process reload smoke test before large tuning
- Do not wait until the end of Step 5 to test checkpoint reload
- Before large tuning begins, confirm at least one checkpoint can be saved and reloaded in a fresh process along with the required preprocessing references
- Fail early if serialization, config, or artifact-linking is broken

V. Determinism / reproducibility requirement
- Given the same data, seed, and config, training behavior should be as stable as reasonably possible
- Centralize seeding in src/utils.py or equivalent
- Set seed before model init, DataLoader creation with shuffle, and training loop execution
- Save the seed used for each run
- Save DataLoader generator seeds as well if they are used
- Do not promise bit-perfect determinism if the stack cannot guarantee it, but aim for strong reproducibility
- Save enough metadata that the run can be recreated faithfully later

W. Suggested CNN tuning workflow
Use a staged, disciplined process such as:
1. run one default CNN to confirm pipeline correctness
2. tune embedding_dim / num_filters / kernel_sizes / dropout / learning_rate / weight_decay on validation
3. compare a small set of candidate runs
4. select the best config using validation macro F1 by default, unless a different metric is explicitly justified and documented
5. retrain if necessary under the chosen config
6. evaluate once on test set
7. save final CNN metrics and summary

X. Recommended initial tuning dimensions
Start with a compact search, for example:
- embedding_dim: a few sizes
- num_filters: a few sizes
- kernel_sizes: a few standard combinations
- dropout_rate: a few reasonable values
- learning_rate: a few values
- weight_decay: off vs a few small values
- maybe trainable vs frozen embeddings only if pretrained embeddings are already in scope

Do not let the search become huge. Keep the number of runs realistic for the project timeline.

Y. Final comparison-table export requirements
- Export a clean CNN result record for later cross-model comparison
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
- Keep this schema stable so baseline and BiLSTM results can be merged without rework

Z. Visualization requirements
Goal:
Produce a small set of practical, report-ready visualizations that help diagnose training behavior, summarize tuning results, and support later comparison with MLP and BiLSTM models.

Implementation requirements:
- Save a training loss curve across epochs
- Save a validation loss curve across epochs
- Save a validation macro F1 curve across epochs
- If useful, also save validation accuracy across epochs, but macro F1 should remain the main plot for model selection
- Mark the best validation epoch on the validation macro F1 plot
- If easy, also mark the early stopping point
- Create at least one compact tuning-summary visualization
- Save a confusion matrix for final test predictions
- Use the exact saved label ordering from artifacts
- If a full confusion matrix is too unreadable, still save it, but also produce a more digestible summary
- Add a top-confused-pairs visualization or summary artifact
- Save a bar chart of the worst-performing classes by F1 if practical
- Treat the class-level worst-classes-by-F1 visualization as required if per-class metrics are already saved
- If OOS is treated as an explicit class and OOS metrics are computed, include a simple OOS-focused visualization
- Structure the code so these plots can later plug into cross-model comparison plots
- Log exactly which plot files were saved
- Save visualization file references in the run summary if practical

Recommended outputs:
- outputs/figures/text_cnn_train_val_loss_curve.png
- outputs/figures/text_cnn_val_macro_f1_curve.png
- outputs/figures/text_cnn_tuning_summary.png
- outputs/figures/text_cnn_confusion_matrix.png
- outputs/figures/text_cnn_top_confused_pairs.png
- outputs/figures/text_cnn_bottom_classes_f1.png
- optionally:
  - outputs/figures/text_cnn_val_accuracy_curve.png
  - outputs/figures/text_cnn_oos_metrics.png
  - outputs/figures/text_cnn_efficiency_summary.png

Validation checks / assertions:
- Ensure plotting does not silently fail if directories do not exist
- Ensure the best epoch shown in plots matches the tracked best checkpoint epoch
- Ensure confusion matrix label ordering matches the saved label-name ordering artifact
- Ensure plots can still be generated in non-interactive environments
- Ensure training-curve plots are based on the saved epoch history rather than recomputed ad hoc values

Definition of done for Step 5 visualizations:
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

AA. Definition of done
Step 5 is only complete if all of the following are true:
- Text CNN model is implemented cleanly in PyTorch
- model input contract matches Step 3 neural preprocessing outputs
- output dimension matches Step 2 label mapping
- CNN training runs end-to-end without errors
- dropout and optional weight decay are implemented and configurable
- early stopping is implemented and used
- best checkpoint is saved based on validation macro F1
- a small, disciplined hyperparameter tuning process was run
- tried configurations and validation results are saved
- validation macro F1 was used as the default early-stopping / model-selection metric while other metrics were still computed and logged
- final best CNN is selected using validation only
- best checkpoint can be loaded in a fresh process
- final test metrics are computed and saved
- parameter count, training time, and inference latency are saved
- artifacts are organized and report-ready
- the chosen OOS strategy is implemented consistently and documented in artifacts
- threshold-based OOS tuning was not performed unless explicitly approved and validation-only
- class-weight policy is documented
- learning-rate scheduling policy is documented
- confusion and top-error artifacts are saved
- a comparison-table export artifact is saved
- visualization artifacts are saved
- preprocessing drift did not occur during tuning
- preprocessing manifest compatibility was verified before training
- one-batch smoke test passed before full training
- fresh-process checkpoint reload was tested before large tuning
- per-class metrics are saved
- trainer remains reusable and not locked to CNN-specific assumptions

AA2. Final Step 5 exit gate
- Before Step 5 is considered closed, confirm all of the following are true:
  - best checkpoint reload works in a fresh process
  - final test metrics are saved
  - comparison row is exported
  - confusion and top-error artifacts are saved
  - per-class metrics are saved
  - core visualizations are saved
  - trainer remains reusable
  - no preprocessing drift occurred

AB. Deliverable quality bar
The Text CNN should be simple, strong, and reproducible. It should not be a messy prototype. It should reflect real care in model selection, regularization, validation monitoring, and artifact generation so that later claims like “Text CNN beat the baseline” or “BiLSTM improved on Text CNN” actually mean something.

AC. Suggested self-check after implementation
After coding, self-check against this exact checklist and confirm:
1. which files/functions implement the Text CNN
2. how Step 3 neural preprocessing artifacts are reused without re-fitting
3. which hyperparameters were tuned
4. that validation macro F1 selected the best model by default, and that any override was explicit and documented
5. how early stopping works
6. where checkpoints and tuning summaries are saved
7. what the final CNN test metrics are
8. that the best checkpoint can be loaded and evaluated in a fresh process
9. which OOS strategy was used and whether it matches the earlier project decision
10. whether class weights were used and how that decision was made
11. whether learning-rate scheduling was used or explicitly not used
12. what the CNN inference latency is
13. where confusion-matrix and top-error artifacts are saved
14. where visualization artifacts are saved
15. that the trainer remains reusable and not locked to one model type
16. that the Conv1d input tensor shape is explicitly correct before convolution
17. that embedding padding_idx behavior is correct
18. that kernel sizes are feasible for the inherited sequence length
19. that truncation and UNK coverage diagnostics were saved
20. that failed or aborted runs were still logged honestly
21. that fresh-process checkpoint reload was tested before large tuning
22. that the final-model selection rule and tie-break policy were explicit
23. that per-class metrics and class-level error plots were saved
24. that device and environment metadata were saved