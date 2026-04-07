Implement and verify Step 6: BiLSTM model for the CLINC150 project.

Goal:
Build a clean, reproducible, well-regularized BiLSTM classifier for CLINC150 using the shared preprocessing pipeline from Step 3. This model should be trained carefully with validation-driven model selection, early stopping, dropout, sensible hyperparameter tuning, gradient clipping, and strong artifact generation so that comparison against the TF-IDF + MLP baseline and the Text CNN is fair, meaningful, and easy to report. Although multiple metrics should be computed and logged, validation macro F1 should be the primary model-selection and early-stopping metric.

Primary outcome:
At the end of this step, I want to have:
1. a working BiLSTM classifier implemented in PyTorch
2. a clean training pipeline that consumes Step 3 neural preprocessing artifacts
3. validation-driven model selection with early stopping
4. regularization handled explicitly, including dropout, optional weight decay, and gradient clipping
5. a small but disciplined hyperparameter tuning workflow
6. best checkpoint saved based on validation macro F1
7. final test-set metrics stored in report-ready artifacts
8. comparison-ready run artifacts aligned with the baseline and Text CNN outputs
9. useful post-training error analysis artifacts for the final report
10. useful visualization artifacts for debugging, tuning, and reporting
11. a clean Step 7 handoff package with final predictions, per-class metrics, confusion artifacts, and comparison-row export

Important scope constraints:
- Do not change preprocessing or label mappings during this step
- Do not use test data for model selection, early stopping, threshold tuning, or hyperparameter tuning
- Do not over-engineer the BiLSTM into a giant architecture search project
- Keep tuning disciplined and limited enough to fit the project timeline
- Use Step 3 saved artifacts rather than rebuilding preprocessing from scratch
- Keep all metrics, label ordering, and evaluation logic consistent with Step 4 and Step 5
- Explicitly follow the OOS strategy chosen earlier in the project; do not improvise a new OOS policy inside Step 6
- Do not perform OOS threshold tuning in Step 6 unless threshold-based OOS was already explicitly approved as the chosen evaluation design and is tuned on validation only
- Never use test data for OOS threshold selection
- This step should focus on a strong, standard BiLSTM rather than highly experimental variants
- Step 6 may not begin until the required Step 3 artifacts are already saved and verified: vocabulary, PAD id, UNK id, max sequence length, label_to_id, id_to_label, tokenizer/normalization settings, and train/validation/test split artifacts
- If vocabulary, max sequence length, tokenization, normalization, or label mapping changes, all prior Step 6 runs become invalid for fair comparison and should not be mixed with the new runs
- Do not peek at test results repeatedly; after the final config is selected, any further debugging or tuning must return to validation only

Out of scope for Step 6:
- attention mechanisms as part of the primary result
- CRF or structured decoding layers
- pretrained embeddings as the primary reported result
- calibration study as a required deliverable
- extensive scheduler search
- huge architecture sweeps
- transformer/tokenizer-specific work that belongs to Step 7 or later

Cross-cutting invariants:
- frozen Step 3 artifacts must be reused exactly
- the same shared metric functions must be used as in earlier steps
- label ordering must remain frozen across metrics, confusion artifacts, and comparison exports
- OOS handling must match the earlier project decision exactly
- all final evaluation artifacts must come from the best saved checkpoint, not the last in-memory epoch state

High-level design requirement:
This step should consume the saved Step 3 neural preprocessing outputs and train a BiLSTM classifier using token-id sequences and sequence DataLoaders. It should include:
- embedding layer
- bidirectional LSTM encoder
- a clear sequence summarization strategy
- dropout
- configurable optimizer / learning rate / weight decay
- gradient clipping
- validation monitoring
- early stopping
- checkpoint saving
- structured experiment logging
- report-ready visualizations

Recommended default implementation path:
- batch_first=True
- fixed-length padded batches
- no packed sequences in the primary implementation
- padding_idx=PAD_ID in nn.Embedding
- randomly initialized trainable embeddings
- one-layer or simple two-layer BiLSTM
- final hidden-state summarization using concatenated last-layer forward/backward hidden states
- classifier-path dropout
- Adam optimizer
- no LR scheduler by default
- gradient clipping enabled by default
- validation macro F1 as the primary model-selection metric

Implementation requirements:

A. Create a clean BiLSTM training entry point
- Add BiLSTM training entry logic in src/train.py and/or main.py
- Keep the BiLSTM workflow easy to run independently
- Prefer explicit functions such as:
  - build_bilstm(...)
  - train_bilstm(...)
  - evaluate_bilstm(...)
  - run_bilstm_experiment(...)
- Make the training path reusable for later comparisons and report generation
- Ensure this step reads preprocessing artifacts from Step 3 rather than re-fitting vocabulary or recreating sequence tensors

A2. Preprocessing manifest and frozen-artifact gate
- Require one saved preprocessing manifest from Step 3 that Step 6 loads before training begins
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
- Step 6 must verify that loaded neural artifacts match this manifest before training starts
- Save the loaded manifest reference in run summaries
- Log the inherited text preprocessing policy clearly, including:
  - lowercase or not
  - punctuation kept or removed
  - numbers kept or normalized
  - whitespace normalization

B. Implement the BiLSTM model cleanly
- Add the model to src/models/bilstm.py
- The model should be a straightforward sequence classifier over token sequences
- Use a standard architecture pattern such as:
  - embedding layer
  - bidirectional LSTM
  - sequence summarization
  - dropout
  - final classifier layer
- Keep the first implementation lightweight and justifiable
- Avoid turning this into a huge experimental architecture
- Make sure output dimension matches the number of classes from Step 2 label mapping
- Use a clean forward() definition returning raw logits for multiclass classification
- Do not apply softmax before CrossEntropyLoss
- Explicitly set and document `batch_first=True` or `False`
- Recommended default: `batch_first=True`
- Compute the classifier input dimension programmatically based on hidden size and bidirectionality rather than hardcoding it
- Add an assertion in model initialization that the classifier input dimension matches the summarization output size

Recommended bias:
- Start with a standard BiLSTM sentence classifier:
  - embeddings -> BiLSTM -> summarize sequence -> dropout -> linear classifier
- Keep the architecture tunable but restrained
- Prefer a simple, defensible summarization strategy over multiple complicated heads
- This should be a standard BiLSTM baseline, not an attention hybrid or highly customized variant

C. Define the neural input contract explicitly
- Load the saved vocabulary, max sequence length, label mappings, and sequence-ready split data from Step 3 artifacts only
- Confirm the token-id inputs match the vocabulary and padding conventions from Step 3
- Reuse Step 2 label mapping and Step 3 neural preprocessing artifacts exactly
- Save the reference paths to preprocessing artifacts in training summary outputs
- Do not recompute label mappings or silently alter vocabulary ordering
- Confirm that embedding input ids align with saved PAD and UNK ids
- Confirm that model output dimension and label mapping exactly match the chosen OOS strategy
- Verify that preprocessing artifact versions or config hashes match what the run expects, so stale Step 3 artifacts are not silently mixed into Step 6
- Confirm fairness with earlier models by keeping label ordering, metric definitions, confusion-artifact format, and comparison-row schema aligned with Step 4 and Step 5 wherever possible
- Explicitly state what each batch contains:
  - input_ids only
  - or input_ids + lengths
  - or input_ids + mask
- Recommended default: input_ids only for fixed-length padded batches
- If lengths or masks are used, document them as part of the model/data contract

C2. Commit to one OOS handling strategy
- Before implementing training logic, explicitly confirm the OOS strategy chosen earlier in the project
- Document whether OOS is handled as:
  - an explicit class in the multiclass output space
  - or a confidence-threshold decision rule applied after training
- For Step 6, prefer the explicit-class approach unless the project has already formally committed to threshold-based OOS detection
- If threshold-based OOS is the chosen strategy, document exactly when and how the threshold is tuned, and ensure it uses validation data only
- Save the chosen OOS strategy in run summaries so earlier models and later analysis can match it exactly
- Explicitly verify that the OOS class index used in Step 6 metrics matches the OOS class index used in Steps 4 and 5

D. Support disciplined hyperparameter configuration
- Centralize BiLSTM hyperparameters in src/config.py
- At minimum make these configurable:
  - vocab_size
  - embedding_dim
  - hidden_dim
  - num_layers
  - bidirectional flag
  - recurrent_dropout if supported by the chosen implementation path
  - dropout_rate
  - learning_rate
  - weight_decay
  - batch_size
  - max_epochs
  - early_stopping_patience
  - optimizer choice if multiple are supported
  - activation function in classifier head if configurable
  - sequence summarization mode
  - whether embeddings are trainable
  - optional use of pretrained embeddings if supported later
  - random seed and DataLoader generator seed policy
  - gradient clipping enabled/disabled
  - max_grad_norm
- Keep defaults simple and defensible
- Avoid scattering BiLSTM-specific magic numbers across files

D2. Seed policy for tuning runs
- Each tuning run must log the training seed
- If a DataLoader generator seed is used, log it as well
- Use one fixed seed policy or one small preset seed list consistently across Step 6
- Do not leave seed choice implicit because it affects fair tuning comparisons
- Do not casually merge results produced under different seed policies into one tuning table without marking that difference

E. Sequence summarization policy
- Explicitly define how the BiLSTM output is converted into a fixed-size classifier input
- Choose one primary strategy for the main Step 6 result
- Recommended default:
  - concatenated final forward/backward hidden states from the last BiLSTM layer
- Alternate strategies such as masked mean pooling or masked max pooling may be compared only as controlled tuning dimensions if needed
- Save the chosen summarization policy in run summaries
- Keep the first implementation simple and standard
- If alternate summarization strategies are tried, treat them as controlled tuning dimensions rather than silent changes

Important implementation note:
- If using `output, (h_n, c_n) = lstm(x)`, prefer extracting final hidden states from `h_n` rather than manually reading the last timestep from `output`
- For a bidirectional LSTM, `h_n` has shape `(num_layers * num_directions, batch, hidden_size)`
- If using the last layer hidden states, extract:
  - forward state: `h_n[-2, :, :]`
  - backward state: `h_n[-1, :, :]`
  for the last BiLSTM layer when bidirectional=True
- Concatenate those two states to form the summarized representation
- Add an explicit shape assertion confirming the summarized feature shape matches the expected classifier input dimension
- If you ever extract states from `output` manually on padded sequences, document exactly how backward-state extraction is handled because the backward final state is not at the same time index as the forward final state

F. Hyperparameter tuning expectations
- Be diligent, but scoped
- Use a small validation-driven tuning workflow rather than random guessing
- Tune only on train/validation splits
- Do not touch test data until the final best configuration is selected
- Prefer a compact experiment grid or staged tuning approach rather than an enormous exhaustive search
- Save every tried configuration and its validation result
- Use a hard cap on total Step 6 tuning runs
- Recommended cap: at most 15 total tuning runs unless explicitly justified
- Prefer a staged tuning order rather than a flat search

Recommended bias:
Use a restrained search over a few plausible settings, for example:
- embedding_dim: a few options
- hidden_dim: a few options
- num_layers: one or two sensible options
- dropout_rate: a few options
- learning_rate: a few options
- weight_decay: off vs a few small values
- maybe one alternate summarization strategy if needed

Recommended staged order:
- first verify one default run
- then tune learning rate and dropout
- then tune hidden dimension and maybe number of layers
- then optionally compare summarization strategies
- failed or unfinished runs must still be logged with config and failure reason

Do not explode the search space. This is a course project, not a benchmark paper.

Recommended default config block:
- embedding_dim: a conservative medium value
- hidden_dim: a conservative medium value
- num_layers: 1 by default
- bidirectional: True
- summarization: concat final forward/backward hidden states
- dropout_rate: moderate classifier dropout
- optimizer: Adam
- learning_rate: modest default
- weight_decay: small or off
- gradient clipping: on
- max_grad_norm: modest default
- no scheduler by default

G. Explicit regularization requirements
- Include dropout in the classifier path and anywhere else appropriate for the chosen BiLSTM implementation
- Support weight decay through the optimizer
- Support gradient clipping through the training loop
- Make regularization choices visible in logs and saved summaries
- Be explicit about why dropout, weight decay, and gradient clipping are included
- Keep these settings tunable and validation-driven
- Do not silently apply regularization without documenting it
- Distinguish between:
  - classifier/output dropout
  - LSTM internal dropout between recurrent layers
- Explicitly note that PyTorch `nn.LSTM(dropout=...)` only applies dropout between layers when `num_layers > 1`
- If `num_layers == 1`, log clearly that LSTM internal dropout is a no-op and confirm classifier-path dropout is still active
- Gradient clipping should be treated as a real safeguard, not a stretch goal
- Recommended default: enable max-norm gradient clipping unless experiments clearly show it is unnecessary

H. Early stopping requirements
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
- If the monitored metric ties exactly with the previous best, use a deterministic tie-break rule:
  - first prefer lower validation loss
  - if still tied, prefer the earlier epoch
- Make sure early stopping is as reproducible as the hardware stack allows given the same seed/config/data
- Do not imply bit-exact determinism on hardware stacks such as MPS that may not guarantee it

I. Checkpointing requirements
- Save the best BiLSTM checkpoint to outputs/checkpoints/
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

J. Training loop requirements
- Use reusable trainer logic if available in src/trainers/trainer.py
- Training loop should log at minimum per epoch:
  - training loss
  - validation loss
  - validation accuracy
  - validation macro F1
  - learning rate
  - whether checkpoint was updated
- If OOS metrics are already wired consistently, also log validation OOS metrics
- Even though multiple validation metrics are logged, validation macro F1 should be treated as the primary validation metric for early stopping and BiLSTM model selection
- Make the tracked primary validation metric explicit in logs and run summaries
- Make sure train/eval mode switches are handled correctly
- Make sure gradient steps, optimizer zeroing, and device placement are handled safely
- If gradient clipping is enabled:
  - log clip policy and threshold
  - optionally log gradient norm or clipping events during default/debug runs
- Confirm one full training run completes without shape/type/device errors

J2. One-batch smoke test before full training
- Before the first full training run, perform a one-batch forward/backward smoke test
- Confirm all of the following on one real batch:
  - input shape is correct
  - embedding output shape is correct
  - BiLSTM output shape is correct
  - h_n shape is correct
  - summarization output shape is correct
  - classifier input shape is correct
  - logits shape is correct
  - loss computes successfully
  - backward pass succeeds
  - optimizer step succeeds
- Fail loudly before expensive tuning begins if this smoke test does not pass
- Save one known-good successful default run as the reference debug run before tuning starts

K. Neural dataset / DataLoader expectations
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
  - batch shape is consistent with the documented `batch_first` convention
- Confirm PAD handling is compatible with the model and embedding layer assumptions
- If sequence lengths or masks are needed for summarization, ensure they are produced consistently and correctly
- Log DataLoader reproducibility settings, including:
  - shuffle on or off for each split
  - drop_last policy
  - num_workers
  - pin_memory if used
  - generator seed if used
- If a custom collate_fn is used for lengths or masks, test it explicitly

L. Variable-length and padding policy
- Explicitly document how padding is handled in the BiLSTM pipeline
- Pass `padding_idx` explicitly to nn.Embedding using the saved PAD id from Step 3
- Verify padding behavior so PAD embeddings do not drift unexpectedly
- Decide whether the model uses:
  - fixed-length padded batches only
  - or packed sequences / masks in a controlled way
- Recommended default for Step 6:
  - fixed-length padded batches
  - no packed sequences in the primary implementation
  - if pooling is used, apply masking correctly
- If using packed sequences, ensure sorting, unsorting, or enforce_sorted behavior is handled correctly
- If using masked pooling, ensure PAD positions do not affect pooled summaries
- Save the padding / sequence-handling policy in run summaries
- If using packed sequences on MPS, explicitly test that forward/backward passes work correctly and do not trigger broken behavior or bad fallbacks on the target hardware
- If MPS shows instability or unsupported behavior, fall back to CPU for the authoritative run and log that choice

Recommended bias:
- Keep the first version simple and reliable
- Fixed-length padded batches with correct masking can be totally fine
- Do not introduce packing complexity unless it clearly improves cleanliness or correctness for your implementation

L2. Embedding policy
- Explicitly document how embeddings are initialized
- Start with randomly initialized trainable embeddings for the primary Step 6 result unless pretrained embeddings are already part of the approved design
- If pretrained embeddings are used, document:
  - source
  - dimensionality
  - whether embeddings are frozen or fine-tuned
- Save embedding policy in run summaries
- If UNK initialization needs to be defined explicitly, document that policy as well, especially if pretrained embeddings are ever used
- The inference path must use the exact same UNK fallback logic as training and evaluation
- If pretrained embeddings are tried at all, treat them as a separate optional experiment rather than the primary Step 6 result

M. Loss function and optimization
- Use a standard multiclass classification loss such as cross-entropy loss
- Ensure logits and targets are shaped correctly for the chosen loss
- Logits must be raw, unnormalized outputs
- Targets must be integer class IDs
- Do not apply softmax before CrossEntropyLoss
- Support at least one sensible optimizer, usually Adam
- Weight decay should be configurable
- Learning rate should be configurable
- Optimizer choice should be logged and saved in run artifacts

M2. Class imbalance policy
- Explicitly decide whether class-weighted cross-entropy is used or not
- If class weights are supported, they must be computed from the training split only
- Keep class weighting configurable but simple
- Log the final class-weight policy in run summaries even if the decision is to use no class weights
- Default to no class weights unless training-split imbalance clearly justifies them

M3. Learning-rate scheduling policy
- Explicitly document whether learning-rate scheduling is used
- It is acceptable for the default BiLSTM to use a fixed learning rate if that choice is justified clearly
- A reasonable default justification is that the search space is intentionally compact and early stopping already controls training length
- If learning-rate scheduling is tried, treat it as an optional tuning dimension and record it in tuning artifacts
- Do not leave the absence or presence of scheduling implicit
- Default recommendation for Step 6: do not use a scheduler unless a clear need appears

N. Metric consistency requirements
- Compute validation metrics using the same shared metric function used by earlier models
- Use the shared function from the common metrics module rather than ad hoc recomputation
- At minimum track:
  - accuracy: overall accuracy
  - macro F1: macro average across the defined class set
  - precision: explicitly define averaging convention
  - recall: explicitly define averaging convention
- Recommended default:
  - report macro precision and macro recall alongside macro F1
- Explicitly define zero-division policy and keep it identical across all models
- If available, also compute OOS precision, recall, and F1 consistently
- Make sure label handling and averaging settings match earlier experiments
- Document clearly which metric is used for early stopping / model selection
- For this model, validation macro F1 should be the default and preferred early-stopping / model-selection metric

Important metric definition block:
- Explicitly define whether validation macro F1 includes the OOS class or only the in-scope classes
- Recommended rule:
  - save one main macro F1 definition and keep it identical across Steps 4, 5, and 6
  - if useful, also save separate OOS F1 as a one-vs-rest metric
- Do not let Step 6 silently change the class set used in macro F1 relative to earlier steps

N2. Sequence-coverage diagnostics
- Save the chosen max sequence length in the run summary
- Save the percentage of train, validation, and test examples that were truncated under the inherited Step 3 max sequence length
- Save UNK token coverage statistics for validation and test, and for train as well if available from Step 3 artifacts
- If BiLSTM performance is weak, these diagnostics should help distinguish preprocessing limits from model limits

O. Small, disciplined experiment tracking
- For every BiLSTM run, save:
  - hyperparameters
  - training curves or epoch history
  - validation metrics
  - best checkpoint path
  - training duration
  - parameter count
- Save a summary table comparing tried BiLSTM configurations
- Example artifact:
  - outputs/reports/bilstm_tuning_results.csv
- Keep run names organized so the best BiLSTM is easy to identify later
- Save results in a schema that can later merge cleanly with MLP and Text CNN experiment outputs
- Save raw epoch history in a machine-readable artifact, not just rendered plots
- Example epoch-history fields:
  - epoch
  - train_loss
  - val_loss
  - val_accuracy
  - val_macro_f1
  - learning_rate
  - checkpoint_updated
- Failed or aborted runs must still save partial metadata including config, epochs completed if any, and failure reason
- Do not compare across changed seed policies without marking that change explicitly

P. Parameter count and model size tracking
- Compute parameter count for the BiLSTM
- Log parameter-count convention explicitly; if frozen embeddings are ever possible, prefer saving both total parameters and trainable parameters
- Save it in the run summary
- Later comparisons will benefit from knowing:
  - parameter count
  - training time
  - final metrics
- Keep this consistent with what other models will report
- Track inference speed for the BiLSTM using one clearly documented procedure that can be reused for all models
- Latency measurements must be done with:
  - model.eval()
  - torch.no_grad()
- Save inference-speed measurements in the run summary and final test metrics artifact

Latency protocol:
- Define one concrete protocol and reuse it for Steps 4, 5, and 6
- The protocol should specify:
  - whether timing is per example or per batch
  - batch size used for timing
  - whether warm-up runs are used
  - device used for timing
  - whether latency starts from raw string input or from pre-tokenized tensors
- Recommended rule for fair comparison:
  - save model-only latency from tensor input for strict architectural comparison
  - if feasible, also save end-to-end latency from raw string to logits as a secondary artifact
- Do not let the protocol vary across models

P2. Device and environment logging
- Save the device used for the run, for example CPU, MPS, or CUDA
- Save Python version and PyTorch version in run summaries
- Save whether deterministic algorithm enforcement was enabled or not, if applicable
- If Apple Silicon MPS is intended for use, document that reproducibility and speed may differ from CPU
- If practical, confirm at least one successful CPU path and one successful MPS path before relying on MPS for the rest of the project
- If MPS proves unstable, unsupported, or misleading for the authoritative run, fall back to CPU and log that choice explicitly

Q. Console output expectations
When BiLSTM training runs, it should print a concise but informative summary including:
- run name
- preprocessing artifact references
- preprocessing manifest reference
- vocabulary size
- max sequence length
- number of classes
- model architecture summary
- batch_first convention
- embedding dimension
- hidden dimension
- number of layers
- summarization policy
- gradient clipping policy and threshold
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

R. Artifact expectations
At the end of Step 6, I expect artifacts similar to these:
- outputs/checkpoints/bilstm_best_*.pt
- outputs/logs/bilstm_training_log_*.json or .csv
- outputs/reports/bilstm_run_summary_*.json
- outputs/reports/bilstm_tuning_results.csv
- outputs/reports/bilstm_test_metrics.json
- outputs/reports/bilstm_comparison_row.json
- outputs/reports/bilstm_confusion_matrix.csv or .json
- outputs/reports/bilstm_top_confusions.json
- outputs/reports/bilstm_top_errors.json
- outputs/reports/bilstm_per_class_metrics.csv or .json
- outputs/reports/bilstm_epoch_history.csv or .json
- outputs/reports/bilstm_final_predictions.csv or .jsonl
- optionally:
  - outputs/reports/bilstm_confidences.npz or .jsonl
  - outputs/figures/bilstm_loss_curve.png
  - outputs/figures/bilstm_val_metric_curve.png

The run summary should contain at minimum:
1. run name
2. config snapshot
3. preprocessing artifact references
4. vocab_size
5. max_sequence_length
6. number_of_classes
7. model architecture details
8. embedding policy
9. hidden dimension
10. num layers
11. bidirectional setting
12. summarization policy
13. parameter count
14. optimizer settings
15. dropout rate
16. weight decay
17. gradient clipping policy
18. max_grad_norm if used
19. max epochs
20. early stopping patience
21. best epoch
22. best validation metric
23. primary model-selection metric used
24. training duration
25. checkpoint path
26. OOS strategy used
27. class-weight policy
28. inference latency summary
29. whether learning-rate scheduling was used
30. shuffle policy for train / validation / test loaders
31. preprocessing artifact refs for final comparison-table export
32. label-name ordering reference
33. preprocessing manifest reference
34. preprocessing config hash or version
35. inherited text preprocessing policy summary
36. sequence truncation percentages by split
37. UNK coverage statistics by split
38. embedding policy field in machine-readable form
39. learning-rate scheduling policy field in machine-readable form
40. sequence-handling policy field in machine-readable form
41. hidden-state initialization policy
42. deterministic-algorithms policy
43. device and environment summary

S. Final test evaluation requirements
- After selecting the best BiLSTM configuration using validation only, load the best checkpoint
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
- Compute and log inference latency on the test split using the shared, documented measurement protocol
- Save a confusion matrix or equivalent per-class confusion artifact
- Confusion matrix outputs should use human-readable label names, not just raw integer IDs
- Save a top-confusions artifact, for example the most confused intent pairs
- Save a top-errors artifact containing representative misclassified texts for later report writing
- Save the exact label-name ordering alongside these metric and confusion artifacts so later analysis cannot drift
- Save per-class precision, recall, F1, and support in a machine-readable artifact
- If storage is manageable, save confidence outputs, probabilities, or logits for the best final run to support later OOS and calibration analysis
- Save final prediction artifacts containing at minimum:
  - y_true
  - y_pred
  - raw text if practical
  - predicted confidence or score if practical
- All final test metrics, confusion artifacts, per-class artifacts, and error artifacts must come from the best saved checkpoint, not from the last in-memory epoch state
- Every saved per-class artifact, confusion artifact, error artifact, and comparison export must embed or reference the exact label ordering artifact

S2. Final model selection and retraining rule
- Do not leave the final-model rule ambiguous
- Choose and document one explicit policy:
  - either use the single best tuning run directly as the final BiLSTM
  - or retrain exactly once using the selected configuration before final test evaluation
- Save the chosen policy in the run summary
- If retraining is used, document the seed policy clearly
- If retraining is used, also document whether it uses:
  - train split only
  - or combined train + validation
- Final test evaluation must always be tied to this explicit final-model rule
- Default recommendation: use the single best validation-selected run directly unless there is a strong reason to retrain
- If two configs are effectively tied on validation macro F1, break ties explicitly using a documented rule such as lower validation loss, smaller model, or faster model

T. Keep the BiLSTM as a serious comparison point
- This model must be treated as a meaningful model, not a throwaway final addition
- Later comparisons should consider:
  - its final test metrics
  - its parameter count
  - its training time
  - its inference latency
  - its validation behavior
  - its tuning budget
- This means Step 6 must be implemented carefully enough that saying it beat or lost to the MLP baseline or Text CNN actually means something

U. Validation checks / assertions
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
- batch input shape is compatible with embedding and recurrent input handling
- h_n shape is compatible with the selected hidden-state extraction policy
- summarized feature shape matches classifier input size
- if using packed sequences, lengths are valid and sorting/unsorting behavior is correct
- if using masking, PAD positions do not leak into pooled summaries
- preprocessing manifest and config hash match the expected Step 3 artifact versions

V. Fresh-process reuse requirement
- Confirm the saved best checkpoint can be loaded in a fresh process
- Confirm the BiLSTM can run evaluation from:
  - saved checkpoint
  - saved preprocessing artifacts
  - saved config summary
- Do not require re-training to reproduce final test evaluation

V2. Trainer generalization check
- If src/trainers/trainer.py is used, verify that it remains model-agnostic rather than tightly coupled to earlier models
- Confirm it can accept arbitrary PyTorch models and compatible DataLoaders without assuming TF-IDF-specific shapes, CNN-specific shapes, or baseline-only device logic
- If the trainer ends Step 6 in an overly BiLSTM-specific state, refactor it before moving to later project steps
- Run a concrete minimal interface check showing the trainer can be called cleanly with a dummy nn.Module and compatible DataLoaders without assuming BiLSTM-specific inputs

V3. Fresh-process reload smoke test before large tuning
- Do not wait until the end of Step 6 to test checkpoint reload
- Before large tuning begins, confirm at least one checkpoint can be saved and reloaded in a fresh process along with the required preprocessing references
- Fail early if serialization, config, or artifact-linking is broken

W. Determinism / reproducibility requirement
- Given the same data, seed, and config, training behavior should be as stable as reasonably possible
- Centralize seeding in src/utils.py or equivalent
- Set seed before model init, DataLoader creation with shuffle, and training loop execution
- Save the seed used for each run
- Save DataLoader generator seeds as well if they are used
- If deterministic algorithm enforcement is enabled, log that choice
- If deterministic algorithm enforcement is not enabled, log that choice too
- Do not promise bit-perfect determinism if the stack cannot guarantee it, but aim for strong reproducibility
- Save enough metadata that the run can be recreated faithfully later

X. Suggested BiLSTM tuning workflow
Use a staged, disciplined process such as:
1. run one default BiLSTM to confirm pipeline correctness
2. tune learning rate and dropout on validation
3. tune hidden dimension and maybe number of layers
4. optionally compare one alternate summarization strategy
5. select the best config using validation macro F1 by default, unless a different metric is explicitly justified and documented
6. retrain if necessary under the chosen config
7. evaluate once on test set
8. save final BiLSTM metrics and summary

Y. Recommended initial tuning dimensions
Start with a compact search, for example:
- embedding_dim: a few sizes
- hidden_dim: a few sizes
- num_layers: one or two sensible options
- dropout_rate: a few reasonable values
- learning_rate: a few values
- weight_decay: off vs a few small values
- maybe one alternate summarization strategy
- maybe trainable vs frozen embeddings only if pretrained embeddings are already in scope

Do not let the search become huge. Keep the number of runs realistic for the project timeline.

Z. Final comparison-table export requirements
- Export a clean BiLSTM result record for later cross-model comparison
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
- Keep this schema stable so baseline and Text CNN results can be merged without rework

AA. Visualization requirements
Goal:
Produce a small set of practical, report-ready visualizations that help diagnose training behavior, summarize tuning results, and support later comparison with MLP and Text CNN models.

Required visualizations:
- train/validation loss curve
- validation macro F1 curve
- confusion matrix
- one tuning summary visualization

Secondary visualizations if time allows:
- top confused pairs figure
- worst-class F1 bar chart
- OOS-focused figure
- efficiency summary figure

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
- outputs/figures/bilstm_train_val_loss_curve.png
- outputs/figures/bilstm_val_macro_f1_curve.png
- outputs/figures/bilstm_tuning_summary.png
- outputs/figures/bilstm_confusion_matrix.png
- outputs/figures/bilstm_top_confused_pairs.png
- outputs/figures/bilstm_bottom_classes_f1.png
- optionally:
  - outputs/figures/bilstm_val_accuracy_curve.png
  - outputs/figures/bilstm_oos_metrics.png
  - outputs/figures/bilstm_efficiency_summary.png

Validation checks / assertions:
- Ensure plotting does not silently fail if directories do not exist
- Ensure the best epoch shown in plots matches the tracked best checkpoint epoch
- Ensure confusion matrix label ordering matches the saved label-name ordering artifact
- Ensure plots can still be generated in non-interactive environments
- Ensure training-curve plots are based on the saved epoch history rather than recomputed ad hoc values

Definition of done for Step 6 visualizations:
- train/validation loss curves are saved
- validation macro F1 curve is saved
- best epoch is visually marked on the primary validation metric plot
- at least one tuning-summary visualization is saved
- confusion matrix visualization is saved
- at least one confusion-summary or top-confused-pairs visualization is saved if time allows
- at least one class-level error analysis visualization is saved or the data for it is saved cleanly
- visualization artifacts are saved in stable output paths
- label ordering is consistent across confusion / class-level plots
- the plots are good enough to drop into the final report or presentation with minimal cleanup

AB. Definition of done
Step 6 is only complete if all of the following are true:
- BiLSTM model is implemented cleanly in PyTorch
- model input contract matches Step 3 neural preprocessing outputs
- output dimension matches Step 2 label mapping
- BiLSTM training runs end-to-end without errors
- dropout and optional weight decay are implemented and configurable
- gradient clipping is implemented and configurable
- early stopping is implemented and used
- best checkpoint is saved based on validation macro F1
- a small, disciplined hyperparameter tuning process was run
- tried configurations and validation results are saved
- validation macro F1 was used as the default early-stopping / model-selection metric while other metrics were still computed and logged
- final best BiLSTM is selected using validation only
- best checkpoint can be loaded in a fresh process
- final test metrics are computed and saved
- parameter count, training time, inference latency, and tuning budget are saved
- artifacts are organized and report-ready
- the chosen OOS strategy is implemented consistently and documented in artifacts
- threshold-based OOS tuning was not performed unless explicitly approved and validation-only
- class-weight policy is documented
- learning-rate scheduling policy is documented
- confusion and top-error artifacts are saved
- final predictions artifact is saved
- a comparison-table export artifact is saved
- visualization artifacts are saved
- preprocessing drift did not occur during tuning
- preprocessing manifest compatibility was verified before training
- one-batch smoke test passed before full training
- fresh-process checkpoint reload was tested before large tuning
- per-class metrics are saved
- trainer remains reusable and not locked to BiLSTM-specific assumptions
- this BiLSTM is strong enough to serve as a meaningful comparison point against the earlier models

AB2. Final Step 6 exit gate
- Before Step 6 is considered closed, confirm all of the following are true:
  - best checkpoint reload works in a fresh process
  - final test metrics are saved
  - comparison row is exported
  - confusion and top-error artifacts are saved
  - final predictions artifact is saved
  - per-class metrics are saved
  - core visualizations are saved
  - trainer remains reusable
  - no preprocessing drift occurred
  - comparison-row schema validates cleanly against the actual Step 4 and Step 5 comparison-row outputs
  - OOS class index and metric computation match the earlier models
  - per-class metrics artifact uses the same label ordering as the earlier models
  - Step 7 handoff artifacts exist:
    - final test metrics
    - per-class metrics
    - confusion artifact
    - label ordering artifact/reference
    - final predictions artifact
    - epoch history
    - comparison row
    - run summary

AC. Deliverable quality bar
The BiLSTM should be simple, strong, and reproducible. It should not be a messy prototype. It should reflect real care in model selection, regularization, validation monitoring, and artifact generation so that later claims like “BiLSTM beat the baseline” or “BiLSTM beat Text CNN” actually mean something.

AD. Suggested self-check after implementation
After coding, self-check against this exact checklist and confirm:
1. which files/functions implement the BiLSTM
2. how Step 3 neural preprocessing artifacts are reused without re-fitting
3. which hyperparameters were tuned
4. that validation macro F1 selected the best model by default, and that any override was explicit and documented
5. how early stopping works
6. where checkpoints and tuning summaries are saved
7. what the final BiLSTM test metrics are
8. that the best checkpoint can be loaded and evaluated in a fresh process
9. which OOS strategy was used and whether it matches the earlier project decision
10. whether class weights were used and how that decision was made
11. whether learning-rate scheduling was used or explicitly not used
12. what the BiLSTM inference latency is
13. where confusion-matrix and top-error artifacts are saved
14. where visualization artifacts are saved
15. that the trainer remains reusable and not locked to one model type
16. that padding / masking / sequence summarization behavior is explicitly correct
17. that final hidden-state extraction is explicitly correct for the chosen BiLSTM summarization path
18. that truncation and UNK coverage diagnostics were saved
19. that failed or aborted runs were still logged honestly
20. that fresh-process checkpoint reload was tested before large tuning
21. that the final-model selection rule and tie-break policy were explicit
22. that per-class metrics and class-level error plots were saved
23. that device and environment metadata were saved
24. that comparison-row export matches the earlier model schemas
25. that gradient clipping policy was active and logged appropriately
26. that batch_first convention was explicit and correct
27. that LSTM internal dropout behavior was understood correctly when num_layers=1