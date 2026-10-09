## Limitations

### Analysis-Level Limitations

- Qualitative, calibration and probability-ranking diagnostics use one validation-selected representative run per model, without measuring their variability across seeds.
- Only the 150 in-scope classes are balanced. The OOS class has 250 training and 1,000 test examples; these supports differ from each in-scope class and from many deployment distributions.
- Taxonomy labels are heuristics: `near_semantic_confusion` means a same-domain misclassification; `short_query_ambiguity` marks errors with at most 5 whitespace tokens. These rules do not establish semantic similarity or query ambiguity.
- Post-hoc calibration was not applied. Test-derived ROC points do not establish validation-selected thresholds or a deployable operating point.
- Length slices use whitespace tokenization; scope slices compare supervised in-scope and OOS classes, not training frequency.

### Project-Level Limitations

- Single benchmark only (CLINC150), with no cross-dataset or domain-transfer evaluation. Its official normalized splits contain 3 train/validation text overlaps (2 with conflicting labels) and 2 train/test overlaps (both with conflicting labels); splits are preserved.
- OOS is a supervised 151st class with labeled training examples. These results describe this explicit-class setting and do not establish general open-set detection.
- Models use different representations and search budgets. Neural models use whitespace tokens and embeddings trained from scratch; no pretrained embeddings or transformers were compared.
- The BiLSTM processes fixed right-PAD positions and concatenates final hidden states. Recurrent transitions remain active on PAD tokens, making the representation sensitive to padding length.
- Means and population standard deviations describe the recorded runs. They are descriptive dispersion measures, not confidence intervals or significance tests.
- Frozen hyperparameters come from limited earlier searches. Advancing tuning-loader RNG state and neural smoke checks made the original trial order part of the search; fair seeded loaders would require new experiments.
- Batched evaluation timing covers loader traversal, device transfer, forward pass, softmax and CPU result collection; it excludes preprocessing, checkpoint loading, array concatenation, metrics and artifact writes. The per-example figures describe batched throughput, with no controlled warmup or repeated timing trials; single-query deployment latency was not measured.
- Model checkpoints not committed to repository; reproduction requires retraining.
