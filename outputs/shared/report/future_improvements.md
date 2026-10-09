## Future Improvements

### High Priority

- **Evaluate additional labeled OOS training data in a separate experiment**: Observed false accepts motivate testing broader OOS coverage; document any new data protocol and retain the current official benchmark for comparability.
- **Manually inspect persistent same-domain confusions**: Domain-based taxonomy tags are heuristics. Inspect examples before drawing semantic conclusions; preserve official benchmark labels in the current comparison.
- **Select OOS or abstention thresholds on validation data**: Specify a target trade-off on validation data and evaluate the fixed threshold on held-out data. Test ROC operating points alone do not establish deployment behavior.
- **Evaluate temperature scaling for post-hoc calibration improvement**: Fit temperature scaling on validation data, then compare held-out calibration. Representative-run ECE differences alone do not establish its benefit across seeds.

### Medium Priority

- **Use fresh seeded loaders for each tuning trial**: Make comparisons independent of advancing loader state and smoke-check shuffles. Changing this search policy requires new tuning and final evaluations.
- **Evaluate length-aware BiLSTM sequence handling**: Packed sequences or length-aware summaries could avoid processing right-PAD positions. This modeling change requires new training, tuning and comparisons.
- **Evaluate pretrained word embeddings**: Test their effect on the neural pipelines through new training and tuning rather than assuming improved generalization.
- **Measure controlled deployment latency separately**: Use a specified device, warmup and repeated single-query measurements before making service-latency claims from batch throughput.

### Lower Priority / Future Work

- **Compare against transformer-based models (DistilBERT, BERT-base)**: Add a separately trained and tuned reference with its own preprocessing and compute budget.
- **Evaluate on additional intent-classification datasets**: Validate whether findings generalize beyond CLINC150.
- **Subword tokenization (BPE, WordPiece) for better OOV handling**: Reduce OOV rates and improve generalization to unseen vocabulary.
- **Evaluate a conventional in-scope-only MSP baseline**: Train a separate classifier without the supervised OOS class; its uncertainty score answers a different question from the current 151-class MSP diagnostic.
- **Expand independently seeded runs and design uncertainty estimates**: Distinguish test-example sampling uncertainty from training-run variation; three seeds and overlapping mean/std ranges alone do not justify significance claims.

(outputs/shared/analysis/error_analysis_summary.json)
