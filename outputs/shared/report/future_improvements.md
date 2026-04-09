## Future Improvements

### High Priority

- **Collect more OOS training examples to reduce false accepts**: OOS false accepts are the dominant error across all models; more OOS training data directly addresses the distribution mismatch.
- **Merge or relabel persistently confused intent pairs within the same domain**: Several intent pairs share near-identical semantics and consistently confuse all models.
- **Apply confidence thresholding in deployment to flag uncertain predictions**: Many errors occur at high confidence; a deployment threshold could redirect uncertain queries to human review.
- **Evaluate temperature scaling for post-hoc calibration improvement**: Calibration varies significantly across models; temperature scaling could improve reliability without retraining.

### Medium Priority

- **Add pretrained word embeddings (GloVe, word2vec) to Text CNN and BiLSTM**: Pretrained embeddings could improve generalization especially for rare words and OOS queries.
- **Systematic hyperparameter tuning (dropout, hidden size, learning rate)**: Current configs use limited grid search; broader exploration may improve all models.
- **Add parameter-count vs accuracy Pareto analysis**: Quantify the efficiency-accuracy trade-off to guide model selection for deployment.
- **Add threshold analysis for OOS detection with operating-point selection**: Enable tunable precision-recall trade-off for OOS detection in production.

### Lower Priority / Future Work

- **Compare against transformer-based models (DistilBERT, BERT-base)**: Establish an upper-bound reference for the lightweight models evaluated.
- **Evaluate on additional intent-classification datasets**: Validate whether findings generalize beyond CLINC150.
- **Subword tokenization (BPE, WordPiece) for better OOV handling**: Reduce OOV rates and improve generalization to unseen vocabulary.
- **Multi-task learning combining intent classification and OOS detection**: Joint training may improve OOS discrimination by explicitly modeling the boundary.
- **Bootstrap confidence intervals for more rigorous statistical comparison**: 3 repeated runs provide limited statistical power; bootstrapping would strengthen claims.

(outputs/shared/analysis/error_analysis_summary.json)
