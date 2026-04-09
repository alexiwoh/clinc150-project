## Key Findings

With only 3 repeated runs, differences between models may not be statistically meaningful. Where models have overlapping mean +/- std ranges, this is noted rather than declaring one superior.

1. **Headline**: TF-IDF + MLP achieved the best aggregate test macro F1 of 0.8727 +/- 0.0031 (outputs/shared/model_comparison_aggregate.json)
2. **Calibration**: Best calibrated: text_cnn (ECE = 0.0374). Worst: bilstm (ECE = 0.1125) (outputs/shared/analysis/calibration_summary.json)
3. **Oos Detection**: Text CNN achieved the best OOS detection with AUROC = 0.9523 and AUPR = 0.8403 (outputs/shared/analysis/oos_threshold_comparison.json)
4. **Confusion**: Dominant error category across all models: `oos_as_inscope`. Dominant per model: TF-IDF + MLP: oos_as_inscope, Text CNN: oos_as_inscope, BiLSTM: oos_as_inscope (outputs/shared/analysis/error_taxonomy_summary.json)
5. **Architecture**: 608 examples (0.1105) misclassified by all models; 464 (0.0844) unique to one model. Agreement on wrong class: 0.3668 (outputs/shared/analysis/cross_model_error_comparison.json)
6. **Length**: Short-query accuracy: TF-IDF + MLP 0.8291, Text CNN 0.8583, BiLSTM 0.8097 (outputs/shared/analysis/length_slice_comparison.png)
7. **Efficiency**: Parameter counts: TF-IDF + MLP 5,197,975, Text CNN 1,930,167, BiLSTM 4,284,311. Inference throughput: TF-IDF + MLP 17783 ex/s, Text CNN 32082 ex/s, BiLSTM 8432 ex/s (outputs/shared/efficiency_summary_table.json)
8. **Recommendation**: Focus on improving OOS detection and addressing semantically ambiguous intent pairs within the same domain. Consider intent merging for persistently confused pairs and confidence thresholding for high-confidence errors. (outputs/shared/analysis/error_analysis_summary.json)
