## Key Findings

Means and population standard deviations describe the recorded runs. They are descriptive dispersion measures, not confidence intervals or significance tests.

Calibration, OOS-probability ranking and qualitative findings use one validation-selected representative run per model. Differences in preprocessing and search budgets prevent an architecture-only causal interpretation.

1. **Headline**: TF-IDF + MLP achieved the best aggregate test macro F1 of 0.8716 +/- 0.0029 (outputs/shared/model_comparison_aggregate.json)
Related figure: `outputs/shared/figures/model_comparison_test_macro_f1.png`

2. **Calibration**: Lowest representative-run ECE: text_cnn (0.0390). Highest: bilstm (0.1232) (outputs/shared/analysis/calibration_summary.json)
Related figure: `outputs/shared/analysis/calibration_comparison.png`

3. **Oos Detection**: Text CNN has the highest representative-run OOS-probability AUROC (0.9491), with AUPR = 0.8273 (outputs/shared/analysis/oos_threshold_comparison.json)
Related figures: `outputs/shared/analysis/oos_roc_comparison.png`, `outputs/shared/analysis/oos_pr_comparison.png`

4. **Confusion**: Representative-run dominant heuristic category per model: TF-IDF + MLP: oos_as_inscope, Text CNN: oos_as_inscope, BiLSTM: oos_as_inscope (outputs/shared/analysis/error_taxonomy_summary.json)
Related figures: `outputs/mlp/figures/representative_top_confused_pairs.png`, `outputs/text_cnn/figures/representative_top_confused_pairs.png`, `outputs/bilstm/figures/representative_top_confused_pairs.png`, `outputs/shared/analysis/error_taxonomy_comparison.png`

5. **Architecture**: 600 examples (0.1091) misclassified by all models; 489 (0.0889) unique to one model. Agreement on wrong class: 0.3633 (outputs/shared/analysis/cross_model_error_comparison.json)
Related figure: `outputs/shared/analysis/cross_model_error_overlap.png`

6. **Length**: Short-query accuracy: TF-IDF + MLP 0.8291, Text CNN 0.8641, BiLSTM 0.8019 (outputs/shared/analysis/error_analysis_summary.json)
Related figure: `outputs/shared/analysis/length_slice_comparison.png`

7. **Efficiency**: Parameter counts: TF-IDF + MLP 5,197,975, Text CNN 1,930,167, BiLSTM 4,284,311. Inference throughput: TF-IDF + MLP 18069 ex/s, Text CNN 28456 ex/s, BiLSTM 7910 ex/s (outputs/shared/efficiency_summary_table.json)
Related figure: `outputs/shared/figures/model_efficiency_comparison.png`

8. **Recommendation**: Focus on improving OOS detection and addressing semantically ambiguous intent pairs within the same domain. Consider intent merging for persistently confused pairs and confidence thresholding for high-confidence errors. (outputs/shared/analysis/error_analysis_summary.json)
Related figures: `outputs/mlp/figures/representative_top_confused_pairs.png`, `outputs/text_cnn/figures/representative_top_confused_pairs.png`, `outputs/bilstm/figures/representative_top_confused_pairs.png`, `outputs/shared/analysis/oos_error_comparison.png`
