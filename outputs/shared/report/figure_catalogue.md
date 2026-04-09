## Figure Catalogue

### Training Diagnostics

- **train_val_loss_curve** (representative): Training and validation loss curves for TF-IDF + MLP. Data: representative run (run_01_seed_42), CLINC150 test set.
  Path: `outputs/mlp/figures/representative_train_val_loss_curve.png`
- **val_macro_f1_curve** (representative): Validation macro F1 progression during training for TF-IDF + MLP. Data: representative run (run_01_seed_42), CLINC150 test set.
  Path: `outputs/mlp/figures/representative_val_macro_f1_curve.png`
- **val_accuracy_curve** (representative): Validation accuracy progression during training for TF-IDF + MLP. Data: representative run (run_01_seed_42), CLINC150 test set.
  Path: `outputs/mlp/figures/representative_val_accuracy_curve.png`
- **train_val_loss_curve** (representative): Training and validation loss curves for Text CNN. Data: representative run (run_03_seed_2024), CLINC150 test set.
  Path: `outputs/text_cnn/figures/representative_train_val_loss_curve.png`
- **val_macro_f1_curve** (representative): Validation macro F1 progression during training for Text CNN. Data: representative run (run_03_seed_2024), CLINC150 test set.
  Path: `outputs/text_cnn/figures/representative_val_macro_f1_curve.png`
- **val_accuracy_curve** (representative): Validation accuracy progression during training for Text CNN. Data: representative run (run_03_seed_2024), CLINC150 test set.
  Path: `outputs/text_cnn/figures/representative_val_accuracy_curve.png`
- **train_val_loss_curve** (representative): Training and validation loss curves for BiLSTM. Data: representative run (run_02_seed_1337), CLINC150 test set.
  Path: `outputs/bilstm/figures/representative_train_val_loss_curve.png`
- **val_macro_f1_curve** (representative): Validation macro F1 progression during training for BiLSTM. Data: representative run (run_02_seed_1337), CLINC150 test set.
  Path: `outputs/bilstm/figures/representative_val_macro_f1_curve.png`
- **val_accuracy_curve** (representative): Validation accuracy progression during training for BiLSTM. Data: representative run (run_02_seed_1337), CLINC150 test set.
  Path: `outputs/bilstm/figures/representative_val_accuracy_curve.png`

### Model Comparison

- **model_comparison_test_accuracy** (aggregate): Aggregate test accuracy comparison with error bars for all models. Data: aggregate over 3 repeated runs, CLINC150 test set.
  Path: `outputs/shared/figures/model_comparison_test_accuracy.png`
- **model_comparison_test_macro_f1** (aggregate): Aggregate test macro F1 comparison with error bars for all models. Data: aggregate over 3 repeated runs, CLINC150 test set.
  Path: `outputs/shared/figures/model_comparison_test_macro_f1.png`
- **model_comparison_oos_f1** (aggregate): Aggregate OOS F1 comparison with error bars for all models. Data: aggregate over 3 repeated runs, CLINC150 test set.
  Path: `outputs/shared/figures/model_comparison_oos_f1.png`
- **oos_metrics_comparison** (aggregate): Aggregate OOS precision, recall, and F1 comparison for all models. Data: aggregate over 3 repeated runs, CLINC150 test set.
  Path: `outputs/shared/figures/oos_metrics_comparison.png`

### OOS Detection

- **oos_metrics** (representative): OOS detection metrics summary (precision, recall, F1) for TF-IDF + MLP. Data: representative run (run_01_seed_42), CLINC150 test set.
  Path: `outputs/mlp/figures/representative_oos_metrics.png`
- **oos_metrics** (representative): OOS detection metrics summary (precision, recall, F1) for Text CNN. Data: representative run (run_03_seed_2024), CLINC150 test set.
  Path: `outputs/text_cnn/figures/representative_oos_metrics.png`
- **oos_metrics** (representative): OOS detection metrics summary (precision, recall, F1) for BiLSTM. Data: representative run (run_02_seed_1337), CLINC150 test set.
  Path: `outputs/bilstm/figures/representative_oos_metrics.png`
- **oos_roc_curve** (representative): OOS detection ROC curve (explicit OOS class probability method) for TF-IDF + MLP. Data: representative run (run_01_seed_42), CLINC150 test set.
  Path: `outputs/mlp/analysis/oos_roc_curve.png`
- **oos_pr_curve** (representative): OOS detection precision-recall curve for TF-IDF + MLP. Data: representative run (run_01_seed_42), CLINC150 test set.
  Path: `outputs/mlp/analysis/oos_pr_curve.png`
- **oos_error_breakdown** (representative): OOS error breakdown: false accepts (OOS as in-scope) vs false rejects for TF-IDF + MLP. Data: representative run (run_01_seed_42), CLINC150 test set.
  Path: `outputs/mlp/analysis/oos_error_breakdown.png`
- **oos_roc_curve** (representative): OOS detection ROC curve (explicit OOS class probability method) for Text CNN. Data: representative run (run_03_seed_2024), CLINC150 test set.
  Path: `outputs/text_cnn/analysis/oos_roc_curve.png`
- **oos_pr_curve** (representative): OOS detection precision-recall curve for Text CNN. Data: representative run (run_03_seed_2024), CLINC150 test set.
  Path: `outputs/text_cnn/analysis/oos_pr_curve.png`
- **oos_error_breakdown** (representative): OOS error breakdown: false accepts (OOS as in-scope) vs false rejects for Text CNN. Data: representative run (run_03_seed_2024), CLINC150 test set.
  Path: `outputs/text_cnn/analysis/oos_error_breakdown.png`
- **oos_roc_curve** (representative): OOS detection ROC curve (explicit OOS class probability method) for BiLSTM. Data: representative run (run_02_seed_1337), CLINC150 test set.
  Path: `outputs/bilstm/analysis/oos_roc_curve.png`
- **oos_pr_curve** (representative): OOS detection precision-recall curve for BiLSTM. Data: representative run (run_02_seed_1337), CLINC150 test set.
  Path: `outputs/bilstm/analysis/oos_pr_curve.png`
- **oos_error_breakdown** (representative): OOS error breakdown: false accepts (OOS as in-scope) vs false rejects for BiLSTM. Data: representative run (run_02_seed_1337), CLINC150 test set.
  Path: `outputs/bilstm/analysis/oos_error_breakdown.png`
- **oos_roc_comparison** (aggregate): OOS ROC curve comparison overlay across models for all models. Data: aggregate over 3 repeated runs, CLINC150 test set.
  Path: `outputs/shared/analysis/oos_roc_comparison.png`
- **oos_pr_comparison** (aggregate): OOS precision-recall curve comparison overlay across models for all models. Data: aggregate over 3 repeated runs, CLINC150 test set.
  Path: `outputs/shared/analysis/oos_pr_comparison.png`
- **oos_error_comparison** (aggregate): OOS false-accept and false-reject pattern comparison across models for all models. Data: aggregate over 3 repeated runs, CLINC150 test set.
  Path: `outputs/shared/analysis/oos_error_comparison.png`

### Confusion Analysis

- **confusion_matrix** (representative): Test-set confusion matrix across 151 intent classes including OOS for TF-IDF + MLP. Data: representative run (run_01_seed_42), CLINC150 test set.
  Path: `outputs/mlp/figures/representative_confusion_matrix.png`
- **top_confused_pairs** (representative): Top confused intent pairs on the test set for TF-IDF + MLP. Data: representative run (run_01_seed_42), CLINC150 test set.
  Path: `outputs/mlp/figures/representative_top_confused_pairs.png`
- **bottom_classes_f1** (representative): Bottom classes by test F1 score for TF-IDF + MLP. Data: representative run (run_01_seed_42), CLINC150 test set.
  Path: `outputs/mlp/figures/representative_bottom_classes_f1.png`
- **confusion_matrix** (representative): Test-set confusion matrix across 151 intent classes including OOS for Text CNN. Data: representative run (run_03_seed_2024), CLINC150 test set.
  Path: `outputs/text_cnn/figures/representative_confusion_matrix.png`
- **top_confused_pairs** (representative): Top confused intent pairs on the test set for Text CNN. Data: representative run (run_03_seed_2024), CLINC150 test set.
  Path: `outputs/text_cnn/figures/representative_top_confused_pairs.png`
- **bottom_classes_f1** (representative): Bottom classes by test F1 score for Text CNN. Data: representative run (run_03_seed_2024), CLINC150 test set.
  Path: `outputs/text_cnn/figures/representative_bottom_classes_f1.png`
- **confusion_matrix** (representative): Test-set confusion matrix across 151 intent classes including OOS for BiLSTM. Data: representative run (run_02_seed_1337), CLINC150 test set.
  Path: `outputs/bilstm/figures/representative_confusion_matrix.png`
- **top_confused_pairs** (representative): Top confused intent pairs on the test set for BiLSTM. Data: representative run (run_02_seed_1337), CLINC150 test set.
  Path: `outputs/bilstm/figures/representative_top_confused_pairs.png`
- **bottom_classes_f1** (representative): Bottom classes by test F1 score for BiLSTM. Data: representative run (run_02_seed_1337), CLINC150 test set.
  Path: `outputs/bilstm/figures/representative_bottom_classes_f1.png`

### Error Analysis

- **error_summary** (representative): Most frequent misclassification categories on the test set for TF-IDF + MLP. Data: representative run (run_01_seed_42), CLINC150 test set.
  Path: `outputs/mlp/figures/representative_error_summary.png`
- **error_summary** (representative): Most frequent misclassification categories on the test set for Text CNN. Data: representative run (run_03_seed_2024), CLINC150 test set.
  Path: `outputs/text_cnn/figures/representative_error_summary.png`
- **error_summary** (representative): Most frequent misclassification categories on the test set for BiLSTM. Data: representative run (run_02_seed_1337), CLINC150 test set.
  Path: `outputs/bilstm/figures/representative_error_summary.png`
- **reliability_diagram** (representative): Reliability diagram showing calibration quality (predicted confidence vs actual accuracy) for TF-IDF + MLP. Data: representative run (run_01_seed_42), CLINC150 test set.
  Path: `outputs/mlp/analysis/reliability_diagram.png`
- **confidence_histogram** (representative): Prediction confidence distribution for correct and incorrect predictions for TF-IDF + MLP. Data: representative run (run_01_seed_42), CLINC150 test set.
  Path: `outputs/mlp/analysis/confidence_histogram.png`
- **confidence_vs_accuracy** (representative): Confidence vs accuracy analysis across confidence bins for TF-IDF + MLP. Data: representative run (run_01_seed_42), CLINC150 test set.
  Path: `outputs/mlp/analysis/confidence_vs_accuracy.png`
- **worst_classes_heatmap** (representative): Confusion heatmap for worst-performing intent classes for TF-IDF + MLP. Data: representative run (run_01_seed_42), CLINC150 test set.
  Path: `outputs/mlp/analysis/worst_classes_confusion_heatmap.png`
- **confusion_stability** (representative): Confusion stability across repeated runs (3 seeds) for TF-IDF + MLP. Data: representative run (run_01_seed_42), CLINC150 test set.
  Path: `outputs/mlp/analysis/confusion_stability.png`
- **reliability_diagram** (representative): Reliability diagram showing calibration quality (predicted confidence vs actual accuracy) for Text CNN. Data: representative run (run_03_seed_2024), CLINC150 test set.
  Path: `outputs/text_cnn/analysis/reliability_diagram.png`
- **confidence_histogram** (representative): Prediction confidence distribution for correct and incorrect predictions for Text CNN. Data: representative run (run_03_seed_2024), CLINC150 test set.
  Path: `outputs/text_cnn/analysis/confidence_histogram.png`
- **confidence_vs_accuracy** (representative): Confidence vs accuracy analysis across confidence bins for Text CNN. Data: representative run (run_03_seed_2024), CLINC150 test set.
  Path: `outputs/text_cnn/analysis/confidence_vs_accuracy.png`
- **worst_classes_heatmap** (representative): Confusion heatmap for worst-performing intent classes for Text CNN. Data: representative run (run_03_seed_2024), CLINC150 test set.
  Path: `outputs/text_cnn/analysis/worst_classes_confusion_heatmap.png`
- **confusion_stability** (representative): Confusion stability across repeated runs (3 seeds) for Text CNN. Data: representative run (run_03_seed_2024), CLINC150 test set.
  Path: `outputs/text_cnn/analysis/confusion_stability.png`
- **reliability_diagram** (representative): Reliability diagram showing calibration quality (predicted confidence vs actual accuracy) for BiLSTM. Data: representative run (run_02_seed_1337), CLINC150 test set.
  Path: `outputs/bilstm/analysis/reliability_diagram.png`
- **confidence_histogram** (representative): Prediction confidence distribution for correct and incorrect predictions for BiLSTM. Data: representative run (run_02_seed_1337), CLINC150 test set.
  Path: `outputs/bilstm/analysis/confidence_histogram.png`
- **confidence_vs_accuracy** (representative): Confidence vs accuracy analysis across confidence bins for BiLSTM. Data: representative run (run_02_seed_1337), CLINC150 test set.
  Path: `outputs/bilstm/analysis/confidence_vs_accuracy.png`
- **worst_classes_heatmap** (representative): Confusion heatmap for worst-performing intent classes for BiLSTM. Data: representative run (run_02_seed_1337), CLINC150 test set.
  Path: `outputs/bilstm/analysis/worst_classes_confusion_heatmap.png`
- **confusion_stability** (representative): Confusion stability across repeated runs (3 seeds) for BiLSTM. Data: representative run (run_02_seed_1337), CLINC150 test set.
  Path: `outputs/bilstm/analysis/confusion_stability.png`
- **calibration_comparison** (aggregate): Calibration comparison (ECE, MCE, Brier, NLL) across models for all models. Data: aggregate over 3 repeated runs, CLINC150 test set.
  Path: `outputs/shared/analysis/calibration_comparison.png`
- **error_taxonomy** (aggregate): Error taxonomy comparison (OOS-as-inscope, inscope-as-OOS, semantic, cross-domain) per model for all models. Data: aggregate over 3 repeated runs, CLINC150 test set.
  Path: `outputs/shared/analysis/error_taxonomy_comparison.png`
- **confidence_vs_accuracy** (aggregate): Confidence vs accuracy analysis across confidence bins for all models. Data: aggregate over 3 repeated runs, CLINC150 test set.
  Path: `outputs/shared/analysis/confidence_accuracy_comparison.png`
- **length_slice** (aggregate): Accuracy by query length (short/medium/long) across models for all models. Data: aggregate over 3 repeated runs, CLINC150 test set.
  Path: `outputs/shared/analysis/length_slice_comparison.png`
- **frequency_slice** (aggregate): Accuracy by class frequency across models for all models. Data: aggregate over 3 repeated runs, CLINC150 test set.
  Path: `outputs/shared/analysis/frequency_slice_comparison.png`
- **cross_model_error_overlap** (aggregate): Cross-model error overlap: examples wrong by all/some/one model for all models. Data: aggregate over 3 repeated runs, CLINC150 test set.
  Path: `outputs/shared/analysis/cross_model_error_overlap.png`

### Efficiency

- **model_efficiency_comparison** (aggregate): Model efficiency comparison (parameters, training time, throughput) for all models. Data: aggregate over 3 repeated runs, CLINC150 test set.
  Path: `outputs/shared/figures/model_efficiency_comparison.png`

Total figures: 62
