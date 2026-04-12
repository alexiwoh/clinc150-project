## Figure Catalogue

### Training Diagnostics

- **train_val_loss_curve** (representative): Training and validation loss by epoch for TF-IDF + MLP. Computed from representative run `run_01_seed_42`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/mlp/figures/representative_train_val_loss_curve.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/epoch_history.json`
- **val_macro_f1_curve** (representative): Validation macro F1 by training epoch for TF-IDF + MLP. Computed from representative run `run_01_seed_42`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/mlp/figures/representative_val_macro_f1_curve.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/epoch_history.json`
- **val_accuracy_curve** (representative): Validation accuracy by training epoch for TF-IDF + MLP. Computed from representative run `run_01_seed_42`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/mlp/figures/representative_val_accuracy_curve.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/epoch_history.json`
- **train_val_loss_curve** (representative): Training and validation loss by epoch for Text CNN. Computed from representative run `run_03_seed_2024`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/text_cnn/figures/representative_train_val_loss_curve.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/epoch_history.json`
- **val_macro_f1_curve** (representative): Validation macro F1 by training epoch for Text CNN. Computed from representative run `run_03_seed_2024`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/text_cnn/figures/representative_val_macro_f1_curve.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/epoch_history.json`
- **val_accuracy_curve** (representative): Validation accuracy by training epoch for Text CNN. Computed from representative run `run_03_seed_2024`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/text_cnn/figures/representative_val_accuracy_curve.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/epoch_history.json`
- **train_val_loss_curve** (representative): Training and validation loss by epoch for BiLSTM. Computed from representative run `run_02_seed_1337`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/bilstm/figures/representative_train_val_loss_curve.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/epoch_history.json`
- **val_macro_f1_curve** (representative): Validation macro F1 by training epoch for BiLSTM. Computed from representative run `run_02_seed_1337`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/bilstm/figures/representative_val_macro_f1_curve.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/epoch_history.json`
- **val_accuracy_curve** (representative): Validation accuracy by training epoch for BiLSTM. Computed from representative run `run_02_seed_1337`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/bilstm/figures/representative_val_accuracy_curve.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/epoch_history.json`

### Model Comparison

- **model_comparison_test_accuracy** (aggregate): Cross-model test accuracy comparison with error bars across all evaluated models. Uses aggregate statistics over 3 repeated runs per model. Uses the CLINC150 test split.
  Figure path: `outputs/shared/figures/model_comparison_test_accuracy.png`
  Source artifacts: `outputs/shared/model_comparison_aggregate.json`
- **model_comparison_test_macro_f1** (aggregate): Cross-model test macro F1 comparison with error bars across all evaluated models. Uses aggregate statistics over 3 repeated runs per model. Uses the CLINC150 test split.
  Figure path: `outputs/shared/figures/model_comparison_test_macro_f1.png`
  Source artifacts: `outputs/shared/model_comparison_aggregate.json`
- **model_comparison_oos_f1** (aggregate): Cross-model OOS F1 comparison with error bars across all evaluated models. Uses aggregate statistics over 3 repeated runs per model. Uses the CLINC150 test split.
  Figure path: `outputs/shared/figures/model_comparison_oos_f1.png`
  Source artifacts: `outputs/shared/model_comparison_aggregate.json`
- **oos_metrics_comparison** (aggregate): Cross-model OOS precision, recall, and F1 comparison across all evaluated models. Uses aggregate statistics over 3 repeated runs per model. Uses the CLINC150 test split. Metrics shown: OOS precision, OOS recall, OOS F1.
  Figure path: `outputs/shared/figures/oos_metrics_comparison.png`
  Source artifacts: `outputs/shared/oos_summary_table.json`

### OOS Detection

- **oos_metrics** (representative): Representative-run OOS precision, recall, and F1 summary for TF-IDF + MLP. Computed from representative run `run_01_seed_42`, selected by highest validation macro f1. Uses the CLINC150 test split. Metrics shown: OOS precision, OOS recall, OOS F1.
  Figure path: `outputs/mlp/figures/representative_oos_metrics.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/test_metrics.json`
- **oos_metrics** (representative): Representative-run OOS precision, recall, and F1 summary for Text CNN. Computed from representative run `run_03_seed_2024`, selected by highest validation macro f1. Uses the CLINC150 test split. Metrics shown: OOS precision, OOS recall, OOS F1.
  Figure path: `outputs/text_cnn/figures/representative_oos_metrics.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/test_metrics.json`
- **oos_metrics** (representative): Representative-run OOS precision, recall, and F1 summary for BiLSTM. Computed from representative run `run_02_seed_1337`, selected by highest validation macro f1. Uses the CLINC150 test split. Metrics shown: OOS precision, OOS recall, OOS F1.
  Figure path: `outputs/bilstm/figures/representative_oos_metrics.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/test_metrics.json`
- **oos_roc_curve** (analysis): ROC curve for explicit-OOS detection for TF-IDF + MLP. Computed from representative-run OOS threshold analysis for representative run `run_01_seed_42`. Uses the CLINC150 test split.
  Figure path: `outputs/mlp/analysis/oos_roc_curve.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/confidences.npz`
- **oos_pr_curve** (analysis): Precision-recall curve for explicit-OOS detection for TF-IDF + MLP. Computed from representative-run OOS threshold analysis for representative run `run_01_seed_42`. Uses the CLINC150 test split.
  Figure path: `outputs/mlp/analysis/oos_pr_curve.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/confidences.npz`
- **oos_error_breakdown** (analysis): False-accept and false-reject intent breakdown for OOS analysis for TF-IDF + MLP. Computed from representative-run OOS error analysis for representative run `run_01_seed_42`. Uses the CLINC150 test split.
  Figure path: `outputs/mlp/analysis/oos_error_breakdown.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/final_predictions.csv`
- **oos_roc_curve** (analysis): ROC curve for explicit-OOS detection for Text CNN. Computed from representative-run OOS threshold analysis for representative run `run_03_seed_2024`. Uses the CLINC150 test split.
  Figure path: `outputs/text_cnn/analysis/oos_roc_curve.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/confidences.npz`
- **oos_pr_curve** (analysis): Precision-recall curve for explicit-OOS detection for Text CNN. Computed from representative-run OOS threshold analysis for representative run `run_03_seed_2024`. Uses the CLINC150 test split.
  Figure path: `outputs/text_cnn/analysis/oos_pr_curve.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/confidences.npz`
- **oos_error_breakdown** (analysis): False-accept and false-reject intent breakdown for OOS analysis for Text CNN. Computed from representative-run OOS error analysis for representative run `run_03_seed_2024`. Uses the CLINC150 test split.
  Figure path: `outputs/text_cnn/analysis/oos_error_breakdown.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/final_predictions.csv`
- **oos_roc_curve** (analysis): ROC curve for explicit-OOS detection for BiLSTM. Computed from representative-run OOS threshold analysis for representative run `run_02_seed_1337`. Uses the CLINC150 test split.
  Figure path: `outputs/bilstm/analysis/oos_roc_curve.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/confidences.npz`
- **oos_pr_curve** (analysis): Precision-recall curve for explicit-OOS detection for BiLSTM. Computed from representative-run OOS threshold analysis for representative run `run_02_seed_1337`. Uses the CLINC150 test split.
  Figure path: `outputs/bilstm/analysis/oos_pr_curve.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/confidences.npz`
- **oos_error_breakdown** (analysis): False-accept and false-reject intent breakdown for OOS analysis for BiLSTM. Computed from representative-run OOS error analysis for representative run `run_02_seed_1337`. Uses the CLINC150 test split.
  Figure path: `outputs/bilstm/analysis/oos_error_breakdown.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/final_predictions.csv`
- **oos_roc_comparison** (analysis): Cross-model ROC comparison for explicit-OOS detection across all evaluated models. Computed from cross-model OOS ROC comparison using one representative run per model. Uses the CLINC150 test split.
  Figure path: `outputs/shared/analysis/oos_roc_comparison.png`
  Source artifacts: `outputs/mlp/analysis/oos_threshold_metrics.json`, `outputs/text_cnn/analysis/oos_threshold_metrics.json`, `outputs/bilstm/analysis/oos_threshold_metrics.json`
- **oos_pr_comparison** (analysis): Cross-model precision-recall comparison for explicit-OOS detection across all evaluated models. Computed from cross-model OOS precision-recall comparison using one representative run per model. Uses the CLINC150 test split.
  Figure path: `outputs/shared/analysis/oos_pr_comparison.png`
  Source artifacts: `outputs/mlp/analysis/oos_threshold_metrics.json`, `outputs/text_cnn/analysis/oos_threshold_metrics.json`, `outputs/bilstm/analysis/oos_threshold_metrics.json`
- **oos_error_comparison** (analysis): Cross-model false-accept versus false-reject comparison across all evaluated models. Computed from cross-model OOS error comparison using one representative run per model. Uses the CLINC150 test split. Metrics shown: false accepts, false rejects.
  Figure path: `outputs/shared/analysis/oos_error_comparison.png`
  Source artifacts: `outputs/mlp/analysis/oos_error_deep_dive.json`, `outputs/text_cnn/analysis/oos_error_deep_dive.json`, `outputs/bilstm/analysis/oos_error_deep_dive.json`

### Confusion Analysis

- **confusion_matrix** (representative): Confusion matrix for intent predictions for TF-IDF + MLP. Computed from representative run `run_01_seed_42`, selected by highest validation macro f1. Uses the CLINC150 test split. Covers 151 intent classes including OOS.
  Figure path: `outputs/mlp/figures/representative_confusion_matrix.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/confusion_matrix.csv`
- **top_confused_pairs** (representative): Most frequent true-label and predicted-label confusion pairs for TF-IDF + MLP. Computed from representative run `run_01_seed_42`, selected by highest validation macro f1. Uses the CLINC150 test split. Highlights the top 10 confusion pairs.
  Figure path: `outputs/mlp/figures/representative_top_confused_pairs.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/top_confusions.json`
- **bottom_classes_f1** (representative): Lowest-F1 intent classes in the representative evaluation for TF-IDF + MLP. Computed from representative run `run_01_seed_42`, selected by highest validation macro f1. Uses the CLINC150 test split.
  Figure path: `outputs/mlp/figures/representative_bottom_classes_f1.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/per_class_metrics.json`
- **confusion_matrix** (representative): Confusion matrix for intent predictions for Text CNN. Computed from representative run `run_03_seed_2024`, selected by highest validation macro f1. Uses the CLINC150 test split. Covers 151 intent classes including OOS.
  Figure path: `outputs/text_cnn/figures/representative_confusion_matrix.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/confusion_matrix.csv`
- **top_confused_pairs** (representative): Most frequent true-label and predicted-label confusion pairs for Text CNN. Computed from representative run `run_03_seed_2024`, selected by highest validation macro f1. Uses the CLINC150 test split. Highlights the top 10 confusion pairs.
  Figure path: `outputs/text_cnn/figures/representative_top_confused_pairs.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/top_confusions.json`
- **bottom_classes_f1** (representative): Lowest-F1 intent classes in the representative evaluation for Text CNN. Computed from representative run `run_03_seed_2024`, selected by highest validation macro f1. Uses the CLINC150 test split.
  Figure path: `outputs/text_cnn/figures/representative_bottom_classes_f1.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/per_class_metrics.json`
- **confusion_matrix** (representative): Confusion matrix for intent predictions for BiLSTM. Computed from representative run `run_02_seed_1337`, selected by highest validation macro f1. Uses the CLINC150 test split. Covers 151 intent classes including OOS.
  Figure path: `outputs/bilstm/figures/representative_confusion_matrix.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/confusion_matrix.csv`
- **top_confused_pairs** (representative): Most frequent true-label and predicted-label confusion pairs for BiLSTM. Computed from representative run `run_02_seed_1337`, selected by highest validation macro f1. Uses the CLINC150 test split. Highlights the top 10 confusion pairs.
  Figure path: `outputs/bilstm/figures/representative_top_confused_pairs.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/top_confusions.json`
- **bottom_classes_f1** (representative): Lowest-F1 intent classes in the representative evaluation for BiLSTM. Computed from representative run `run_02_seed_1337`, selected by highest validation macro f1. Uses the CLINC150 test split.
  Figure path: `outputs/bilstm/figures/representative_bottom_classes_f1.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/per_class_metrics.json`

### Error Analysis

- **error_summary** (representative): Representative-run summary of the most frequent error patterns for TF-IDF + MLP. Computed from representative run `run_01_seed_42`, selected by highest validation macro f1. Uses the CLINC150 test split. Summarizes 20 ranked error examples or categories.
  Figure path: `outputs/mlp/figures/representative_error_summary.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/top_errors.json`
- **error_summary** (representative): Representative-run summary of the most frequent error patterns for Text CNN. Computed from representative run `run_03_seed_2024`, selected by highest validation macro f1. Uses the CLINC150 test split. Summarizes 20 ranked error examples or categories.
  Figure path: `outputs/text_cnn/figures/representative_error_summary.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/top_errors.json`
- **error_summary** (representative): Representative-run summary of the most frequent error patterns for BiLSTM. Computed from representative run `run_02_seed_1337`, selected by highest validation macro f1. Uses the CLINC150 test split. Summarizes 20 ranked error examples or categories.
  Figure path: `outputs/bilstm/figures/representative_error_summary.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/top_errors.json`
- **reliability_diagram** (analysis): Reliability diagram comparing confidence with empirical accuracy for TF-IDF + MLP. Computed from representative-run calibration analysis for representative run `run_01_seed_42`. Uses the CLINC150 test split.
  Figure path: `outputs/mlp/analysis/reliability_diagram.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/confidences.npz`
- **confidence_histogram** (analysis): Confidence distribution for correct versus incorrect predictions for TF-IDF + MLP. Computed from representative-run calibration analysis for representative run `run_01_seed_42`. Uses the CLINC150 test split.
  Figure path: `outputs/mlp/analysis/confidence_histogram.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/confidences.npz`
- **confidence_vs_accuracy** (analysis): Accuracy across confidence buckets for TF-IDF + MLP. Computed from representative-run confidence stratification for representative run `run_01_seed_42`. Uses the CLINC150 test split.
  Figure path: `outputs/mlp/analysis/confidence_vs_accuracy.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/confidences.npz`
- **worst_classes_heatmap** (analysis): Confusion heatmap for the weakest intent classes for TF-IDF + MLP. Computed from representative-run worst-class analysis for representative run `run_01_seed_42`. Uses the CLINC150 test split.
  Figure path: `outputs/mlp/analysis/worst_classes_confusion_heatmap.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/confusion_matrix.csv`
- **confusion_stability** (analysis): Run-to-run stability of the highest-count confusion pairs for TF-IDF + MLP. Computed from cross-run confusion stability analysis. Uses the CLINC150 test split. Tracks the top 20 confusion pairs across 3 completed runs.
  Figure path: `outputs/mlp/analysis/confusion_stability.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/confusion_matrix.csv`, `outputs/mlp/final_runs/run_02_seed_1337/confusion_matrix.csv`, `outputs/mlp/final_runs/run_03_seed_2024/confusion_matrix.csv`
- **reliability_diagram** (analysis): Reliability diagram comparing confidence with empirical accuracy for Text CNN. Computed from representative-run calibration analysis for representative run `run_03_seed_2024`. Uses the CLINC150 test split.
  Figure path: `outputs/text_cnn/analysis/reliability_diagram.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/confidences.npz`
- **confidence_histogram** (analysis): Confidence distribution for correct versus incorrect predictions for Text CNN. Computed from representative-run calibration analysis for representative run `run_03_seed_2024`. Uses the CLINC150 test split.
  Figure path: `outputs/text_cnn/analysis/confidence_histogram.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/confidences.npz`
- **confidence_vs_accuracy** (analysis): Accuracy across confidence buckets for Text CNN. Computed from representative-run confidence stratification for representative run `run_03_seed_2024`. Uses the CLINC150 test split.
  Figure path: `outputs/text_cnn/analysis/confidence_vs_accuracy.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/confidences.npz`
- **worst_classes_heatmap** (analysis): Confusion heatmap for the weakest intent classes for Text CNN. Computed from representative-run worst-class analysis for representative run `run_03_seed_2024`. Uses the CLINC150 test split.
  Figure path: `outputs/text_cnn/analysis/worst_classes_confusion_heatmap.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/confusion_matrix.csv`
- **confusion_stability** (analysis): Run-to-run stability of the highest-count confusion pairs for Text CNN. Computed from cross-run confusion stability analysis. Uses the CLINC150 test split. Tracks the top 20 confusion pairs across 3 completed runs.
  Figure path: `outputs/text_cnn/analysis/confusion_stability.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_01_seed_42/confusion_matrix.csv`, `outputs/text_cnn/final_runs/run_02_seed_1337/confusion_matrix.csv`, `outputs/text_cnn/final_runs/run_03_seed_2024/confusion_matrix.csv`
- **reliability_diagram** (analysis): Reliability diagram comparing confidence with empirical accuracy for BiLSTM. Computed from representative-run calibration analysis for representative run `run_02_seed_1337`. Uses the CLINC150 test split.
  Figure path: `outputs/bilstm/analysis/reliability_diagram.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/confidences.npz`
- **confidence_histogram** (analysis): Confidence distribution for correct versus incorrect predictions for BiLSTM. Computed from representative-run calibration analysis for representative run `run_02_seed_1337`. Uses the CLINC150 test split.
  Figure path: `outputs/bilstm/analysis/confidence_histogram.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/confidences.npz`
- **confidence_vs_accuracy** (analysis): Accuracy across confidence buckets for BiLSTM. Computed from representative-run confidence stratification for representative run `run_02_seed_1337`. Uses the CLINC150 test split.
  Figure path: `outputs/bilstm/analysis/confidence_vs_accuracy.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/confidences.npz`
- **worst_classes_heatmap** (analysis): Confusion heatmap for the weakest intent classes for BiLSTM. Computed from representative-run worst-class analysis for representative run `run_02_seed_1337`. Uses the CLINC150 test split.
  Figure path: `outputs/bilstm/analysis/worst_classes_confusion_heatmap.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/confusion_matrix.csv`
- **confusion_stability** (analysis): Run-to-run stability of the highest-count confusion pairs for BiLSTM. Computed from cross-run confusion stability analysis. Uses the CLINC150 test split. Tracks the top 20 confusion pairs across 3 completed runs.
  Figure path: `outputs/bilstm/analysis/confusion_stability.png`
  Source artifacts: `outputs/bilstm/final_runs/run_01_seed_42/confusion_matrix.csv`, `outputs/bilstm/final_runs/run_02_seed_1337/confusion_matrix.csv`, `outputs/bilstm/final_runs/run_03_seed_2024/confusion_matrix.csv`
- **calibration_comparison** (analysis): Cross-model calibration comparison across all evaluated models. Computed from cross-model calibration comparison using one representative run per model. Uses the CLINC150 test split. Metrics shown: ECE, MCE, Brier score, NLL.
  Figure path: `outputs/shared/analysis/calibration_comparison.png`
  Source artifacts: `outputs/mlp/analysis/calibration_metrics.json`, `outputs/text_cnn/analysis/calibration_metrics.json`, `outputs/bilstm/analysis/calibration_metrics.json`
- **error_taxonomy** (analysis): Cross-model comparison of error-taxonomy fractions across all evaluated models. Computed from cross-model error taxonomy comparison using representative runs. Uses the CLINC150 test split.
  Figure path: `outputs/shared/analysis/error_taxonomy_comparison.png`
  Source artifacts: `outputs/mlp/analysis/error_taxonomy.json`, `outputs/text_cnn/analysis/error_taxonomy.json`, `outputs/bilstm/analysis/error_taxonomy.json`
- **confidence_vs_accuracy** (analysis): Accuracy across confidence buckets across all evaluated models. Computed from cross-model confidence stratification comparison using representative runs. Uses the CLINC150 test split.
  Figure path: `outputs/shared/analysis/confidence_accuracy_comparison.png`
  Source artifacts: `outputs/mlp/analysis/confidence_stratification.json`, `outputs/text_cnn/analysis/confidence_stratification.json`, `outputs/bilstm/analysis/confidence_stratification.json`
- **length_slice** (analysis): Cross-model accuracy comparison by utterance length across all evaluated models. Computed from cross-model slice comparison using representative runs. Uses the CLINC150 test split.
  Figure path: `outputs/shared/analysis/length_slice_comparison.png`
  Source artifacts: `outputs/mlp/analysis/length_slice_analysis.json`, `outputs/text_cnn/analysis/length_slice_analysis.json`, `outputs/bilstm/analysis/length_slice_analysis.json`
- **frequency_slice** (analysis): Cross-model accuracy comparison by class-frequency tier across all evaluated models. Computed from cross-model slice comparison using representative runs. Uses the CLINC150 test split.
  Figure path: `outputs/shared/analysis/frequency_slice_comparison.png`
  Source artifacts: `outputs/mlp/analysis/frequency_slice_analysis.json`, `outputs/text_cnn/analysis/frequency_slice_analysis.json`, `outputs/bilstm/analysis/frequency_slice_analysis.json`
- **cross_model_error_overlap** (analysis): Overlap between shared and model-specific prediction failures across all evaluated models. Computed from cross-model error overlap using one representative run per model. Uses the CLINC150 test split.
  Figure path: `outputs/shared/analysis/cross_model_error_overlap.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/final_predictions.csv`, `outputs/text_cnn/final_runs/run_03_seed_2024/final_predictions.csv`, `outputs/bilstm/final_runs/run_02_seed_1337/final_predictions.csv`

### Efficiency

- **model_efficiency_comparison** (aggregate): Cross-model efficiency comparison across all evaluated models. Uses aggregate statistics over 3 repeated runs per model. Uses CLINC150 artifacts. Metrics shown: parameter count, training time, inference throughput.
  Figure path: `outputs/shared/figures/model_efficiency_comparison.png`
  Source artifacts: `outputs/shared/efficiency_summary_table.json`

Total figures: 62
