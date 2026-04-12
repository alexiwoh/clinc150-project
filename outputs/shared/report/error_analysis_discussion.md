## Error Analysis Discussion

### Error Taxonomy

- **TF-IDF + MLP**: dominant error category is `oos_as_inscope` (497 errors, 0.5648 of all errors) (outputs/shared/analysis/error_taxonomy_summary.json)
- **Text CNN**: dominant error category is `oos_as_inscope` (627 errors, 0.6372 of all errors) (outputs/shared/analysis/error_taxonomy_summary.json)
- **BiLSTM**: dominant error category is `oos_as_inscope` (672 errors, 0.5705 of all errors) (outputs/shared/analysis/error_taxonomy_summary.json)

All models share `oos_as_inscope` as the dominant error category, indicating that OOS false accepts are the primary failure mode across architectures.

Related figure: `outputs/shared/analysis/error_taxonomy_comparison.png`

### Calibration and Confidence

Best-calibrated model: Text CNN (ECE = 0.0374). Worst-calibrated: BiLSTM (ECE = 0.1125) (outputs/shared/analysis/calibration_summary.json).

Related figures: `outputs/mlp/analysis/reliability_diagram.png`, `outputs/mlp/analysis/confidence_histogram.png`, `outputs/text_cnn/analysis/reliability_diagram.png`, `outputs/text_cnn/analysis/confidence_histogram.png`, `outputs/bilstm/analysis/reliability_diagram.png`, `outputs/bilstm/analysis/confidence_histogram.png`, `outputs/shared/analysis/calibration_comparison.png`

### OOS Detection Deep Dive

Best OOS detector by AUROC: Text CNN (AUROC = 0.9523) (outputs/shared/analysis/oos_threshold_comparison.json). The main failure mode across all models is `oos_as_inscope` (false accepts).

Related figures: `outputs/mlp/analysis/oos_error_breakdown.png`, `outputs/text_cnn/analysis/oos_error_breakdown.png`, `outputs/bilstm/analysis/oos_error_breakdown.png`, `outputs/shared/analysis/oos_roc_comparison.png`, `outputs/shared/analysis/oos_error_comparison.png`

### Cross-Model Error Overlap

Of 5,500 test examples:
- All models correct: 4,051 (0.7365)
- All models wrong: 608 (0.1105)
- Model-specific errors: 464 (0.0844)
- Partial overlap: 377 (0.0685)

Of universally wrong examples, 0.3668 predict the same incorrect class (outputs/shared/analysis/cross_model_error_comparison.json). See also `outputs/shared/analysis/universally_misclassified_examples.csv`.

Related figure: `outputs/shared/analysis/cross_model_error_overlap.png`

### Worst-Class Analysis

Classes consistently worst across all models (shared): `income`, `oos`, `order`, `recipe`, `smart_home`, `yes` (outputs/shared/analysis/worst_classes_comparison.json).

Model-specific worst classes:
- **TF-IDF + MLP**: `calculator`, `calendar`, `how_busy`, `shopping_list`, `w2`
- **Text CNN**: `bill_balance`, `order_status`, `translate`, `travel_suggestion`, `who_do_you_work_for`
- **BiLSTM**: `current_location`, `goodbye`, `weather`

Related figures: `outputs/mlp/analysis/worst_classes_confusion_heatmap.png`, `outputs/text_cnn/analysis/worst_classes_confusion_heatmap.png`, `outputs/bilstm/analysis/worst_classes_confusion_heatmap.png`

### Confused Pairs (OOS False-Accept Targets)

Intents that persistently capture OOS examples across 2+ models: `recipe`, `directions`, `income`, `smart_home`, `travel_suggestion`, `restaurant_suggestion` (outputs/shared/most_confused_pairs_table.json). These span multiple domains (travel, food, finance), suggesting OOS queries are topically diverse.

Related figures: `outputs/mlp/figures/representative_top_confused_pairs.png`, `outputs/text_cnn/figures/representative_top_confused_pairs.png`, `outputs/bilstm/figures/representative_top_confused_pairs.png`

### Length and Frequency Slices

Short-query accuracy per model (representative run):
- **TF-IDF + MLP**: 0.8291
- **Text CNN**: 0.8583
- **BiLSTM**: 0.8097

(outputs/shared/analysis/error_analysis_summary.json)

Related figure: `outputs/shared/analysis/length_slice_comparison.png`

### Confusion Matrix

![Test-set confusion matrix across 151 intent classes including OOS for TF-IDF + MLP. Data: representative run, CLINC150 test set.](outputs/mlp/figures/representative_confusion_matrix.png)
