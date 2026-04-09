## OOS Detection Results

### Aggregate OOS Metrics

*Aggregate over 3 runs (mean +/- std)*

| Model        | OOS Precision     | OOS Recall        | OOS F1            |
| ------------ | ----------------- | ----------------- | ----------------- |
| TF-IDF + MLP | 0.8840 +/- 0.0132 | 0.4867 +/- 0.0444 | 0.6261 +/- 0.0348 |
| Text CNN     | 0.9414 +/- 0.0102 | 0.3563 +/- 0.0401 | 0.5154 +/- 0.0416 |
| BiLSTM       | 0.8817 +/- 0.0104 | 0.2947 +/- 0.0236 | 0.4413 +/- 0.0273 |

### OOS Threshold Analysis

*Representative run only (single seed)*

| Model        | AUROC  | AUPR   | FPR@95TPR | FPR@90TPR | MSP AUROC | MSP AUPR |
| ------------ | ------ | ------ | --------- | --------- | --------- | -------- |
| TF-IDF + MLP | 0.9415 | 0.7984 | 0.2389    | 0.1531    | 0.8901    | 0.6180   |
| Text CNN     | 0.9523 | 0.8403 | 0.2082    | 0.1387    | 0.9059    | 0.6560   |
| BiLSTM       | 0.9436 | 0.8055 | 0.2309    | 0.1436    | 0.8747    | 0.5453   |

Text CNN achieved the highest AUROC (0.9523), indicating the best threshold-independent OOS discrimination (outputs/shared/analysis/oos_threshold_comparison.json). The explicit OOS class probability method substantially outperforms the maximum softmax probability (MSP) baseline across all models.

### OOS Distribution Challenge

The test split contains ~18% OOS examples versus ~1.6% in training, creating a severe distribution shift that challenges all models. All models show low OOS recall, indicating difficulty detecting OOS examples despite reasonable OOS precision.

### False-Accept Patterns

- **TF-IDF + MLP**: 497 false accepts. Top capturing intents: `w2` (19), `calculator` (19), `recipe` (16)
- **Text CNN**: 627 false accepts. Top capturing intents: `travel_suggestion` (27), `directions` (23), `todo_list` (21)
- **BiLSTM**: 672 false accepts. Top capturing intents: `smart_home` (39), `income` (30), `current_location` (28)

![OOS ROC Comparison](outputs/shared/analysis/oos_roc_comparison.png)
