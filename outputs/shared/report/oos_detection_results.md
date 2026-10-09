## OOS Detection Results

OOS is a supervised 151st class with labeled training examples. These results describe this explicit-class setting and do not establish general open-set detection.

### Aggregate OOS Metrics

*Aggregate over 3 runs (mean +/- std)*

| Model        | OOS Precision     | OOS Recall        | OOS F1            |
| ------------ | ----------------- | ----------------- | ----------------- |
| TF-IDF + MLP | 0.8877 +/- 0.0112 | 0.4667 +/- 0.0425 | 0.6103 +/- 0.0352 |
| Text CNN     | 0.9326 +/- 0.0145 | 0.3810 +/- 0.0375 | 0.5397 +/- 0.0357 |
| BiLSTM       | 0.8865 +/- 0.0203 | 0.2610 +/- 0.0406 | 0.4019 +/- 0.0511 |

TF-IDF + MLP achieved the highest aggregate argmax OOS F1 (0.6103 +/- 0.0352).

Related figure: `outputs/shared/figures/oos_metrics_comparison.png`

### OOS Threshold Analysis

*Representative run only (single seed)*

| Model        | AUROC  | AUPR   | FPR@95TPR | FPR@90TPR | MSP AUROC | MSP AUPR |
| ------------ | ------ | ------ | --------- | --------- | --------- | -------- |
| TF-IDF + MLP | 0.9415 | 0.7984 | 0.2389    | 0.1531    | 0.8901    | 0.6180   |
| Text CNN     | 0.9491 | 0.8273 | 0.2322    | 0.1458    | 0.9029    | 0.6423   |
| BiLSTM       | 0.9455 | 0.8050 | 0.2367    | 0.1478    | 0.8799    | 0.5819   |

Text CNN has the highest representative-run OOS-probability AUROC (0.9491) among the compared runs. (outputs/shared/analysis/oos_threshold_comparison.json). This probability ranking measures different behavior from aggregate argmax OOS F1.

The MSP diagnostic uses `1 - max(p)` over all 151 softmax classes, including OOS. A confidently correct OOS prediction can therefore receive a low MSP OOS score. This is a diagnostic of the supervised classifier, rather than a conventional in-scope-only MSP detector.

FPR@95TPR and FPR@90TPR are descriptive points on the representative test ROC curve. They do not establish a deployable operating point: thresholds and calibration must be selected on validation data before separate held-out evaluation.

Related figures: `outputs/shared/analysis/oos_roc_comparison.png`, `outputs/shared/analysis/oos_pr_comparison.png`

### OOS Distribution Challenge

The test split contains ~18% OOS examples versus ~1.6% in training, with 250 labeled OOS training examples and 1,000 OOS test examples. Interpret the observed precision and recall within this benchmark support pattern.

### False-Accept Patterns

- **TF-IDF + MLP**: 497 false accepts. Top capturing intents: `w2` (19), `calculator` (19), `recipe` (16)
- **Text CNN**: 636 false accepts. Top capturing intents: `directions` (33), `recipe` (25), `todo_list` (24)
- **BiLSTM**: 716 false accepts. Top capturing intents: `order` (48), `smart_home` (32), `whisper_mode` (27)

The representative-run false-accept counts identify in-scope intents that capture labeled OOS examples. (outputs/shared/analysis/oos_false_accept_comparison.json).

Related figure: `outputs/shared/analysis/oos_error_comparison.png`

![OOS ROC Comparison](../analysis/oos_roc_comparison.png)
