## Main Results

### Table 1: Main Model Comparison

*Aggregate over 3 runs (mean +/- std)*

| Model        | Test Accuracy     | Test Macro F1     | Test Precision    | Test Recall       |
| ------------ | ----------------- | ----------------- | ----------------- | ----------------- |
| TF-IDF + MLP | 0.8373 +/- 0.0066 | 0.8727 +/- 0.0031 | 0.8464 +/- 0.0065 | 0.9123 +/- 0.0015 |
| Text CNN     | 0.8176 +/- 0.0084 | 0.8635 +/- 0.0057 | 0.8278 +/- 0.0094 | 0.9163 +/- 0.0016 |
| BiLSTM       | 0.7751 +/- 0.0088 | 0.8284 +/- 0.0071 | 0.8005 +/- 0.0062 | 0.8780 +/- 0.0068 |

### Table 2: OOS Detection Metrics

*Aggregate over 3 runs (mean +/- std)*

| Model        | OOS Precision     | OOS Recall        | OOS F1            |
| ------------ | ----------------- | ----------------- | ----------------- |
| TF-IDF + MLP | 0.8840 +/- 0.0132 | 0.4867 +/- 0.0444 | 0.6261 +/- 0.0348 |
| Text CNN     | 0.9414 +/- 0.0102 | 0.3563 +/- 0.0401 | 0.5154 +/- 0.0416 |
| BiLSTM       | 0.8817 +/- 0.0104 | 0.2947 +/- 0.0236 | 0.4413 +/- 0.0273 |

### Table 3: Extended Metrics

*Representative run only (single seed)*

| Model        | Macro F1 | Micro F1 | Weighted F1 | Macro Precision | Macro Recall |
| ------------ | -------- | -------- | ----------- | --------------- | ------------ |
| TF-IDF + MLP | 0.8742   | 0.8400   | 0.8326      | 0.8492          | 0.9122       |
| Text CNN     | 0.8658   | 0.8211   | 0.8072      | 0.8315          | 0.9170       |
| BiLSTM       | 0.8354   | 0.7858   | 0.7727      | 0.8062          | 0.8838       |

### Table 4: Calibration Metrics

*Representative run only (single seed)*

| Model        | ECE    | MCE    | Brier Score | NLL    |
| ------------ | ------ | ------ | ----------- | ------ |
| TF-IDF + MLP | 0.1027 | 0.2580 | 0.2532      | 0.7303 |
| Text CNN     | 0.0374 | 0.1835 | 0.2567      | 0.8647 |
| BiLSTM       | 0.1125 | 0.4094 | 0.3235      | 1.2145 |

### Table 5: Efficiency Comparison

*Aggregate over 3 runs (mean +/- std)*

| Model        | Parameters | Training Time (s) | Inference (ms/example) | Throughput (ex/s)    |
| ------------ | ---------- | ----------------- | ---------------------- | -------------------- |
| TF-IDF + MLP | 5,197,975  | 60.62 +/- 10.80   | 0.0563 +/- 0.0013      | 17782.89 +/- 390.80  |
| Text CNN     | 1,930,167  | 136.48 +/- 7.70   | 0.0313 +/- 0.0021      | 32082.23 +/- 2138.80 |
| BiLSTM       | 4,284,311  | 91.74 +/- 23.20   | 0.1205 +/- 0.0150      | 8431.56 +/- 1079.83  |

### Key Figures

![Aggregate test macro F1 comparison with error bars across all models. Data: aggregate over 3 repeated runs, CLINC150 test set.](outputs/shared/figures/model_comparison_test_macro_f1.png)

![Validation macro F1 progression during training for TF-IDF + MLP. Data: representative run, CLINC150.](outputs/mlp/figures/representative_val_macro_f1_curve.png)
