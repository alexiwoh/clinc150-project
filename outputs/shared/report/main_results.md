## Main Results

Means and population standard deviations describe the recorded runs. They are descriptive dispersion measures, not confidence intervals or significance tests.

### Table 1: Main Model Comparison

*Aggregate over 3 runs (mean +/- std)*

| Model        | Test Accuracy     | Test Macro F1     | Test Precision    | Test Recall       |
| ------------ | ----------------- | ----------------- | ----------------- | ----------------- |
| TF-IDF + MLP | 0.8345 +/- 0.0063 | 0.8716 +/- 0.0029 | 0.8439 +/- 0.0059 | 0.9132 +/- 0.0014 |
| Text CNN     | 0.8210 +/- 0.0067 | 0.8645 +/- 0.0031 | 0.8299 +/- 0.0050 | 0.9153 +/- 0.0012 |
| BiLSTM       | 0.7744 +/- 0.0065 | 0.8283 +/- 0.0046 | 0.7944 +/- 0.0074 | 0.8844 +/- 0.0040 |

TF-IDF + MLP achieved the highest aggregate test macro F1 (0.8716 +/- 0.0029).

Related figure: `outputs/shared/figures/model_comparison_test_macro_f1.png`

### Table 2: OOS Detection Metrics

*Aggregate over 3 runs (mean +/- std)*

OOS precision/recall/F1 score argmax predictions in the supervised 151-class classifier.

| Model        | OOS Precision     | OOS Recall        | OOS F1            |
| ------------ | ----------------- | ----------------- | ----------------- |
| TF-IDF + MLP | 0.8877 +/- 0.0112 | 0.4667 +/- 0.0425 | 0.6103 +/- 0.0352 |
| Text CNN     | 0.9326 +/- 0.0145 | 0.3810 +/- 0.0375 | 0.5397 +/- 0.0357 |
| BiLSTM       | 0.8865 +/- 0.0203 | 0.2610 +/- 0.0406 | 0.4019 +/- 0.0511 |

TF-IDF + MLP achieved the highest aggregate argmax OOS F1 (0.6103 +/- 0.0352).

Related figure: `outputs/shared/figures/oos_metrics_comparison.png`

### Table 3: Extended Metrics

*Representative run only (single seed)*

| Model        | Macro F1 | Micro F1 | Weighted F1 | Macro Precision | Macro Recall |
| ------------ | -------- | -------- | ----------- | --------------- | ------------ |
| TF-IDF + MLP | 0.8742   | 0.8400   | 0.8326      | 0.8492          | 0.9122       |
| Text CNN     | 0.8628   | 0.8167   | 0.8027      | 0.8290          | 0.9137       |
| BiLSTM       | 0.8347   | 0.7816   | 0.7639      | 0.8037          | 0.8882       |

TF-IDF + MLP has the highest representative-run macro F1 (0.8742) in the extended-metrics table.

### Table 4: Calibration Metrics

*Representative run only (single seed)*

| Model        | ECE    | MCE    | Brier Score | NLL    |
| ------------ | ------ | ------ | ----------- | ------ |
| TF-IDF + MLP | 0.1027 | 0.2580 | 0.2532      | 0.7303 |
| Text CNN     | 0.0390 | 0.1772 | 0.2595      | 0.9015 |
| BiLSTM       | 0.1232 | 0.4381 | 0.3345      | 1.3284 |

Text CNN is the best calibrated representative run (ECE = 0.0390).

Related figure: `outputs/shared/analysis/calibration_comparison.png`

### Table 5: Efficiency Comparison

*Aggregate over 3 runs (mean +/- std)*

| Model        | Parameters | Training Time (s) | Inference (ms/example) | Throughput (ex/s)    |
| ------------ | ---------- | ----------------- | ---------------------- | -------------------- |
| TF-IDF + MLP | 5,197,975  | 82.69 +/- 6.85    | 0.0557 +/- 0.0048      | 18069.35 +/- 1459.73 |
| Text CNN     | 1,930,167  | 168.54 +/- 18.36  | 0.0353 +/- 0.0023      | 28455.69 +/- 1795.24 |
| BiLSTM       | 4,284,311  | 144.16 +/- 18.48  | 0.1281 +/- 0.0150      | 7909.59 +/- 862.64   |

Text CNN has the highest mean inference throughput (28456 ex/s).

Batched evaluation timing covers loader traversal, device transfer, forward pass, softmax and CPU result collection; it excludes preprocessing, checkpoint loading, array concatenation, metrics and artifact writes. The per-example figures describe batched throughput, with no controlled warmup or repeated timing trials; single-query deployment latency was not measured.

Related figure: `outputs/shared/figures/model_efficiency_comparison.png`

### Key Figures

![Aggregate test macro F1 comparison with error bars across all models. Data: aggregate over 3 repeated runs, CLINC150 test set.](../figures/model_comparison_test_macro_f1.png)

![Validation macro F1 progression during training for TF-IDF + MLP. Data: representative run, CLINC150.](../../mlp/figures/representative_val_macro_f1_curve.png)
