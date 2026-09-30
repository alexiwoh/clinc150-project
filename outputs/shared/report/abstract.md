## Abstract

Evaluate lightweight deep learning models for intent classification and supervised out-of-scope (OOS) classification on the CLINC150 dataset.

Comparison of a sparse-feature baseline (TF-IDF + MLP), a convolutional sequence model (Text CNN), and a recurrent sequence model (BiLSTM).

OOS is a supervised 151st class with labeled training examples. These results describe this explicit-class setting and do not establish general open-set detection.

Three models are compared: TF-IDF + MLP, Text CNN, BiLSTM.

**Headline Result**: TF-IDF + MLP achieved the highest aggregate test macro F1 of 0.8716 +/- 0.0029 across 3 repeated runs (outputs/shared/model_comparison_aggregate.json).

**OOS Classification**: TF-IDF + MLP achieved the highest aggregate argmax OOS F1 of 0.6103 +/- 0.0352 (outputs/shared/oos_summary_table.json).

**Key Finding**: Inspect representative OOS false accepts and same-domain confusions as heuristic error groups. Preserve official benchmark labels and splits. Select any proposed calibration or rejection threshold on validation data before final test evaluation. (outputs/shared/analysis/error_analysis_summary.json).

Means and population standard deviations describe the recorded runs. They are descriptive dispersion measures, not confidence intervals or significance tests.
