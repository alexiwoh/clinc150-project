## Abstract

Evaluate lightweight deep learning models for intent classification and out-of-scope (OOS) detection on the CLINC150 dataset.

Comparison of a sparse-feature baseline (TF-IDF + MLP), a convolutional sequence model (Text CNN), and a recurrent sequence model (BiLSTM).

Three models are compared: TF-IDF + MLP, Text CNN, BiLSTM.

**Headline Result**: TF-IDF + MLP achieved the highest aggregate test macro F1 of 0.8727 +/- 0.0031 across 3 repeated runs (outputs/shared/model_comparison_aggregate.json).

**OOS Detection**: TF-IDF + MLP achieved the highest aggregate OOS F1 of 0.6261 +/- 0.0348 (outputs/shared/oos_summary_table.json).

**Key Finding**: Focus on improving OOS detection and addressing semantically ambiguous intent pairs within the same domain. Consider intent merging for persistently confused pairs and confidence thresholding for high-confidence errors. (outputs/shared/analysis/error_analysis_summary.json).
