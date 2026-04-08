# Error Analysis Discussion Notes

## Overview

The best-performing model overall is **mlp** with aggregate test macro F1 of 0.8727 (±0.0031) across repeated runs. This analysis examines calibration, OOS detection, systematic confusions, and architecture-specific failure modes to understand not just how well each model performs, but why and where it fails.

## Calibration and Confidence

The best-calibrated model is **text_cnn** (ECE = 0.0374), while the worst-calibrated is **bilstm** (ECE = 0.1125). See `outputs/shared/analysis/calibration_summary.json` for full calibration comparison.

## OOS Detection Quality

The best OOS detector is **text_cnn** with AUROC = 0.9523 and AUPR = 0.8403. See `outputs/shared/analysis/oos_threshold_comparison.json` for threshold comparison.

## Systematic Confusions

- **mlp**: dominant error category is `oos_as_inscope`
- **text_cnn**: dominant error category is `oos_as_inscope`
- **bilstm**: dominant error category is `oos_as_inscope`

See `outputs/shared/analysis/error_taxonomy_summary.json` for per-model taxonomy.

## Architecture-Specific vs Shared Errors

Across all models, **608** examples (11.1%) are misclassified by all three models, while **464** examples (8.4%) are unique to a single model. Of universally wrong examples, 36.7% predict the same incorrect class. See `outputs/shared/analysis/cross_model_error_comparison.json` for overlap analysis.

## Limitations

- Qualitative analysis uses a single representative run per model, not all seeds.
- CLINC150 is balanced; real-world class distributions may differ significantly.
- No interpretability analysis (attention, saliency) was performed.
- Post-hoc calibration (e.g., temperature scaling) was not applied.
- Length and frequency slicing uses simple whitespace tokenization.

## Recommendations

- **Top recommendation**: Focus on improving OOS detection and addressing semantically ambiguous intent pairs within the same domain. Consider intent merging for persistently confused pairs and confidence thresholding for high-confidence errors.
- Collect more OOS training examples to reduce false accepts.
- Consider merging or relabeling persistently confused intent pairs within the same domain.
- Apply confidence thresholding in deployment to flag uncertain predictions for human review.
- Evaluate temperature scaling to improve calibration without retraining.