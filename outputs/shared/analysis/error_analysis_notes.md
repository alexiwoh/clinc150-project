# Error Analysis Discussion Notes

## Overview

Among the configured pipelines, **mlp** has the highest mean test macro F1 of 0.8716 (±0.0029) across repeated runs. The following calibration, OOS probability, confusion and overlap diagnostics use one validation-selected representative run per model. They describe observed errors without establishing architecture causality or variability across seeds.

## Calibration and Confidence

The lowest representative-run ECE is from **text_cnn** (ECE = 0.0390); the highest is from **bilstm** (ECE = 0.1232). See `outputs/shared/analysis/calibration_summary.json` for full calibration comparison.

## OOS Detection Quality

The highest representative-run OOS probability AUROC is from **text_cnn** (AUROC = 0.9491, AUPR = 0.8273). OOS is a supervised 151st class. These test-set ROC points are diagnostics, not deployable rejection thresholds or general open-set evidence. See `outputs/shared/analysis/oos_threshold_comparison.json` for threshold comparison.

## Systematic Confusions

- **mlp**: dominant heuristic error category is `oos_as_inscope`
- **text_cnn**: dominant heuristic error category is `oos_as_inscope`
- **bilstm**: dominant heuristic error category is `oos_as_inscope`

See `outputs/shared/analysis/error_taxonomy_summary.json` for per-model taxonomy.
Same-domain and short-query tags do not establish semantic similarity or query ambiguity.

## Model-Specific and Shared Errors

Across all models, **600** examples (10.9%) are misclassified by all three models, while **489** examples (8.9%) are unique to a single model. Of universally wrong examples, 36.3% predict the same incorrect class. See `outputs/shared/analysis/cross_model_error_comparison.json` for overlap analysis.

## Limitations

- Qualitative analysis uses a single representative run per model, not all seeds.
- Only in-scope intent classes are balanced; supervised OOS has different train/test supports.
- No interpretability analysis (attention, saliency) was performed.
- Post-hoc calibration (e.g., temperature scaling) was not applied.
- Length slices use whitespace tokenization; scope slices compare supervised in-scope and OOS classes, not training frequency.

## Recommendations

- **Top recommendation**: Inspect representative OOS false accepts and same-domain confusions as heuristic error groups. Preserve official benchmark labels and splits. Select any proposed calibration or rejection threshold on validation data before final test evaluation.
- Investigate representative OOS false accepts and same-domain groups without changing benchmark labels.
- Evaluate any additional OOS collection as a separate experiment with explicit provenance.
- Choose calibration and threshold settings on validation data, then freeze them before test evaluation.
- Compare additional independent runs before claiming improved calibration or rejection performance.
