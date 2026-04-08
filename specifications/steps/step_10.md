Implement and verify Step 10: error analysis for the CLINC150 project.

Goal:
Perform a rigorous, structured error analysis across all three models (TF-IDF + MLP, Text CNN, BiLSTM) that goes beyond headline metrics. This step should uncover systematic failure patterns, quantify model calibration and uncertainty behavior, evaluate OOS detection quality under threshold variation, compare error profiles across architectures, and produce report-ready qualitative examples and discussion notes for Step 11. All analysis must be grounded in saved artifacts from Steps 7-9 with no model retraining or re-inference required.

Primary outcome:
At the end of this step, I want to have:
1. extended evaluation metrics (micro/weighted F1, calibration, entropy, margin) computed from saved artifacts
2. OOS threshold analysis with ROC/PR curves, AUROC, AUPR, and FPR@95TPR per model
3. a structured error taxonomy categorizing representative misclassifications by failure type
4. confidence-stratified error analysis showing where models are confidently wrong vs uncertainly correct
5. semantic confusion analysis identifying persistent intent-pair collisions across models and seeds
6. OOS-specific error breakdown separating false accepts from false rejects with confidence context
7. slice-based analysis by utterance length and class frequency
8. cross-model comparative error analysis identifying shared vs architecture-specific failure modes
9. per-class deep dive into worst-performing intents with root-cause annotations
10. curated representative examples ready for the report discussion section
11. a structured error analysis summary with quantitative claims tied to aggregate tables
12. machine-readable analysis artifacts and figures for Step 11 consumption

Important scope constraints:
- Do not retrain or re-run inference for any model
- All analysis must derive from saved artifacts: `final_predictions.csv`, `confidences.npz`, `per_class_metrics.json`, `confusion_matrix.csv`, `top_confusions.json`, `top_errors.json`, and Step 8/9 aggregate tables
- Keep aggregate statistical claims grounded in repeated-run mean/std from `model_comparison_aggregate.json`
- Use representative-run artifacts only for qualitative examples and within-model diagnostics
- Do not conflate representative-run examples with aggregate performance claims
- Do not fabricate class-level aggregate statistics from a single representative run
- Do not add model-interpretability methods (attention visualization, saliency maps, integrated gradients) unless a later step explicitly requests them
- Do not over-engineer interactive tooling; static figures and tables are sufficient

Out of scope for Step 10:
- model retraining or architecture changes based on error findings
- attention or saliency-based interpretability
- data augmentation or label correction
- interactive error-exploration dashboards
- post-hoc calibration (temperature scaling); report the calibration state as-is and note temperature scaling as a potential improvement for Step 12 or future work
- statistical significance tests beyond repeated-run mean/std (bootstrap CIs, paired tests are optional stretch items)

Cross-cutting invariants:
- aggregate figures and claims must come from aggregate repeated-run artifacts
- representative-run artifacts are for qualitative illustration only
- all plots must respect the saved label ordering from `label_order.json`
- all analysis code must consume saved artifacts, not raw training code internals
- figure filenames must make analysis type and scope obvious
- the OOS class identity and index must come from saved metadata, not hardcoded

Cross-step ownership and data flow:
- Step 7 owns:
  - per-run artifacts (`final_predictions.csv`, `confidences.npz`, `per_class_metrics.json`, etc.)
  - per-model aggregate metrics
  - representative-run selection
- Step 8 owns:
  - schema validation and provenance
  - cross-model comparison tables (`model_comparison_aggregate.json`, `oos_summary_table.json`)
- Step 9 owns:
  - representative-run figures
  - aggregate comparison figures
  - figure manifest
  - error analysis handoff bundles (`error_analysis_handoff.json`)
  - `most_confused_pairs_table.json`
  - `representative_examples_index.json`
  - `representative_misclassifications.csv` per model
- Step 10 owns:
  - extended metric computation (calibration, micro/weighted F1, entropy, margin, OOS curves)
  - structured error taxonomy and categorization
  - confidence-stratified and slice-based analysis
  - cross-model error comparison
  - curated report examples and discussion notes
  - error analysis summary artifacts
- Data flow summary:
  - Step 9 handoff -> `error_analysis_handoff.json` per model, shared tables and indices
  - Step 7 per-run artifacts -> `confidences.npz`, `final_predictions.csv` for extended metric computation
  - Step 10 -> extended metrics, analysis artifacts, curated examples, discussion notes for Step 11

Shared protocol metadata and canonical naming:
- Canonical model IDs:
  - `mlp`
  - `text_cnn`
  - `bilstm`
- Canonical display names:
  - `TF-IDF + MLP`
  - `Text CNN`
  - `BiLSTM`
- Canonical model order for shared tables and plots:
  - `mlp`
  - `text_cnn`
  - `bilstm`
- Every machine-readable artifact must include:
  - `schema_version`
  - `protocol_version`
- Every machine-readable path reference must be repo-relative and must resolve under the repository root

High-level design requirement:
Step 10 should produce two complementary deliverable families:
- quantitative extended analysis artifacts (calibration metrics, OOS curves, slice tables, extended metric tables) that strengthen the numerical results chapter
- qualitative analysis artifacts (error taxonomy, curated examples, discussion notes) that support the discussion and limitations sections of the report

The analysis path must be fully artifact-driven:
- resolve the representative run from `error_analysis_handoff.json` per model
- load `confidences.npz` from the representative run directory for calibration, entropy, and OOS threshold analysis
- load `final_predictions.csv` from the representative run for text-level analysis
- load aggregate tables from `outputs/shared/` for cross-model statistical claims
- generate new analysis artifacts and figures without any model or data-loader dependency

Recommended execution story:
- Use a dedicated Step 10 entry point such as `scripts/run_error_analysis.py`
- The entry point should accept an optional `--models` argument to run analysis for specific models or default to all three
- Reuse shared analysis logic in a module such as `src/analysis/error_analysis.py` or `src/error_analysis.py`
- The script should read `error_analysis_handoff.json` for each model to resolve all artifact paths
- Keep the analysis module importable and testable independent of the script entry point
- The pipeline runner `scripts/run_model_pipeline.py` may optionally invoke error analysis after Step 9, but Step 10 should also be runnable standalone

Recommended default implementation path:
- create a new analysis module under `src/` (e.g. `src/error_analysis.py` or `src/analysis/error_analysis.py`)
- keep calibration, OOS threshold, and confidence utilities as pure functions that accept numpy arrays
- keep taxonomy and curation logic separate from metric computation
- reuse `src/metrics/classification_metrics.py` where applicable (e.g. for recomputing micro/weighted F1 from predictions)
- place new figures under `outputs/<model_name>/analysis/` for per-model outputs and `outputs/shared/analysis/` for cross-model outputs
- save machine-readable summaries as JSON alongside figures

Implementation requirements:

A. Preflight validation
- Before running any analysis, validate that all required input artifacts exist
- For each model, verify the following resolve from `error_analysis_handoff.json`:
  - `confidences.npz` exists in the representative run directory
  - `final_predictions.csv` exists and has expected columns: `text`, `true_label_id`, `true_label_name`, `predicted_label_id`, `predicted_label_name`, `run_id`, `max_confidence`
  - `per_class_metrics.json` exists and contains `classes` with `label_name`, `precision`, `recall`, `f1`, `support`, `is_oos`
  - `confusion_matrix.csv` exists
  - `top_confusions.json` exists
  - `top_errors.json` exists
  - `label_order.json` exists
- Verify shared artifacts exist:
  - `outputs/shared/model_comparison_aggregate.json`
  - `outputs/shared/oos_summary_table.json`
  - `outputs/shared/most_confused_pairs_table.json`
  - `outputs/shared/representative_examples_index.json`
- If any required artifact is missing, fail with a clear error message identifying the missing file
- Do not silently skip analysis for a model with incomplete artifacts

B. Compute extended classification metrics
- For each model's representative run, compute and save additional metrics not currently in `test_metrics.json`:
  - `micro_f1`: micro-averaged F1 across all classes
  - `weighted_f1`: support-weighted F1 across all classes
  - `micro_precision`, `micro_recall`: micro-averaged precision and recall
  - `weighted_precision`, `weighted_recall`: support-weighted precision and recall
- Compute these from `final_predictions.csv` using the true and predicted label columns
- Save per-model extended metrics to `outputs/<model_name>/analysis/extended_test_metrics.json`
- Save a cross-model extended metrics comparison table to `outputs/shared/analysis/extended_metrics_comparison.json`
- The cross-model table must use representative-run values and must clearly state that these are single-run values, not aggregate means
- Include the original macro metrics from `test_metrics.json` alongside the new metrics for a complete single-table view

B2. Extended metrics artifact contract
- `outputs/<model_name>/analysis/extended_test_metrics.json` must include:
  - `schema_version`, `protocol_version`
  - `model_id`, `model_name`
  - `representative_run_id`
  - `macro_f1`, `micro_f1`, `weighted_f1`
  - `macro_precision`, `macro_recall`
  - `micro_precision`, `micro_recall`
  - `weighted_precision`, `weighted_recall`
  - `accuracy`
  - `source_artifact`: repo-relative path to `final_predictions.csv`

C. Compute confidence and calibration metrics
- For each model's representative run, load `confidences.npz` which contains:
  - `probabilities`: full softmax probability matrix, shape `(n_examples, n_classes)`
  - `predictions`: predicted class indices
  - `targets`: true class indices
  - `label_names`: ordered label name array
- Compute the following calibration metrics:
  - Expected Calibration Error (ECE) using equal-width bins (default 15 bins)
  - Maximum Calibration Error (MCE): worst-bin calibration gap
  - Brier score (multi-class): mean squared error between one-hot targets and predicted probabilities
  - Negative log-likelihood (NLL): mean cross-entropy loss on the test set
- Generate the following calibration figures per model:
  - reliability diagram (calibration curve): binned accuracy vs binned mean confidence with bin counts
  - confidence histogram: distribution of max-class confidence for correct vs incorrect predictions
- Save calibration metrics to `outputs/<model_name>/analysis/calibration_metrics.json`
- Save calibration figures to:
  - `outputs/<model_name>/analysis/reliability_diagram.png`
  - `outputs/<model_name>/analysis/confidence_histogram.png`
- Generate one shared cross-model calibration comparison figure:
  - `outputs/shared/analysis/calibration_comparison.png` showing reliability curves for all three models overlaid
- Generate a shared calibration summary table:
  - `outputs/shared/analysis/calibration_summary.json` with ECE, MCE, Brier, and NLL per model

C2. Calibration metric conventions
- ECE formula: weighted average of |accuracy_b - confidence_b| over bins b, weighted by bin sample count / total samples
- Use equal-width bins over [0, 1] for the confidence axis
- Report the bin count used (default 15) in the saved artifact
- Brier score: use the multi-class formulation, mean of sum of squared differences between predicted probability vector and one-hot target vector
- NLL: use the standard cross-entropy, -log(p_true_class), averaged over examples
- Save all metrics as raw floats, not percentages

C3. Confidence distribution statistics
- For each model's representative run, also compute:
  - mean and std of max confidence for correct predictions
  - mean and std of max confidence for incorrect predictions
  - prediction entropy: -sum(p * log(p)) for each example, then report mean and std
  - confidence margin: (max_prob - second_max_prob) for each example, then report mean and std for correct vs incorrect
- Save these to the same `calibration_metrics.json` under a `confidence_statistics` sub-object

D. Compute OOS threshold analysis
- For each model's representative run, evaluate OOS detection as a binary classification problem
- Use the OOS class probability from `confidences.npz` as the positive-class score
- Determine the OOS class index from `label_order.json` (the class where `is_oos` is true)
- Compute the following:
  - ROC curve (FPR vs TPR) for OOS detection
  - PR curve (precision vs recall) for OOS detection
  - AUROC: area under the ROC curve
  - AUPR: area under the precision-recall curve
  - FPR@95TPR: false positive rate at 95% true positive rate
  - FPR@90TPR: false positive rate at 90% true positive rate
- Generate the following figures per model:
  - `outputs/<model_name>/analysis/oos_roc_curve.png`
  - `outputs/<model_name>/analysis/oos_pr_curve.png`
- Generate shared cross-model OOS threshold figures:
  - `outputs/shared/analysis/oos_roc_comparison.png` with all three models overlaid
  - `outputs/shared/analysis/oos_pr_comparison.png` with all three models overlaid
- Save OOS threshold metrics to `outputs/<model_name>/analysis/oos_threshold_metrics.json`
- Save a shared OOS threshold comparison table to `outputs/shared/analysis/oos_threshold_comparison.json`

D2. OOS threshold metric conventions
- Define the OOS detection task as: given the softmax probability for the OOS class, classify an example as OOS or in-scope
- True positives: OOS examples correctly identified as OOS
- False positives: in-scope examples incorrectly identified as OOS
- The ROC and PR curves should sweep the OOS probability threshold from 0 to 1
- Report AUROC and AUPR as raw floats in [0, 1]
- FPR@95TPR: the FPR when the threshold is set so that 95% of true OOS examples are correctly detected

D3. Alternative OOS scoring methods
- The primary OOS score is the softmax probability assigned to the OOS class (the explicit-class approach)
- Optionally, also compute a maximum softmax probability (MSP) baseline: classify as OOS when max(softmax) < threshold
- If MSP is computed, save its AUROC and AUPR alongside the explicit-class metrics for comparison
- Clearly label which method each metric corresponds to in the saved artifact

E. Structured error taxonomy
- For each model's representative run, classify every misclassified example into one of the following error categories:
  - `oos_as_inscope`: true label is OOS, predicted as an in-scope intent
  - `inscope_as_oos`: true label is in-scope, predicted as OOS
  - `near_semantic_confusion`: true and predicted labels are semantically related intents (within the same domain or with overlapping vocabulary)
  - `cross_domain_confusion`: true and predicted labels come from unrelated intent domains
  - `short_query_ambiguity`: misclassified example has very few tokens (threshold: 5 tokens or fewer after whitespace split)
- The `near_semantic_confusion` vs `cross_domain_confusion` distinction requires a domain or relatedness mapping
- Recommended approach: use the CLINC150 domain groupings (the dataset organizes 150 intents into 10 domains of 15 intents each) to determine whether two intents share a domain
- If the domain mapping is not already saved as an artifact, build it from the dataset metadata and save it as `outputs/shared/analysis/intent_domain_mapping.json`
- An example may belong to multiple categories (e.g. both `oos_as_inscope` and `short_query_ambiguity`); assign the primary category based on the order above, then record secondary tags
- Save per-model taxonomy results to `outputs/<model_name>/analysis/error_taxonomy.json`
- Save a cross-model taxonomy summary to `outputs/shared/analysis/error_taxonomy_summary.json`

E2. Error taxonomy artifact contract
- `outputs/<model_name>/analysis/error_taxonomy.json` must include:
  - `schema_version`, `protocol_version`
  - `model_id`, `model_name`, `representative_run_id`
  - `total_misclassified`: total number of misclassified examples
  - `total_test_examples`: total test set size
  - `error_rate`: `total_misclassified / total_test_examples`
  - `categories`: object mapping each category name to:
    - `count`: number of examples in this category
    - `fraction_of_errors`: this category's count / total misclassified
    - `fraction_of_test_set`: this category's count / total test examples
  - `examples`: list of misclassified examples (from `final_predictions.csv`) each with:
    - `text`, `true_label_name`, `predicted_label_name`, `max_confidence`, `primary_category`, `secondary_categories`
- `outputs/shared/analysis/error_taxonomy_summary.json` must include:
  - per-model category counts and fractions in the canonical model order
  - enough information to generate a grouped bar chart of error categories across models

E3. Error taxonomy figure
- Generate one cross-model error taxonomy comparison figure:
  - `outputs/shared/analysis/error_taxonomy_comparison.png`
  - grouped bar chart showing the fraction of errors in each category, grouped by model
  - use the canonical model order and display names

F. OOS error deep dive
- For each model, separate OOS-related errors into:
  - false accepts (`oos_as_inscope`): OOS examples misclassified as in-scope intents
  - false rejects (`inscope_as_oos`): in-scope examples misclassified as OOS
- For false accepts, identify:
  - which in-scope intents most frequently capture OOS examples (top 10 by count)
  - the mean and std of max confidence on these false accepts
  - representative text examples (top 5 by highest confidence, showing dangerously confident false accepts)
- For false rejects, identify:
  - which in-scope intents are most frequently rejected as OOS (top 10 by count)
  - the mean and std of max confidence on these false rejects
  - representative text examples (top 5 by lowest confidence, showing the most uncertain correct-class examples that got pushed to OOS)
- Save per-model OOS deep dive to `outputs/<model_name>/analysis/oos_error_deep_dive.json`
- Generate a per-model OOS error breakdown figure:
  - `outputs/<model_name>/analysis/oos_error_breakdown.png`
  - two-panel figure: top intents capturing OOS (false accepts) and top intents rejected as OOS (false rejects)
- Generate a shared cross-model OOS error comparison:
  - `outputs/shared/analysis/oos_false_accept_comparison.json` comparing which intents capture OOS across models
  - `outputs/shared/analysis/oos_error_comparison.png` showing false accept and false reject counts per model

G. Confidence-stratified error analysis
- For each model's representative run, stratify predictions into confidence bins and compute accuracy per bin
- Use 5 confidence strata: [0.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]
- For each stratum, compute:
  - number of examples
  - accuracy within that stratum
  - number of errors
  - proportion of total errors contributed by this stratum
- Identify high-confidence errors specifically:
  - examples where `max_confidence >= 0.8` and the prediction is wrong
  - these are the most dangerous errors in a deployed system
  - save the top 20 high-confidence errors per model ranked by confidence descending
- Save per-model confidence analysis to `outputs/<model_name>/analysis/confidence_stratification.json`
- Generate per-model confidence-accuracy figure:
  - `outputs/<model_name>/analysis/confidence_vs_accuracy.png`
  - bar chart showing accuracy per confidence bin with example counts annotated
- Generate shared cross-model figure:
  - `outputs/shared/analysis/confidence_accuracy_comparison.png` comparing accuracy-per-bin across models

G2. High-confidence error artifact contract
- The `confidence_stratification.json` must include a `high_confidence_errors` list with:
  - `text`, `true_label_name`, `predicted_label_name`, `max_confidence`, `prediction_entropy`, `confidence_margin`
- `prediction_entropy` is computed from the full softmax vector: -sum(p * log(p))
- `confidence_margin` is max_prob - second_max_prob
- These fields help distinguish overconfident narrow predictions from overconfident but uncertain-tailed predictions

H. Slice-based analysis
- Slice the representative-run test set along two dimensions and compute metrics per slice

H1. Utterance length slices
- Split test examples by token count (whitespace-split) into buckets:
  - short: 1-4 tokens
  - medium: 5-8 tokens
  - long: 9+ tokens
- For each bucket and each model, compute:
  - accuracy
  - error count and error rate
  - mean confidence on correct vs incorrect predictions
- Save per-model length analysis to `outputs/<model_name>/analysis/length_slice_analysis.json`
- Generate shared cross-model length-slice figure:
  - `outputs/shared/analysis/length_slice_comparison.png`
  - grouped bar chart: accuracy per length bucket, grouped by model

H2. Class frequency slices
- Split test examples by the training-set frequency of their true class:
  - low-frequency: classes in the bottom quartile of training support
  - medium-frequency: classes in the middle two quartiles
  - high-frequency: classes in the top quartile
- If exact training-set frequencies are not saved, use test-set `support` from `per_class_metrics.json` as a proxy (CLINC150 is balanced, so this is a reasonable approximation; note the assumption explicitly in the artifact)
- For each frequency bucket and each model, compute accuracy and error rate
- Save per-model frequency analysis to `outputs/<model_name>/analysis/frequency_slice_analysis.json`
- Generate shared cross-model frequency-slice figure:
  - `outputs/shared/analysis/frequency_slice_comparison.png`

I. Cross-model comparative error analysis
- For each pair of examples in the representative-run test sets, determine whether each model got it right or wrong
- Because representative runs may differ across models (different seeds), use `final_predictions.csv` keyed on `example_index` or `text` to align examples across models
- Categorize each test example into:
  - `all_correct`: all three models predict correctly
  - `all_wrong`: all three models predict incorrectly
  - `model_specific_error`: only one model gets it wrong
  - `partial_error`: exactly two models get it wrong
- Compute the count and fraction of the test set in each category
- For `all_wrong` examples, examine whether the three models make the same wrong prediction or different wrong predictions
- For `model_specific_error` examples, identify which model is uniquely wrong and whether those errors cluster in specific intents
- Save cross-model comparison to `outputs/shared/analysis/cross_model_error_comparison.json`
- Generate a cross-model error Venn or upset-style summary figure:
  - `outputs/shared/analysis/cross_model_error_overlap.png`
- Save the `all_wrong` examples as a curated list for report discussion:
  - `outputs/shared/analysis/universally_misclassified_examples.csv`

I2. Cross-model comparison artifact contract
- `cross_model_error_comparison.json` must include:
  - `schema_version`, `protocol_version`
  - `models_compared`: list of model IDs in canonical order
  - `representative_run_ids`: mapping of model ID to representative run ID
  - `total_test_examples`: count
  - `categories`: object with `all_correct`, `all_wrong`, `model_specific_error`, `partial_error` each having `count` and `fraction`
  - `model_specific_errors`: per-model counts of examples only that model got wrong
  - `all_wrong_agreement`: fraction of `all_wrong` examples where all three models predicted the same incorrect class

J. Per-class deep dive
- For each model, identify the bottom 15 classes by F1 from `per_class_metrics.json`
- For these worst classes, compute:
  - precision, recall, F1, and support (already available)
  - the top 3 classes that examples of this class are most often confused with (from `confusion_matrix.csv`)
  - whether the confusion targets share a domain with the source class
  - the mean confidence on misclassified examples of this class
- Save per-model deep dive to `outputs/<model_name>/analysis/worst_classes_deep_dive.json`
- Generate a per-model worst-classes summary figure:
  - `outputs/<model_name>/analysis/worst_classes_confusion_heatmap.png`
  - a small heatmap showing confusion counts between the bottom 15 classes and their top confusion targets

J2. Cross-model worst-class comparison
- Identify classes that appear in the bottom 15 for all three models
- Identify classes that appear in the bottom 15 for only one model
- Save this comparison to `outputs/shared/analysis/worst_classes_comparison.json`
- This helps distinguish inherently ambiguous intents from architecture-sensitive intents

K. Cross-run confusion stability
- For each model, compare `confusion_matrix.csv` across all available runs (not just the representative run)
- For each off-diagonal cell (true_class, predicted_class), record the confusion count across runs
- Identify:
  - stable confusions: pairs that appear in the top 20 confused pairs for all runs of a model
  - unstable confusions: pairs that appear in the top 20 for only one run
- Stable confusions indicate systematic model or data issues
- Unstable confusions indicate seed-sensitive behavior
- Save per-model stability analysis to `outputs/<model_name>/analysis/confusion_stability.json`
- Generate a per-model stability summary figure:
  - `outputs/<model_name>/analysis/confusion_stability.png`
  - scatter or bar chart showing confusion count variability (mean +/- std across runs) for the top pairs

L. Curate representative examples for the report
- Select a final set of curated misclassification examples for inclusion in the report discussion section
- Selection criteria:
  - 3-5 examples per error taxonomy category per model (aim for ~50-75 total curated examples across all models and categories)
  - prioritize examples that illustrate a generalizable pattern, not one-off oddities
  - include at least 2 high-confidence errors per model (from Section G)
  - include at least 2 OOS false accepts per model (from Section F)
  - include at least 2 semantically ambiguous in-scope confusions per model (from Section E)
- For each curated example, record:
  - `text`, `true_label_name`, `predicted_label_name`, `max_confidence`, `primary_category`, `model_id`
  - a brief annotation tag (1-3 words) describing why this example is illustrative (e.g. "semantic overlap", "short query", "confident false accept")
- Save curated examples to `outputs/shared/analysis/curated_report_examples.csv`
- Save a machine-readable version to `outputs/shared/analysis/curated_report_examples.json`

L2. Curated example quality bar
- Every curated example must be traceable to a specific model, run, and `final_predictions.csv` row
- Do not fabricate or paraphrase example text; use the exact text from the saved predictions
- Do not include duplicate examples across models unless the same text appears in multiple models' test sets and the comparison is the point

M. Error analysis summary and report notes
- Produce a structured summary artifact that ties together all analyses into a narrative-ready format
- The summary must include:
  - headline finding: which model has the best overall error profile and why
  - calibration finding: which model is best/worst calibrated and by how much (ECE comparison)
  - OOS finding: which model handles OOS best by AUROC and AUPR, and what the main OOS failure mode is
  - confusion finding: what the most persistent confused pairs are across all models and whether they share a domain
  - architecture comparison finding: what fraction of errors are shared across all models vs architecture-specific
  - length finding: whether short queries disproportionately cause errors
  - top recommendation: the single most impactful improvement for error reduction (e.g. more OOS training data, intent merging, confidence thresholding)
- Every claim in the summary must reference the specific artifact and metric it is based on
- Save to `outputs/shared/analysis/error_analysis_summary.json`
- Save a human-readable markdown version to `outputs/shared/analysis/error_analysis_notes.md`
- The markdown version should be structured as discussion-section draft notes, not a finished paper section

M2. Report notes format
- The markdown notes should use the following structure:
  - Overview: 2-3 sentence summary of overall error analysis findings
  - Calibration and Confidence: paragraph on model calibration comparison with ECE numbers
  - OOS Detection Quality: paragraph on OOS threshold behavior with AUROC/AUPR numbers
  - Systematic Confusions: paragraph on persistent confused pairs and domain analysis
  - Architecture-Specific vs Shared Errors: paragraph on cross-model error overlap
  - Limitations: bullet list of analysis limitations (single representative run for qualitative, balanced dataset may not reflect real-world class distribution, no interpretability analysis, etc.)
  - Recommendations: bullet list of actionable improvements

N. Build Step 11 handoff
- At the end of Step 10, save one `outputs/shared/analysis/step11_handoff.json` that references all Step 10 deliverables
- This handoff must include:
  - paths to all per-model analysis artifacts
  - paths to all shared analysis artifacts and figures
  - paths to curated examples
  - path to error analysis summary and notes
  - path to extended metrics comparison
  - path to calibration summary
  - path to OOS threshold comparison
  - path to cross-model error comparison
- Step 11 should be able to start report writing directly from this handoff without additional artifact discovery

N2. Update figure manifest
- Append all new Step 10 figures to `outputs/shared/figure_manifest.json`
- New figure-manifest entries must follow the same contract as Step 9 entries:
  - `schema_version`, `protocol_version`, `figure_path`, `figure_type`, `scope`, `model_name` if applicable, `source_artifact_paths`
- Use `figure_type` values such as:
  - `reliability_diagram`
  - `confidence_histogram`
  - `oos_roc_curve`
  - `oos_pr_curve`
  - `error_taxonomy`
  - `confidence_vs_accuracy`
  - `length_slice`
  - `frequency_slice`
  - `cross_model_error_overlap`
  - `confusion_stability`
  - `worst_classes_heatmap`
  - `oos_error_breakdown`

O. Figure and artifact expectations

Recommended per-model analysis outputs:
- `outputs/<model_name>/analysis/extended_test_metrics.json`
- `outputs/<model_name>/analysis/calibration_metrics.json`
- `outputs/<model_name>/analysis/reliability_diagram.png`
- `outputs/<model_name>/analysis/confidence_histogram.png`
- `outputs/<model_name>/analysis/oos_threshold_metrics.json`
- `outputs/<model_name>/analysis/oos_roc_curve.png`
- `outputs/<model_name>/analysis/oos_pr_curve.png`
- `outputs/<model_name>/analysis/error_taxonomy.json`
- `outputs/<model_name>/analysis/oos_error_deep_dive.json`
- `outputs/<model_name>/analysis/oos_error_breakdown.png`
- `outputs/<model_name>/analysis/confidence_stratification.json`
- `outputs/<model_name>/analysis/confidence_vs_accuracy.png`
- `outputs/<model_name>/analysis/length_slice_analysis.json`
- `outputs/<model_name>/analysis/frequency_slice_analysis.json`
- `outputs/<model_name>/analysis/worst_classes_deep_dive.json`
- `outputs/<model_name>/analysis/worst_classes_confusion_heatmap.png`
- `outputs/<model_name>/analysis/confusion_stability.json`
- `outputs/<model_name>/analysis/confusion_stability.png`

Recommended shared analysis outputs:
- `outputs/shared/analysis/extended_metrics_comparison.json`
- `outputs/shared/analysis/calibration_summary.json`
- `outputs/shared/analysis/calibration_comparison.png`
- `outputs/shared/analysis/oos_threshold_comparison.json`
- `outputs/shared/analysis/oos_roc_comparison.png`
- `outputs/shared/analysis/oos_pr_comparison.png`
- `outputs/shared/analysis/intent_domain_mapping.json`
- `outputs/shared/analysis/error_taxonomy_summary.json`
- `outputs/shared/analysis/error_taxonomy_comparison.png`
- `outputs/shared/analysis/oos_false_accept_comparison.json`
- `outputs/shared/analysis/oos_error_comparison.png`
- `outputs/shared/analysis/confidence_accuracy_comparison.png`
- `outputs/shared/analysis/length_slice_comparison.png`
- `outputs/shared/analysis/frequency_slice_comparison.png`
- `outputs/shared/analysis/cross_model_error_comparison.json`
- `outputs/shared/analysis/cross_model_error_overlap.png`
- `outputs/shared/analysis/universally_misclassified_examples.csv`
- `outputs/shared/analysis/worst_classes_comparison.json`
- `outputs/shared/analysis/curated_report_examples.csv`
- `outputs/shared/analysis/curated_report_examples.json`
- `outputs/shared/analysis/error_analysis_summary.json`
- `outputs/shared/analysis/error_analysis_notes.md`
- `outputs/shared/analysis/step11_handoff.json`

P. Validation checks and assertions
- Assert that `confidences.npz` contains `probabilities`, `predictions`, `targets`, and `label_names` arrays with compatible shapes
- Assert that the number of examples in `confidences.npz` matches `final_predictions.csv` row count
- Assert that predictions in `confidences.npz` match `final_predictions.csv` predicted label IDs
- Assert that ECE bin probabilities sum to approximately 1.0 across bins
- Assert that AUROC is in [0, 1] and not degenerate (exactly 0 or 1 suggests a bug)
- Assert that error taxonomy category counts sum to `total_misclassified`
- Assert that cross-model example alignment (by `example_index`) covers the full test set
- Assert that all figure paths in the updated figure manifest exist on disk
- Assert that all artifact paths in `step11_handoff.json` exist on disk
- Assert that curated examples trace back to real rows in `final_predictions.csv`
- Assert that OOS class index is consistent across all analysis code and matches `label_order.json`

Q. Test requirements
- Add or update tests covering:
  - ECE computation with known inputs and expected output
  - Brier score computation with known inputs
  - reliability diagram bin computation
  - OOS ROC/PR curve computation with a small synthetic example
  - error taxonomy categorization logic
  - confidence stratification bin assignment
  - utterance length bucketing
  - cross-model example alignment logic
  - Step 11 handoff generation and path validation
  - preflight validation failures for missing artifacts
- Tests should use small synthetic data (not full model outputs) for speed
- Extend existing test modules where possible rather than creating isolated test files

R. Definition of done
Step 10 is only complete if all of the following are true:
- extended metrics (micro/weighted F1) are computed and saved for all three models
- calibration metrics (ECE, MCE, Brier, NLL) and figures (reliability diagram, confidence histogram) exist for all three models
- OOS threshold metrics (AUROC, AUPR, FPR@95TPR) and figures (ROC, PR curves) exist for all three models
- a structured error taxonomy is saved for all three models with category counts and annotated examples
- OOS error deep dive artifacts exist for all three models with false accept and false reject analysis
- confidence-stratified analysis exists for all three models with high-confidence errors identified
- slice-based analysis (length, frequency) exists for all three models
- cross-model error comparison exists with overlap categories quantified
- per-class deep dive exists for all three models with worst-class confusion analysis
- cross-run confusion stability exists for all three models
- curated report examples are saved with annotation tags
- error analysis summary and discussion notes exist and reference specific metrics
- Step 11 handoff exists and all referenced paths resolve
- figure manifest is updated with all new figures
- no model retraining or re-inference was required
- all analysis code reads saved artifacts only

R2. Final Step 10 exit gate
- Before Step 10 is considered closed, confirm all of the following are true:
  - calibration comparison shows which model is best calibrated
  - OOS comparison shows which model is best at OOS detection by AUROC
  - error taxonomy reveals the dominant error category for each model
  - cross-model analysis quantifies the fraction of errors shared across all models
  - at least 50 curated examples are saved for the report
  - discussion notes contain at least one actionable recommendation
  - Step 11 can begin report writing directly from the Step 10 handoff

S. Deliverable quality bar
Step 10 should make the project analytically rigorous. A reader should be able to understand not just how well each model performs, but why it fails, where it fails, and how confidently it fails. The analysis should distinguish between errors that are fixable with better data versus errors that reflect fundamental limitations of the model architecture or task definition. The curated examples and discussion notes should be directly insertable into a report discussion section with minimal editing.

T. Suggested self-check after implementation
After coding, self-check against this exact checklist and confirm:
1. that all analysis reads `confidences.npz` and `final_predictions.csv` from saved artifacts, not from model re-inference
2. where extended metrics (micro/weighted F1) are saved per model and in the cross-model table
3. where calibration metrics and figures are saved per model and in the cross-model comparison
4. where OOS threshold metrics and curves are saved per model and in the cross-model comparison
5. that ECE is computed with the correct formula and reported bin count
6. that AUROC and AUPR are computed for OOS detection using the correct positive class
7. where the error taxonomy is saved and that category counts sum to total misclassified
8. where OOS false accept and false reject details are saved
9. where high-confidence errors are saved and how many per model
10. where length-slice and frequency-slice analyses are saved
11. that cross-model example alignment correctly handles different representative seeds
12. where the cross-model error overlap artifact and figure are saved
13. where curated report examples are saved and that they trace to real prediction rows
14. where the error analysis summary and discussion notes are saved
15. that the Step 11 handoff exists and all referenced paths resolve
16. that the figure manifest is updated with all new Step 10 figures
17. that aggregate claims in the summary reference `model_comparison_aggregate.json`, not representative-run-only metrics
18. that no model was retrained or re-evaluated during Step 10
