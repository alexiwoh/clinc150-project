Implement and verify Step 11: final report support for the CLINC150 project.

Goal:
Assemble all project artifacts from Steps 1-10 into structured, report-ready section drafts for the final paper. This step should transform machine-readable JSON artifacts, saved figures, and analysis notes into a complete set of markdown section drafts that a report author can use directly with minimal additional editing. Every claim, table, and figure reference in the report drafts must trace back to a specific saved artifact. No new analysis, model training, or re-inference is required.

Primary outcome:
At the end of this step, I want to have:
1. a preflight-validated inventory confirming all upstream artifacts exist
2. a structured abstract / project summary with headline results
3. a dataset description section draft
4. a preprocessing summary section draft
5. a model architecture comparison section draft with per-model detail and a unified comparison table
6. an experimental setup section draft documenting the repeated-run protocol, metric definitions, and evaluation methodology
7. formatted main results tables (aggregate metrics, extended metrics, calibration, efficiency)
8. a dedicated OOS detection results section draft
9. a figure catalogue with captions and report-section assignments for every figure in the manifest
10. an error analysis discussion section draft synthesizing Step 10 findings
11. a representative examples section with curated qualitative misclassifications
12. a key findings narrative with every claim citing its source artifact
13. a limitations section combining analysis-level and project-level limitations
14. a future improvements section with prioritized recommendations
15. a reproducibility section documenting how to reproduce the full pipeline
16. a report structure manifest mapping every section to its source artifacts
17. one assembled markdown report draft combining all sections with a table of contents

Important scope constraints:
- Do not retrain or re-run inference for any model
- Do not compute new metrics or generate new figures; consume only what Steps 1-10 already produced
- All narrative text must cite the specific artifact it derives from
- Do not modify any upstream artifact files; Step 11 outputs go under `outputs/shared/report/` only
- Keep formatting consistent with the canonical metric conventions from earlier steps (raw ratios in [0.0, 1.0], mean +/- std for aggregate values)
- Do not invent claims not supported by saved artifacts
- Do not mix aggregate repeated-run metrics with representative-run values without explicit labeling

Out of scope for Step 11:
- actual final paper writing, polishing, or editing beyond structured section drafts
- LaTeX compilation or submission formatting
- literature review or related work authoring
- new visualizations or figures not already in the figure manifest
- new statistical analysis beyond what Step 10 produced
- interactive report viewers or dashboards

Cross-cutting invariants:
- aggregate claims must come from aggregate repeated-run artifacts (`model_comparison_aggregate.json`)
- representative-run artifacts are for qualitative illustration only and must be labeled as such
- all model references must use the canonical model order and display names
- all artifact paths in report outputs must be repo-relative
- every machine-readable output must include `schema_version` and `protocol_version`
- the OOS class identity must come from saved metadata, not hardcoded

Cross-step ownership and data flow:
- Steps 2-3 own:
  - dataset summary artifacts (`data/artifacts/dataset_summary.json`, `data/artifacts/dataset_summary.md`)
  - preprocessing summary (`data/artifacts/preprocessing_summary.json`)
- Steps 4-6 own:
  - model source code (`src/models/{mlp.py, text_cnn.py, bilstm.py}`)
- Step 7 owns:
  - frozen configs (`outputs/<model_name>/frozen_final_config.json`)
  - per-run artifacts
  - per-model aggregate metrics
  - representative-run selection
- Step 8 owns:
  - evaluation protocol manifest (`outputs/shared/evaluation_protocol.json`)
  - cross-model comparison tables (`model_comparison_aggregate.json`, `oos_summary_table.json`, `efficiency_summary_table.json`)
  - representative examples index (`outputs/shared/representative_examples_index.json`)
  - most confused pairs table (`outputs/shared/most_confused_pairs_table.json`)
- Step 9 owns:
  - figure manifest (`outputs/shared/figure_manifest.json`)
  - per-model representative-run figures
  - shared aggregate comparison figures
- Step 10 owns:
  - extended metrics, calibration, OOS threshold analysis
  - error taxonomy, confidence stratification, slice analysis
  - cross-model error comparison
  - curated report examples
  - error analysis summary and discussion notes
  - Step 11 handoff (`outputs/shared/analysis/step11_handoff.json`)
- Step 11 owns:
  - report section drafts under `outputs/shared/report/`
  - report structure manifest
  - assembled report draft
  - figure catalogue with captions
- Data flow summary:
  - Steps 2-10 -> saved artifacts (JSON, CSV, PNG, MD)
  - Step 10 -> `step11_handoff.json` (master index to analysis artifacts)
  - Step 11 -> report section drafts, assembled report, figure catalogue

Shared protocol metadata and canonical naming:
- Canonical model IDs:
  - `mlp`
  - `text_cnn`
  - `bilstm`
- Canonical display names:
  - `TF-IDF + MLP`
  - `Text CNN`
  - `BiLSTM`
- Canonical model order for all tables and narrative:
  - `mlp`
  - `text_cnn`
  - `bilstm`
- Every machine-readable artifact must include:
  - `schema_version`
  - `protocol_version`
- Every machine-readable path reference must be repo-relative and must resolve under the repository root

Recommended execution story:
- Use a dedicated Step 11 entry point such as `scripts/run_report_generation.py`
- The entry point should accept an optional `--sections` argument to generate specific sections or default to all
- Reuse shared report-generation logic in a module such as `src/report/generator.py` or a package `src/report/`
- The script should read `step11_handoff.json` to discover all Step 10 artifacts, plus known paths for Steps 2-9 artifacts
- Keep the report module importable and testable independent of the script entry point
- The pipeline runner `scripts/run_model_pipeline.py` may optionally invoke report generation after Step 10, but Step 11 should also be runnable standalone

Recommended default implementation path:
- create a new report module under `src/report/` (e.g. `src/report/__init__.py`, `src/report/generator.py`, `src/report/sections.py`, `src/report/tables.py`, `src/report/figures.py`)
- keep table formatting as pure functions that accept dictionaries and return formatted markdown strings
- keep section generation separate from the assembled report to allow independent section regeneration
- place all outputs under `outputs/shared/report/`
- save both machine-readable JSON and human-readable markdown for each section

Implementation requirements:

A. Preflight validation
- Before generating any report section, validate that all required input artifacts exist
- Verify the following Step 10 artifacts via `outputs/shared/analysis/step11_handoff.json`:
  - all per-model analysis artifact paths resolve
  - all shared analysis artifact paths resolve
- Verify the following Step 2-3 artifacts exist:
  - `data/artifacts/dataset_summary.json`
  - `data/artifacts/dataset_summary.md`
  - `data/artifacts/preprocessing_summary.json`
- Verify the following Step 7-8 per-model artifacts exist for each of `mlp`, `text_cnn`, `bilstm`:
  - `outputs/<model_name>/frozen_final_config.json`
  - `outputs/<model_name>/aggregate/aggregate_metrics.json`
- Verify the following Step 8 shared artifacts exist:
  - `outputs/shared/evaluation_protocol.json`
  - `outputs/shared/model_comparison_aggregate.json`
  - `outputs/shared/oos_summary_table.json`
  - `outputs/shared/efficiency_summary_table.json`
  - `outputs/shared/most_confused_pairs_table.json`
  - `outputs/shared/representative_examples_index.json`
- Verify the following Step 9 artifact exists:
  - `outputs/shared/figure_manifest.json`
- If any required artifact is missing, fail with a clear error message identifying the missing file
- Do not silently skip sections for missing artifacts

B. Abstract / project summary
- Generate a concise project summary suitable for an abstract or executive summary
- Content must include:
  - project goal: lightweight deep learning models for intent classification and OOS detection on CLINC150
  - project framing: comparison of sparse feature baselines, convolutional sequence models, and recurrent sequence models
  - models compared: TF-IDF + MLP, Text CNN, BiLSTM
  - headline result: best model by aggregate test macro F1 with mean and std (from `model_comparison_aggregate.json`)
  - OOS headline: best model by OOS F1 with mean and std (from `oos_summary_table.json`)
  - key finding: one-sentence summary from `error_analysis_summary.json` `top_recommendation`
- Save to `outputs/shared/report/abstract.md` and `outputs/shared/report/abstract.json`
- The JSON must include `schema_version`, `protocol_version`, and `source_artifacts` listing all artifacts consumed

B2. Abstract artifact contract
- `abstract.json` must include:
  - `schema_version`, `protocol_version`
  - `project_goal`: string
  - `project_framing`: string
  - `models`: list of model display names in canonical order
  - `headline_result`: object with `best_model`, `metric`, `mean`, `std`
  - `oos_headline`: object with `best_model`, `metric`, `mean`, `std`
  - `key_finding`: string
  - `source_artifacts`: list of repo-relative paths consumed

C. Dataset description
- Generate a dataset description section draft from `data/artifacts/dataset_summary.json`
- Content must include:
  - dataset source and subset (`clinc/clinc_oos`, `plus` subset)
  - split sizes: train, validation, test with exact counts
  - number of intent classes (151 including OOS)
  - OOS handling: explicit OOS class (label 42), OOS counts per split
  - class balance: in-scope classes are balanced; OOS has different support across splits
  - distribution quirks: test split has ~18% OOS vs ~1% in train (heavy distribution shift)
  - in-scope vs OOS count breakdown per split
- Format split sizes and OOS counts as a markdown table
- Save to `outputs/shared/report/dataset_description.md` and `outputs/shared/report/dataset_description.json`

C2. Dataset description artifact contract
- `dataset_description.json` must include:
  - `schema_version`, `protocol_version`
  - `dataset_source`, `subset`
  - `split_sizes`: object with `train`, `validation`, `test` counts
  - `num_classes`: integer (151)
  - `oos_label_name`, `oos_label_id`
  - `oos_counts`: object with per-split counts
  - `in_scope_counts`: object with per-split counts
  - `class_balance_note`: string
  - `distribution_quirks`: list of strings
  - `source_artifact`: repo-relative path to `data/artifacts/dataset_summary.json`

D. Preprocessing summary
- Generate a preprocessing section draft from `data/artifacts/preprocessing_summary.json`
- Content must include:
  - text cleaning policy: lowercase, strip, normalize whitespace; preserves punctuation, contractions, digits
  - tokenization: whitespace split
  - vocabulary: size (6161), special tokens (`<PAD>` = 0, `<UNK>` = 1), built from training data only
  - sequence handling: max length (20), padding/truncation policy, truncation rates per split
  - OOV rates per split
  - sequence length statistics: mean, median, p90, p95, max, min
  - TF-IDF configuration: max features (10000), n-gram range (1, 2), fitted on training data only
  - note that TF-IDF is used only for the MLP baseline; Text CNN and BiLSTM use token-ID sequences
- Format OOV rates and truncation stats as markdown tables
- Save to `outputs/shared/report/preprocessing_summary_report.md` and `outputs/shared/report/preprocessing_summary_report.json`

D2. Preprocessing artifact contract
- `preprocessing_summary_report.json` must include:
  - `schema_version`, `protocol_version`
  - `cleaning_policy`: object
  - `tokenizer`: string
  - `vocab_size`: integer
  - `special_tokens`: object
  - `max_seq_length`: integer
  - `sequence_length_stats`: object
  - `oov_rates`: object with per-split rates
  - `truncation_rates`: object with per-split rates
  - `tfidf_config`: object
  - `source_artifact`: repo-relative path to `data/artifacts/preprocessing_summary.json`

E. Model architecture summaries
- Generate an architecture comparison section with both a unified table and per-model detail blocks
- For each model, read `outputs/<model_name>/frozen_final_config.json` and `outputs/shared/efficiency_summary_table.json`
- Unified comparison table columns:
  - model display name
  - input type (TF-IDF vectors vs token-ID sequences)
  - key architectural components (e.g. "2-layer MLP", "multi-kernel CNN", "2-layer BiLSTM")
  - total parameter count
  - trainable parameter count
  - embedding dimension (if applicable)
  - key hyperparameters: dropout rate, hidden dim
- Per-model detail blocks must include:
  - MLP: input dimension (10000 TF-IDF features), hidden dim (512), single hidden layer, ReLU activation, dropout (0.2)
  - Text CNN: embedding dim (256), 3 kernel sizes (3, 4, 5), 100 filters per kernel, ReLU, max pooling, dropout (0.5), trainable embeddings
  - BiLSTM: embedding dim (256), hidden dim (256), 2 layers, bidirectional, concat final hidden summarization, gradient clipping (max norm 1.0), dropout (0.3), trainable embeddings
- Save to `outputs/shared/report/model_architectures.md` and `outputs/shared/report/model_architectures.json`

E2. Architecture artifact contract
- `model_architectures.json` must include:
  - `schema_version`, `protocol_version`
  - `models`: list in canonical order, each with:
    - `model_id`, `model_name`, `display_name`
    - `input_type`
    - `architecture_summary`: short string description
    - `hyperparameters`: object with all frozen hyperparameters
    - `parameter_count`, `trainable_parameter_count`
  - `comparison_table`: list of rows for the unified table
  - `source_artifacts`: list of repo-relative paths to frozen configs and efficiency table

F. Experimental setup
- Generate an experimental methodology section from `outputs/shared/evaluation_protocol.json` and frozen configs
- Content must include:
  - evaluation protocol: repeated-run with 3 seeds (42, 1337, 2024)
  - representative-run selection rule: highest validation macro F1
  - optimizer: Adam for all models
  - learning rate: per-model from frozen configs
  - weight decay: per-model from frozen configs
  - batch size: 64 for all models
  - early stopping: patience 10, monitoring validation macro F1
  - max epochs: 100
  - seed derivation: training_seed = seed, dataloader_seed = seed + 1
  - metric definitions:
    - accuracy: correct / total
    - macro F1: macro-averaged across all 151 classes including OOS
    - precision / recall: macro-averaged including OOS
    - OOS metrics: one-vs-rest precision, recall, F1 for OOS class
    - zero-division policy: zero_division=0
    - value range: raw ratios in [0.0, 1.0]
  - OOS evaluation policy: explicit class (label 42), OOS is positive class, all in-scope are negative
  - timing policy: includes DataLoader overhead
  - device: Apple Silicon MPS when available, CPU fallback
- Save to `outputs/shared/report/experimental_setup.md` and `outputs/shared/report/experimental_setup.json`

F2. Experimental setup artifact contract
- `experimental_setup.json` must include:
  - `schema_version`, `protocol_version`
  - `run_count`, `seed_list`
  - `representative_run_rule`
  - `optimizer`: string
  - `per_model_lr`: object mapping model_id to learning rate
  - `per_model_weight_decay`: object mapping model_id to weight decay
  - `batch_size`, `max_epochs`, `early_stopping_patience`
  - `monitor_metric`
  - `metric_definitions`: object
  - `oos_evaluation_policy`: object
  - `timing_policy`: string
  - `source_artifacts`: list of repo-relative paths

G. Main results tables
- Generate formatted results tables from aggregate and analysis artifacts
- Required tables:
  1. Main model comparison table from `model_comparison_aggregate.json`:
     - columns: Model, Test Accuracy, Test Macro F1, Test Precision, Test Recall
     - values: mean +/- std (e.g. "0.8373 +/- 0.0066")
     - label as "Aggregate over 3 runs"
  2. OOS metrics table from `oos_summary_table.json`:
     - columns: Model, OOS Precision, OOS Recall, OOS F1
     - values: mean +/- std
     - label as "Aggregate over 3 runs"
  3. Extended metrics table from `extended_metrics_comparison.json`:
     - columns: Model, Macro F1, Micro F1, Weighted F1, Macro Precision, Macro Recall
     - label as "Representative run only"
  4. Calibration metrics table from `calibration_summary.json`:
     - columns: Model, ECE, MCE, Brier Score, NLL
     - label as "Representative run only"
  5. Efficiency comparison table from `efficiency_summary_table.json`:
     - columns: Model, Parameters, Training Time (s), Inference (ms/example), Throughput (ex/s)
     - training time: mean +/- std
     - parameter count: exact integer
- Each table must explicitly state whether it reports aggregate or representative-run values
- Save to `outputs/shared/report/main_results.md` and `outputs/shared/report/main_results.json`

G2. Table formatting conventions
- Use 4 decimal places for all ratio-valued metrics (e.g. 0.8727)
- Use mean +/- std format for aggregate metrics: `0.8727 +/- 0.0031`
- Use 2 decimal places for time values in seconds
- Use comma-separated integers for parameter counts (e.g. 5,197,975)
- Use 4 decimal places for ms/example timing
- Maintain canonical model order in all tables: MLP, Text CNN, BiLSTM
- Align numeric columns right in markdown tables where possible

G3. Main results artifact contract
- `main_results.json` must include:
  - `schema_version`, `protocol_version`
  - `tables`: list of table objects, each with:
    - `table_id`: unique identifier
    - `title`: descriptive title
    - `data_scope`: `"aggregate"` or `"representative"`
    - `columns`: list of column names
    - `rows`: list of row objects with model_id and values
    - `source_artifact`: repo-relative path
  - `source_artifacts`: list of all repo-relative paths consumed

H. OOS detection results
- Generate a dedicated OOS detection section draft from OOS-specific artifacts
- Content must include:
  - aggregate OOS precision, recall, F1 per model from `oos_summary_table.json` (already in main results, but repeated here with narrative context)
  - OOS threshold analysis from `oos_threshold_comparison.json`:
    - AUROC, AUPR per model (explicit OOS class probability method)
    - FPR@95TPR, FPR@90TPR per model
    - MSP baseline AUROC and AUPR for comparison
  - narrative interpreting which model handles OOS best and why
  - OOS distribution challenge: 18% OOS in test vs 1.6% in train
  - OOS false-accept analysis from `oos_false_accept_comparison.json`:
    - which in-scope intents most frequently capture OOS examples per model
    - cross-model comparison of false-accept patterns
  - reference to OOS-specific figures from the figure manifest:
    - per-model OOS ROC curves, PR curves, OOS error breakdown
    - shared OOS ROC comparison, PR comparison, OOS error comparison
- Save to `outputs/shared/report/oos_detection_results.md` and `outputs/shared/report/oos_detection_results.json`

H2. OOS detection artifact contract
- `oos_detection_results.json` must include:
  - `schema_version`, `protocol_version`
  - `aggregate_oos_metrics`: per-model OOS precision, recall, F1 (mean +/- std)
  - `threshold_metrics`: per-model AUROC, AUPR, FPR@95TPR, FPR@90TPR
  - `msp_baseline`: per-model MSP AUROC, MSP AUPR
  - `best_oos_model`: model_id and rationale
  - `false_accept_summary`: per-model top intents capturing OOS
  - `figure_references`: list of repo-relative figure paths
  - `source_artifacts`: list of repo-relative paths consumed

I. Figure and plot catalogue
- Generate an organized figure catalogue from `outputs/shared/figure_manifest.json`
- For each figure in the manifest, produce:
  - a descriptive caption suitable for a report figure legend
  - a section assignment (which report section the figure belongs to)
  - a scope tag: `aggregate`, `representative`, or `analysis`
  - the repo-relative path to the figure file
- Organize figures by report section:
  - Training Diagnostics: representative training curves (loss, macro F1, accuracy per model)
  - Model Comparison: aggregate test accuracy, macro F1, OOS F1 comparison bar charts
  - OOS Detection: per-model and shared ROC/PR curves, OOS error breakdown/comparison
  - Confusion Analysis: representative confusion matrices, top confused pairs, bottom classes by F1
  - Error Analysis: calibration comparison, reliability diagrams, confidence histograms, error taxonomy comparison, confidence vs accuracy, length/frequency slice comparisons, cross-model error overlap, worst-class heatmaps, confusion stability
  - Efficiency: model efficiency comparison
- Each caption must describe what the figure shows, not just repeat the filename
- Save to `outputs/shared/report/figure_catalogue.md` and `outputs/shared/report/figure_catalogue.json`

I2. Figure catalogue artifact contract
- `figure_catalogue.json` must include:
  - `schema_version`, `protocol_version`
  - `total_figure_count`: integer
  - `sections`: list of section objects, each with:
    - `section_name`: string
    - `figures`: list of figure objects, each with:
      - `figure_path`: repo-relative path
      - `caption`: descriptive caption string
      - `scope`: `"aggregate"`, `"representative"`, or `"analysis"`
      - `model_name`: model_id or `null` for shared figures
      - `figure_type`: from figure manifest
  - `source_artifact`: path to `outputs/shared/figure_manifest.json`

J. Error analysis discussion
- Generate an error analysis discussion section from Step 10 analysis artifacts
- Content must include:
  - error taxonomy overview from `error_taxonomy_summary.json`:
    - per-model error category breakdown (oos_as_inscope, inscope_as_oos, near_semantic_confusion, cross_domain_confusion, short_query_ambiguity)
    - which category dominates for each model
    - cross-model comparison
  - calibration and confidence analysis from `calibration_summary.json`:
    - which model is best/worst calibrated (ECE comparison)
    - high-confidence error patterns
    - reference to reliability diagrams and confidence histograms
  - OOS-specific error deep dive:
    - false accept vs false reject patterns from error taxonomy
    - which intents most frequently capture OOS examples
    - reference to OOS error breakdown figures
  - cross-model error overlap from `cross_model_error_comparison.json`:
    - fraction of test set where all models are correct vs all wrong
    - model-specific error fraction
    - whether universally wrong examples agree on the same incorrect prediction
    - reference to `universally_misclassified_examples.csv`
  - worst-class analysis from `worst_classes_comparison.json`:
    - classes that are consistently worst across all models (inherently ambiguous)
    - classes that are worst for only one model (architecture-sensitive)
  - confused pairs analysis from `most_confused_pairs_table.json`:
    - persistent intent-pair confusions across models
    - whether confused pairs share a domain
  - length and frequency slice findings:
    - whether short queries disproportionately cause errors
    - reference to length/frequency slice comparison figures
- Every claim must cite its source artifact in parentheses
- Save to `outputs/shared/report/error_analysis_discussion.md` and `outputs/shared/report/error_analysis_discussion.json`

J2. Error analysis discussion artifact contract
- `error_analysis_discussion.json` must include:
  - `schema_version`, `protocol_version`
  - `taxonomy_summary`: per-model dominant category and counts
  - `calibration_finding`: best/worst model and ECE values
  - `oos_finding`: best model by AUROC and main failure mode
  - `cross_model_overlap`: all_correct/all_wrong/model_specific counts and fractions
  - `worst_classes_finding`: shared vs model-specific worst classes
  - `confused_pairs_finding`: most persistent pairs across models
  - `length_finding`: short-query error rates per model
  - `source_artifacts`: list of all repo-relative paths consumed

K. Representative examples
- Generate a formatted qualitative examples section from `outputs/shared/analysis/curated_report_examples.json`
- Content must include:
  - a formatted markdown table of curated examples grouped by error category
  - columns: Text, True Label, Predicted Label, Model, Confidence, Category, Annotation
  - select a subset (15-25 examples) that best illustrate the key error patterns:
    - at least 3 OOS false accepts (high-confidence examples where OOS was predicted as in-scope)
    - at least 3 semantic confusions (within-domain intent pairs)
    - at least 3 cross-domain confusions
    - at least 3 short-query ambiguity examples
    - at least 2 examples from each model
  - each example must be traceable: include model_id so the reader can verify against `final_predictions.csv`
  - a brief narrative paragraph introducing the examples and explaining the selection criteria
- Label clearly that these come from representative runs, not aggregate statistics
- Save to `outputs/shared/report/representative_examples.md` and `outputs/shared/report/representative_examples.json`

K2. Representative examples artifact contract
- `representative_examples.json` must include:
  - `schema_version`, `protocol_version`
  - `selection_criteria`: string explaining how examples were chosen
  - `total_curated_count`: integer (from full curated set)
  - `selected_count`: integer (subset included in report)
  - `examples`: list of example objects, each with:
    - `text`, `true_label_name`, `predicted_label_name`, `model_id`, `max_confidence`, `primary_category`, `annotation`
  - `data_scope`: `"representative"` (not aggregate)
  - `source_artifact`: path to `outputs/shared/analysis/curated_report_examples.json`

L. Key findings
- Generate a structured key findings narrative from `outputs/shared/analysis/error_analysis_summary.json` and aggregate comparison tables
- Content must include:
  - headline finding: best overall model with aggregate test macro F1 and std, citing `model_comparison_aggregate.json`
  - calibration finding: best/worst calibrated model with ECE values, citing `calibration_summary.json`
  - OOS detection finding: best OOS model by AUROC and AUPR, citing `oos_threshold_comparison.json`
  - confusion finding: dominant error category across models, persistent confused pairs, citing `error_taxonomy_summary.json`
  - architecture comparison finding: fraction of errors shared vs model-specific, citing `cross_model_error_comparison.json`
  - length finding: whether short queries disproportionately cause errors, with per-model short-query accuracy, citing `error_analysis_summary.json`
  - efficiency finding: parameter counts and inference throughput comparison, citing `efficiency_summary_table.json`
  - top recommendation: the single most impactful improvement, citing `error_analysis_summary.json`
- Format as a numbered list of findings, each with a parenthetical artifact citation
- Save to `outputs/shared/report/key_findings.md` and `outputs/shared/report/key_findings.json`

L2. Key findings artifact contract
- `key_findings.json` must include:
  - `schema_version`, `protocol_version`
  - `findings`: list of finding objects, each with:
    - `finding_id`: string
    - `category`: string (e.g. "headline", "calibration", "oos_detection", "confusion", "architecture", "length", "efficiency", "recommendation")
    - `claim`: string
    - `metric_value`: number or null
    - `metric_std`: number or null
    - `source_artifact`: repo-relative path
  - `source_artifacts`: list of all repo-relative paths consumed

M. Limitations
- Generate a limitations section combining analysis-level and project-level limitations
- Analysis-level limitations (from `error_analysis_notes.md`):
  - qualitative analysis uses a single representative run per model, not all seeds
  - CLINC150 is balanced; real-world class distributions may differ significantly
  - no interpretability analysis (attention, saliency) was performed
  - post-hoc calibration (temperature scaling) was not applied
  - length and frequency slicing uses simple whitespace tokenization
- Project-level limitations:
  - single dataset only (CLINC150); results may not generalize to other intent-classification benchmarks
  - no pretrained word embeddings (GloVe, word2vec) used; embeddings trained from scratch
  - whitespace tokenizer rather than subword tokenization (BPE, WordPiece)
  - no transformer-based models compared (BERT, DistilBERT, etc.)
  - 3 repeated runs provide limited statistical power; 5+ runs would strengthen variance estimates
  - OOS training data is sparse relative to test distribution (250 train vs 1000 test OOS examples)
  - no cross-dataset validation or domain-transfer evaluation
  - model checkpoints not committed to repository; reproduction requires retraining
- Format as a structured bullet list under two subsections
- Save to `outputs/shared/report/limitations.md` and `outputs/shared/report/limitations.json`

M2. Limitations artifact contract
- `limitations.json` must include:
  - `schema_version`, `protocol_version`
  - `analysis_limitations`: list of strings
  - `project_limitations`: list of strings
  - `source_artifacts`: list of repo-relative paths (at minimum `outputs/shared/analysis/error_analysis_notes.md`)

N. Future improvements
- Generate a prioritized future improvements section from Step 10 recommendations and IMPLEMENTATION_SPEC.md Step 12 stretch goals
- High-priority improvements (from Step 10 `error_analysis_summary.json`):
  - collect more OOS training examples to reduce false accepts
  - merge or relabel persistently confused intent pairs within the same domain
  - apply confidence thresholding in deployment to flag uncertain predictions
  - evaluate temperature scaling for post-hoc calibration improvement
- Medium-priority improvements (from Step 12 stretch goals):
  - add pretrained word embeddings (GloVe, word2vec) to Text CNN and BiLSTM
  - systematic hyperparameter tuning (dropout, hidden size, learning rate)
  - add parameter-count vs accuracy Pareto analysis
  - add threshold analysis for OOS detection with operating-point selection
- Lower-priority / future work:
  - compare against transformer-based models (DistilBERT, BERT-base)
  - evaluate on additional intent-classification datasets
  - subword tokenization (BPE, WordPiece) for better OOV handling
  - multi-task learning combining intent classification and OOS detection
  - bootstrap confidence intervals for more rigorous statistical comparison
- Format as a prioritized list with brief rationale for each item
- Save to `outputs/shared/report/future_improvements.md` and `outputs/shared/report/future_improvements.json`

N2. Future improvements artifact contract
- `future_improvements.json` must include:
  - `schema_version`, `protocol_version`
  - `high_priority`: list of improvement objects (each with `improvement`, `rationale`)
  - `medium_priority`: list of improvement objects
  - `lower_priority`: list of improvement objects
  - `source_artifacts`: list of repo-relative paths

O. Reproducibility
- Generate a reproducibility section documenting how to reproduce the full pipeline from scratch
- Content must include:
  - environment setup:
    - Python version requirement (>=3.11); include exact version used if available from `pyproject.toml` or runtime metadata
    - dependency installation: `uv sync` or `pip install -r requirements.txt`
    - note that pinned dependency versions are available in `uv.lock` or `requirements.txt` for exact reproduction
    - PyTorch with MPS support for Apple Silicon (optional, CPU fallback available)
  - dataset acquisition:
    - CLINC150 loaded via HuggingFace `datasets` library (`clinc/clinc_oos`, `plus` subset)
    - no manual download required; loaded at runtime
  - full pipeline execution:
    - canonical command: `python scripts/run_model_pipeline.py --model all --run-count 3`
    - this runs preprocessing, tuning, repeated evaluation, tracking, visualization, and error analysis for all three models
  - seed list: [42, 1337, 2024]
  - expected output structure: brief description of `outputs/` directory layout
  - artifact verification: how to verify outputs match expected artifact schemas
  - estimated runtime: approximate wall-clock time per model on Apple Silicon (from efficiency table); include hardware context for quoted runtimes
  - nondeterminism note: MPS backend nondeterminism, DataLoader worker ordering; exact numeric reproduction across hardware is not guaranteed
  - numbered step-by-step reproduction checklist: (1) clone repo, (2) install dependencies, (3) run pipeline command, (4) verify outputs
- Save to `outputs/shared/report/reproducibility.md` and `outputs/shared/report/reproducibility.json`

O2. Reproducibility artifact contract
- `reproducibility.json` must include:
  - `schema_version`, `protocol_version`
  - `python_version`: string
  - `install_command`: string
  - `pinned_versions_available`: boolean
  - `dataset_source`: string
  - `pipeline_command`: string
  - `seed_list`: list of integers
  - `run_count`: integer
  - `approximate_runtime_minutes`: object with per-model estimates
  - `hardware_context`: string (hardware used for quoted runtimes)
  - `nondeterminism_notes`: list of strings
  - `reproduction_steps`: list of strings (ordered checklist)
  - `source_artifacts`: list of repo-relative paths

P. Report structure manifest
- Generate a master JSON manifest mapping every report section to its source artifacts, output files, and associated figures
- The manifest must include:
  - `schema_version`, `protocol_version`
  - `report_title`: string
  - `sections`: ordered list of section objects, each with:
    - `section_id`: unique identifier matching the implementation section letter
    - `section_title`: human-readable title
    - `output_md`: repo-relative path to the markdown section file
    - `output_json`: repo-relative path to the JSON section file
    - `source_artifacts`: list of repo-relative paths consumed by this section
    - `figures`: list of repo-relative figure paths assigned to this section
    - `tables`: list of table IDs defined in this section
  - `assembled_report`: path to `full_report_draft.md`
  - `total_sections`: integer
  - `total_figures`: integer
  - `total_tables`: integer
- Save to `outputs/shared/report/report_structure.json`

Q. Assembled markdown report draft
- Generate one complete markdown report by assembling all section drafts in logical order
- Report structure:
  1. Title: "Lightweight Deep Learning Models for Intent Classification and Out-of-Scope Detection on CLINC150"
  2. Table of contents (with section numbers)
  3. Abstract (from Section B)
  4. Dataset Description (from Section C)
  5. Preprocessing (from Section D)
  6. Model Architectures (from Section E)
  7. Experimental Setup (from Section F)
  8. Results (from Section G)
  9. OOS Detection (from Section H)
  10. Error Analysis (from Section J)
  11. Representative Examples (from Section K)
  12. Key Findings (from Section L)
  13. Limitations (from Section M)
  14. Future Improvements (from Section N)
  15. Reproducibility (from Section O)
  16. Figure Catalogue (from Section I, as an appendix)
- Each section must include proper markdown heading levels (## for main sections, ### for subsections)
- Figure references must use the format: `![Caption](repo-relative-path)`
- Table references must use standard markdown table syntax
- The assembled report must be self-contained: readable as a single document without needing to open individual section files
- Save to `outputs/shared/report/full_report_draft.md`

Q2. Assembled report requirements
- The report must use consistent heading hierarchy: `#` for title, `##` for major sections, `###` for subsections
- The table of contents must link to section anchors
- Every metric cited in narrative text must match the corresponding table value exactly
- Every figure reference must point to an existing file
- The report must clearly distinguish aggregate claims from representative-run observations
- Core figures must be embedded or referenced in the main narrative per the figure quality standards (confusion matrix in Error Analysis, training curve in Results, aggregate comparison in Results, OOS curve in OOS Detection)
- Include a "Data Sources" footnote or note at the end listing all upstream artifacts consumed

R. Output validation
- After generating all report outputs, validate:
  - all expected files exist under `outputs/shared/report/`
  - all JSON files parse cleanly
  - all `source_artifact` references in JSON files resolve to real files
  - all figure paths in the figure catalogue resolve to real files
  - all table values in `main_results.json` match the upstream aggregate artifacts exactly
  - the assembled report `full_report_draft.md` exists and is non-empty
  - the report structure manifest references all section files
  - total figure count in catalogue matches figure manifest count
- If any validation fails, report the failure clearly

Aggregate vs representative labeling convention:
- Every table in the report must include a scope annotation:
  - `"Aggregate over N runs (mean +/- std)"` for tables built from `model_comparison_aggregate.json`, `oos_summary_table.json`, or `efficiency_summary_table.json`
  - `"Representative run only (single seed)"` for tables built from `extended_metrics_comparison.json`, `calibration_summary.json`, or `oos_threshold_comparison.json`
- Every narrative claim about model performance must state whether it is an aggregate or representative observation
- Qualitative examples (Section K) must always be labeled as representative-run data
- Do not present representative-run metrics as aggregate results or vice versa

Statistical reporting policy:
- When aggregate artifacts provide mean and standard deviation, both must appear in every table cell and narrative claim that references the metric
- Do not claim statistical significance or rank models as "significantly better" without a formal test; with only 3 runs, differences may not be statistically meaningful -- state this explicitly if model performances are close
- When reporting representative-run metrics (calibration, extended metrics, threshold analysis), note that these reflect a single seed and may vary across runs
- Avoid language implying causal relationships (e.g. "BiLSTM causes better OOS detection"); prefer descriptive comparisons ("BiLSTM achieved higher OOS AUROC")
- If two models have overlapping mean +/- std ranges on a metric, note the overlap rather than declaring one superior

Table formatting conventions:
- Use 4 decimal places for all ratio-valued metrics (e.g. 0.8727, not 87.27%)
- Use mean +/- std format for aggregate metrics: `0.8727 +/- 0.0031`
- Use 2 decimal places for time values in seconds (e.g. 60.62)
- Use comma-separated integers for parameter counts (e.g. 5,197,975)
- Use 4 decimal places for ms/example timing (e.g. 0.0563)
- Maintain canonical model order in all tables: TF-IDF + MLP, Text CNN, BiLSTM
- All metrics are raw ratios in [0.0, 1.0] unless explicitly labeled otherwise

Figure quality standards:
- Every figure caption must be self-contained: a reader should understand what the figure shows without reading surrounding text
- Captions must state the data scope (aggregate, representative, or analysis), the dataset split (train/val/test), and which model(s) the figure covers
- Core figures must be referenced in the main narrative of their assigned section, not merely listed in the appendix catalogue; at minimum, the assembled report must embed or reference:
  - at least one confusion matrix (from Confusion Analysis figures) in the Error Analysis section
  - at least one training curve (from Training Diagnostics figures) in the Results or Experimental Setup section
  - at least one aggregate comparison bar chart (from Model Comparison figures) in the Results section
  - at least one OOS ROC or PR curve (from OOS Detection figures) in the OOS Detection section
- Figures in the catalogue appendix that are not referenced in the main narrative should still have complete captions but are considered supplementary

Reproducibility depth standards:
- The reproducibility section must include the exact Python version used (from `pyproject.toml` or runtime metadata if available), not just the minimum requirement
- Include a note about known sources of nondeterminism: MPS backend nondeterminism, DataLoader worker ordering, and any CUDA nondeterminism if applicable
- Include the hardware context for quoted runtimes (e.g. "Apple M-series, X GB RAM") so readers can calibrate expectations
- Present a numbered step-by-step reproduction checklist: (1) clone repo, (2) install dependencies, (3) run pipeline command, (4) verify outputs
- If `uv.lock` or `requirements.txt` pins exact dependency versions, note that pinned versions are available for exact reproduction

Recommended report output structure:
- `outputs/shared/report/report_structure.json`
- `outputs/shared/report/abstract.md`
- `outputs/shared/report/abstract.json`
- `outputs/shared/report/dataset_description.md`
- `outputs/shared/report/dataset_description.json`
- `outputs/shared/report/preprocessing_summary_report.md`
- `outputs/shared/report/preprocessing_summary_report.json`
- `outputs/shared/report/model_architectures.md`
- `outputs/shared/report/model_architectures.json`
- `outputs/shared/report/experimental_setup.md`
- `outputs/shared/report/experimental_setup.json`
- `outputs/shared/report/main_results.md`
- `outputs/shared/report/main_results.json`
- `outputs/shared/report/oos_detection_results.md`
- `outputs/shared/report/oos_detection_results.json`
- `outputs/shared/report/figure_catalogue.md`
- `outputs/shared/report/figure_catalogue.json`
- `outputs/shared/report/error_analysis_discussion.md`
- `outputs/shared/report/error_analysis_discussion.json`
- `outputs/shared/report/representative_examples.md`
- `outputs/shared/report/representative_examples.json`
- `outputs/shared/report/key_findings.md`
- `outputs/shared/report/key_findings.json`
- `outputs/shared/report/limitations.md`
- `outputs/shared/report/limitations.json`
- `outputs/shared/report/future_improvements.md`
- `outputs/shared/report/future_improvements.json`
- `outputs/shared/report/reproducibility.md`
- `outputs/shared/report/reproducibility.json`
- `outputs/shared/report/full_report_draft.md`

Validation checks and assertions:
- Assert that all paths in `step11_handoff.json` resolve to real files before starting
- Assert that all per-model frozen configs exist and parse cleanly
- Assert that `model_comparison_aggregate.json` has rows for all three canonical models
- Assert that metric values extracted for tables match the source artifacts exactly (no rounding drift)
- Assert that all figure paths in the catalogue exist on disk
- Assert that all `source_artifact` lists in JSON outputs contain only paths that resolve
- Assert that the assembled report references every section file
- Assert that the report structure manifest total counts match actual file counts
- Assert that aggregate tables use aggregate data and representative tables use representative data (no cross-contamination)
- Assert that curated examples trace back to real rows in `curated_report_examples.json`
- Assert that the OOS class identity in all sections matches `evaluation_protocol.json`

Test requirements:
- Add or update tests covering:
  - preflight validation logic (missing artifacts produce clear errors)
  - table formatting functions (correct decimal places, mean +/- std format, comma-separated integers)
  - section generation from known test inputs (small synthetic artifacts)
  - report structure manifest generation and validation
  - figure catalogue generation from a small test manifest
  - assembled report structure (correct section order, heading hierarchy, TOC anchors)
  - metric extraction accuracy (values match source artifacts)
  - aggregate vs representative scope tagging correctness
- Tests should use small synthetic data for speed
- Extend existing test modules where possible

Definition of done:
Step 11 is only complete if all of the following are true:
- preflight validation passes for all upstream artifacts
- abstract exists with headline results citing aggregate metrics
- dataset description exists with split sizes, class counts, OOS handling, and distribution quirks
- preprocessing summary exists with cleaning policy, tokenizer, vocabulary, TF-IDF, OOV rates, and truncation stats
- model architecture comparison exists with unified table and per-model detail blocks
- experimental setup exists with protocol, seeds, optimizer, metric definitions, and OOS policy
- main results tables exist for aggregate metrics, OOS, extended metrics, calibration, and efficiency
- dedicated OOS detection section exists with threshold analysis, false-accept patterns, and figure references
- figure catalogue exists with captions and section assignments for all figures in the manifest
- error analysis discussion exists synthesizing taxonomy, calibration, OOS, cross-model overlap, worst classes, and confused pairs
- representative examples section exists with curated qualitative examples and annotation tags
- key findings section exists with every claim citing its source artifact
- limitations section exists combining analysis-level and project-level limitations
- future improvements section exists with prioritized recommendations
- reproducibility section exists with environment setup, pipeline command, and seed list
- report structure manifest exists mapping every section to its sources
- assembled markdown report exists with table of contents, all sections, and figure references
- all JSON outputs have `schema_version` and `protocol_version`
- all artifact path references resolve to real files
- no model was retrained or re-evaluated
- no new figures or metrics were computed; only existing artifacts were consumed

Final Step 11 exit gate:
- Before Step 11 is considered closed, confirm all of the following are true:
  - the assembled report is readable as a self-contained document
  - every metric in the report matches its source artifact value exactly
  - every figure in the catalogue has a self-contained descriptive caption stating scope, split, and model(s)
  - core figures (confusion matrix, training curve, aggregate comparison, OOS curve) are referenced in the main narrative, not only in the appendix
  - aggregate and representative data are never mixed without labeling; overlapping mean +/- std ranges are noted rather than declaring one model superior
  - the report structure manifest accounts for every section file
  - a reader could write the final paper directly from these outputs
  - the report covers all items listed in IMPLEMENTATION_SPEC.md Step 11: dataset description, preprocessing summary, model architecture summary, training setup, metrics table, plots and confusion matrix, key findings, limitations, future improvements
  - the reproducibility section includes a numbered reproduction checklist, hardware context, and nondeterminism notes

Deliverable quality bar:
Step 11 should make the project paper-ready. A reader should be able to take the assembled report draft and, with editorial polish and literature-review additions, produce a complete course project paper. The section drafts should be factually precise, properly cited to artifacts, and structured for easy consumption. Tables should be formatted consistently and figures should have informative captions. The report must not require the reader to open JSON files, re-run scripts, or hunt through directories to understand the results.

Suggested self-check after implementation:
After coding, self-check against this exact checklist and confirm:
1. that the preflight validation checks all upstream artifacts before generating any section
2. where the abstract is saved and that it cites aggregate headline metrics
3. where the dataset description is saved and that it includes OOS distribution quirks
4. where the preprocessing summary is saved and that it covers TF-IDF and sequence details
5. where the architecture comparison is saved and that it has both a unified table and per-model detail
6. where the experimental setup is saved and that it documents seeds, optimizer, early stopping, and metric definitions
7. where the main results tables are saved and that aggregate tables use mean +/- std
8. where the OOS detection section is saved and that it includes threshold analysis and false-accept patterns
9. where the figure catalogue is saved and that every figure has a self-contained caption stating scope, split, and model(s)
10. where the error analysis discussion is saved and that every claim cites its source artifact
11. where the representative examples are saved and that they are labeled as representative-run data
12. where the key findings are saved and that each finding has an artifact citation
13. where the limitations are saved and that they cover both analysis-level and project-level issues
14. where the future improvements are saved and that they are prioritized
15. where the reproducibility section is saved and that it includes the canonical pipeline command, numbered reproduction checklist, hardware context, and nondeterminism notes
16. where the report structure manifest is saved and that it maps every section to its sources
17. where the assembled report is saved and that it has a table of contents and all sections
18. that core figures (confusion matrix, training curve, aggregate comparison, OOS curve) are referenced in the main narrative sections, not only in the appendix
19. that no model was retrained and no new analysis was computed
20. that aggregate and representative metrics are never conflated; overlapping mean +/- std ranges are acknowledged
21. that all JSON outputs include schema_version and protocol_version
22. that no unsupported significance claims are made; close model performances note the limited statistical power of 3 runs
