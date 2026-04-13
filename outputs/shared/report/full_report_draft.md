# Lightweight Deep Learning Models for Intent Classification and Out-of-Scope Detection on CLINC150

## Table of Contents

1. [Abstract](#abstract)
2. [Dataset Description](#dataset-description)
3. [Preprocessing Summary](#preprocessing-summary)
4. [Model Architectures](#model-architectures)
5. [Experimental Setup](#experimental-setup)
6. [Main Results](#main-results)
7. [OOS Detection Results](#oos-detection-results)
8. [Error Analysis Discussion](#error-analysis-discussion)
9. [Representative Examples](#representative-examples)
10. [Key Findings](#key-findings)
11. [Limitations](#limitations)
12. [Future Improvements](#future-improvements)
13. [Reproducibility](#reproducibility)
14. [Figure Catalogue](#figure-catalogue)

## Abstract

Evaluate lightweight deep learning models for intent classification and out-of-scope (OOS) detection on the CLINC150 dataset.

Comparison of a sparse-feature baseline (TF-IDF + MLP), a convolutional sequence model (Text CNN), and a recurrent sequence model (BiLSTM).

Three models are compared: TF-IDF + MLP, Text CNN, BiLSTM.

**Headline Result**: TF-IDF + MLP achieved the highest aggregate test macro F1 of 0.8727 +/- 0.0031 across 3 repeated runs (outputs/shared/model_comparison_aggregate.json).

**OOS Detection**: TF-IDF + MLP achieved the highest aggregate OOS F1 of 0.6261 +/- 0.0348 (outputs/shared/oos_summary_table.json).

**Key Finding**: Focus on improving OOS detection and addressing semantically ambiguous intent pairs within the same domain. Consider intent merging for persistently confused pairs and confidence thresholding for high-confidence errors. (outputs/shared/analysis/error_analysis_summary.json).

## Dataset Description

**Source**: `clinc/clinc_oos` (subset: `plus`) (data/artifacts/dataset_summary.json).

The dataset contains 151 intent classes (150 in-scope + 1 OOS). The OOS class uses label name `oos` (label ID 42).

### Split Sizes

| Split      | Total  | In-Scope | OOS   |
| ---------- | ------ | -------- | ----- |
| Train      | 15,250 | 15,000   | 250   |
| Validation | 3,100  | 3,000    | 100   |
| Test       | 5,500  | 4,500    | 1,000 |

### Class Balance

In-scope classes are balanced in the `plus` subset at 100 training examples per class. OOS support varies across splits: train has 250 OOS examples (~1.6%), while test has 1000 OOS examples (~18.2%), creating a significant distribution shift.

### Distribution Quirks

- Test split has ~18.2% OOS examples vs ~1.6% in train, creating a substantial distribution shift.
- In-scope classes are perfectly balanced in the 'plus' subset (100 training examples per class).
- OOS has 250 training examples, 2.50x the per-class in-scope training count (100).

## Preprocessing Summary

**Source**: data/artifacts/preprocessing_summary.json

### Text Cleaning

Strategy: strip + normalize whitespace + optional lowercase. Preserves: punctuation, contractions, digits.

### Tokenization and Vocabulary

Tokenizer: whitespace split. Vocabulary size: 6161 (built from training data only). Special tokens: `<PAD>` = 0, `<UNK>` = 1.

### Sequence Handling

Max sequence length: 20. Statistics (training set): mean 8.32, median 8.0, p90 13, p95 14, max 28, min 1.

### OOV Rates

| Split      | Total Tokens | Unknown Tokens | OOV Rate |
| ---------- | ------------ | -------------- | -------- |
| Train      | 126,920      | 0              | 0.0000   |
| Validation | 25,674       | 868            | 0.0338   |
| Test       | 45,606       | 2,362          | 0.0518   |

### Truncation Rates

| Split      | Total Sequences | Truncated | Truncation Rate |
| ---------- | --------------- | --------- | --------------- |
| Train      | 15,250          | 32        | 0.0021          |
| Validation | 3,100           | 20        | 0.0065          |
| Test       | 5,500           | 10        | 0.0018          |

### TF-IDF Configuration

Max features: 10,000. N-gram range: (1, 2). Fitted on training data only. Note: TF-IDF is used only for the MLP baseline; Text CNN and BiLSTM use token-ID sequences.

## Model Architectures

### Unified Comparison

| Model        | Input Type         | Architecture                      | Params    | Trainable | Embed Dim | Dropout | Hidden Dim |
| ------------ | ------------------ | --------------------------------- | --------- | --------- | --------- | ------- | ---------- |
| TF-IDF + MLP | TF-IDF vectors     | 2-layer MLP (1 hidden layer)      | 5,197,975 | 5,197,975 | N/A       | 0.2     | 512        |
| Text CNN     | Token-ID sequences | Multi-kernel CNN (3 kernel sizes) | 1,930,167 | 1,930,167 | 256       | 0.5     | N/A        |
| BiLSTM       | Token-ID sequences | 2-layer bidirectional LSTM        | 4,284,311 | 4,284,311 | 256       | 0.3     | 256        |

### TF-IDF + MLP

Input: 10,000 TF-IDF features (10,000-dimensional). Single hidden layer with 512 units, ReLU activation, dropout 0.2.

### Text CNN

Embedding dimension: 256. Kernel sizes: [3, 4, 5] with 100 filters each. ReLU activation, max-over-time pooling, dropout 0.5. Trainable embeddings.

### BiLSTM

Embedding dimension: 256. Hidden dimension: 256, 2 layers, bidirectional. Summarization: concat_final_hidden. Gradient clipping (max norm 1.0). Dropout 0.3. Trainable embeddings.

## Experimental Setup

**Source**: outputs/shared/evaluation_protocol.json

### Evaluation Protocol

Repeated-run evaluation with 3 seeds: [42, 1337, 2024]. Representative run selection: highest_validation_macro_f1 (highest validation macro F1).

### Training Configuration

- **Optimizer**: Adam (all models)
- **Learning rate**: TF-IDF + MLP: 0.0005, Text CNN: 0.001, BiLSTM: 0.001
- **Weight decay**: TF-IDF + MLP: 0.0001, Text CNN: 0.0001, BiLSTM: 0.0
- **Batch size**: 64
- **Max epochs**: 100
- **Early stopping**: patience 10, monitoring val_macro_f1
- **Seed behavior**: The repeated-evaluation pipeline derives `training_seed = seed` and `dataloader_seed = seed + 1`. `dataloader_seed` controls train-batch shuffling, but model initialization currently depends on `config.random_seed`, not necessarily the nominal seed.

### Metric Definitions

- **accuracy**: overall accuracy (correct / total)
- **macro_f1**: macro-averaged F1 across all classes including OOS
- **precision**: macro-averaged precision across all classes including OOS
- **recall**: macro-averaged recall across all classes including OOS
- **oos_precision**: one-vs-rest precision for OOS class
- **oos_recall**: one-vs-rest recall for OOS class
- **oos_f1**: one-vs-rest F1 for OOS class
- **zero_division_policy**: zero_division=0 (sklearn convention)
- **value_range**: [0.0, 1.0] (raw ratios, not percentages)

### OOS Evaluation Policy

- OOS class: `oos` (label ID 42)
- Method: explicit_class
- Rule: oos is the positive class; all in-scope labels are negative

### Timing and Device

Timing includes DataLoader overhead: True. Device: Apple Silicon MPS when available, CPU fallback.

## Main Results

### Table 1: Main Model Comparison

*Aggregate over 3 runs (mean +/- std)*

| Model        | Test Accuracy     | Test Macro F1     | Test Precision    | Test Recall       |
| ------------ | ----------------- | ----------------- | ----------------- | ----------------- |
| TF-IDF + MLP | 0.8373 +/- 0.0066 | 0.8727 +/- 0.0031 | 0.8464 +/- 0.0065 | 0.9123 +/- 0.0015 |
| Text CNN     | 0.8176 +/- 0.0084 | 0.8635 +/- 0.0057 | 0.8278 +/- 0.0094 | 0.9163 +/- 0.0016 |
| BiLSTM       | 0.7751 +/- 0.0088 | 0.8284 +/- 0.0071 | 0.8005 +/- 0.0062 | 0.8780 +/- 0.0068 |

TF-IDF + MLP achieved the highest aggregate test macro F1 (0.8727 +/- 0.0031).

Related figure: `outputs/shared/figures/model_comparison_test_macro_f1.png`

### Table 2: OOS Detection Metrics

*Aggregate over 3 runs (mean +/- std)*

| Model        | OOS Precision     | OOS Recall        | OOS F1            |
| ------------ | ----------------- | ----------------- | ----------------- |
| TF-IDF + MLP | 0.8840 +/- 0.0132 | 0.4867 +/- 0.0444 | 0.6261 +/- 0.0348 |
| Text CNN     | 0.9414 +/- 0.0102 | 0.3563 +/- 0.0401 | 0.5154 +/- 0.0416 |
| BiLSTM       | 0.8817 +/- 0.0104 | 0.2947 +/- 0.0236 | 0.4413 +/- 0.0273 |

TF-IDF + MLP achieved the highest aggregate OOS F1 (0.6261 +/- 0.0348).

Related figure: `outputs/shared/figures/oos_metrics_comparison.png`

### Table 3: Extended Metrics

*Representative run only (single seed)*

| Model        | Macro F1 | Micro F1 | Weighted F1 | Macro Precision | Macro Recall |
| ------------ | -------- | -------- | ----------- | --------------- | ------------ |
| TF-IDF + MLP | 0.8742   | 0.8400   | 0.8326      | 0.8492          | 0.9122       |
| Text CNN     | 0.8658   | 0.8211   | 0.8072      | 0.8315          | 0.9170       |
| BiLSTM       | 0.8354   | 0.7858   | 0.7727      | 0.8062          | 0.8838       |

TF-IDF + MLP has the highest representative-run macro F1 (0.8742) in the extended-metrics table.

### Table 4: Calibration Metrics

*Representative run only (single seed)*

| Model        | ECE    | MCE    | Brier Score | NLL    |
| ------------ | ------ | ------ | ----------- | ------ |
| TF-IDF + MLP | 0.1027 | 0.2580 | 0.2532      | 0.7303 |
| Text CNN     | 0.0374 | 0.1835 | 0.2567      | 0.8647 |
| BiLSTM       | 0.1125 | 0.4094 | 0.3235      | 1.2145 |

Text CNN is the best calibrated representative run (ECE = 0.0374).

Related figure: `outputs/shared/analysis/calibration_comparison.png`

### Table 5: Efficiency Comparison

*Aggregate over 3 runs (mean +/- std)*

| Model        | Parameters | Training Time (s) | Inference (ms/example) | Throughput (ex/s)    |
| ------------ | ---------- | ----------------- | ---------------------- | -------------------- |
| TF-IDF + MLP | 5,197,975  | 60.62 +/- 10.80   | 0.0563 +/- 0.0013      | 17782.89 +/- 390.80  |
| Text CNN     | 1,930,167  | 136.48 +/- 7.70   | 0.0313 +/- 0.0021      | 32082.23 +/- 2138.80 |
| BiLSTM       | 4,284,311  | 91.74 +/- 23.20   | 0.1205 +/- 0.0150      | 8431.56 +/- 1079.83  |

Text CNN has the highest mean inference throughput (32082 ex/s).

Related figure: `outputs/shared/figures/model_efficiency_comparison.png`

### Key Figures

![Aggregate test macro F1 comparison with error bars across all models. Data: aggregate over 3 repeated runs, CLINC150 test set.](outputs/shared/figures/model_comparison_test_macro_f1.png)

![Validation macro F1 progression during training for TF-IDF + MLP. Data: representative run, CLINC150.](outputs/mlp/figures/representative_val_macro_f1_curve.png)

## OOS Detection Results

### Aggregate OOS Metrics

*Aggregate over 3 runs (mean +/- std)*

| Model        | OOS Precision     | OOS Recall        | OOS F1            |
| ------------ | ----------------- | ----------------- | ----------------- |
| TF-IDF + MLP | 0.8840 +/- 0.0132 | 0.4867 +/- 0.0444 | 0.6261 +/- 0.0348 |
| Text CNN     | 0.9414 +/- 0.0102 | 0.3563 +/- 0.0401 | 0.5154 +/- 0.0416 |
| BiLSTM       | 0.8817 +/- 0.0104 | 0.2947 +/- 0.0236 | 0.4413 +/- 0.0273 |

TF-IDF + MLP achieved the highest aggregate OOS F1 (0.6261 +/- 0.0348).

Related figure: `outputs/shared/figures/oos_metrics_comparison.png`

### OOS Threshold Analysis

*Representative run only (single seed)*

| Model        | AUROC  | AUPR   | FPR@95TPR | FPR@90TPR | MSP AUROC | MSP AUPR |
| ------------ | ------ | ------ | --------- | --------- | --------- | -------- |
| TF-IDF + MLP | 0.9415 | 0.7984 | 0.2389    | 0.1531    | 0.8901    | 0.6180   |
| Text CNN     | 0.9523 | 0.8403 | 0.2082    | 0.1387    | 0.9059    | 0.6560   |
| BiLSTM       | 0.9436 | 0.8055 | 0.2309    | 0.1436    | 0.8747    | 0.5453   |

Text CNN achieved the highest AUROC (0.9523), indicating the strongest threshold-independent OOS discrimination. (outputs/shared/analysis/oos_threshold_comparison.json). The explicit OOS class probability method substantially outperforms the maximum softmax probability (MSP) baseline across all models.

Related figures: `outputs/shared/analysis/oos_roc_comparison.png`, `outputs/shared/analysis/oos_pr_comparison.png`

### OOS Distribution Challenge

The test split contains ~18% OOS examples versus ~1.6% in training, creating a severe distribution shift that challenges all models. All models show low OOS recall, indicating difficulty detecting OOS examples despite reasonable OOS precision.

### False-Accept Patterns

- **TF-IDF + MLP**: 497 false accepts. Top capturing intents: `w2` (19), `calculator` (19), `recipe` (16)
- **Text CNN**: 627 false accepts. Top capturing intents: `travel_suggestion` (27), `directions` (23), `todo_list` (21)
- **BiLSTM**: 672 false accepts. Top capturing intents: `smart_home` (39), `income` (30), `current_location` (28)

False accepts remain the dominant OOS failure pattern, with a small set of intents repeatedly capturing OOS examples. (outputs/shared/analysis/oos_false_accept_comparison.json).

Related figure: `outputs/shared/analysis/oos_error_comparison.png`

![OOS ROC Comparison](outputs/shared/analysis/oos_roc_comparison.png)

## Error Analysis Discussion

### Error Taxonomy

- **TF-IDF + MLP**: dominant error category is `oos_as_inscope` (497 errors, 0.5648 of all errors) (outputs/shared/analysis/error_taxonomy_summary.json)
- **Text CNN**: dominant error category is `oos_as_inscope` (627 errors, 0.6372 of all errors) (outputs/shared/analysis/error_taxonomy_summary.json)
- **BiLSTM**: dominant error category is `oos_as_inscope` (672 errors, 0.5705 of all errors) (outputs/shared/analysis/error_taxonomy_summary.json)

All models share `oos_as_inscope` as the dominant error category, indicating that OOS false accepts are the primary failure mode across architectures.

Related figure: `outputs/shared/analysis/error_taxonomy_comparison.png`

### Calibration and Confidence

Best-calibrated model: Text CNN (ECE = 0.0374). Worst-calibrated: BiLSTM (ECE = 0.1125) (outputs/shared/analysis/calibration_summary.json).

Related figures: `outputs/mlp/analysis/reliability_diagram.png`, `outputs/mlp/analysis/confidence_histogram.png`, `outputs/text_cnn/analysis/reliability_diagram.png`, `outputs/text_cnn/analysis/confidence_histogram.png`, `outputs/bilstm/analysis/reliability_diagram.png`, `outputs/bilstm/analysis/confidence_histogram.png`, `outputs/shared/analysis/calibration_comparison.png`

### OOS Detection Deep Dive

Best OOS detector by AUROC: Text CNN (AUROC = 0.9523) (outputs/shared/analysis/oos_threshold_comparison.json). The main failure mode across all models is `oos_as_inscope` (false accepts).

Related figures: `outputs/mlp/analysis/oos_error_breakdown.png`, `outputs/text_cnn/analysis/oos_error_breakdown.png`, `outputs/bilstm/analysis/oos_error_breakdown.png`, `outputs/shared/analysis/oos_roc_comparison.png`, `outputs/shared/analysis/oos_error_comparison.png`

### Cross-Model Error Overlap

Of 5,500 test examples:
- All models correct: 4,051 (0.7365)
- All models wrong: 608 (0.1105)
- Model-specific errors: 464 (0.0844)
- Partial overlap: 377 (0.0685)

Of universally wrong examples, 0.3668 predict the same incorrect class (outputs/shared/analysis/cross_model_error_comparison.json). See also `outputs/shared/analysis/universally_misclassified_examples.csv`.

Related figure: `outputs/shared/analysis/cross_model_error_overlap.png`

### Worst-Class Analysis

Classes consistently worst across all models (shared): `income`, `oos`, `order`, `recipe`, `smart_home`, `yes` (outputs/shared/analysis/worst_classes_comparison.json).

Model-specific worst classes:
- **TF-IDF + MLP**: `calculator`, `calendar`, `how_busy`, `shopping_list`, `w2`
- **Text CNN**: `bill_balance`, `order_status`, `translate`, `travel_suggestion`, `who_do_you_work_for`
- **BiLSTM**: `current_location`, `goodbye`, `weather`

Related figures: `outputs/mlp/analysis/worst_classes_confusion_heatmap.png`, `outputs/text_cnn/analysis/worst_classes_confusion_heatmap.png`, `outputs/bilstm/analysis/worst_classes_confusion_heatmap.png`

### Confused Pairs (OOS False-Accept Targets)

Intents that persistently capture OOS examples across 2+ models: `recipe`, `directions`, `income`, `smart_home`, `travel_suggestion`, `restaurant_suggestion` (outputs/shared/most_confused_pairs_table.json). These span multiple domains (travel, food, finance), suggesting OOS queries are topically diverse.

Related figures: `outputs/mlp/figures/representative_top_confused_pairs.png`, `outputs/text_cnn/figures/representative_top_confused_pairs.png`, `outputs/bilstm/figures/representative_top_confused_pairs.png`

### Length and Frequency Slices

Short-query accuracy per model (representative run):
- **TF-IDF + MLP**: 0.8291
- **Text CNN**: 0.8583
- **BiLSTM**: 0.8097

(outputs/shared/analysis/error_analysis_summary.json)

Related figure: `outputs/shared/analysis/length_slice_comparison.png`

### Confusion Matrix

![Test-set confusion matrix across 151 intent classes including OOS for TF-IDF + MLP. Data: representative run, CLINC150 test set.](outputs/mlp/figures/representative_confusion_matrix.png)

## Representative Examples

*Representative run only (single seed)*

Examples selected to cover key error patterns: at least 3 OOS false accepts, 3 semantic confusions, 3 cross-domain confusions, 3 short-query ambiguity, and 2+ examples per model. Sorted by category then confidence.

### Coverage in Selected Examples

- Selected examples include 12 OOS false-accept cases.
- Selected examples include 7 within-domain semantic-confusion cases.
- Selected examples include 5 cross-domain confusion cases.
- Selected examples include 0 short-query ambiguity cases.

Related figures: `outputs/mlp/analysis/oos_error_breakdown.png`, `outputs/text_cnn/analysis/oos_error_breakdown.png`, `outputs/bilstm/analysis/oos_error_breakdown.png`, `outputs/shared/analysis/oos_error_comparison.png`

Related figures: `outputs/mlp/figures/representative_confusion_matrix.png`, `outputs/mlp/figures/representative_top_confused_pairs.png`, `outputs/text_cnn/figures/representative_confusion_matrix.png`, `outputs/text_cnn/figures/representative_top_confused_pairs.png`, `outputs/bilstm/figures/representative_confusion_matrix.png`, `outputs/bilstm/figures/representative_top_confused_pairs.png`

Related figures: `outputs/mlp/figures/representative_error_summary.png`, `outputs/text_cnn/figures/representative_error_summary.png`, `outputs/bilstm/figures/representative_error_summary.png`, `outputs/shared/analysis/cross_model_error_overlap.png`

Related figure: `outputs/shared/analysis/length_slice_comparison.png`

| Text                                                            | True Label                | Predicted            | Model        | Confidence | Category                | Annotation             |
| --------------------------------------------------------------- | ------------------------- | -------------------- | ------------ | ---------- | ----------------------- | ---------------------- |
| what other countries speak the english language                 | oos                       | change_language      | BiLSTM       | 1.0000     | oos_as_inscope          | confident false accept |
| how many calories does doing 20 situps burn                     | oos                       | calories             | Text CNN     | 1.0000     | oos_as_inscope          | confident false accept |
| give me the weather forecast for today                          | oos                       | weather              | Text CNN     | 1.0000     | oos_as_inscope          | confident false accept |
| look up the conversion rate for the euro to dollar exchange     | oos                       | exchange_rate        | BiLSTM       | 1.0000     | oos_as_inscope          | confident false accept |
| give me the weather forecast for today                          | oos                       | weather              | BiLSTM       | 1.0000     | oos_as_inscope          | confident false accept |
| how many calories does jumping up and down burn                 | oos                       | calories             | Text CNN     | 0.9999     | oos_as_inscope          | confident false accept |
| check the status of my amazon orders for me                     | oos                       | order_status         | BiLSTM       | 0.9999     | oos_as_inscope          | confident false accept |
| call an uber to take me to the closest grocery store            | oos                       | uber                 | Text CNN     | 0.9999     | oos_as_inscope          | confident false accept |
| when should i remove my snow tires                              | oos                       | tire_change          | BiLSTM       | 0.9999     | oos_as_inscope          | confident false accept |
| what's the current prevailing interest rate for mortgages in... | oos                       | interest_rate        | Text CNN     | 0.9995     | oos_as_inscope          | confident false accept |
| ignore call                                                     | oos                       | make_call            | TF-IDF + MLP | 0.9980     | oos_as_inscope          | confident false accept |
| deny incoming phone call                                        | oos                       | make_call            | TF-IDF + MLP | 0.9972     | oos_as_inscope          | confident false accept |
| have i told you to add washing dishes to my todo list           | todo_list                 | todo_list_update     | BiLSTM       | 0.9999     | near_semantic_confusion | semantic overlap       |
| thanks for your help, goodbye!                                  | goodbye                   | thank_you            | BiLSTM       | 0.9999     | near_semantic_confusion | semantic overlap       |
| give me a recipe for tacos                                      | ingredients_list          | recipe               | BiLSTM       | 0.9999     | near_semantic_confusion | semantic overlap       |
| have i added my doctor's appointment to my calendar             | calendar                  | calendar_update      | BiLSTM       | 0.9999     | near_semantic_confusion | semantic overlap       |
| how can i request a new credit card                             | replacement_card_duration | new_card             | BiLSTM       | 0.9999     | near_semantic_confusion | semantic overlap       |
| who is responsible for your employment                          | who_do_you_work_for       | who_made_you         | Text CNN     | 0.9986     | near_semantic_confusion | semantic overlap       |
| i'm trying to raise my credit score can you tell me what it ... | credit_score              | improve_credit_score | Text CNN     | 0.9985     | near_semantic_confusion | semantic overlap       |
| what time is it in phoenix                                      | timezone                  | time                 | BiLSTM       | 1.0000     | cross_domain_confusion  | cross-domain mix-up    |
| what is my current location                                     | share_location            | current_location     | BiLSTM       | 1.0000     | cross_domain_confusion  | cross-domain mix-up    |
| repeat what the weather will be like                            | transfer                  | weather              | BiLSTM       | 1.0000     | cross_domain_confusion  | cross-domain mix-up    |
| i want to be reminded to pay the electric bill                  | reminder_update           | pay_bill             | BiLSTM       | 0.9999     | cross_domain_confusion  | cross-domain mix-up    |
| i don't want to forget to call mom                              | reminder_update           | make_call            | BiLSTM       | 0.9998     | cross_domain_confusion  | cross-domain mix-up    |
| what is the price of bluetooth speakers on amazon               | order                     | oos                  | BiLSTM       | 0.9995     | inscope_as_oos          | false rejection        |

Total curated examples: 60; selected for report: 25 (outputs/shared/analysis/curated_report_examples.json).

## Key Findings

With only 3 repeated runs, differences between models may not be statistically meaningful. Where models have overlapping mean +/- std ranges, this is noted rather than declaring one superior.

1. **Headline**: TF-IDF + MLP achieved the best aggregate test macro F1 of 0.8727 +/- 0.0031 (outputs/shared/model_comparison_aggregate.json)
Related figure: `outputs/shared/figures/model_comparison_test_macro_f1.png`

2. **Calibration**: Best calibrated: text_cnn (ECE = 0.0374). Worst: bilstm (ECE = 0.1125) (outputs/shared/analysis/calibration_summary.json)
Related figure: `outputs/shared/analysis/calibration_comparison.png`

3. **Oos Detection**: Text CNN achieved the best OOS detection with AUROC = 0.9523 and AUPR = 0.8403 (outputs/shared/analysis/oos_threshold_comparison.json)
Related figures: `outputs/shared/analysis/oos_roc_comparison.png`, `outputs/shared/analysis/oos_pr_comparison.png`

4. **Confusion**: Dominant error category across all models: `oos_as_inscope`. Dominant per model: TF-IDF + MLP: oos_as_inscope, Text CNN: oos_as_inscope, BiLSTM: oos_as_inscope (outputs/shared/analysis/error_taxonomy_summary.json)
Related figures: `outputs/mlp/figures/representative_top_confused_pairs.png`, `outputs/text_cnn/figures/representative_top_confused_pairs.png`, `outputs/bilstm/figures/representative_top_confused_pairs.png`, `outputs/shared/analysis/error_taxonomy_comparison.png`

5. **Architecture**: 608 examples (0.1105) misclassified by all models; 464 (0.0844) unique to one model. Agreement on wrong class: 0.3668 (outputs/shared/analysis/cross_model_error_comparison.json)
Related figure: `outputs/shared/analysis/cross_model_error_overlap.png`

6. **Length**: Short-query accuracy: TF-IDF + MLP 0.8291, Text CNN 0.8583, BiLSTM 0.8097 (outputs/shared/analysis/error_analysis_summary.json)
Related figure: `outputs/shared/analysis/length_slice_comparison.png`

7. **Efficiency**: Parameter counts: TF-IDF + MLP 5,197,975, Text CNN 1,930,167, BiLSTM 4,284,311. Inference throughput: TF-IDF + MLP 17783 ex/s, Text CNN 32082 ex/s, BiLSTM 8432 ex/s (outputs/shared/efficiency_summary_table.json)
Related figure: `outputs/shared/figures/model_efficiency_comparison.png`

8. **Recommendation**: Focus on improving OOS detection and addressing semantically ambiguous intent pairs within the same domain. Consider intent merging for persistently confused pairs and confidence thresholding for high-confidence errors. (outputs/shared/analysis/error_analysis_summary.json)
Related figures: `outputs/mlp/figures/representative_top_confused_pairs.png`, `outputs/text_cnn/figures/representative_top_confused_pairs.png`, `outputs/bilstm/figures/representative_top_confused_pairs.png`, `outputs/shared/analysis/oos_error_comparison.png`

## Limitations

### Analysis-Level Limitations

- Qualitative analysis uses a single representative run per model, not all seeds.
- CLINC150 is balanced; real-world class distributions may differ significantly.
- No interpretability analysis (attention, saliency) was performed.
- Post-hoc calibration (temperature scaling) was not applied.
- Length and frequency slicing uses simple whitespace tokenization.

### Project-Level Limitations

- Single dataset only (CLINC150); results may not generalize to other intent-classification benchmarks.
- No pretrained word embeddings (GloVe, word2vec) used; embeddings trained from scratch.
- Whitespace tokenizer rather than subword tokenization (BPE, WordPiece).
- No transformer-based models compared (BERT, DistilBERT, etc.).
- 3 repeated runs provide limited statistical power; 5+ runs would strengthen variance estimates.
- OOS training data is sparse relative to test distribution (250 train vs 1,000 test OOS examples).
- No cross-dataset validation or domain-transfer evaluation.
- Model checkpoints not committed to repository; reproduction requires retraining.

## Future Improvements

### High Priority

- **Collect more OOS training examples to reduce false accepts**: OOS false accepts are the dominant error across all models; more OOS training data directly addresses the distribution mismatch.
- **Merge or relabel persistently confused intent pairs within the same domain**: Several intent pairs share near-identical semantics and consistently confuse all models.
- **Apply confidence thresholding in deployment to flag uncertain predictions**: Many errors occur at high confidence; a deployment threshold could redirect uncertain queries to human review.
- **Evaluate temperature scaling for post-hoc calibration improvement**: Calibration varies significantly across models; temperature scaling could improve reliability without retraining.

### Medium Priority

- **Add pretrained word embeddings (GloVe, word2vec) to Text CNN and BiLSTM**: Pretrained embeddings could improve generalization especially for rare words and OOS queries.
- **Systematic hyperparameter tuning (dropout, hidden size, learning rate)**: Current configs use limited grid search; broader exploration may improve all models.
- **Add parameter-count vs accuracy Pareto analysis**: Quantify the efficiency-accuracy trade-off to guide model selection for deployment.
- **Add threshold analysis for OOS detection with operating-point selection**: Enable tunable precision-recall trade-off for OOS detection in production.

### Lower Priority / Future Work

- **Compare against transformer-based models (DistilBERT, BERT-base)**: Establish an upper-bound reference for the lightweight models evaluated.
- **Evaluate on additional intent-classification datasets**: Validate whether findings generalize beyond CLINC150.
- **Subword tokenization (BPE, WordPiece) for better OOV handling**: Reduce OOV rates and improve generalization to unseen vocabulary.
- **Multi-task learning combining intent classification and OOS detection**: Joint training may improve OOS discrimination by explicitly modeling the boundary.
- **Bootstrap confidence intervals for more rigorous statistical comparison**: 3 repeated runs provide limited statistical power; bootstrapping would strengthen claims.

(outputs/shared/analysis/error_analysis_summary.json)

## Reproducibility

### Environment Setup

- Python version: >=3.11
- Install: `uv sync` (recommended) or `pip install -r requirements.txt`
- Pinned dependency versions available: `uv.lock`
- PyTorch with MPS support for Apple Silicon (optional; CPU fallback available)

### Dataset Acquisition

CLINC150 is loaded via the HuggingFace `datasets` library (`clinc/clinc_oos`, `plus` subset). No manual download is required; the dataset is fetched at runtime.

### Full Pipeline Execution

```bash
python scripts/run_model_pipeline.py --model all --run-count 3
```

Seeds: [42, 1337, 2024]

### Expected Output Structure

```
outputs/
  mlp/           # MLP model artifacts (tuning, final_runs, aggregate, figures, analysis)
  text_cnn/      # Text CNN model artifacts
  bilstm/        # BiLSTM model artifacts
  shared/        # Cross-model comparisons, figures, analysis, report
```

### Approximate Runtime

- **TF-IDF + MLP** (3 runs): ~3.0 min (training only; 60.62 s/run)
- **Text CNN** (3 runs): ~6.8 min (training only; 136.48 s/run)
- **BiLSTM** (3 runs): ~4.6 min (training only; 91.74 s/run)

Hardware context: Apple Silicon (MPS). Runtimes will vary on different hardware.

### Nondeterminism Notes

- MPS backend nondeterminism on Apple Silicon (PyTorch does not guarantee deterministic MPS operations).
- DataLoader worker ordering may vary across runs.
- Exact numeric reproduction across different hardware is not guaranteed.

### Reproduction Checklist

1. Clone the repository: `git clone <repo-url> && cd clinc150-project`
2. Install dependencies: `uv sync` (or `pip install -r requirements.txt`)
3. Run the full pipeline: `python scripts/run_model_pipeline.py --model all --run-count 3`
4. Verify outputs exist under `outputs/` with the expected directory structure

## Figure Catalogue

### Training Diagnostics

- **train_val_loss_curve** (representative): Training and validation loss by epoch for TF-IDF + MLP. Computed from representative run `run_01_seed_42`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/mlp/figures/representative_train_val_loss_curve.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/epoch_history.json`
- **val_macro_f1_curve** (representative): Validation macro F1 by training epoch for TF-IDF + MLP. Computed from representative run `run_01_seed_42`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/mlp/figures/representative_val_macro_f1_curve.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/epoch_history.json`
- **val_accuracy_curve** (representative): Validation accuracy by training epoch for TF-IDF + MLP. Computed from representative run `run_01_seed_42`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/mlp/figures/representative_val_accuracy_curve.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/epoch_history.json`
- **train_val_loss_curve** (representative): Training and validation loss by epoch for Text CNN. Computed from representative run `run_03_seed_2024`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/text_cnn/figures/representative_train_val_loss_curve.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/epoch_history.json`
- **val_macro_f1_curve** (representative): Validation macro F1 by training epoch for Text CNN. Computed from representative run `run_03_seed_2024`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/text_cnn/figures/representative_val_macro_f1_curve.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/epoch_history.json`
- **val_accuracy_curve** (representative): Validation accuracy by training epoch for Text CNN. Computed from representative run `run_03_seed_2024`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/text_cnn/figures/representative_val_accuracy_curve.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/epoch_history.json`
- **train_val_loss_curve** (representative): Training and validation loss by epoch for BiLSTM. Computed from representative run `run_02_seed_1337`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/bilstm/figures/representative_train_val_loss_curve.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/epoch_history.json`
- **val_macro_f1_curve** (representative): Validation macro F1 by training epoch for BiLSTM. Computed from representative run `run_02_seed_1337`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/bilstm/figures/representative_val_macro_f1_curve.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/epoch_history.json`
- **val_accuracy_curve** (representative): Validation accuracy by training epoch for BiLSTM. Computed from representative run `run_02_seed_1337`, selected by highest validation macro f1. Uses the CLINC150 validation split.
  Figure path: `outputs/bilstm/figures/representative_val_accuracy_curve.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/epoch_history.json`

### Model Comparison

- **model_comparison_test_accuracy** (aggregate): Cross-model test accuracy comparison with error bars across all evaluated models. Uses aggregate statistics over 3 repeated runs per model. Uses the CLINC150 test split.
  Figure path: `outputs/shared/figures/model_comparison_test_accuracy.png`
  Source artifacts: `outputs/shared/model_comparison_aggregate.json`
- **model_comparison_test_macro_f1** (aggregate): Cross-model test macro F1 comparison with error bars across all evaluated models. Uses aggregate statistics over 3 repeated runs per model. Uses the CLINC150 test split.
  Figure path: `outputs/shared/figures/model_comparison_test_macro_f1.png`
  Source artifacts: `outputs/shared/model_comparison_aggregate.json`
- **model_comparison_oos_f1** (aggregate): Cross-model OOS F1 comparison with error bars across all evaluated models. Uses aggregate statistics over 3 repeated runs per model. Uses the CLINC150 test split.
  Figure path: `outputs/shared/figures/model_comparison_oos_f1.png`
  Source artifacts: `outputs/shared/model_comparison_aggregate.json`
- **oos_metrics_comparison** (aggregate): Cross-model OOS precision, recall, and F1 comparison across all evaluated models. Uses aggregate statistics over 3 repeated runs per model. Uses the CLINC150 test split. Metrics shown: OOS precision, OOS recall, OOS F1.
  Figure path: `outputs/shared/figures/oos_metrics_comparison.png`
  Source artifacts: `outputs/shared/oos_summary_table.json`

### OOS Detection

- **oos_metrics** (representative): Representative-run OOS precision, recall, and F1 summary for TF-IDF + MLP. Computed from representative run `run_01_seed_42`, selected by highest validation macro f1. Uses the CLINC150 test split. Metrics shown: OOS precision, OOS recall, OOS F1.
  Figure path: `outputs/mlp/figures/representative_oos_metrics.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/test_metrics.json`
- **oos_metrics** (representative): Representative-run OOS precision, recall, and F1 summary for Text CNN. Computed from representative run `run_03_seed_2024`, selected by highest validation macro f1. Uses the CLINC150 test split. Metrics shown: OOS precision, OOS recall, OOS F1.
  Figure path: `outputs/text_cnn/figures/representative_oos_metrics.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/test_metrics.json`
- **oos_metrics** (representative): Representative-run OOS precision, recall, and F1 summary for BiLSTM. Computed from representative run `run_02_seed_1337`, selected by highest validation macro f1. Uses the CLINC150 test split. Metrics shown: OOS precision, OOS recall, OOS F1.
  Figure path: `outputs/bilstm/figures/representative_oos_metrics.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/test_metrics.json`
- **oos_roc_curve** (analysis): ROC curve for explicit-OOS detection for TF-IDF + MLP. Computed from representative-run OOS threshold analysis for representative run `run_01_seed_42`. Uses the CLINC150 test split.
  Figure path: `outputs/mlp/analysis/oos_roc_curve.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/confidences.npz`
- **oos_pr_curve** (analysis): Precision-recall curve for explicit-OOS detection for TF-IDF + MLP. Computed from representative-run OOS threshold analysis for representative run `run_01_seed_42`. Uses the CLINC150 test split.
  Figure path: `outputs/mlp/analysis/oos_pr_curve.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/confidences.npz`
- **oos_error_breakdown** (analysis): False-accept and false-reject intent breakdown for OOS analysis for TF-IDF + MLP. Computed from representative-run OOS error analysis for representative run `run_01_seed_42`. Uses the CLINC150 test split.
  Figure path: `outputs/mlp/analysis/oos_error_breakdown.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/final_predictions.csv`
- **oos_roc_curve** (analysis): ROC curve for explicit-OOS detection for Text CNN. Computed from representative-run OOS threshold analysis for representative run `run_03_seed_2024`. Uses the CLINC150 test split.
  Figure path: `outputs/text_cnn/analysis/oos_roc_curve.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/confidences.npz`
- **oos_pr_curve** (analysis): Precision-recall curve for explicit-OOS detection for Text CNN. Computed from representative-run OOS threshold analysis for representative run `run_03_seed_2024`. Uses the CLINC150 test split.
  Figure path: `outputs/text_cnn/analysis/oos_pr_curve.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/confidences.npz`
- **oos_error_breakdown** (analysis): False-accept and false-reject intent breakdown for OOS analysis for Text CNN. Computed from representative-run OOS error analysis for representative run `run_03_seed_2024`. Uses the CLINC150 test split.
  Figure path: `outputs/text_cnn/analysis/oos_error_breakdown.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/final_predictions.csv`
- **oos_roc_curve** (analysis): ROC curve for explicit-OOS detection for BiLSTM. Computed from representative-run OOS threshold analysis for representative run `run_02_seed_1337`. Uses the CLINC150 test split.
  Figure path: `outputs/bilstm/analysis/oos_roc_curve.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/confidences.npz`
- **oos_pr_curve** (analysis): Precision-recall curve for explicit-OOS detection for BiLSTM. Computed from representative-run OOS threshold analysis for representative run `run_02_seed_1337`. Uses the CLINC150 test split.
  Figure path: `outputs/bilstm/analysis/oos_pr_curve.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/confidences.npz`
- **oos_error_breakdown** (analysis): False-accept and false-reject intent breakdown for OOS analysis for BiLSTM. Computed from representative-run OOS error analysis for representative run `run_02_seed_1337`. Uses the CLINC150 test split.
  Figure path: `outputs/bilstm/analysis/oos_error_breakdown.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/final_predictions.csv`
- **oos_roc_comparison** (analysis): Cross-model ROC comparison for explicit-OOS detection across all evaluated models. Computed from cross-model OOS ROC comparison using one representative run per model. Uses the CLINC150 test split.
  Figure path: `outputs/shared/analysis/oos_roc_comparison.png`
  Source artifacts: `outputs/mlp/analysis/oos_threshold_metrics.json`, `outputs/text_cnn/analysis/oos_threshold_metrics.json`, `outputs/bilstm/analysis/oos_threshold_metrics.json`
- **oos_pr_comparison** (analysis): Cross-model precision-recall comparison for explicit-OOS detection across all evaluated models. Computed from cross-model OOS precision-recall comparison using one representative run per model. Uses the CLINC150 test split.
  Figure path: `outputs/shared/analysis/oos_pr_comparison.png`
  Source artifacts: `outputs/mlp/analysis/oos_threshold_metrics.json`, `outputs/text_cnn/analysis/oos_threshold_metrics.json`, `outputs/bilstm/analysis/oos_threshold_metrics.json`
- **oos_error_comparison** (analysis): Cross-model false-accept versus false-reject comparison across all evaluated models. Computed from cross-model OOS error comparison using one representative run per model. Uses the CLINC150 test split. Metrics shown: false accepts, false rejects.
  Figure path: `outputs/shared/analysis/oos_error_comparison.png`
  Source artifacts: `outputs/mlp/analysis/oos_error_deep_dive.json`, `outputs/text_cnn/analysis/oos_error_deep_dive.json`, `outputs/bilstm/analysis/oos_error_deep_dive.json`

### Confusion Analysis

- **confusion_matrix** (representative): Confusion matrix for intent predictions for TF-IDF + MLP. Computed from representative run `run_01_seed_42`, selected by highest validation macro f1. Uses the CLINC150 test split. Covers 151 intent classes including OOS.
  Figure path: `outputs/mlp/figures/representative_confusion_matrix.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/confusion_matrix.csv`
- **top_confused_pairs** (representative): Most frequent true-label and predicted-label confusion pairs for TF-IDF + MLP. Computed from representative run `run_01_seed_42`, selected by highest validation macro f1. Uses the CLINC150 test split. Highlights the top 10 confusion pairs.
  Figure path: `outputs/mlp/figures/representative_top_confused_pairs.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/top_confusions.json`
- **bottom_classes_f1** (representative): Lowest-F1 intent classes in the representative evaluation for TF-IDF + MLP. Computed from representative run `run_01_seed_42`, selected by highest validation macro f1. Uses the CLINC150 test split.
  Figure path: `outputs/mlp/figures/representative_bottom_classes_f1.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/per_class_metrics.json`
- **confusion_matrix** (representative): Confusion matrix for intent predictions for Text CNN. Computed from representative run `run_03_seed_2024`, selected by highest validation macro f1. Uses the CLINC150 test split. Covers 151 intent classes including OOS.
  Figure path: `outputs/text_cnn/figures/representative_confusion_matrix.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/confusion_matrix.csv`
- **top_confused_pairs** (representative): Most frequent true-label and predicted-label confusion pairs for Text CNN. Computed from representative run `run_03_seed_2024`, selected by highest validation macro f1. Uses the CLINC150 test split. Highlights the top 10 confusion pairs.
  Figure path: `outputs/text_cnn/figures/representative_top_confused_pairs.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/top_confusions.json`
- **bottom_classes_f1** (representative): Lowest-F1 intent classes in the representative evaluation for Text CNN. Computed from representative run `run_03_seed_2024`, selected by highest validation macro f1. Uses the CLINC150 test split.
  Figure path: `outputs/text_cnn/figures/representative_bottom_classes_f1.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/per_class_metrics.json`
- **confusion_matrix** (representative): Confusion matrix for intent predictions for BiLSTM. Computed from representative run `run_02_seed_1337`, selected by highest validation macro f1. Uses the CLINC150 test split. Covers 151 intent classes including OOS.
  Figure path: `outputs/bilstm/figures/representative_confusion_matrix.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/confusion_matrix.csv`
- **top_confused_pairs** (representative): Most frequent true-label and predicted-label confusion pairs for BiLSTM. Computed from representative run `run_02_seed_1337`, selected by highest validation macro f1. Uses the CLINC150 test split. Highlights the top 10 confusion pairs.
  Figure path: `outputs/bilstm/figures/representative_top_confused_pairs.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/top_confusions.json`
- **bottom_classes_f1** (representative): Lowest-F1 intent classes in the representative evaluation for BiLSTM. Computed from representative run `run_02_seed_1337`, selected by highest validation macro f1. Uses the CLINC150 test split.
  Figure path: `outputs/bilstm/figures/representative_bottom_classes_f1.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/per_class_metrics.json`

### Error Analysis

- **error_summary** (representative): Representative-run summary of the most frequent error patterns for TF-IDF + MLP. Computed from representative run `run_01_seed_42`, selected by highest validation macro f1. Uses the CLINC150 test split. Summarizes 20 ranked error examples or categories.
  Figure path: `outputs/mlp/figures/representative_error_summary.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/top_errors.json`
- **error_summary** (representative): Representative-run summary of the most frequent error patterns for Text CNN. Computed from representative run `run_03_seed_2024`, selected by highest validation macro f1. Uses the CLINC150 test split. Summarizes 20 ranked error examples or categories.
  Figure path: `outputs/text_cnn/figures/representative_error_summary.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/top_errors.json`
- **error_summary** (representative): Representative-run summary of the most frequent error patterns for BiLSTM. Computed from representative run `run_02_seed_1337`, selected by highest validation macro f1. Uses the CLINC150 test split. Summarizes 20 ranked error examples or categories.
  Figure path: `outputs/bilstm/figures/representative_error_summary.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/top_errors.json`
- **reliability_diagram** (analysis): Reliability diagram comparing confidence with empirical accuracy for TF-IDF + MLP. Computed from representative-run calibration analysis for representative run `run_01_seed_42`. Uses the CLINC150 test split.
  Figure path: `outputs/mlp/analysis/reliability_diagram.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/confidences.npz`
- **confidence_histogram** (analysis): Confidence distribution for correct versus incorrect predictions for TF-IDF + MLP. Computed from representative-run calibration analysis for representative run `run_01_seed_42`. Uses the CLINC150 test split.
  Figure path: `outputs/mlp/analysis/confidence_histogram.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/confidences.npz`
- **confidence_vs_accuracy** (analysis): Accuracy across confidence buckets for TF-IDF + MLP. Computed from representative-run confidence stratification for representative run `run_01_seed_42`. Uses the CLINC150 test split.
  Figure path: `outputs/mlp/analysis/confidence_vs_accuracy.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/confidences.npz`
- **worst_classes_heatmap** (analysis): Confusion heatmap for the weakest intent classes for TF-IDF + MLP. Computed from representative-run worst-class analysis for representative run `run_01_seed_42`. Uses the CLINC150 test split.
  Figure path: `outputs/mlp/analysis/worst_classes_confusion_heatmap.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/confusion_matrix.csv`
- **confusion_stability** (analysis): Run-to-run stability of the highest-count confusion pairs for TF-IDF + MLP. Computed from cross-run confusion stability analysis. Uses the CLINC150 test split. Tracks the top 20 confusion pairs across 3 completed runs.
  Figure path: `outputs/mlp/analysis/confusion_stability.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/confusion_matrix.csv`, `outputs/mlp/final_runs/run_02_seed_1337/confusion_matrix.csv`, `outputs/mlp/final_runs/run_03_seed_2024/confusion_matrix.csv`
- **reliability_diagram** (analysis): Reliability diagram comparing confidence with empirical accuracy for Text CNN. Computed from representative-run calibration analysis for representative run `run_03_seed_2024`. Uses the CLINC150 test split.
  Figure path: `outputs/text_cnn/analysis/reliability_diagram.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/confidences.npz`
- **confidence_histogram** (analysis): Confidence distribution for correct versus incorrect predictions for Text CNN. Computed from representative-run calibration analysis for representative run `run_03_seed_2024`. Uses the CLINC150 test split.
  Figure path: `outputs/text_cnn/analysis/confidence_histogram.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/confidences.npz`
- **confidence_vs_accuracy** (analysis): Accuracy across confidence buckets for Text CNN. Computed from representative-run confidence stratification for representative run `run_03_seed_2024`. Uses the CLINC150 test split.
  Figure path: `outputs/text_cnn/analysis/confidence_vs_accuracy.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/confidences.npz`
- **worst_classes_heatmap** (analysis): Confusion heatmap for the weakest intent classes for Text CNN. Computed from representative-run worst-class analysis for representative run `run_03_seed_2024`. Uses the CLINC150 test split.
  Figure path: `outputs/text_cnn/analysis/worst_classes_confusion_heatmap.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_03_seed_2024/confusion_matrix.csv`
- **confusion_stability** (analysis): Run-to-run stability of the highest-count confusion pairs for Text CNN. Computed from cross-run confusion stability analysis. Uses the CLINC150 test split. Tracks the top 20 confusion pairs across 3 completed runs.
  Figure path: `outputs/text_cnn/analysis/confusion_stability.png`
  Source artifacts: `outputs/text_cnn/final_runs/run_01_seed_42/confusion_matrix.csv`, `outputs/text_cnn/final_runs/run_02_seed_1337/confusion_matrix.csv`, `outputs/text_cnn/final_runs/run_03_seed_2024/confusion_matrix.csv`
- **reliability_diagram** (analysis): Reliability diagram comparing confidence with empirical accuracy for BiLSTM. Computed from representative-run calibration analysis for representative run `run_02_seed_1337`. Uses the CLINC150 test split.
  Figure path: `outputs/bilstm/analysis/reliability_diagram.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/confidences.npz`
- **confidence_histogram** (analysis): Confidence distribution for correct versus incorrect predictions for BiLSTM. Computed from representative-run calibration analysis for representative run `run_02_seed_1337`. Uses the CLINC150 test split.
  Figure path: `outputs/bilstm/analysis/confidence_histogram.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/confidences.npz`
- **confidence_vs_accuracy** (analysis): Accuracy across confidence buckets for BiLSTM. Computed from representative-run confidence stratification for representative run `run_02_seed_1337`. Uses the CLINC150 test split.
  Figure path: `outputs/bilstm/analysis/confidence_vs_accuracy.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/confidences.npz`
- **worst_classes_heatmap** (analysis): Confusion heatmap for the weakest intent classes for BiLSTM. Computed from representative-run worst-class analysis for representative run `run_02_seed_1337`. Uses the CLINC150 test split.
  Figure path: `outputs/bilstm/analysis/worst_classes_confusion_heatmap.png`
  Source artifacts: `outputs/bilstm/final_runs/run_02_seed_1337/confusion_matrix.csv`
- **confusion_stability** (analysis): Run-to-run stability of the highest-count confusion pairs for BiLSTM. Computed from cross-run confusion stability analysis. Uses the CLINC150 test split. Tracks the top 20 confusion pairs across 3 completed runs.
  Figure path: `outputs/bilstm/analysis/confusion_stability.png`
  Source artifacts: `outputs/bilstm/final_runs/run_01_seed_42/confusion_matrix.csv`, `outputs/bilstm/final_runs/run_02_seed_1337/confusion_matrix.csv`, `outputs/bilstm/final_runs/run_03_seed_2024/confusion_matrix.csv`
- **calibration_comparison** (analysis): Cross-model calibration comparison across all evaluated models. Computed from cross-model calibration comparison using one representative run per model. Uses the CLINC150 test split. Metrics shown: ECE, MCE, Brier score, NLL.
  Figure path: `outputs/shared/analysis/calibration_comparison.png`
  Source artifacts: `outputs/mlp/analysis/calibration_metrics.json`, `outputs/text_cnn/analysis/calibration_metrics.json`, `outputs/bilstm/analysis/calibration_metrics.json`
- **error_taxonomy** (analysis): Cross-model comparison of error-taxonomy fractions across all evaluated models. Computed from cross-model error taxonomy comparison using representative runs. Uses the CLINC150 test split.
  Figure path: `outputs/shared/analysis/error_taxonomy_comparison.png`
  Source artifacts: `outputs/mlp/analysis/error_taxonomy.json`, `outputs/text_cnn/analysis/error_taxonomy.json`, `outputs/bilstm/analysis/error_taxonomy.json`
- **confidence_vs_accuracy** (analysis): Accuracy across confidence buckets across all evaluated models. Computed from cross-model confidence stratification comparison using representative runs. Uses the CLINC150 test split.
  Figure path: `outputs/shared/analysis/confidence_accuracy_comparison.png`
  Source artifacts: `outputs/mlp/analysis/confidence_stratification.json`, `outputs/text_cnn/analysis/confidence_stratification.json`, `outputs/bilstm/analysis/confidence_stratification.json`
- **length_slice** (analysis): Cross-model accuracy comparison by utterance length across all evaluated models. Computed from cross-model slice comparison using representative runs. Uses the CLINC150 test split.
  Figure path: `outputs/shared/analysis/length_slice_comparison.png`
  Source artifacts: `outputs/mlp/analysis/length_slice_analysis.json`, `outputs/text_cnn/analysis/length_slice_analysis.json`, `outputs/bilstm/analysis/length_slice_analysis.json`
- **frequency_slice** (analysis): Cross-model accuracy comparison by class-frequency tier across all evaluated models. Computed from cross-model slice comparison using representative runs. Uses the CLINC150 test split.
  Figure path: `outputs/shared/analysis/frequency_slice_comparison.png`
  Source artifacts: `outputs/mlp/analysis/frequency_slice_analysis.json`, `outputs/text_cnn/analysis/frequency_slice_analysis.json`, `outputs/bilstm/analysis/frequency_slice_analysis.json`
- **cross_model_error_overlap** (analysis): Overlap between shared and model-specific prediction failures across all evaluated models. Computed from cross-model error overlap using one representative run per model. Uses the CLINC150 test split.
  Figure path: `outputs/shared/analysis/cross_model_error_overlap.png`
  Source artifacts: `outputs/mlp/final_runs/run_01_seed_42/final_predictions.csv`, `outputs/text_cnn/final_runs/run_03_seed_2024/final_predictions.csv`, `outputs/bilstm/final_runs/run_02_seed_1337/final_predictions.csv`

### Efficiency

- **model_efficiency_comparison** (aggregate): Cross-model efficiency comparison across all evaluated models. Uses aggregate statistics over 3 repeated runs per model. Uses CLINC150 artifacts. Metrics shown: parameter count, training time, inference throughput.
  Figure path: `outputs/shared/figures/model_efficiency_comparison.png`
  Source artifacts: `outputs/shared/efficiency_summary_table.json`

Total figures: 62

---

## Data Sources

The following upstream artifacts were consumed to generate this report:

- `data/artifacts/dataset_summary.json`
- `data/artifacts/preprocessing_summary.json`
- `outputs/bilstm/frozen_final_config.json`
- `outputs/mlp/frozen_final_config.json`
- `outputs/shared/analysis/calibration_summary.json`
- `outputs/shared/analysis/cross_model_error_comparison.json`
- `outputs/shared/analysis/curated_report_examples.json`
- `outputs/shared/analysis/error_analysis_notes.md`
- `outputs/shared/analysis/error_analysis_summary.json`
- `outputs/shared/analysis/error_taxonomy_summary.json`
- `outputs/shared/analysis/extended_metrics_comparison.json`
- `outputs/shared/analysis/oos_false_accept_comparison.json`
- `outputs/shared/analysis/oos_threshold_comparison.json`
- `outputs/shared/analysis/worst_classes_comparison.json`
- `outputs/shared/efficiency_summary_table.json`
- `outputs/shared/evaluation_protocol.json`
- `outputs/shared/figure_manifest.json`
- `outputs/shared/model_comparison_aggregate.json`
- `outputs/shared/most_confused_pairs_table.json`
- `outputs/shared/oos_summary_table.json`
- `outputs/text_cnn/frozen_final_config.json`
