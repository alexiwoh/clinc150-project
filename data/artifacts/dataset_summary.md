# CLINC150 Dataset Summary

**Source:** `clinc/clinc_oos` (subset: `plus`)
**Text field:** `text`  
**Label field:** `intent`  
**Total classes:** 151 (150 in-scope + 1 OOS)
**OOS label:** `oos` (id 42)

## Split Sizes

| Split | Total | In-scope | OOS | OOS % |
|-------|------:|---------:|----:|------:|
| train | 15,250 | 15,000 | 250 | 1.6% |
| validation | 3,100 | 3,000 | 100 | 3.2% |
| test | 5,500 | 4,500 | 1,000 | 18.2% |

## Training Class Balance

- Min count per class: 100
- Max count per class: 250
- Mean count per class: 101.0

## OOS Evaluation Strategy

Multiclass classification with an explicit OOS class (label 42). All models classify into 151 classes; OOS precision/recall/F1 computed directly from class-42 predictions. Threshold-based OOS detection is deferred to stretch goals (Step 12).

## Quirks and Caveats

- Test split has ~18% OOS examples vs ~1% in train — heavy distribution shift.
- In-scope classes are perfectly balanced in the 'plus' subset.
- OOS has 100 train examples (2x any single in-scope class in 'small'; ~0.67x in 'plus').
