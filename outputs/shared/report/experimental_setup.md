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
- **Seed derivation**: training_seed = seed, dataloader_seed = seed + 1

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
