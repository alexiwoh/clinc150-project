## Experimental Setup

**Source**: outputs/shared/evaluation_protocol.json

### Evaluation Protocol

Repeated-run evaluation with 3 seeds: [42, 1337, 2024]. Representative run selection: highest_validation_macro_f1 (highest validation macro F1).

Means and population standard deviations describe the recorded runs. They are descriptive dispersion measures, not confidence intervals or significance tests.

### Training Configuration

- **Optimizer**: Adam (all models)
- **Learning rate**: TF-IDF + MLP: 0.0005, Text CNN: 0.001, BiLSTM: 0.001
- **Weight decay**: TF-IDF + MLP: 0.0001, Text CNN: 0.0001, BiLSTM: 0.0
- **Batch size**: 64
- **Max epochs**: 100
- **Early stopping**: patience 10, monitoring val_macro_f1
- **Seed behavior**: Each run copies the frozen model configuration with `random_seed = seed` and `dataloader_seed = seed + 1`. The training seed controls initialization and training randomness; the dataloader seed controls train-batch shuffling.

Final evaluations reuse frozen hyperparameters from limited model-specific searches. The original tuning reused advancing loader RNG state across trials, and neural smoke checks also consumed a train shuffle, making trial order part of that search. Preprocessing and tuning budgets differ across models; this compares complete pipelines rather than isolating architecture.

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

OOS is a supervised 151st class with labeled training examples. These results describe this explicit-class setting and do not establish general open-set detection.

Aggregate OOS precision/recall/F1 use the 151-class argmax prediction. Probability-based AUROC and calibration diagnostics use one validation-selected representative run per model, without averaging across seeds.

### Timing and Device

Timing includes DataLoader overhead: True. Automatic device selection checks MPS, then CUDA, then CPU. Refreshed run metadata records the actual device, hardware model, processor and RAM; historical metadata may lack these fields.

Batched evaluation timing covers loader traversal, device transfer, forward pass, softmax and CPU result collection; it excludes preprocessing, checkpoint loading, array concatenation, metrics and artifact writes. The per-example figures describe batched throughput, with no controlled warmup or repeated timing trials; single-query deployment latency was not measured.
