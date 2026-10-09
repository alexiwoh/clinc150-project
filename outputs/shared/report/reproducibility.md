## Reproducibility

### Environment Setup

- Python version: >=3.11
- Install from the repository root: `uv sync --frozen`
- Secondary route: activate a virtual environment, then `python -m pip install -r requirements.txt`. The portable export includes `-e .` to install the project.
- Pinned dependency versions available: `uv.lock`
- PyTorch with MPS support for Apple Silicon (optional; CPU fallback available)

### Dataset Acquisition

CLINC150 is loaded via the HuggingFace `datasets` library (`clinc/clinc_oos`, `plus` subset). The configured revision is pinned and future runs record ordered split hashes. The first uncached download requires network access; official split membership is preserved.

### Frozen Final Evaluations and Report

```bash
uv run --frozen python scripts/run_repeated_evaluation.py --model all --run-count 3
uv run --frozen python scripts/run_experiment_tracking.py
uv run --frozen python scripts/run_report_figures.py
uv run --frozen python scripts/run_error_analysis.py
uv run --frozen python scripts/run_report_generation.py
```

Seeds: [42, 1337, 2024]

To rebuild preprocessing from source first run `uv run --frozen python scripts/explore_dataset.py` and `uv run --frozen python scripts/run_preprocessing.py`. The evaluation commands reuse existing frozen hyperparameters; they do not retune. See the repository README for explicit tuning-source selection.

### Expected Output Structure

```
outputs/
  mlp/           # MLP model artifacts (tuning, final_runs, aggregate, figures, analysis)
  text_cnn/      # Text CNN model artifacts
  bilstm/        # BiLSTM model artifacts
  shared/        # Cross-model comparisons, figures, analysis, report
```

### Approximate Runtime

- **TF-IDF + MLP** (3 runs): ~4.1 min (training only; 82.69 s/run)
- **Text CNN** (3 runs): ~8.4 min (training only; 168.54 s/run)
- **BiLSTM** (3 runs): ~7.2 min (training only; 144.16 s/run)

Hardware, device, source commit, input hashes and effective settings are recorded per run. These times cover the training loop including validation/checkpoint writes, excluding setup, preprocessing, neural smoke checks, tuning and final training-log serialization. Refreshed runs use `time.perf_counter` and record the exact hardware model, processor and RAM in `run_metadata.json`; historical outputs without these identities remain explicitly unknown.

Batched evaluation timing covers loader traversal, device transfer, forward pass, softmax and CPU result collection; it excludes preprocessing, checkpoint loading, array concatenation, metrics and artifact writes. The per-example figures describe batched throughput, with no controlled warmup or repeated timing trials; single-query deployment latency was not measured.

### Nondeterminism Notes

- MPS backend nondeterminism on Apple Silicon (PyTorch does not guarantee deterministic MPS operations).
- DataLoaders use num_workers=0 and the recorded per-run shuffle seed.
- Exact numeric reproduction across different hardware is not guaranteed.

### Reproduction Checklist

1. Clone https://github.com/alexiwoh/clinc150-project into a separate checkout; generated outputs are replaced in place.
2. Install locked dependencies from the repository root with `uv sync --frozen`.
3. Run frozen final evaluations, then tracking, report figures, error analysis and report generation.
4. For source reconstruction, rebuild dataset summaries and train-only preprocessing before evaluation.
5. Run offline tests with `uv run --frozen pytest -q`; generated-artifact checks require trained checkpoints.
6. Verify outputs exist under `outputs/` with the expected directory structure

### References and Licensing

- [Larson et al. (2019), An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction](https://aclanthology.org/D19-1131/)
- [Kim (2014), Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181/)
- [CLINC150 dataset card](https://huggingface.co/datasets/clinc/clinc_oos)
- [GPLv3 source license](../../../LICENSE)

The CLINC150 dataset card metadata lists CC BY 3.0 for the dataset. Dataset licensing is separate from this repository's GPLv3 source license. The sentence-CNN design follows Kim (2014); the benchmark is attributed to Larson et al. (2019).
