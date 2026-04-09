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
