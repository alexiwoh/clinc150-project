# CLINC150 Intent Classification and Out-of-Scope Detection
Deep Learning for Intent Classification and Out-of-Scope Detection: A Comparative Study of Lightweight Neural Architectures

This project explores how lightweight deep learning models can be used to understand assistant-style user queries. The main goal is to compare multiple neural network architectures on the task of **intent classification** and **out-of-scope (OOS) detection** using the **CLINC150** dataset.

CLINC150 is a public benchmark dataset containing short natural-language queries across many intent classes, along with out-of-scope examples that do not belong to any supported intent. This makes it a strong fit for studying problems related to AI assistants, query routing, tool selection, and guardrails.

The project is implemented in **Python with PyTorch** and compares three model families:

- **TF-IDF + MLP baseline**
- **Text CNN**
- **BiLSTM**

The purpose of the project is not only to measure raw classification accuracy, but also to study how well these models can distinguish valid in-scope intents from unknown or unsupported queries. This mirrors real-world AI engineering problems where systems must both route known requests correctly and avoid confidently mishandling unknown ones.

The final output of the project includes:

- training and validation loss curves
- test accuracy and macro F1 score
- out-of-scope precision, recall, and F1
- confusion matrix for the best-performing model
- comparison of model size, training time, and common error patterns

Overall, this project provides a practical introduction to deep learning for NLP while staying closely connected to real-world assistant and agent systems.

## Setup and checks

Run commands from the repository root. Python 3.11 or newer is required;
`pyproject.toml` and `uv.lock` define the environment.

```bash
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

The default suite is offline and uses small synthetic fixtures. The other suites
are explicit:

```bash
# Downloads the public benchmark, or uses the local Hugging Face cache.
uv run --frozen pytest -m network -q
# Run after training all nine runs; the model checkpoints are intentionally untracked.
uv run --frozen pytest -m generated_artifacts -q
```

`./scripts/bootstrap.sh` also installs the environment and exports
`requirements.txt` from the lock. The secondary pip route installs the project
through the portable `-e .` entry, so script imports work:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_repeated_evaluation.py --help
```

## Reproduce the experiments

Use a separate clone for reruns: the commands write generated outputs and
checkpoints at their canonical paths. Initial setup and dataset downloads need
network access. Dataset loading uses the official `plus` train, validation, and
test splits. No examples are moved between splits.

To reuse the committed preprocessing artifacts and frozen winning settings:

```bash
uv run --frozen python scripts/run_repeated_evaluation.py --model all --run-count 3
uv run --frozen python scripts/run_experiment_tracking.py
uv run --frozen python scripts/run_report_figures.py
uv run --frozen python scripts/run_error_analysis.py
uv run --frozen python scripts/run_report_generation.py
```

To rebuild preprocessing from the public source and then evaluate the existing
frozen settings, first run:

```bash
uv run --frozen python scripts/explore_dataset.py
uv run --frozen python scripts/run_preprocessing.py
```

Then run the same five evaluation/report commands above. The fitted vocabulary
and TF-IDF vectorizer use only training examples. The supplied pickle vectorizer
and locally generated Torch checkpoints are trusted local artifacts; only load
artifacts you trust.

`uv run --frozen python main.py --model all --run-count 3` is a convenience route
that extracts frozen settings from existing tuning tables, trains, tracks, and
produces figures and analysis. Report generation remains the final explicit
command. Run counts are 1 through 3, with seeds 42, 1337, and 2024. The pipeline
reuses tuning; it does not launch a hyperparameter search.
If canonical and legacy tuning CSVs differ, extraction stops before writing.
Choose `--tuning-source canonical` or `--tuning-source legacy` explicitly after
inspecting them; both original CSVs are preserved. The dedicated repeated-run
command above reads the committed frozen settings directly.

New run metadata records the pinned `clinc/clinc_oos` revision, ordered raw
split and label hashes, fitted preprocessing and encoded-input hashes, runtime
source hashes and Git state, effective model and loader settings, and hardware.
Historical metadata without these identities remains unknown. Current run
ledgers define downstream membership, so leftover run directories are preserved
and excluded. Tracking rejects mixed or disagreeing generation identities.

Device selection is MPS, then CUDA, then CPU. These are full final training runs,
not a quick inference demo; runtime depends on hardware and early stopping.
Training-loop timing includes validation and checkpoint writes, and excludes
preprocessing, setup, and tuning. Per-example evaluation timing is batched
throughput and excludes preprocessing and model loading.

Predictions and reports are committed for inspection. Model weights are ignored;
checkpoint integrity checks and new inference require training. To open the
notebook environment, run `uv run --frozen jupyter notebook`.
