# CLINC150 Intent Classification and Out-of-Scope Detection

A Python/PyTorch comparison of three lightweight text-classification pipelines: **TF-IDF + MLP**, **Text CNN**, and **BiLSTM**. The project includes frozen experiment settings, repeated training runs, probability diagnostics, error analysis, and a generated report.

The official CLINC150 `plus` splits contain 150 supported intents and a labeled out-of-scope (OOS) class. These experiments train a **151-class supervised classifier**, including OOS examples in training. The splits contain 15,250 training, 3,100 validation, and 5,500 test examples. Vocabulary and TF-IDF fitting use training examples only.

## Results

Each model was retrained with the existing frozen winning hyperparameters and seeds **42, 1337, and 2024**. Checkpoints are selected by validation macro F1, with validation loss and earliest epoch as tie-breakers. Values below are the mean ± population standard deviation across three runs.

| Model | Test accuracy | Test macro F1 | OOS F1 |
| --- | ---: | ---: | ---: |
| TF-IDF + MLP | 0.8345 ± 0.0063 | 0.8716 ± 0.0029 | 0.6103 ± 0.0352 |
| Text CNN | 0.8210 ± 0.0067 | 0.8645 ± 0.0031 | 0.5397 ± 0.0357 |
| BiLSTM | 0.7744 ± 0.0065 | 0.8283 ± 0.0046 | 0.4019 ± 0.0511 |

![Test macro F1 across three independently seeded runs per model](outputs/shared/figures/model_comparison_test_macro_f1.png)

TF-IDF + MLP has the highest mean test macro F1 in these recorded runs. This comparison describes the three configured pipelines; different preprocessing and tuning budgets prevent attributing the difference solely to architecture. Three seeds describe run variation and do not establish statistical significance.

OOS precision/recall/F1 use the classifier's argmax OOS label across all three runs. Probability-based AUROC and calibration diagnostics use one validation-selected representative run per model. Those diagnostics do not establish a deployable rejection threshold or general open-world OOS performance.

Read the [full report](outputs/shared/report/full_report_draft.md) for methods, numerical tables, representative examples, limitations, and references. The [figure catalogue](outputs/shared/report/figure_catalogue.md) identifies each figure's data and scope. Limitations include the original tuning trial order, official split text overlaps with conflicting labels, heuristic error categories, and the BiLSTM's sensitivity to fixed right padding.

## Artifact guide

- [Canonical comparison](outputs/shared/model_comparison_aggregate.json): full-precision metrics and run counts; [CSV](outputs/shared/model_comparison_aggregate.csv) for inspection.
- [MLP](outputs/mlp/), [Text CNN](outputs/text_cnn/), and [BiLSTM](outputs/bilstm/): frozen settings, current run ledgers, per-run probabilities/predictions, aggregates, and representative diagnostics.
- [Shared analysis](outputs/shared/analysis/) and [figures](outputs/shared/figures/): cross-model diagnostics and aggregate comparisons.
- [Historical outputs](outputs/reports/README.md) and [small archive](outputs_small_archive/README.md): preserved earlier single-run experiments and tuning evidence, with their original scope.

Model weights are untracked. Retrain to obtain checkpoints and run their integrity checks. `src/predict.py` is a labeled placeholder for future inference work; the implemented workflow trains, evaluates, and reports saved experiments.

## References and license

The benchmark is from [Larson et al. (2019), An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction](https://aclanthology.org/D19-1131/). The sentence CNN follows [Kim (2014), Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181/).

Repository source is licensed under [GPLv3](LICENSE). The [CLINC150 dataset card](https://huggingface.co/datasets/clinc/clinc_oos) lists CC BY 3.0 for the dataset; dataset licensing is separate from the source license.

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
