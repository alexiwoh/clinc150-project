# Implementation Spec / Project To-Do

## Project Goal
Build and compare lightweight deep learning models for **assistant-style intent classification** and **out-of-scope detection** on the **CLINC150** dataset using **PyTorch**.

This project is intentionally scoped to be realistic for a 4-week timeline while also leaving room to learn core PyTorch concepts. The focus is on building a clean, reproducible NLP deep learning pipeline for assistant-style query understanding rather than exploring too many architectures or datasets.

---

## Project Framing
This project can be framed as a study of lightweight NLP architectures for assistant-style query understanding. The comparison loosely reflects different eras of practical NLP modeling: sparse feature baselines, convolutional sequence models, and recurrent sequence models. The main focus is not historical completeness, but understanding how different representation and sequence-modeling choices affect intent classification and out-of-scope robustness.

---

## Runtime and Hardware Notes
- Prefer Apple Silicon GPU acceleration through PyTorch MPS when available, but do not make MPS a hard requirement for the repository to run.
- Device selection should be handled centrally and safely, with fallback behavior if MPS is unavailable.
- Data loading and batching should be configurable through project settings rather than hardcoded.
- If parallel data loading is used, keep it configurable and benchmarked for local stability on macOS rather than assuming one fixed worker count is always optimal.

---

## Scope
Keep the project intentionally focused:

- Use **one dataset**: CLINC150
- Implement **three models**:
  - TF-IDF + MLP
  - Text CNN
  - BiLSTM
- Evaluate both:
  - intent classification performance
  - out-of-scope detection performance
- Generate results suitable for the final report:
  - metrics tables
  - plots
  - confusion matrix
  - error analysis

Do **not** add transformers, retrieval pipelines, or multiple datasets unless everything else is already complete.

---

## Recommended Project Structure

```text
clinc150-project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── artifacts/
├── notebooks/
│   └── exploration.ipynb
├── scripts/
│   └── bootstrap.sh
├── src/
│   ├── config.py
│   ├── constants.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── utils.py
│   ├── models/
│   │   ├── mlp.py
│   │   ├── text_cnn.py
│   │   └── bilstm.py
│   ├── metrics/
│   │   └── classification_metrics.py
│   └── trainers/
│       └── trainer.py
├── outputs/
│   ├── shared/              # cross-model protocol, comparison tables, shared figures
│   ├── mlp/                 # MLP tuning, final runs, aggregate, figures
│   ├── text_cnn/            # Text CNN tuning, final runs, aggregate, figures
│   ├── bilstm/              # BiLSTM tuning, final runs, aggregate, figures
│   ├── checkpoints/         # legacy single-run checkpoints (Steps 4-6)
│   ├── figures/             # legacy single-run figures (Steps 4-6)
│   ├── logs/                # legacy single-run logs (Steps 4-6)
│   └── reports/             # legacy single-run reports (Steps 4-6)
├── tests/
├── requirements.txt
├── README.md
├── IMPLEMENTATION_SPEC.md
└── main.py
```

---

## File Responsibilities

- `src/config.py`: central configuration for hyperparameters, file paths, training settings, and model options
- `src/constants.py`: shared constants such as label names, special tokens, and default paths
- `src/dataset.py`: dataset loading, split handling, label mapping, and PyTorch dataset wrappers
- `src/preprocessing.py`: text cleaning, tokenization, vocabulary building, sequence padding, and TF-IDF feature preparation
- `src/train.py`: model training entry logic for launching experiments
- `src/evaluate.py`: final model evaluation, metric computation, and confusion matrix generation
- `src/predict.py`: helper functions for running inference on new text examples
- `src/utils.py`: shared utilities such as seeding, logging helpers, checkpoint helpers, and device selection
- `src/models/mlp.py`: TF-IDF + MLP baseline model definition
- `src/models/text_cnn.py`: Text CNN model definition
- `src/models/bilstm.py`: BiLSTM model definition
- `src/metrics/classification_metrics.py`: accuracy, precision, recall, macro F1, and OOS metric helpers
- `src/trainers/trainer.py`: reusable training and validation loop logic shared across models
- `main.py`: top-level project entry point for training, evaluation, or experiment execution
- `scripts/run_model_pipeline.py`: canonical user-facing pipeline runner for tuning reuse, repeated evaluation, tracking, and visualization
- `notebooks/exploration.ipynb`: optional notebook for exploratory analysis, data inspection, and quick visualizations
- `outputs/shared/`: cross-model protocol manifest, comparison tables, and shared figures
- `outputs/<model_name>/`: per-model tuning, final runs (with co-located checkpoints and logs), aggregate summaries, and figures
- `outputs/checkpoints/`: legacy single-run checkpoints from Steps 4-6
- `outputs/figures/`: legacy single-run figures from Steps 4-6
- `outputs/logs/`: legacy single-run logs from Steps 4-6
- `outputs/reports/`: legacy single-run reports from Steps 4-6

---

## Recommended Execution Path
- Use `python scripts/run_model_pipeline.py --model all --run-count 3` as the canonical common-case command
- The pipeline runner should accept:
  - `--model {mlp,text_cnn,bilstm,all}`
  - `--run-count N`
  - optional `--retune`
- Default behavior for each selected model:
  - validate required preprocessing artifacts
  - reuse valid tuning outputs if they already exist
  - retune only when artifacts are missing, invalid, or `--retune` is passed
  - freeze the winning validation-selected config
  - run repeated final training and evaluation
  - build aggregate and representative-run artifacts
  - run tracking and visualization stages
  - emit `step10_handoff.json`
- `main.py` may delegate to this runner, but the pipeline contract should remain explicit and stage-specific scripts should still be available for targeted reruns

---

## Step 1: Project setup
- Create the project folder structure
- Create a Python virtual environment
- Install dependencies:
  - torch
  - datasets
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - jupyter
  - tqdm
- Verify PyTorch works on local machine
- If available, test Apple Silicon MPS acceleration
- Keep device selection and optional parallel data loading configurable rather than hardcoded
- Add a `requirements.txt` file and confirm the environment is reproducible
- Define a reproducible multi-run evaluation protocol with configurable seeds and run counts

---

## Step 2: Dataset loading and exploration
- Download/load CLINC150
- Inspect train, validation, and test splits
- Identify:
  - number of intent classes
  - label names
  - out-of-scope label handling
- Explore class balance
- Print sample queries from different classes
- Decide how OOS will be evaluated:
  - explicit OOS class
  - or threshold-based detection
- Save a short dataset summary for the final report

---

## Step 3: Preprocessing pipeline
- Clean and normalize text if needed
- Build tokenization pipeline for neural models
- Build vocabulary from training data
- Convert text to integer token sequences
- Pad/truncate sequences to a fixed max length
- Build PyTorch Dataset and DataLoader objects
- Separately build TF-IDF features for the MLP baseline
- Keep preprocessing logic reusable and consistent across experiments

---

## Step 4: Baseline model
- Implement TF-IDF + MLP baseline
- Train baseline model
- Record:
  - training loss
  - validation loss
  - validation accuracy
- Save best checkpoint
- Run on test set
- Store results for comparison table
- Use this model as the first reference point for all later comparisons

---

## Step 5: Text CNN model
- Implement Text CNN in PyTorch
- Suggested components:
  - embedding layer
  - multiple convolution filter sizes
  - max pooling
  - dropout
  - linear classifier
- Train and validate
- Save best checkpoint
- Evaluate on test set
- Store metrics and training time

---

## Step 6: BiLSTM model
- Implement BiLSTM in PyTorch
- Suggested components:
  - embedding layer
  - bidirectional LSTM
  - pooling or final hidden state
  - dropout
  - linear classifier
- Train and validate
- Save best checkpoint
- Evaluate on test set
- Store metrics and training time

---

## Step 7: Evaluation metrics
- Compute for each model:
  - validation and test accuracy
  - validation and test macro F1
  - validation and test precision
  - validation and test recall
- Compute OOS metrics:
  - OOS precision
  - OOS recall
  - OOS F1
- Freeze one validation-selected final config per model from the existing tuning outputs
- Run each model multiple times with the same shared seed list
- Save per-run artifacts and per-model aggregate mean/std summaries
- Choose one representative run per model for confusion matrices and qualitative error analysis
- Identify most commonly confused intent pairs
- Make sure metrics are computed consistently across all models
- Keep aggregate reporting separate from representative-run diagnostics
- Use the same seed list and evaluation protocol for all models to keep comparisons fair

---

## Step 8: Experiment tracking
- Save:
  - frozen configs
  - model checkpoints
  - training logs
  - per-run metadata and metrics
  - aggregate summaries and ledgers
- Validate and organize artifacts under the canonical run-first directory layout
- Create shared comparison tables comparing all models
- Track:
  - parameter count
  - trainable parameter count
  - training and inference timing
  - final test and OOS metrics
- Keep experiment outputs organized so they can be reused directly in the report
- Save seed, run index, and model configuration for every run
- Keep per-run metrics separate from aggregated summary metrics
- Save final comparison tables with both mean metrics and variability across runs
- Do not retrain or reevaluate models in this step

---

## Repeated-Run Evaluation Protocol
- Use repeated runs to reduce dependence on a single random seed
- Start with **3 runs per model** as the default protocol
- Increase to **5 runs per model** if runtime allows and metric variance appears meaningful
- Change the seed across runs while keeping the data split and evaluation procedure fixed
- Use aggregated results for the final comparison tables in the report
- Identify one representative run per model for confusion matrix generation and qualitative error analysis, while clearly separating this from the aggregated reporting protocol

---

## Step 9: Visualization
- Generate representative training curves, confusion figures, bottom-classes-by-F1 diagnostics, and OOS diagnostics from saved artifacts
- Generate aggregate cross-model comparison figures from the shared comparison tables
- Save all figures in report-ready format under canonical filenames
- Save `figure_manifest.json`, `representative_examples_index.json`, and one `step10_handoff.json` per model
- Do not retrain models to generate figures

---

## Additional Report-Ready Outputs
In addition to the core metrics and plots, generate the following:
- bottom-classes-by-F1 figure
- table of the most frequently confused intent pairs
- summary table focused specifically on OOS precision, recall, and F1
- a small set of representative misclassified examples for qualitative discussion
- efficiency comparison figure

These outputs are intended to strengthen the Numerical Results chapter without increasing modeling complexity.

---

## Step 10: Error analysis
- Review representative misclassified samples from the Step 9 handoff bundle
- Identify patterns such as:
  - semantically similar intents
  - short ambiguous queries
  - OOS queries predicted as valid intents
- Keep aggregate claims grounded in the repeated-run comparison tables
- Write notes for report discussion section
- Save representative examples for later inclusion in the final write-up

---

## Step 11: Final report support
Prepare outputs for the final paper:
- dataset description
- preprocessing summary
- model architecture summary
- training setup
- metrics table
- plots and confusion matrix
- key findings
- limitations
- future improvements

---

## Step 12: Stretch goals only if ahead of schedule
- Add pretrained embeddings
- Tune dropout / hidden size / learning rate
- Add parameter-count comparison
- Add threshold analysis for OOS detection
- Add simple confidence-threshold analysis for OOS detection

Do not start stretch goals until the main three-model comparison is already complete.

---

## Final Deliverables
- Clean Python project
- Trained model checkpoints
- Evaluation scripts
- Saved plots and confusion matrix
- Final metrics table
- Per-run and aggregated metrics tables with mean and standard deviation
- Report-ready findings
- A reproducible implementation flow that matches the intended project scope
