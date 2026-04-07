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
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   └── reports/
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
- `notebooks/exploration.ipynb`: optional notebook for exploratory analysis, data inspection, and quick visualizations
- `outputs/checkpoints/`: saved trained model checkpoints
- `outputs/figures/`: saved plots, charts, and confusion matrices
- `outputs/logs/`: experiment logs and run summaries
- `outputs/reports/`: report-ready tables, exported metrics, and related artifacts

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
  - test accuracy
  - macro F1
  - precision
  - recall
- Compute OOS metrics:
  - OOS precision
  - OOS recall
  - OOS F1
- Generate confusion matrix for best model
- Identify most commonly confused intent pairs
- Make sure metrics are computed consistently across all models
- Run each model multiple times with different random seeds
- Record per-run validation and test metrics separately
- Compute aggregated metrics across runs, including mean and standard deviation
- Use the same seed list and evaluation protocol for all models to keep comparisons fair

---

## Step 8: Experiment tracking
- Save:
  - hyperparameters
  - model checkpoints
  - training logs
  - evaluation metrics
- Create a summary results table comparing all models
- Track:
  - parameter count
  - training time
  - final test metrics
- Keep experiment outputs organized so they can be reused directly in the report
- Save seed, run index, and model configuration for every run
- Keep per-run metrics separate from aggregated summary metrics
- Save final comparison tables with both mean metrics and variability across runs

---

## Repeated-Run Evaluation Protocol
- Use repeated runs to reduce dependence on a single random seed
- Start with **3 runs per model** as the default protocol
- Increase to **5 runs per model** if runtime allows and metric variance appears meaningful
- Change the seed across runs while keeping the data split and evaluation procedure fixed
- Use aggregated results for the final comparison tables in the report
- Optionally identify the single best run per model for confusion matrix generation and qualitative error analysis, while clearly separating this from the aggregated reporting protocol

---

## Step 9: Visualization
- Plot training loss curves
- Plot validation accuracy curves
- Save confusion matrix image
- Optionally create bar chart comparing model performance
- Optionally create a per-class accuracy bar chart
- Save all figures in report-ready format

---

## Additional Report-Ready Outputs
In addition to the core metrics and plots, generate the following:
- per-class accuracy bar chart
- table of the most frequently confused intent pairs
- summary table focused specifically on OOS precision, recall, and F1
- a small set of representative misclassified examples for qualitative discussion

These outputs are intended to strengthen the Numerical Results chapter without increasing modeling complexity.

---

## Step 10: Error analysis
- Review misclassified samples from best model
- Identify patterns such as:
  - semantically similar intents
  - short ambiguous queries
  - OOS queries predicted as valid intents
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
