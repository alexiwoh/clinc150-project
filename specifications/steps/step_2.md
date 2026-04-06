Implement and verify Step 2: Dataset loading and exploration for the CLINC150 project.

Goal:
Build a clean, reproducible dataset exploration workflow that confirms the dataset schema, split structure, label space, out-of-scope handling, and report-ready summary artifacts. This step should be mostly read-only and should not yet perform heavy preprocessing, vocabulary building, sequence padding, or model training.

Primary outcome:
At the end of this step, I want to know with certainty:
1. how CLINC150 is loaded
2. what the train / validation / test splits contain
3. how many intent labels exist
4. how out-of-scope examples are represented
5. how I will evaluate OOS for the main version of the project
6. what the class balance looks like
7. what representative examples look like
8. that all of this is saved in reusable artifacts for later steps and the final report

Important scope constraints:
- Do not do Step 3 work yet
- Do not build vocabulary
- Do not tokenize for training
- Do not pad/truncate sequences
- Do not create DataLoaders for model training
- Do not train any models
- Do not silently modify the original split boundaries
- Do not hardcode assumptions about CLINC150 without verifying them from the loaded dataset

Implementation requirements:

A. Create a clear dataset inspection entry point
- Add a dedicated function or script for dataset loading and exploration
- Prefer something like:
  - src/dataset.py for reusable loading helpers
  - and/or a script entry such as main.py or a dedicated inspect_dataset function
- The exploration logic should be easy to rerun and should print useful information in a readable way
- Keep the loading code modular so later steps can reuse the exact same source and split definitions

B. Load CLINC150 explicitly and safely
- Load the CLINC150 dataset from the chosen source
- Print or log exactly which source is being used
- Verify the dataset loads without errors
- Inspect the dataset object and confirm:
  - available splits
  - field names / column names
  - feature types
- Do not assume text and label column names. Confirm them from the loaded data

C. Inspect split structure carefully
- Confirm that train, validation, and test splits all exist
- Confirm the number of examples in each split
- Save these counts in a reusable artifact
- Do not reshuffle or recreate splits unless there is a verified reason
- Preserve provided splits as-is
- Make sure later steps can reuse these exact split definitions

D. Identify text field and label field precisely
- Determine which column contains the user query text
- Determine which column contains the intent label
- Check whether there are other useful metadata fields
- Print a few raw examples from each split to verify structure
- Confirm there are no null or malformed entries in the core text/label fields
- If malformed or missing values exist, report them clearly and count them

E. Determine the label space
- Extract all unique labels from the training split
- Extract all unique labels from validation and test splits
- Verify whether all splits share the same label space or whether any differences exist
- Count total number of unique labels
- Save:
  - sorted label list
  - label count
- Produce label mapping artifacts:
  - data/artifacts/label_to_id.json
  - data/artifacts/id_to_label.json
- Make sure label ordering is deterministic and reproducible

F. Inspect OOS handling explicitly
- Determine how out-of-scope examples are represented in the dataset
- Verify whether OOS examples appear in train / validation / test
- Verify the exact OOS label name(s)
- Count how many OOS examples exist in each split
- Count how many in-scope examples exist in each split
- Save these counts in a dataset summary artifact
- Make no assumptions here. Confirm everything from the actual loaded data

G. Decide and document the primary OOS evaluation strategy
- For the main project implementation, use explicit OOS class handling as the default / primary strategy
- Document this clearly in the dataset summary
- Mention that threshold-based OOS detection may be considered later as a stretch goal, but should not be the main Step 2 output
- Save a note in the summary artifact stating that the primary framing for the base project is multiclass intent classification with an explicit OOS label

H. Analyze class balance
- Compute per-class counts for each split
- Compute:
  - top N largest classes
  - top N smallest classes
  - OOS vs in-scope counts
- Save per-class distributions to a CSV artifact, for example:
  - data/artifacts/train_label_distribution.csv
  - data/artifacts/val_label_distribution.csv
  - data/artifacts/test_label_distribution.csv
- Also produce a concise human-readable summary of balance findings
- Do not over-engineer rebalancing here. This is inspection only

I. Print and save representative sample queries
- For several different in-scope labels, print a handful of sample utterances
- Print a handful of OOS samples as well
- Try to include examples from:
  - clearly distinct intents
  - semantically similar intents if obvious
  - OOS examples
- Save representative examples to an artifact for later report writing and error analysis
- Example artifact:
  - data/artifacts/sample_queries_by_label.json
- Keep these examples raw and readable

J. Produce a report-ready dataset summary artifact
Create a concise reusable summary artifact, such as:
- data/artifacts/dataset_summary.json
and/or
- data/artifacts/dataset_summary.md

The summary should contain at minimum:
1. dataset source
2. split names
3. split sizes
4. confirmed text field name
5. confirmed label field name
6. total number of labels
7. label names or path to label file
8. OOS label name
9. OOS counts by split
10. in-scope counts by split
11. brief class balance observations
12. brief note that the primary project framing uses explicit OOS as a class
13. any dataset quirks or caveats discovered during inspection

K. Keep code reusable for later steps
- Expose reusable functions such as:
  - load_clinc150_dataset(...)
  - get_label_mappings(...)
  - summarize_dataset(...)
  - export_dataset_artifacts(...)
- Avoid mixing exploration code with future preprocessing logic
- The output of this step should serve Step 3 and later without rewriting dataset code

L. Add validation checks / assertions
Include sanity checks so Step 2 fails loudly if something is wrong:
- dataset contains expected splits
- text field exists
- label field exists
- no empty splits
- label mapping is deterministic
- OOS label can be identified
- exported artifacts are written successfully
- split counts in summary match the loaded dataset sizes

M. Logging / console output expectations
When this step runs, it should print a concise but informative summary including:
- dataset source
- split sizes
- field names
- total label count
- OOS label identification
- OOS counts by split
- a short class balance summary
- a few sample utterances
- where artifacts were saved

N. File / artifact expectations
At the end of Step 2, I expect artifacts similar to these:
- data/artifacts/label_to_id.json
- data/artifacts/id_to_label.json
- data/artifacts/dataset_summary.json
- data/artifacts/dataset_summary.md
- data/artifacts/train_label_distribution.csv
- data/artifacts/val_label_distribution.csv
- data/artifacts/test_label_distribution.csv
- data/artifacts/sample_queries_by_label.json

O. Definition of done
Step 2 is only complete if all of the following are true:
- CLINC150 loads successfully from the chosen source
- train / validation / test splits are confirmed and counted
- text field and label field are confirmed from the real dataset
- all label names are extracted and saved
- deterministic label mappings are created and saved
- OOS handling is explicitly identified and counted
- the main project framing is documented as using explicit OOS as a class
- class distributions are computed and saved
- representative sample queries are saved
- a report-ready dataset summary artifact exists
- the code is reusable and separate from Step 3 preprocessing
- no accidental split modifications were introduced

P. Deliverable quality bar
The implementation should be clean, minimal, and reproducible. Prefer simple and correct over clever. This step should establish trust in the dataset setup so later model comparisons are fair and reportable.

After implementing, self-check against this exact checklist and confirm which files/functions/artifacts satisfy each section.

Step 2 Visualization Checklist

1. Split size bar chart
   - Plot the number of examples in the train, validation, and test splits.
   - Usefulness:
     - Confirms the dataset was loaded correctly and split sizes match expectations.
     - Gives an immediate sanity check before any preprocessing or modeling.
     - Helps document dataset scale in the report.

2. Intent class distribution bar chart
   - Plot the count of examples for each intent class, ideally sorted in descending order.
   - Usefulness:
     - Verifies whether the dataset is balanced across intent labels.
     - Helps identify any unexpected skew or missing labels.
     - Provides context for later interpretation of model performance and per-class errors.

3. OOS vs in-scope distribution bar chart
   - Plot counts of out-of-scope examples versus in-scope examples, preferably broken down by split.
   - Usefulness:
     - Directly supports one of the core project goals: out-of-scope detection.
     - Confirms how OOS examples are represented in the dataset.
     - Makes later OOS evaluation choices easier to justify.

4. Query length histogram
   - Plot a histogram of utterance lengths.
   - Length should be measured in tokens if tokenization is already available in Step 2, otherwise use whitespace-separated word count.
   - Usefulness:
     - Helps understand how short or long CLINC150 utterances typically are.
     - Provides evidence for later preprocessing choices such as maximum sequence length.
     - Helps anticipate truncation risk in the neural models.

5. Query length boxplot by split
   - Plot query length distributions separately for train, validation, and test.
   - Usefulness:
     - Checks whether the splits have similar linguistic length distributions.
     - Helps detect accidental inconsistencies across splits.
     - Strengthens documentation that dataset splits are comparable, not just similar in count.

6. Representative sample queries table
   - Save a small table showing a few example queries with their labels.
   - Include both in-scope and OOS examples.
   - Usefulness:
     - Makes the dataset concrete and interpretable for humans.
     - Helps verify that labels and OOS examples were parsed correctly.
     - Useful for the report and for later qualitative error analysis.

Documentation Notes for the Coding Agent

- Each visualization function should include a short docstring explaining:
  - what is being plotted,
  - which dataset field(s) it uses,
  - and why the plot is useful for dataset verification or later modeling decisions.

- Save all figures to a dedicated output directory for Step 2 so they can be reused in the report later.

- Log key summary numbers used to create the plots, not just the images.
  - Example: split sizes, number of intent classes, OOS counts, min/mean/median/max query length.

- Keep plotting code modular and reusable.
  - Prefer one function per visualization rather than one large notebook-only block.

- Use clear file names for saved outputs.
  - Example:
    - split_sizes.png
    - intent_class_distribution.png
    - oos_vs_inscope_distribution.png
    - query_length_histogram.png
    - query_length_by_split_boxplot.png
    - representative_queries.csv

- If class distribution has too many labels to display cleanly:
  - still save the full plot,
  - but also save a second plot showing the top N classes by count for readability.

- Keep styling simple and consistent.
  - Titles, axis labels, tick rotation where needed, and tight layout should be included.
  - The goal is readability and report reuse.
