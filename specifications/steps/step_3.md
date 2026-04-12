Implement and verify Step 3: Preprocessing pipeline for the CLINC150 project.

Goal:
Build a clean, reproducible preprocessing pipeline that converts the raw CLINC150 text data into model-ready inputs for both:
1. neural sequence models (Text CNN and BiLSTM)
2. the TF-IDF + MLP baseline

This step should transform the verified dataset setup from Step 2 into reusable training inputs while preserving split integrity, label consistency, and experiment fairness.

Primary outcome:
At the end of this step, I want to have:
1. a clear text normalization / cleaning strategy
2. a reusable tokenization pipeline for neural models
3. a vocabulary built from training data only
4. train / validation / test texts converted to integer token sequences
5. padded / truncated fixed-length tensors ready for neural training
6. PyTorch Dataset and DataLoader objects for neural models
7. TF-IDF feature matrices built separately for the MLP baseline
8. preprocessing artifacts saved for reproducibility
9. one shared preprocessing contract that all later experiments can reuse consistently
10. proof that preprocessing can be rerun deterministically and reused at inference time without re-fitting

Important scope constraints:
- Do not train any models yet
- Do not evaluate models yet
- Do not tune hyperparameters based on test data
- Do not rebuild or modify dataset splits
- Do not use validation or test data to fit vocabulary, tokenizer statistics, or TF-IDF vectorizer
- Do not silently change label mappings created in Step 2
- Do not mix baseline-specific feature prep and neural sequence prep in a confusing way
- Keep preprocessing reusable, deterministic, and split-safe

High-level design requirement:
Use one central preprocessing pipeline with two branches:
- Branch A: neural text pipeline -> token ids + padding + DataLoaders
- Branch B: baseline text pipeline -> TF-IDF features for MLP

Both branches must use the same underlying dataset splits and same label mappings.

Implementation requirements:

A. Create a clear preprocessing entry point
- Add reusable preprocessing code in src/preprocessing.py
- Keep orchestration logic easy to rerun from main.py or a dedicated script / function
- Prefer functions that are cleanly separable, for example:
  - clean_text(...)
  - tokenize_text(...)
  - build_vocab(...)
  - numericalize_texts(...)
  - pad_sequences(...)
  - build_tfidf_features(...)
  - create_dataloaders(...)
  - export_preprocessing_artifacts(...)
  - load_preprocessing_artifacts(...)
- Make preprocessing easy to rerun without manually editing code
- Ensure artifact output directories are created automatically if missing

B. Define and document the text cleaning strategy
- Decide whether text cleaning should be minimal or moderate
- Because CLINC150 is short assistant-style text, prefer conservative cleaning
- Preserve meaning-bearing information
- Do not aggressively strip useful punctuation if it may affect intent
- Explicitly decide how to handle:
  - lowercasing
  - whitespace normalization
  - punctuation
  - apostrophes / contractions
  - digits / numbers
  - special characters
- Apply the same cleaning policy consistently across splits
- Save this policy in a preprocessing summary artifact
- Make the cleaning function deterministic and testable

Recommended bias:
- Start with light normalization:
  - strip leading/trailing whitespace
  - normalize repeated spaces
  - optionally lowercase
- Avoid over-cleaning unless justified

C. Build a reusable tokenization pipeline for neural models
- Implement a tokenization step for sequence models
- Keep it simple and reproducible
- Avoid introducing unnecessary external tokenizer complexity unless clearly needed
- Tokenization should work consistently for:
  - training
  - validation
  - test
  - future inference
- Confirm tokenization output on several example queries
- Save example tokenized outputs for sanity checking
- Ensure tokenization happens after any selected cleaning step

D. Build vocabulary from training data only
- Build the neural-model vocabulary using training texts only
- Never use validation or test texts to fit the vocabulary
- Include special tokens such as:
  - PAD
  - UNK
- Optionally include others only if truly needed, but keep it minimal
- Make vocabulary ordering deterministic
- If token frequencies tie, break ties deterministically, for example alphabetically
- Save vocabulary artifact, for example:
  - data/artifacts/vocab.json
- Save vocabulary size in a summary artifact
- Verify PAD and UNK ids are fixed and do not drift across reruns
- Confirm special token ids are fixed and documented

E. Numericalize text into integer token sequences
- Convert tokenized texts into integer id sequences using the saved vocabulary
- Unknown tokens from validation/test should map to UNK
- Do this separately for train / validation / test
- Save a few example before/after transformations for sanity checks
- Include some representative val/test examples that demonstrate unknown-token behavior
- Verify no sequence contains invalid token ids
- Verify special token handling works as expected
- Record OOV / UNK token rate for train, validation, and test splits
- Save OOV / UNK statistics in preprocessing artifacts

F. Choose and document max sequence length
- Determine an appropriate fixed max length using the training split only
- Inspect sequence length distribution on training data
- Save useful summary stats such as:
  - mean length
  - median length
  - 90th percentile
  - 95th percentile
  - max length
- Use this analysis to select a reasonable max length
- Document why this max length was chosen
- Keep it configurable through src/config.py rather than hardcoded deep in the code

G. Pad and truncate sequences consistently
- Pad or truncate all neural model sequences to the selected fixed max length
- Make padding side configurable
- Make truncation side configurable
- Use a documented default padding/truncation strategy unless there is a clear reason otherwise
- Apply the same sequence length policy to train / validation / test
- Ensure padded output is consistent in shape
- Verify:
  - all resulting tensors have the same sequence length
  - padding token id is correct
  - truncation does not break the pipeline
- Record percentage of sequences truncated for train, validation, and test
- Save truncation statistics in preprocessing artifacts
- Save sample padded sequences for sanity checks

H. Convert labels using existing Step 2 mappings
- Reuse the deterministic label_to_id / id_to_label artifacts from Step 2
- Do not recreate labels differently in Step 3
- Convert split labels to integer targets using the existing mapping
- Verify label ids are valid across all splits
- Save or log target tensor shapes and class counts
- Confirm OOS label remains handled consistently
- Add an explicit check that label tensor lengths exactly match input example counts for every split
- Save the path or reference to the Step 2 label mapping artifacts inside the preprocessing summary

I. Build PyTorch Dataset objects for neural models
- Create reusable PyTorch Dataset wrappers in src/dataset.py or a similarly appropriate location
- Dataset objects should expose:
  - padded sequence tensor
  - label tensor
- Optionally include raw text for debugging only if useful, but keep it clean
- Make sure Dataset classes are simple and reusable
- Confirm __len__ and __getitem__ behave correctly
- Test on a few samples manually

J. Build DataLoader objects for neural models
- Create train / validation / test DataLoaders
- Batch size should come from config rather than hardcoded values
- Shuffle training loader only
- Validation/test loaders should not shuffle
- num_workers should be configurable because of local macOS stability concerns
- pin_memory behavior should be handled sensibly depending on device support
- Keep DataLoader creation centralized and reusable
- Random seed / determinism settings must be centralized and applied before creating any DataLoader with shuffle=True
- Print batch shape sanity checks:
  - input tensor shape
  - label tensor shape
- Confirm one batch can be iterated successfully without error
- Add a check that DataLoader batch counts are non-zero for every split

K. Build TF-IDF features separately for the MLP baseline
- Build a separate baseline preprocessing branch for TF-IDF
- Use training texts only to fit the TF-IDF vectorizer
- Transform validation and test texts using that fitted vectorizer
- Do not leak validation/test information into TF-IDF fitting
- Keep TF-IDF feature prep separate from neural token-id sequence prep
- Save vectorizer artifact, for example:
  - data/artifacts/tfidf_vectorizer.pkl
- Save feature matrix shape summary for:
  - X_train_tfidf
  - X_val_tfidf
  - X_test_tfidf
- Save the actual fitted TF-IDF vocabulary size, not just configured max_features
- Save the actual fitted feature dimension after transform
- Optionally save this under a named field such as:
  - tfidf_vocab_size
  - tfidf_input_dim
- Verify train / val / test all share the exact same TF-IDF feature ordering
- Document key vectorizer settings, such as:
  - ngram range
  - max features
  - lowercase behavior
  - stop word handling if any
- Explicitly document whether TF-IDF lowercasing relies on shared cleaning policy or duplicates it internally
- Keep the first version simple and justifiable

Recommended bias:
- Use a simple, defensible TF-IDF setup first
- Avoid over-tuning vectorizer settings in Step 3

L. Optional TF-IDF dataset / loader symmetry
- If it improves consistency, add a TFIDFDataset class in src/dataset.py
- It can wrap (X_tfidf[i], y[i]) pairs as a torch.utils.data.Dataset
- If implemented, also add TF-IDF DataLoader creation
- Reuse the same batch size, num_workers, and pin_memory conventions where appropriate
- Add TF-IDF DataLoader batch shape to console sanity checks
- This is optional, not mandatory, if plain matrices are cleaner for the baseline

M. Preserve consistency across experiment branches
- Ensure both preprocessing branches use:
  - the same raw split membership
  - the same cleaned text policy unless intentionally separated and documented
  - the same label mapping
- The only difference should be representation:
  - neural branch -> token sequences
  - baseline branch -> TF-IDF vectors

N. Explicitly document shared vs branch-specific behavior
- Add one explicit section in the preprocessing summary documenting:
  - what is shared across all models
  - what is neural-specific
  - what is baseline-specific
- Add one explicit section documenting what later steps are allowed to reuse from Step 3
- Examples of reusable outputs:
  - label mappings from Step 2
  - saved vocabulary
  - saved vectorizer
  - max sequence length
  - cleaning config
  - processed tensor shapes
  - DataLoader creation functions
- Explicitly require inference-time preprocessing to reuse saved artifacts rather than re-fit anything

O. Save preprocessing artifacts for reproducibility
Save reusable artifacts such as:
- data/artifacts/vocab.json
- data/artifacts/preprocessing_summary.json
- data/artifacts/preprocessing_summary.md
- data/artifacts/sequence_length_stats.json
- data/artifacts/tfidf_vectorizer.pkl
- optionally sample transformed outputs for sanity checks

The preprocessing summary should contain at minimum:
1. text cleaning policy
2. whether lowercasing was used
3. tokenizer description
4. vocabulary size
5. special tokens and ids
6. selected max sequence length
7. sequence length statistics
8. OOV / UNK statistics by split
9. truncation statistics by split
10. TF-IDF configuration
11. fitted TF-IDF vocabulary size
12. fitted TF-IDF feature dimension
13. output tensor / matrix shapes by split
14. confirmation that vocab/vectorizer were fit on training data only
15. path/reference to Step 2 label mapping artifacts
16. artifact output directory
17. note that inference must reuse saved artifacts instead of re-fitting

P. Keep preprocessing configurable
- Centralize preprocessing-related settings in src/config.py
- Likely settings include:
  - lowercase flag
  - max sequence length
  - min token frequency if used
  - batch size
  - TF-IDF max features
  - TF-IDF ngram range
  - DataLoader worker count
  - padding side
  - truncation side
  - artifact output directory
  - random seed / determinism settings if not already centralized elsewhere
- Do not scatter magic numbers across files

Q. Add validation checks / assertions
Include explicit sanity checks so preprocessing fails loudly if something is wrong:
- no split is empty
- vocabulary contains PAD and UNK
- special token ids are valid
- PAD and UNK ids do not drift across reruns
- only training data is used to fit vocab
- only training data is used to fit TF-IDF vectorizer
- numericalized sequences contain only valid token ids
- padded sequences all share the selected fixed length
- labels map correctly using Step 2 artifacts
- label tensor lengths exactly match input counts for every split
- DataLoader batch iteration works
- DataLoader batch counts are non-zero for every split
- TF-IDF train / val / test feature dimensions match
- TF-IDF train / val / test feature ordering matches
- no NaN / inf values appear in padded tensors
- no NaN / inf values appear in TF-IDF matrices
- preprocessing artifacts write successfully
- artifact directories are created automatically if missing

R. Console output expectations
When Step 3 runs, it should print a concise but informative summary including:
- cleaning policy
- tokenizer summary
- vocabulary size
- special tokens and ids
- sequence length stats
- chosen max sequence length
- OOV / UNK rates by split
- truncation rates by split
- padded tensor shapes by split
- one example raw text -> cleaned text -> tokens -> ids
- include at least one val/test example demonstrating unknown-token behavior if present
- TF-IDF feature matrix shapes by split
- TF-IDF fitted feature dimension
- DataLoader batch shapes
- artifact save locations

S. File / artifact expectations
At the end of Step 3, I expect artifacts similar to these:
- data/artifacts/vocab.json
- data/artifacts/sequence_length_stats.json
- data/artifacts/preprocessing_summary.json
- data/artifacts/preprocessing_summary.md
- data/artifacts/tfidf_vectorizer.pkl
- optionally:
  - data/artifacts/sample_preprocessed_examples.json

Depending on design, I also expect in-memory outputs or saved serialized outputs for:
- train / val / test padded sequences
- train / val / test label tensors
- train / val / test TF-IDF matrices
- train / val / test DataLoaders or DataLoader builders
- optionally tfidf_input_dim as an explicit artifact field for later model construction

T. Fresh-process reuse requirement
- Prove that saved preprocessing artifacts can be loaded in a fresh process and reused successfully
- Confirm inference-time preprocessing can run from saved artifacts without re-fitting vocabulary or TF-IDF
- Add a simple smoke test for loading:
  - label mappings
  - vocabulary
  - TF-IDF vectorizer
  - config-derived preprocessing settings

U. Determinism requirement
- Given the same data, seed, and config, rerunning preprocessing should produce identical outputs
- Confirm this explicitly for:
  - vocabulary construction
  - special token ids
  - numericalized outputs
  - TF-IDF feature dimension and ordering
- Save or log a short determinism confirmation in the summary

V. Definition of done
Step 3 is only complete if all of the following are true:
- text cleaning / normalization policy is implemented and documented
- neural tokenization pipeline works consistently
- vocabulary is built from training data only
- vocabulary ordering is deterministic, including tie cases
- PAD / UNK ids are fixed and stable across reruns
- token sequences are converted to integer ids correctly
- OOV / UNK statistics are saved and reviewed
- max sequence length is chosen from training-data analysis and documented
- truncation statistics are saved and reviewed
- sequences are padded / truncated consistently
- labels reuse the Step 2 mapping without drift
- PyTorch Dataset objects work correctly
- PyTorch DataLoaders work correctly
- TF-IDF vectorizer is fit on training data only
- TF-IDF features exist for train / validation / test
- TF-IDF fitted feature dimension is saved
- preprocessing artifacts are saved
- saved artifacts can be loaded in a fresh process and reused successfully
- inference-time preprocessing uses saved artifacts rather than re-fitting
- preprocessing rerun produces identical outputs given the same seed/config/data
- the pipeline is reusable and consistent across experiments
- no split leakage was introduced

W. Deliverable quality bar
The implementation should be clean, minimal, reproducible, and easy to reason about. Prefer simple and correct over clever. The output of Step 3 should make Step 4, Step 5, and Step 6 feel straightforward because every model can rely on the same trusted preprocessing contract.

X. Suggested self-check after implementation
After coding, self-check against this exact checklist and confirm:
1. which functions implement each preprocessing responsibility
2. which files store each artifact
3. how split leakage is prevented
4. how sequence length was selected
5. how OOV / truncation statistics were measured
6. how TF-IDF fitting was kept separate but consistent
7. how inference-time reuse works without re-fitting
8. that one end-to-end preprocessing run completes successfully without training any model
9. that a rerun with the same seed/config/data produces identical outputs