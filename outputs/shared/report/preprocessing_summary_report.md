## Preprocessing Summary

**Source**: data/artifacts/preprocessing_summary.json

### Text Cleaning

Strategy: strip + normalize whitespace + optional lowercase. Preserves: punctuation, contractions, digits.

### Tokenization and Vocabulary

Tokenizer: whitespace split. Vocabulary size: 6161 (built from training data only). Special tokens: `<PAD>` = 0, `<UNK>` = 1.

### Sequence Handling

Max sequence length: 20. Statistics (training set): mean 8.32, median 8.0, p90 13, p95 14, max 28, min 1.

### OOV Rates

| Split      | Total Tokens | Unknown Tokens | OOV Rate |
| ---------- | ------------ | -------------- | -------- |
| Train      | 126,920      | 0              | 0.0000   |
| Validation | 25,674       | 868            | 0.0338   |
| Test       | 45,606       | 2,362          | 0.0518   |

### Truncation Rates

| Split      | Total Sequences | Truncated | Truncation Rate |
| ---------- | --------------- | --------- | --------------- |
| Train      | 15,250          | 32        | 0.0021          |
| Validation | 3,100           | 20        | 0.0065          |
| Test       | 5,500           | 10        | 0.0018          |

### TF-IDF Configuration

Max features: 10,000. N-gram range: (1, 2). Fitted on training data only. Note: TF-IDF is used only for the MLP baseline; Text CNN and BiLSTM use token-ID sequences.
