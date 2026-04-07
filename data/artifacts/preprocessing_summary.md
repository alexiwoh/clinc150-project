# Preprocessing Summary

## text_cleaning_policy

- **lowercase**: True
- **strategy**: strip + normalize whitespace + optional lowercase
- **preserves**: punctuation, contractions, digits
- **tokenizer**: whitespace split
- **vocabulary_size**: 6161
## special_tokens

- **<PAD>**: 0
- **<UNK>**: 1
- **max_seq_length**: 20
## sequence_length_stats

- **mean**: 8.32
- **median**: 8.0
- **p90**: 13
- **p95**: 14
- **max**: 28
- **min**: 1
- **count**: 15250
## oov_stats

- **train**: {'total_tokens': 126920, 'unknown_tokens': 0, 'oov_rate': 0.0}
- **validation**: {'total_tokens': 25674, 'unknown_tokens': 868, 'oov_rate': 0.033809}
- **test**: {'total_tokens': 45606, 'unknown_tokens': 2362, 'oov_rate': 0.051791}
## truncation_stats

- **train**: {'total_sequences': 15250, 'truncated': 32, 'truncation_rate': 0.002098}
- **validation**: {'total_sequences': 3100, 'truncated': 20, 'truncation_rate': 0.006452}
- **test**: {'total_sequences': 5500, 'truncated': 10, 'truncation_rate': 0.001818}
## tfidf_config

- **max_features**: 10000
- **ngram_range**: [1, 2]
- **lowercase**: False
- **stop_words**: None
- **note**: Lowercase handled by shared cleaning, not duplicated by vectorizer
- **tfidf_fitted_vocab_size**: 10000
- **tfidf_fitted_feature_dim**: 10000
## output_shapes

- **train**: {'sequences': [15250, 20], 'labels': [15250], 'tfidf': [15250, 10000]}
- **validation**: {'sequences': [3100, 20], 'labels': [3100], 'tfidf': [3100, 10000]}
- **test**: {'sequences': [5500, 20], 'labels': [5500], 'tfidf': [5500, 10000]}
- **vocab_fitted_on**: training data only
- **tfidf_fitted_on**: training data only
## step2_label_mapping_artifacts

- **label_to_id**: /Users/alexanderiwoh/Programming-Local/GitHub/clinc150-project/data/artifacts/label_to_id.json
- **id_to_label**: /Users/alexanderiwoh/Programming-Local/GitHub/clinc150-project/data/artifacts/id_to_label.json
- **artifact_output_directory**: /Users/alexanderiwoh/Programming-Local/GitHub/clinc150-project/data/artifacts
- **inference_reuse_note**: Inference must reuse saved vocab and TF-IDF vectorizer; never re-fit.
- **timestamp**: 2026-04-07T02:08:24.271730+00:00
