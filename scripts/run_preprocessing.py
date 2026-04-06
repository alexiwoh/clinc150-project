"""Step 3 entry point: run the full preprocessing pipeline, save artifacts, verify."""

from __future__ import annotations

import json

import torch

from src.config import DATASET_CONFIG, NUM_WORKERS, PREPROCESSING_CONFIG, PreprocessingConfig, get_device
from src.constants import ARTIFACTS_DIR
from src.dataset import IntentDataset, TFIDFDataset, create_dataloaders
from src.preprocessing import (
    _build_sample_examples,
    _SPLIT_NAMES,
    clean_text,
    export_preprocessing_artifacts,
    load_preprocessing_artifacts,
    numericalize,
    print_preprocessing_summary,
    run_preprocessing,
    tokenize_text,
)


def _load_label_mappings() -> tuple[dict[str, int], dict[int, str]]:
    """Load Step 2 label mappings from saved artifacts."""
    label_to_id_path = ARTIFACTS_DIR / "label_to_id.json"
    id_to_label_path = ARTIFACTS_DIR / "id_to_label.json"
    assert label_to_id_path.exists(), f"Missing Step 2 artifact: {label_to_id_path}"
    assert id_to_label_path.exists(), f"Missing Step 2 artifact: {id_to_label_path}"

    label_to_id: dict[str, int] = json.loads(label_to_id_path.read_text())
    id_to_label: dict[int, str] = {int(k): v for k, v in json.loads(id_to_label_path.read_text()).items()}
    return label_to_id, id_to_label


def _extract_splits(ds):  # noqa: ANN001
    """Extract raw texts and integer labels from CLINCDataset for each split.
    
    Args:
        ds: The CLINCDataset instance.

    Returns:
        A tuple of dictionaries containing the raw texts and integer labels for each split.
    """
    texts: dict[str, list[str]] = {}
    labels: dict[str, list[int]] = {}
    for split in _SPLIT_NAMES:
        hf_split = ds[split]
        texts[split] = hf_split["text"]
        labels[split] = hf_split["intent"]
    return texts, labels


def _verify_dataloaders(loaders: dict[str, object], label: str) -> None:
    """Iterate one batch from each loader and print shapes."""
    for split, loader in loaders.items():
        batch_count = len(loader)
        assert batch_count > 0, f"{label} {split} DataLoader has 0 batches"
        batch = next(iter(loader))
        x, y = batch
        print(f"  {label} {split}: input={list(x.shape)}, labels={list(y.shape)}, batches={batch_count}")


def _run_smoke_test() -> None:
    """Load saved artifacts in a fresh context and verify they're usable."""
    print("\n--- Smoke test: loading artifacts from disk ---")
    artifacts = load_preprocessing_artifacts(ARTIFACTS_DIR)
    vocab = artifacts["vocab"]
    vectorizer = artifacts["vectorizer"]

    test_text = "what is the weather like today"
    cleaned = clean_text(test_text, lowercase=True)
    tokens = tokenize_text(cleaned)
    ids = numericalize(tokens, vocab)
    print(f"  Text: {test_text!r}")
    print(f"  Cleaned: {cleaned!r}")
    print(f"  Tokens: {tokens}")
    print(f"  IDs: {ids}")

    tfidf_vec = vectorizer.transform([cleaned])
    print(f"  TF-IDF shape: {tfidf_vec.shape}")
    print("  Smoke test PASSED")


def _run_determinism_check(
    texts: dict[str, list[str]],
    labels: dict[str, list[int]],
    config: PreprocessingConfig,
) -> None:
    """Re-run preprocessing and verify outputs match."""
    print("\n--- Determinism check ---")
    r1 = run_preprocessing(texts, labels, config)
    r2 = run_preprocessing(texts, labels, config)

    assert r1.vocab.token_to_id == r2.vocab.token_to_id, "Vocabulary mismatch"
    assert r1.vocab.id_to_token == r2.vocab.id_to_token, "Vocabulary id_to_token mismatch"
    for split in _SPLIT_NAMES:
        assert torch.equal(r1.sequences[split], r2.sequences[split]), f"Sequence mismatch in {split}"
        assert torch.equal(r1.labels[split], r2.labels[split]), f"Label mismatch in {split}"
    print("  Determinism check PASSED")


def main() -> None:
    from src.dataset import CLINCDataset

    print("=" * 60)
    print("  Step 3: Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load dataset
    print("\nLoading CLINC150 dataset...")
    ds = CLINCDataset.load(DATASET_CONFIG)
    texts, labels = _extract_splits(ds)

    # 2. Load Step 2 label mappings and verify consistency
    label_to_id, id_to_label = _load_label_mappings()
    assert ds.num_classes == len(label_to_id), "Label count mismatch with Step 2 artifacts"
    for name in ds.label_names:
        assert name in label_to_id, f"Label {name!r} missing from Step 2 label_to_id"
        assert label_to_id[name] == ds.label_id(name), f"Label id mismatch for {name!r}"

    config = PREPROCESSING_CONFIG

    # 3. Run preprocessing
    print("\nRunning preprocessing pipeline...")
    result = run_preprocessing(texts, labels, config)

    # 4. Build sample examples
    cleaned: dict[str, list[str]] = {}
    tokenized: dict[str, list[list[str]]] = {}
    numericalized: dict[str, list[list[int]]] = {}
    for split in _SPLIT_NAMES:
        cleaned[split] = [clean_text(t, lowercase=config.lowercase) for t in texts[split]]
        tokenized[split] = [tokenize_text(t) for t in cleaned[split]]
        numericalized[split] = [numericalize(toks, result.vocab) for toks in tokenized[split]]

    sample_examples = _build_sample_examples(texts, cleaned, tokenized, numericalized, result.vocab, n=3)

    # 5. Export artifacts
    print("\nSaving artifacts...")
    artifact_paths = export_preprocessing_artifacts(
        vocab=result.vocab,
        tfidf_result=result.tfidf,
        seq_length_stats=result.seq_length_stats,
        summary=result.summary,
        sample_examples=sample_examples,
        artifacts_dir=config.artifacts_dir,
    )

    # 6. Print console summary
    print_preprocessing_summary(result, artifact_paths)

    # 7. Print sample before/after examples
    print(f"\n{'=' * 60}")
    print("  Sample Preprocessed Examples")
    print("=" * 60)
    for ex in sample_examples:
        unk_marker = " [CONTAINS UNK]" if ex["contains_unk"] else ""
        print(f"\n  [{ex['split']}][{ex['index']}]{unk_marker}")
        print(f"    Raw:     {ex['raw_text']!r}")
        print(f"    Cleaned: {ex['cleaned_text']!r}")
        print(f"    Tokens:  {ex['tokens']}")
        print(f"    IDs:     {ex['token_ids']}")

    # 8. Build and verify DataLoaders
    device = get_device()
    pin_memory = device.type == "cuda"

    print(f"\n{'=' * 60}")
    print("  DataLoader Verification")
    print("=" * 60)

    neural_datasets = {split: IntentDataset(result.sequences[split], result.labels[split]) for split in _SPLIT_NAMES}
    neural_loaders = create_dataloaders(
        neural_datasets,
        batch_size=config.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        random_seed=config.random_seed,
    )
    _verify_dataloaders(neural_loaders, "Neural")

    tfidf_datasets = {split: TFIDFDataset(getattr(result.tfidf, split), result.labels[split]) for split in _SPLIT_NAMES}
    tfidf_loaders = create_dataloaders(
        tfidf_datasets,
        batch_size=config.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        random_seed=config.random_seed,
    )
    _verify_dataloaders(tfidf_loaders, "TF-IDF")

    # 9. Smoke test
    _run_smoke_test()

    # 10. Determinism check
    _run_determinism_check(texts, labels, config)

    print(f"\n{'=' * 60}")
    print("  Step 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
