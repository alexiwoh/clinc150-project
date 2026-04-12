"""Step 4 entry point: train, tune, and evaluate the TF-IDF + MLP baseline.

Also runs a Step 5/6 readiness smoke test to verify token-sequence
DataLoaders can still be reconstructed from saved preprocessing artifacts.
"""

from __future__ import annotations

import logging
import sys

import torch
from torch.utils.data.dataloader import DataLoader

from src.config import DATASET_CONFIG, PREPROCESSING_CONFIG, PreprocessingConfig
from src.dataset import CLINCDataset, IntentDataset, create_dataloaders
from src.preprocessing import _SPLIT_NAMES, PreprocessingResult, clean_text, run_preprocessing
from src.train import run_mlp_experiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def _step56_smoke_test() -> None:
    """Verify token-sequence DataLoaders can be reconstructed (spec U2).

    Token-sequence tensors are not saved to disk. This smoke test
    reconstructs them by running the preprocessing pipeline and wrapping
    the outputs in IntentDataset + create_dataloaders(), then confirms
    batch shapes and dtypes.
    """
    print(f"\n{'=' * 60}")
    print("  Step 5/6 Readiness Smoke Test")
    print("=" * 60)

    ds: CLINCDataset = CLINCDataset.load(DATASET_CONFIG)
    texts: dict[str, list[str]] = {}
    labels: dict[str, list[int]] = {}
    for split in _SPLIT_NAMES:
        raw = ds[split]
        texts[split] = [clean_text(t) for t in raw["text"]]
        labels[split] = raw["intent"]

    config: PreprocessingConfig = PREPROCESSING_CONFIG
    result: PreprocessingResult = run_preprocessing(texts, labels, config)

    datasets: dict[str, IntentDataset] = {
        split: IntentDataset(result.sequences[split], result.labels[split]) for split in _SPLIT_NAMES
    }
    loaders: dict[str, DataLoader] = create_dataloaders(
        datasets,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    expected_seq_len = config.max_seq_length
    for split in _SPLIT_NAMES:
        batch_x, batch_y = next(iter(loaders[split]))
        assert batch_x.shape[1] == expected_seq_len, (
            f"Expected seq_len={expected_seq_len}, got {batch_x.shape[1]} in {split}"
        )
        assert batch_x.dtype == torch.long, f"Expected input dtype=torch.long, got {batch_x.dtype} in {split}"
        assert batch_y.dtype == torch.long, f"Expected label dtype=torch.long, got {batch_y.dtype} in {split}"
        print(
            f"  {split}: input={list(batch_x.shape)} dtype={batch_x.dtype}"
            f" | label={list(batch_y.shape)} dtype={batch_y.dtype}"
        )

    print("  Step 5/6 smoke test PASSED")


def main() -> None:
    print("=" * 60)
    print("  Step 4: TF-IDF + MLP Baseline")
    print("=" * 60)

    experiment_results = run_mlp_experiment()

    test_results = experiment_results["test_results"]
    print(f"\n{'=' * 60}")
    print("  Step 4 COMPLETE — Final Summary")
    print(f"{'=' * 60}")
    print(f"  Best run:            {experiment_results['best_run_name']}")
    print(f"  Test accuracy:       {test_results['test_accuracy']:.4f}")
    print(f"  Test macro F1:       {test_results['test_macro_f1']:.4f}")
    print(f"  Test precision:      {test_results['test_precision']:.4f}")
    print(f"  Test recall:         {test_results['test_recall']:.4f}")
    print(f"  OOS F1:              {test_results['oos_f1']:.4f}")
    print(f"  Inference:           {test_results['inference_latency']['avg_ms_per_example']:.2f} ms/example")
    print(f"  Total params:        {test_results['total_parameters']:,}")
    print(f"  Trainable params:    {test_results['trainable_parameters']:,}")
    print(f"  Checkpoint:          {test_results['checkpoint_path']}")

    _step56_smoke_test()

    print(f"\n{'=' * 60}")
    print("  Step 4 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
