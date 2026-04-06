"""Generate Step 2 dataset visualizations for CLINC150.

Run with:  uv run python scripts/visualize_dataset.py

Produces six figures and one CSV in ``outputs/figures/``, plus summary
statistics logged to stdout.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean, median

from src.config import DATASET_CONFIG
from src.constants import FIGURES_DIR, OOS_LABEL_ID, OOS_LABEL_NAME
from src.dataset import CLINCDataset
from src.visualizers import DatasetVisualizer

_SPLIT_NAMES = ("train", "validation", "test")

_REPRESENTATIVE_INTENTS = [
    "greeting",
    "goodbye",
    "book_flight",
    "weather",
    "translate",
    "balance",
    "shopping_list",
    "restaurant_reservation",
    "alarm",
    "play_music",
]
_SAMPLES_PER_INTENT = 3


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _assert_written(path: Path) -> None:
    assert path.exists() and path.stat().st_size > 0, f"Output missing or empty: {path}"


def _word_lengths(ds: CLINCDataset, split: str) -> list[int]:
    """Compute whitespace-based word counts for every text in *split*."""
    return [len(text.split()) for text in ds[split]["text"]]


def _build_representative_rows(ds: CLINCDataset) -> list[dict[str, str]]:
    """Collect a small set of representative queries across intents and splits."""
    rows: list[dict[str, str]] = []
    for split in _SPLIT_NAMES:
        for intent_name in _REPRESENTATIVE_INTENTS:
            label_id = ds.label_id(intent_name)
            for query in ds.sample_examples(split, label_id, n=_SAMPLES_PER_INTENT):
                rows.append({"split": split, "label": intent_name, "scope": "in_scope", "query": query})
        for query in ds.sample_examples(split, OOS_LABEL_ID, n=_SAMPLES_PER_INTENT):
            rows.append({"split": split, "label": OOS_LABEL_NAME, "scope": "oos", "query": query})
    return rows


def main() -> None:
    ds = CLINCDataset.load(DATASET_CONFIG)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []

    # ---- 1. split sizes ----
    _section("1. Split sizes")
    split_sizes = ds.split_sizes()
    for s in _SPLIT_NAMES:
        print(f"  {s:>12s}: {split_sizes[s]:>6,}")

    path = FIGURES_DIR / "split_sizes.png"
    DatasetVisualizer.plot_split_sizes(split_sizes, path)
    output_paths.append(path)

    # ---- 2. class distribution ----
    _section("2. Intent class distribution")
    train_dist = ds.class_distribution("train")
    label_names = [ds.label_name(i) for i in range(ds.num_classes)]
    counts = [train_dist[i] for i in range(ds.num_classes)]
    print(f"  Classes: {ds.num_classes}")
    print(f"  Min count:  {min(counts)}")
    print(f"  Max count:  {max(counts)}")
    print(f"  Mean count: {mean(counts):.1f}")

    full_path = FIGURES_DIR / "intent_class_distribution.png"
    DatasetVisualizer.plot_class_distribution(label_names, counts, full_path)
    output_paths.append(full_path)

    top_path = FIGURES_DIR / "intent_class_distribution_top20.png"
    DatasetVisualizer.plot_class_distribution(label_names, counts, top_path, top_n=20)
    output_paths.append(top_path)

    # ---- 3. OOS vs in-scope ----
    _section("3. OOS vs in-scope distribution")
    oos = ds.oos_counts()
    in_scope = ds.in_scope_counts()
    for s in _SPLIT_NAMES:
        total = oos[s] + in_scope[s]
        pct = 100.0 * oos[s] / total
        print(f"  {s:>12s}: {in_scope[s]:>5,} in-scope | {oos[s]:>5,} OOS ({pct:.1f}%)")

    path = FIGURES_DIR / "oos_vs_inscope_distribution.png"
    DatasetVisualizer.plot_oos_vs_inscope(oos, in_scope, path)
    output_paths.append(path)

    # ---- 4. query length histogram ----
    _section("4. Query length statistics (all splits)")
    all_lengths: list[int] = []
    lengths_by_split: dict[str, list[int]] = {}
    for s in _SPLIT_NAMES:
        split_lengths = _word_lengths(ds, s)
        lengths_by_split[s] = split_lengths
        all_lengths.extend(split_lengths)

    print(f"  Total queries: {len(all_lengths):,}")
    print(f"  Min length:    {min(all_lengths)}")
    print(f"  Max length:    {max(all_lengths)}")
    print(f"  Mean length:   {mean(all_lengths):.1f}")
    print(f"  Median length: {median(all_lengths):.1f}")

    path = FIGURES_DIR / "query_length_histogram.png"
    DatasetVisualizer.plot_query_length_histogram(all_lengths, path)
    output_paths.append(path)

    # ---- 5. query length boxplot by split ----
    _section("5. Query length by split")
    for s in _SPLIT_NAMES:
        sl = lengths_by_split[s]
        print(f"  {s:>12s}: min={min(sl):>2}  max={max(sl):>2}  mean={mean(sl):.1f}  median={median(sl):.1f}")

    path = FIGURES_DIR / "query_length_by_split_boxplot.png"
    DatasetVisualizer.plot_query_length_boxplot(lengths_by_split, path)
    output_paths.append(path)

    # ---- 6. representative queries ----
    _section("6. Representative queries")
    rows = _build_representative_rows(ds)
    print(f"  Total rows: {len(rows)}")

    path = FIGURES_DIR / "representative_queries.csv"
    DatasetVisualizer.save_representative_queries(rows, path)
    output_paths.append(path)

    # ---- verify outputs ----
    _section("Output verification")
    for p in output_paths:
        _assert_written(p)
        print(f"  OK  {p.name:>45s}  ({p.stat().st_size:,} bytes)")
    print(f"\n  All {len(output_paths)} outputs verified.")


if __name__ == "__main__":
    main()
