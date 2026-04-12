"""Inspect the CLINC150 dataset: splits, class balance, OOS stats, and sample queries.

Run with:  uv run python scripts/explore_dataset.py
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import DATASET_CONFIG
from src.constants import ARTIFACTS_DIR, OOS_LABEL_ID, OOS_LABEL_NAME
from src.dataset import CLINCDataset

SAMPLE_INTENTS = [
    "greeting",
    "book_flight",
    "flight_status",
    "weather",
    "translate",
    "balance",
    "shopping_list",
    "shopping_list_update",
    "pto_request",
    "pto_request_status",
]
SAMPLES_PER_INTENT = 5

OOS_EVAL_STRATEGY = (
    "Multiclass classification with an explicit OOS class (label 42). "
    "All models classify into 151 classes; OOS precision/recall/F1 computed "
    "directly from class-42 predictions. "
    "Threshold-based OOS detection is deferred to stretch goals (Step 12)."
)


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _assert_written(path: Path) -> None:
    assert path.exists() and path.stat().st_size > 0, f"Artifact missing or empty: {path}"


# ------------------------------------------------------------------
# Artifact writers
# ------------------------------------------------------------------


def _write_label_maps(ds: CLINCDataset) -> tuple[Path, Path]:
    label_to_id_path = ARTIFACTS_DIR / "label_to_id.json"
    id_to_label_path = ARTIFACTS_DIR / "id_to_label.json"
    _write_json(label_to_id_path, {n: ds.label_id(n) for n in ds.label_names})
    _write_json(id_to_label_path, {str(i): ds.label_name(i) for i in range(ds.num_classes)})
    return label_to_id_path, id_to_label_path


def _write_distribution_csv(ds: CLINCDataset, split: str) -> Path:
    dist = ds.class_distribution(split)
    path = ARTIFACTS_DIR / f"{split}_label_distribution.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label_id", "label_name", "count"])
        for label_id in range(ds.num_classes):
            writer.writerow([label_id, ds.label_name(label_id), dist[label_id]])
    return path


def _write_sample_queries(ds: CLINCDataset) -> Path:
    samples: dict[str, list[str]] = {}
    for intent_name in SAMPLE_INTENTS:
        label_id = ds.label_id(intent_name)
        samples[intent_name] = ds.sample_examples("train", label_id, n=SAMPLES_PER_INTENT)
    samples[OOS_LABEL_NAME] = ds.sample_examples("train", OOS_LABEL_ID, n=SAMPLES_PER_INTENT)
    path = ARTIFACTS_DIR / "sample_queries_by_label.json"
    _write_json(path, samples)
    return path


def _build_quirks_and_caveats(ds: CLINCDataset) -> list[str]:
    """Compute dataset notes directly from the active subset statistics."""

    sizes = ds.split_sizes()
    oos_counts = ds.oos_counts()
    train_distribution = ds.class_distribution("train")
    in_scope_train_counts = [count for label_id, count in train_distribution.items() if label_id != OOS_LABEL_ID]

    train_oos_pct = 100.0 * oos_counts["train"] / sizes["train"]
    test_oos_pct = 100.0 * oos_counts["test"] / sizes["test"]

    quirks = [
        (
            f"Test split has ~{test_oos_pct:.1f}% OOS examples vs "
            f"~{train_oos_pct:.1f}% in train, creating a substantial distribution shift."
        )
    ]

    in_scope_min = min(in_scope_train_counts)
    in_scope_max = max(in_scope_train_counts)
    if in_scope_min == in_scope_max:
        quirks.append(
            f"In-scope classes are perfectly balanced in the '{ds.subset}' subset "
            f"({in_scope_min} training examples per class)."
        )
    else:
        quirks.append(
            f"In-scope training counts vary between {in_scope_min} and {in_scope_max} "
            f"examples per class in the '{ds.subset}' subset."
        )

    oos_train_count = oos_counts["train"]
    reference_count = in_scope_train_counts[0]
    ratio = oos_train_count / reference_count
    if abs(ratio - 1.0) < 0.005:
        quirks.append(f"OOS has {oos_train_count} training examples, matching the per-class in-scope training count.")
    else:
        quirks.append(
            f"OOS has {oos_train_count} training examples, {ratio:.2f}x the per-class in-scope training count "
            f"({reference_count})."
        )

    return quirks


def _write_summary_json(ds: CLINCDataset, train_counts: list[int]) -> Path:
    sizes = ds.split_sizes()
    quirks_and_caveats = _build_quirks_and_caveats(ds)
    summary = {
        "dataset_source": ds.source,
        "subset": ds.subset,
        "text_field": ds.text_field,
        "label_field": ds.label_field,
        "split_sizes": sizes,
        "num_classes": ds.num_classes,
        "label_names": ds.label_names,
        "oos_label_name": OOS_LABEL_NAME,
        "oos_label_id": OOS_LABEL_ID,
        "oos_counts": ds.oos_counts(),
        "in_scope_counts": ds.in_scope_counts(),
        "train_class_count_min": train_counts[0],
        "train_class_count_max": train_counts[-1],
        "train_class_count_mean": round(sum(train_counts) / len(train_counts), 2),
        "oos_evaluation_strategy": OOS_EVAL_STRATEGY,
        "quirks_or_caveats": quirks_and_caveats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = ARTIFACTS_DIR / "dataset_summary.json"
    _write_json(path, summary)
    return path


def _write_summary_md(ds: CLINCDataset, train_counts: list[int]) -> Path:
    sizes = ds.split_sizes()
    oos = ds.oos_counts()
    quirks_and_caveats = _build_quirks_and_caveats(ds)
    in_scope = ds.in_scope_counts()

    lines = [
        "# CLINC150 Dataset Summary",
        "",
        f"**Source:** `{ds.source}` (subset: `{ds.subset}`)",
        f"**Text field:** `{ds.text_field}`  ",
        f"**Label field:** `{ds.label_field}`  ",
        f"**Total classes:** {ds.num_classes} (150 in-scope + 1 OOS)",
        f"**OOS label:** `{OOS_LABEL_NAME}` (id {OOS_LABEL_ID})",
        "",
        "## Split Sizes",
        "",
        "| Split | Total | In-scope | OOS | OOS % |",
        "|-------|------:|---------:|----:|------:|",
    ]
    for s in ("train", "validation", "test"):
        pct = 100.0 * oos[s] / sizes[s]
        lines.append(f"| {s} | {sizes[s]:,} | {in_scope[s]:,} | {oos[s]:,} | {pct:.1f}% |")

    lines += [
        "",
        "## Training Class Balance",
        "",
        f"- Min count per class: {train_counts[0]}",
        f"- Max count per class: {train_counts[-1]}",
        f"- Mean count per class: {sum(train_counts) / len(train_counts):.1f}",
        "",
        "## OOS Evaluation Strategy",
        "",
        OOS_EVAL_STRATEGY,
        "",
        "## Quirks and Caveats",
        "",
    ]
    for note in quirks_and_caveats:
        lines.append(f"- {note}")

    path = ARTIFACTS_DIR / "dataset_summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    ds = CLINCDataset.load(DATASET_CONFIG)

    # ---- dataset source ----
    _section("Dataset source")
    print(f"  Source:      {ds.source}")
    print(f"  Subset:      {ds.subset}")
    print(f"  Columns:     {ds.column_names}")
    print(f"  Text field:  {ds.text_field}")
    print(f"  Label field: {ds.label_field}")
    print(f"  Features:    {ds.feature_types}")

    # ---- data quality ----
    _section("Data quality validation")
    report = ds.validate()
    print(f"  Null texts:           {report.null_text_counts}")
    print(f"  Empty texts:          {report.empty_text_counts}")
    print(f"  Out-of-range labels:  {report.out_of_range_label_counts}")
    print(f"  Clean: {report.is_clean}")

    # ---- split sizes ----
    _section("Split sizes")
    sizes = ds.split_sizes()
    oos = ds.oos_counts()
    in_scope = ds.in_scope_counts()
    for split in ("train", "validation", "test"):
        pct = 100.0 * oos[split] / sizes[split]
        print(
            f"  {split:>12s}: {sizes[split]:>5,} total"
            f"  |  {in_scope[split]:>5,} in-scope"
            f"  |  {oos[split]:>4,} OOS ({pct:.1f}%)"
        )

    # ---- label overview ----
    _section("Label overview")
    print(f"  Total classes: {ds.num_classes}")
    print(f"  Label names:   {ds.label_names[:10]} ... (showing first 10)")

    # ---- class balance (training set) ----
    _section("Training set class distribution")
    train_dist = ds.class_distribution("train")
    train_counts = sorted(count for label_id, count in train_dist.items() if label_id != OOS_LABEL_ID)
    print(f"  In-scope min count:  {train_counts[0]}")
    print(f"  In-scope max count:  {train_counts[-1]}")
    print(f"  In-scope mean count: {sum(train_counts) / len(train_counts):.1f}")
    print("\n  Bottom-5 classes by count:")
    for label_id, count in train_dist.most_common()[:-6:-1]:
        print(f"    {ds.label_name(label_id):>30s} (id={label_id}): {count}")
    print("  Top-5 classes by count:")
    for label_id, count in train_dist.most_common(5):
        print(f"    {ds.label_name(label_id):>30s} (id={label_id}): {count}")

    # ---- sample queries ----
    _section("Sample queries")
    for intent_name in SAMPLE_INTENTS:
        label_id = ds.label_id(intent_name)
        examples = ds.sample_examples("train", label_id, n=3)
        print(f"\n  [{intent_name}] (id={label_id}):")
        for ex in examples:
            print(f"    - {ex}")

    oos_examples = ds.sample_examples("train", OOS_LABEL_ID, n=3)
    print(f"\n  [{OOS_LABEL_NAME}] (id={OOS_LABEL_ID}):")
    for ex in oos_examples:
        print(f"    - {ex}")

    # ---- OOS evaluation strategy ----
    _section("OOS evaluation strategy")
    print(f"  {OOS_EVAL_STRATEGY}")

    # ---- save artifacts ----
    _section("Saving artifacts")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    artifact_paths: list[Path] = []

    l2i, i2l = _write_label_maps(ds)
    artifact_paths.extend([l2i, i2l])

    for split in ("train", "validation", "test"):
        artifact_paths.append(_write_distribution_csv(ds, split))

    artifact_paths.append(_write_sample_queries(ds))
    artifact_paths.append(_write_summary_json(ds, train_counts))
    artifact_paths.append(_write_summary_md(ds, train_counts))

    # ---- verify all artifacts ----
    _section("Artifact verification")
    for path in artifact_paths:
        _assert_written(path)
        print(f"  OK  {path.name:>40s}  ({path.stat().st_size:,} bytes)")

    # ---- verify split counts match ----
    summary_raw = json.loads((ARTIFACTS_DIR / "dataset_summary.json").read_text())
    for split in ("train", "validation", "test"):
        assert summary_raw["split_sizes"][split] == sizes[split], f"Summary split count mismatch for {split}"
    print("\n  Split counts in summary match loaded dataset.")
    print("  All artifacts verified.")


if __name__ == "__main__":
    main()
