"""Report section generators (Spec Sections B–H, J–O).

Each public function loads its source artifacts, generates a Markdown section
draft and a JSON-serializable metadata dict matching the artifact contract,
and returns ``(markdown, metadata)``.

Sections are independent: each can be regenerated in isolation.
"""

from __future__ import annotations

from typing import Any

from src.constants import (
    PROJECT_ROOT,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
)
from src.enums import ModelID
from src.report.artifact_loader import load_json, resolve_repo_path
from src.report.figure_metadata import (
    StructuredClaim,
    build_structured_claim,
    load_figure_manifest_entries,
    render_related_figure_note,
)
from src.report.tables import (
    build_markdown_table,
    format_mean_std,
    format_param_count,
    format_ratio,
    format_time_seconds,
)

SectionOutput = tuple[str, dict[str, Any]]

# ---------------------------------------------------------------------------
# Well-known artifact repo-relative paths
# ---------------------------------------------------------------------------

_ARTIFACTS: dict[str, str] = {
    "model_comparison_aggregate": "outputs/shared/model_comparison_aggregate.json",
    "oos_summary_table": "outputs/shared/oos_summary_table.json",
    "efficiency_summary_table": "outputs/shared/efficiency_summary_table.json",
    "evaluation_protocol": "outputs/shared/evaluation_protocol.json",
    "figure_manifest": "outputs/shared/figure_manifest.json",
    "dataset_summary": "data/artifacts/dataset_summary.json",
    "preprocessing_summary": "data/artifacts/preprocessing_summary.json",
    "extended_metrics_comparison": "outputs/shared/analysis/extended_metrics_comparison.json",
    "calibration_summary": "outputs/shared/analysis/calibration_summary.json",
    "error_taxonomy_summary": "outputs/shared/analysis/error_taxonomy_summary.json",
    "oos_threshold_comparison": "outputs/shared/analysis/oos_threshold_comparison.json",
    "oos_false_accept_comparison": "outputs/shared/analysis/oos_false_accept_comparison.json",
    "cross_model_error_comparison": "outputs/shared/analysis/cross_model_error_comparison.json",
    "worst_classes_comparison": "outputs/shared/analysis/worst_classes_comparison.json",
    "most_confused_pairs_table": "outputs/shared/most_confused_pairs_table.json",
    "curated_report_examples": "outputs/shared/analysis/curated_report_examples.json",
    "error_analysis_summary": "outputs/shared/analysis/error_analysis_summary.json",
    "error_analysis_notes": "outputs/shared/analysis/error_analysis_notes.md",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load(key: str) -> dict[str, Any]:
    """Load a well-known JSON artifact by registry key."""
    return load_json(resolve_repo_path(_ARTIFACTS[key]))


def _frozen_path(mid: ModelID) -> str:
    return f"outputs/{mid}/frozen_final_config.json"


def _load_frozen(mid: ModelID) -> dict[str, Any]:
    return load_json(resolve_repo_path(_frozen_path(mid)))


def _best_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return max(rows, key=lambda r: r[key])


def _rows_by_model(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["model_id"]: r for r in rows}


def _meta(**kwargs: Any) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "protocol_version": PROTOCOL_VERSION, **kwargs}


def _scope_note(scope: str) -> str:
    if scope == "aggregate":
        return "*Aggregate over 3 runs (mean +/- std)*"
    return "*Representative run only (single seed)*"


def _append_related_figure_note(lines: list[str], claim: StructuredClaim) -> None:
    """Append a markdown note for resolved figure links when present."""

    note = render_related_figure_note(claim["related_figure_paths"])
    if note is None:
        return
    lines.append(note)
    lines.append("")


# ---------------------------------------------------------------------------
# Section B: Abstract / project summary
# ---------------------------------------------------------------------------


def generate_abstract() -> SectionOutput:
    """Spec section B / B2."""
    agg = _load("model_comparison_aggregate")
    oos = _load("oos_summary_table")
    eas = _load("error_analysis_summary")

    sources = [
        _ARTIFACTS["model_comparison_aggregate"],
        _ARTIFACTS["oos_summary_table"],
        _ARTIFACTS["error_analysis_summary"],
    ]

    best_f1 = _best_row(agg["rows"], "test_macro_f1_mean")
    best_oos = _best_row(oos["rows"], "oos_f1_mean")
    models = [m.display_name for m in ModelID]
    run_count: int = best_f1["run_count"]

    goal = (
        "Evaluate lightweight deep learning models for intent classification "
        "and out-of-scope (OOS) detection on the CLINC150 dataset."
    )
    framing = (
        "Comparison of a sparse-feature baseline (TF-IDF + MLP), a convolutional "
        "sequence model (Text CNN), and a recurrent sequence model (BiLSTM)."
    )
    key_finding: str = eas["top_recommendation"]

    f1_fmt = format_mean_std(best_f1["test_macro_f1_mean"], best_f1["test_macro_f1_std"])
    oos_fmt = format_mean_std(best_oos["oos_f1_mean"], best_oos["oos_f1_std"])

    md = "\n".join(
        [
            "## Abstract",
            "",
            goal,
            "",
            framing,
            "",
            f"Three models are compared: {', '.join(models)}.",
            "",
            f"**Headline Result**: {best_f1['display_name']} achieved the highest aggregate test macro F1 "
            f"of {f1_fmt} across {run_count} repeated runs ({sources[0]}).",
            "",
            f"**OOS Detection**: {best_oos['display_name']} achieved the highest aggregate OOS F1 "
            f"of {oos_fmt} ({sources[1]}).",
            "",
            f"**Key Finding**: {key_finding} ({sources[2]}).",
            "",
        ]
    )

    metadata = _meta(
        project_goal=goal,
        project_framing=framing,
        models=models,
        headline_result={
            "best_model": best_f1["model_id"],
            "metric": "test_macro_f1",
            "mean": best_f1["test_macro_f1_mean"],
            "std": best_f1["test_macro_f1_std"],
        },
        oos_headline={
            "best_model": best_oos["model_id"],
            "metric": "oos_f1",
            "mean": best_oos["oos_f1_mean"],
            "std": best_oos["oos_f1_std"],
        },
        key_finding=key_finding,
        source_artifacts=sources,
    )
    return md, metadata


# ---------------------------------------------------------------------------
# Section C: Dataset description
# ---------------------------------------------------------------------------


def generate_dataset_description() -> SectionOutput:
    """Spec section C / C2."""
    ds = _load("dataset_summary")
    source = _ARTIFACTS["dataset_summary"]
    splits = ds["split_sizes"]
    oos_c = ds["oos_counts"]
    ins_c = ds["in_scope_counts"]

    split_table = build_markdown_table(
        ["Split", "Total", "In-Scope", "OOS"],
        [
            ["Train", f"{splits['train']:,}", f"{ins_c['train']:,}", f"{oos_c['train']:,}"],
            ["Validation", f"{splits['validation']:,}", f"{ins_c['validation']:,}", f"{oos_c['validation']:,}"],
            ["Test", f"{splits['test']:,}", f"{ins_c['test']:,}", f"{oos_c['test']:,}"],
        ],
    )

    train_oos_pct = oos_c["train"] / splits["train"] * 100
    test_oos_pct = oos_c["test"] / splits["test"] * 100
    quirks: list[str] = ds.get("quirks_or_caveats", [])
    quirk_lines = [f"- {q}" for q in quirks]
    in_scope_train_min = ds.get("train_class_count_min")
    in_scope_train_max = ds.get("train_class_count_max")
    if in_scope_train_min == in_scope_train_max:
        class_balance_note = (
            f"In-scope classes are balanced in the `{ds['subset']}` subset at "
            f"{in_scope_train_min} training examples per class."
        )
    else:
        class_balance_note = (
            f"In-scope class counts in the `{ds['subset']}` subset range from "
            f"{in_scope_train_min} to {in_scope_train_max} training examples."
        )

    md = "\n".join(
        [
            "## Dataset Description",
            "",
            f"**Source**: `{ds['dataset_source']}` (subset: `{ds['subset']}`) ({source}).",
            "",
            f"The dataset contains {ds['num_classes']} intent classes (150 in-scope + 1 OOS). "
            f"The OOS class uses label name `{ds['oos_label_name']}` (label ID {ds['oos_label_id']}).",
            "",
            "### Split Sizes",
            "",
            split_table,
            "",
            "### Class Balance",
            "",
            f"{class_balance_note} "
            f"OOS support varies across splits: train has {oos_c['train']} OOS examples "
            f"(~{train_oos_pct:.1f}%), while test has {oos_c['test']} OOS examples "
            f"(~{test_oos_pct:.1f}%), creating a significant distribution shift.",
            "",
            "### Distribution Quirks",
            "",
            *quirk_lines,
            "",
        ]
    )

    metadata = _meta(
        dataset_source=ds["dataset_source"],
        subset=ds["subset"],
        split_sizes=splits,
        num_classes=ds["num_classes"],
        oos_label_name=ds["oos_label_name"],
        oos_label_id=ds["oos_label_id"],
        oos_counts=oos_c,
        in_scope_counts=ins_c,
        class_balance_note=class_balance_note,
        distribution_quirks=quirks,
        source_artifact=source,
    )
    return md, metadata


# ---------------------------------------------------------------------------
# Section D: Preprocessing summary
# ---------------------------------------------------------------------------


def generate_preprocessing_summary() -> SectionOutput:
    """Spec section D / D2."""
    pp = _load("preprocessing_summary")
    source = _ARTIFACTS["preprocessing_summary"]
    oov = pp["oov_stats"]
    trunc = pp["truncation_stats"]
    seq = pp["sequence_length_stats"]
    tfidf = pp["tfidf_config"]
    cleaning = pp["text_cleaning_policy"]

    oov_table = build_markdown_table(
        ["Split", "Total Tokens", "Unknown Tokens", "OOV Rate"],
        [
            [
                s.title(),
                f"{oov[s]['total_tokens']:,}",
                f"{oov[s]['unknown_tokens']:,}",
                format_ratio(oov[s]["oov_rate"]),
            ]
            for s in ("train", "validation", "test")
        ],
    )

    trunc_table = build_markdown_table(
        ["Split", "Total Sequences", "Truncated", "Truncation Rate"],
        [
            [
                s.title(),
                f"{trunc[s]['total_sequences']:,}",
                f"{trunc[s]['truncated']:,}",
                format_ratio(trunc[s]["truncation_rate"]),
            ]
            for s in ("train", "validation", "test")
        ],
    )

    md = "\n".join(
        [
            "## Preprocessing Summary",
            "",
            f"**Source**: {source}",
            "",
            "### Text Cleaning",
            "",
            f"Strategy: {cleaning['strategy']}. Preserves: {cleaning['preserves']}.",
            "",
            "### Tokenization and Vocabulary",
            "",
            f"Tokenizer: {pp['tokenizer']}. "
            f"Vocabulary size: {pp['vocabulary_size']} (built from training data only). "
            f"Special tokens: `<PAD>` = {pp['special_tokens']['<PAD>']}, "
            f"`<UNK>` = {pp['special_tokens']['<UNK>']}.",
            "",
            "### Sequence Handling",
            "",
            f"Max sequence length: {pp['max_seq_length']}. "
            f"Statistics (training set): mean {seq['mean']:.2f}, median {seq['median']:.1f}, "
            f"p90 {seq['p90']}, p95 {seq['p95']}, max {seq['max']}, min {seq['min']}.",
            "",
            "### OOV Rates",
            "",
            oov_table,
            "",
            "### Truncation Rates",
            "",
            trunc_table,
            "",
            "### TF-IDF Configuration",
            "",
            f"Max features: {tfidf['max_features']:,}. "
            f"N-gram range: ({tfidf['ngram_range'][0]}, {tfidf['ngram_range'][1]}). "
            f"Fitted on training data only. "
            f"Note: TF-IDF is used only for the MLP baseline; Text CNN and BiLSTM use token-ID sequences.",
            "",
        ]
    )

    metadata = _meta(
        cleaning_policy=cleaning,
        tokenizer=pp["tokenizer"],
        vocab_size=pp["vocabulary_size"],
        special_tokens=pp["special_tokens"],
        max_seq_length=pp["max_seq_length"],
        sequence_length_stats=seq,
        oov_rates={s: oov[s]["oov_rate"] for s in ("train", "validation", "test")},
        truncation_rates={s: trunc[s]["truncation_rate"] for s in ("train", "validation", "test")},
        tfidf_config=tfidf,
        source_artifact=source,
    )
    return md, metadata


# ---------------------------------------------------------------------------
# Section E: Model architectures
# ---------------------------------------------------------------------------


def generate_model_architectures() -> SectionOutput:
    """Spec section E / E2."""
    eff = _load("efficiency_summary_table")
    pp = _load("preprocessing_summary")
    eff_by = _rows_by_model(eff["rows"])
    configs = {mid: _load_frozen(mid) for mid in ModelID}
    tfidf_input_dim: int = pp["tfidf_config"]["max_features"]
    sources = [_frozen_path(m) for m in ModelID] + [
        _ARTIFACTS["efficiency_summary_table"],
        _ARTIFACTS["preprocessing_summary"],
    ]

    _INPUT_TYPES: dict[str, str] = {"tfidf": "TF-IDF vectors", "token_ids": "Token-ID sequences"}

    _ARCH_SUMMARIES: dict[str, str] = {
        "mlp": "2-layer MLP (1 hidden layer)",
        "text_cnn": "Multi-kernel CNN (3 kernel sizes)",
        "bilstm": "2-layer bidirectional LSTM",
    }

    table_rows: list[list[str]] = []
    model_details: list[dict[str, Any]] = []

    for mid in ModelID:
        cfg = configs[mid]
        hp = cfg["hyperparameters"]
        er = eff_by[str(mid)]
        embed_dim = hp.get("embedding_dim", "N/A")
        hidden = hp.get("hidden_dim", "N/A")

        table_rows.append(
            [
                mid.display_name,
                _INPUT_TYPES.get(mid.input_type, mid.input_type),
                _ARCH_SUMMARIES[str(mid)],
                format_param_count(er["parameter_count"]),
                format_param_count(er["trainable_parameter_count"]),
                str(embed_dim),
                format_ratio(hp["dropout_rate"], 1),
                str(hidden),
            ]
        )

        model_details.append(
            {
                "model_id": str(mid),
                "model_name": cfg["model_name"],
                "display_name": mid.display_name,
                "input_type": _INPUT_TYPES.get(mid.input_type, mid.input_type),
                "architecture_summary": _ARCH_SUMMARIES[str(mid)],
                "hyperparameters": hp,
                "parameter_count": er["parameter_count"],
                "trainable_parameter_count": er["trainable_parameter_count"],
            }
        )

    comp_table = build_markdown_table(
        ["Model", "Input Type", "Architecture", "Params", "Trainable", "Embed Dim", "Dropout", "Hidden Dim"],
        table_rows,
    )

    mlp_hp = configs[ModelID.MLP]["hyperparameters"]
    cnn_hp = configs[ModelID.TEXT_CNN]["hyperparameters"]
    lstm_hp = configs[ModelID.BILSTM]["hyperparameters"]

    md = "\n".join(
        [
            "## Model Architectures",
            "",
            "### Unified Comparison",
            "",
            comp_table,
            "",
            "### TF-IDF + MLP",
            "",
            f"Input: {tfidf_input_dim:,} TF-IDF features ({tfidf_input_dim:,}-dimensional). "
            f"Single hidden layer with {mlp_hp['hidden_dim']} units, ReLU activation, "
            f"dropout {mlp_hp['dropout_rate']}.",
            "",
            "### Text CNN",
            "",
            f"Embedding dimension: {cnn_hp['embedding_dim']}. "
            f"Kernel sizes: {cnn_hp['kernel_sizes']} with {cnn_hp['num_filters']} filters each. "
            f"ReLU activation, max-over-time pooling, dropout {cnn_hp['dropout_rate']}. "
            "Trainable embeddings.",
            "",
            "### BiLSTM",
            "",
            f"Embedding dimension: {lstm_hp['embedding_dim']}. "
            f"Hidden dimension: {lstm_hp['hidden_dim']}, {lstm_hp['num_layers']} layers, bidirectional. "
            f"Summarization: {lstm_hp.get('summarization_mode', 'concat_final_hidden')}. "
            f"Gradient clipping (max norm {lstm_hp.get('max_grad_norm', 1.0)}). "
            f"Dropout {lstm_hp['dropout_rate']}. Trainable embeddings.",
            "",
        ]
    )

    metadata = _meta(
        models=model_details,
        comparison_table=table_rows,
        source_artifacts=sources,
    )
    return md, metadata


# ---------------------------------------------------------------------------
# Section F: Experimental setup
# ---------------------------------------------------------------------------


def generate_experimental_setup() -> SectionOutput:
    """Spec section F / F2."""
    proto = _load("evaluation_protocol")
    pc = proto["protocol_config"]
    configs = {mid: _load_frozen(mid) for mid in ModelID}
    sources = [_ARTIFACTS["evaluation_protocol"]] + [_frozen_path(m) for m in ModelID]

    per_model_lr = {str(m): configs[m]["hyperparameters"]["learning_rate"] for m in ModelID}
    per_model_wd = {str(m): configs[m]["hyperparameters"]["weight_decay"] for m in ModelID}
    seeds = pc["seed_list"]
    metric_defs = proto["metric_definitions"]
    oos_policy = proto["oos_evaluation_policy"]
    seed_policy = proto["seed_policy"]
    seed_behavior_note = seed_policy.get(
        "report_note",
        "The repeated-evaluation pipeline derives `training_seed = seed` and "
        "`dataloader_seed = seed + 1`. `dataloader_seed` controls train-batch "
        "shuffling, but model initialization currently depends on "
        "`config.random_seed`, not necessarily the nominal seed.",
    )

    lr_items = ", ".join(f"{ModelID(k).display_name}: {v}" for k, v in per_model_lr.items())
    wd_items = ", ".join(f"{ModelID(k).display_name}: {v}" for k, v in per_model_wd.items())

    md = "\n".join(
        [
            "## Experimental Setup",
            "",
            f"**Source**: {sources[0]}",
            "",
            "### Evaluation Protocol",
            "",
            f"Repeated-run evaluation with {pc['run_count']} seeds: {seeds}. "
            f"Representative run selection: {pc['representative_run_rule']} "
            "(highest validation macro F1).",
            "",
            "### Training Configuration",
            "",
            "- **Optimizer**: Adam (all models)",
            f"- **Learning rate**: {lr_items}",
            f"- **Weight decay**: {wd_items}",
            f"- **Batch size**: {configs[ModelID.MLP]['hyperparameters']['batch_size']}",
            f"- **Max epochs**: {configs[ModelID.MLP]['hyperparameters']['max_epochs']}",
            f"- **Early stopping**: patience {configs[ModelID.MLP]['hyperparameters']['early_stopping_patience']}, "
            f"monitoring {configs[ModelID.MLP]['hyperparameters']['monitor_metric']}",
            f"- **Seed behavior**: {seed_behavior_note}",
            "",
            "### Metric Definitions",
            "",
            *[f"- **{k}**: {v}" for k, v in metric_defs.items()],
            "",
            "### OOS Evaluation Policy",
            "",
            f"- OOS class: `{oos_policy['oos_class_name']}` (label ID {oos_policy['oos_class_id']})",
            f"- Method: {oos_policy['evaluation_method']}",
            f"- Rule: {oos_policy['one_vs_rest_rule']}",
            "",
            "### Timing and Device",
            "",
            f"Timing includes DataLoader overhead: {pc['timing_includes_dataloader_overhead']}. "
            "Device: Apple Silicon MPS when available, CPU fallback.",
            "",
        ]
    )

    metadata = _meta(
        run_count=pc["run_count"],
        seed_list=seeds,
        representative_run_rule=pc["representative_run_rule"],
        optimizer="adam",
        per_model_lr=per_model_lr,
        per_model_weight_decay=per_model_wd,
        batch_size=configs[ModelID.MLP]["hyperparameters"]["batch_size"],
        max_epochs=configs[ModelID.MLP]["hyperparameters"]["max_epochs"],
        early_stopping_patience=configs[ModelID.MLP]["hyperparameters"]["early_stopping_patience"],
        monitor_metric=configs[ModelID.MLP]["hyperparameters"]["monitor_metric"],
        metric_definitions=metric_defs,
        oos_evaluation_policy=oos_policy,
        timing_policy="includes DataLoader overhead",
        source_artifacts=sources,
    )
    return md, metadata


# ---------------------------------------------------------------------------
# Section G: Main results tables
# ---------------------------------------------------------------------------


def generate_main_results() -> SectionOutput:
    """Spec section G / G2 / G3."""
    agg = _load("model_comparison_aggregate")
    oos = _load("oos_summary_table")
    ext = _load("extended_metrics_comparison")
    cal = _load("calibration_summary")
    eff = _load("efficiency_summary_table")
    figure_entries = load_figure_manifest_entries()

    source_keys = [
        "model_comparison_aggregate",
        "oos_summary_table",
        "extended_metrics_comparison",
        "calibration_summary",
        "efficiency_summary_table",
        "figure_manifest",
    ]
    sources = [_ARTIFACTS[k] for k in source_keys]
    agg_by = _rows_by_model(agg["rows"])
    oos_by = _rows_by_model(oos["rows"])
    ext_by = _rows_by_model(ext["rows"])
    cal_by = _rows_by_model(cal["rows"])
    eff_by = _rows_by_model(eff["rows"])
    best_agg = _best_row(agg["rows"], "test_macro_f1_mean")
    best_oos = _best_row(oos["rows"], "oos_f1_mean")
    best_ext = max(ext["rows"], key=lambda row: row["macro_f1"])
    best_cal = min(cal["rows"], key=lambda row: row["ece"])
    fastest = max(eff["rows"], key=lambda row: row["inference_examples_per_sec_mean"])

    def _agg_row(mid: ModelID, metrics: list[str]) -> list[str]:
        r = agg_by[str(mid)]
        return [mid.display_name] + [format_mean_std(r[f"{m}_mean"], r[f"{m}_std"]) for m in metrics]

    def _oos_row(mid: ModelID) -> list[str]:
        r = oos_by[str(mid)]
        return [
            mid.display_name,
            format_mean_std(r["oos_precision_mean"], r["oos_precision_std"]),
            format_mean_std(r["oos_recall_mean"], r["oos_recall_std"]),
            format_mean_std(r["oos_f1_mean"], r["oos_f1_std"]),
        ]

    # Table 1: Main comparison (aggregate)
    t1_headers = ["Model", "Test Accuracy", "Test Macro F1", "Test Precision", "Test Recall"]
    t1_metrics = ["test_accuracy", "test_macro_f1", "test_precision", "test_recall"]
    t1_rows = [_agg_row(m, t1_metrics) for m in ModelID]
    t1_md = build_markdown_table(t1_headers, t1_rows)

    # Table 2: OOS metrics (aggregate)
    t2_headers = ["Model", "OOS Precision", "OOS Recall", "OOS F1"]
    t2_rows = [_oos_row(m) for m in ModelID]
    t2_md = build_markdown_table(t2_headers, t2_rows)

    # Table 3: Extended metrics (representative)
    t3_headers = ["Model", "Macro F1", "Micro F1", "Weighted F1", "Macro Precision", "Macro Recall"]
    t3_rows = [
        [
            mid.display_name,
            format_ratio(ext_by[str(mid)]["macro_f1"]),
            format_ratio(ext_by[str(mid)]["micro_f1"]),
            format_ratio(ext_by[str(mid)]["weighted_f1"]),
            format_ratio(ext_by[str(mid)]["macro_precision"]),
            format_ratio(ext_by[str(mid)]["macro_recall"]),
        ]
        for mid in ModelID
    ]
    t3_md = build_markdown_table(t3_headers, t3_rows)

    # Table 4: Calibration (representative)
    t4_headers = ["Model", "ECE", "MCE", "Brier Score", "NLL"]
    t4_rows = [
        [
            mid.display_name,
            format_ratio(cal_by[str(mid)]["ece"]),
            format_ratio(cal_by[str(mid)]["mce"]),
            format_ratio(cal_by[str(mid)]["brier_score"]),
            format_ratio(cal_by[str(mid)]["nll"]),
        ]
        for mid in ModelID
    ]
    t4_md = build_markdown_table(t4_headers, t4_rows)

    # Table 5: Efficiency (aggregate)
    t5_headers = ["Model", "Parameters", "Training Time (s)", "Inference (ms/example)", "Throughput (ex/s)"]
    t5_rows = [
        [
            mid.display_name,
            format_param_count(eff_by[str(mid)]["parameter_count"]),
            format_mean_std(
                eff_by[str(mid)]["training_time_seconds_mean"],
                eff_by[str(mid)]["training_time_seconds_std"],
                decimals=2,
            ),
            format_mean_std(
                eff_by[str(mid)]["inference_avg_ms_per_example_mean"],
                eff_by[str(mid)]["inference_avg_ms_per_example_std"],
            ),
            format_mean_std(
                eff_by[str(mid)]["inference_examples_per_sec_mean"],
                eff_by[str(mid)]["inference_examples_per_sec_std"],
                decimals=2,
            ),
        ]
        for mid in ModelID
    ]
    t5_md = build_markdown_table(t5_headers, t5_rows)

    t1_claim = build_structured_claim(
        claim_id="main_comparison",
        claim_text=(
            f"{ModelID(best_agg['model_id']).display_name} achieved the highest aggregate test macro F1 "
            f"({format_mean_std(best_agg['test_macro_f1_mean'], best_agg['test_macro_f1_std'])})."
        ),
        figure_entries=figure_entries,
        source_artifacts=[_ARTIFACTS["model_comparison_aggregate"]],
        related_figure_types=["model_comparison_test_macro_f1"],
        scopes=["aggregate"],
    )
    t2_claim = build_structured_claim(
        claim_id="oos_metrics",
        claim_text=(
            f"{ModelID(best_oos['model_id']).display_name} achieved the highest aggregate OOS F1 "
            f"({format_mean_std(best_oos['oos_f1_mean'], best_oos['oos_f1_std'])})."
        ),
        figure_entries=figure_entries,
        source_artifacts=[_ARTIFACTS["oos_summary_table"]],
        related_figure_types=["oos_metrics_comparison", "model_comparison_oos_f1"],
        scopes=["aggregate"],
    )
    t3_claim = build_structured_claim(
        claim_id="extended_metrics",
        claim_text=(
            f"{ModelID(best_ext['model_id']).display_name} has the highest representative-run macro F1 "
            f"({format_ratio(best_ext['macro_f1'])}) in the extended-metrics table."
        ),
        figure_entries=figure_entries,
        source_artifacts=[_ARTIFACTS["extended_metrics_comparison"]],
        related_figure_types=[],
        scopes=["representative", "analysis"],
    )
    t4_claim = build_structured_claim(
        claim_id="calibration",
        claim_text=(
            f"{ModelID(best_cal['model_id']).display_name} is the best calibrated representative run "
            f"(ECE = {format_ratio(best_cal['ece'])})."
        ),
        figure_entries=figure_entries,
        source_artifacts=[_ARTIFACTS["calibration_summary"]],
        related_figure_types=["calibration_comparison"],
        scopes=["analysis"],
    )
    t5_claim = build_structured_claim(
        claim_id="efficiency",
        claim_text=(
            f"{ModelID(fastest['model_id']).display_name} has the highest mean inference throughput "
            f"({fastest['inference_examples_per_sec_mean']:.0f} ex/s)."
        ),
        figure_entries=figure_entries,
        source_artifacts=[_ARTIFACTS["efficiency_summary_table"]],
        related_figure_types=["model_efficiency_comparison"],
        scopes=["aggregate"],
    )
    claims = [t1_claim, t2_claim, t3_claim, t4_claim, t5_claim]

    md_parts: list[str] = [
        "## Main Results",
        "",
        "### Table 1: Main Model Comparison",
        "",
        _scope_note("aggregate"),
        "",
        t1_md,
        "",
        t1_claim["claim_text"],
        "",
    ]
    _append_related_figure_note(md_parts, t1_claim)
    md_parts.extend(
        [
            "### Table 2: OOS Detection Metrics",
            "",
            _scope_note("aggregate"),
            "",
            t2_md,
            "",
            t2_claim["claim_text"],
            "",
        ]
    )
    _append_related_figure_note(md_parts, t2_claim)
    md_parts.extend(
        [
            "### Table 3: Extended Metrics",
            "",
            _scope_note("representative"),
            "",
            t3_md,
            "",
            t3_claim["claim_text"],
            "",
        ]
    )
    _append_related_figure_note(md_parts, t3_claim)
    md_parts.extend(
        [
            "### Table 4: Calibration Metrics",
            "",
            _scope_note("representative"),
            "",
            t4_md,
            "",
            t4_claim["claim_text"],
            "",
        ]
    )
    _append_related_figure_note(md_parts, t4_claim)
    md_parts.extend(
        [
            "### Table 5: Efficiency Comparison",
            "",
            _scope_note("aggregate"),
            "",
            t5_md,
            "",
            t5_claim["claim_text"],
            "",
        ]
    )
    _append_related_figure_note(md_parts, t5_claim)

    md = "\n".join(md_parts)

    agg_fig = next(
        (f for f in figure_entries if f["figure_type"] == "model_comparison_test_macro_f1"),
        None,
    )
    train_fig = next(
        (f for f in figure_entries if f["figure_type"] == "val_macro_f1_curve"),
        None,
    )
    fig_parts: list[str] = []
    if agg_fig or train_fig:
        fig_parts.extend(["### Key Figures", ""])
    if agg_fig:
        fig_parts.append(
            f"![Aggregate test macro F1 comparison with error bars across all models. "
            f"Data: aggregate over 3 repeated runs, CLINC150 test set.]({agg_fig['figure_path']})"
        )
        fig_parts.append("")
    if train_fig:
        fn = ModelID(train_fig.get("model_name", "mlp")).display_name
        fig_parts.append(
            f"![Validation macro F1 progression during training for {fn}. "
            f"Data: representative run, CLINC150.]({train_fig['figure_path']})"
        )
        fig_parts.append("")
    if fig_parts:
        md += "\n" + "\n".join(fig_parts)

    # Build JSON table objects with raw values for validation
    tables_json: list[dict[str, Any]] = []
    for table_id, title, scope, cols, src_key in [
        ("main_comparison", "Main Model Comparison", "aggregate", t1_headers, "model_comparison_aggregate"),
        ("oos_metrics", "OOS Detection Metrics", "aggregate", t2_headers, "oos_summary_table"),
        ("extended_metrics", "Extended Metrics", "representative", t3_headers, "extended_metrics_comparison"),
        ("calibration", "Calibration Metrics", "representative", t4_headers, "calibration_summary"),
        ("efficiency", "Efficiency Comparison", "aggregate", t5_headers, "efficiency_summary_table"),
    ]:
        src_data = {
            "model_comparison_aggregate": agg,
            "oos_summary_table": oos,
            "extended_metrics_comparison": ext,
            "calibration_summary": cal,
            "efficiency_summary_table": eff,
        }[src_key]
        tables_json.append(
            {
                "table_id": table_id,
                "title": title,
                "data_scope": scope,
                "columns": cols,
                "rows": src_data["rows"],
                "source_artifact": _ARTIFACTS[src_key],
            }
        )

    metadata = _meta(tables=tables_json, claims=claims, source_artifacts=sources)
    return md, metadata


# ---------------------------------------------------------------------------
# Section H: OOS detection results
# ---------------------------------------------------------------------------


def generate_oos_detection() -> SectionOutput:
    """Spec section H / H2."""
    oos = _load("oos_summary_table")
    thresh = _load("oos_threshold_comparison")
    fa = _load("oos_false_accept_comparison")
    figure_entries = load_figure_manifest_entries()

    sources = [
        _ARTIFACTS["oos_summary_table"],
        _ARTIFACTS["oos_threshold_comparison"],
        _ARTIFACTS["oos_false_accept_comparison"],
        _ARTIFACTS["figure_manifest"],
    ]

    oos_by = _rows_by_model(oos["rows"])
    thresh_by = _rows_by_model(thresh["rows"])

    # Find best OOS model by AUROC
    best_thresh = _best_row(thresh["rows"], "auroc")
    best_oos_id = best_thresh["model_id"]
    best_display = ModelID(best_oos_id).display_name
    best_oos_row = _best_row(oos["rows"], "oos_f1_mean")

    # Collect OOS-related figure paths
    oos_fig_types = {
        "oos_roc_curve",
        "oos_pr_curve",
        "oos_error_breakdown",
        "oos_roc_comparison",
        "oos_pr_comparison",
        "oos_error_comparison",
    }
    figure_refs = [f["figure_path"] for f in figure_entries if f["figure_type"] in oos_fig_types]

    # Aggregate OOS table
    agg_oos_table = build_markdown_table(
        ["Model", "OOS Precision", "OOS Recall", "OOS F1"],
        [
            [
                mid.display_name,
                format_mean_std(oos_by[str(mid)]["oos_precision_mean"], oos_by[str(mid)]["oos_precision_std"]),
                format_mean_std(oos_by[str(mid)]["oos_recall_mean"], oos_by[str(mid)]["oos_recall_std"]),
                format_mean_std(oos_by[str(mid)]["oos_f1_mean"], oos_by[str(mid)]["oos_f1_std"]),
            ]
            for mid in ModelID
        ],
    )

    # Threshold table
    thresh_table = build_markdown_table(
        ["Model", "AUROC", "AUPR", "FPR@95TPR", "FPR@90TPR", "MSP AUROC", "MSP AUPR"],
        [
            [
                mid.display_name,
                format_ratio(thresh_by[str(mid)]["auroc"]),
                format_ratio(thresh_by[str(mid)]["aupr"]),
                format_ratio(thresh_by[str(mid)]["fpr_at_95tpr"]),
                format_ratio(thresh_by[str(mid)]["fpr_at_90tpr"]),
                format_ratio(thresh_by[str(mid)]["msp_auroc"]),
                format_ratio(thresh_by[str(mid)]["msp_aupr"]),
            ]
            for mid in ModelID
        ],
    )

    # False-accept summary
    fa_lines: list[str] = []
    for mid in ModelID:
        model_fa = fa["models"][str(mid)]
        top3 = model_fa["top_capturing_intents"][:3]
        intents = ", ".join(f"`{i['intent']}` ({i['count']})" for i in top3)
        fa_lines.append(
            f"- **{mid.display_name}**: {model_fa['false_accept_count']} false accepts. "
            f"Top capturing intents: {intents}"
        )

    # Shared ROC comparison figure reference
    roc_comp_path = next(
        (f["figure_path"] for f in figure_entries if f["figure_type"] == "oos_roc_comparison"),
        None,
    )
    aggregate_claim = build_structured_claim(
        claim_id="aggregate_oos_metrics",
        claim_text=(
            f"{ModelID(best_oos_row['model_id']).display_name} achieved the highest aggregate OOS F1 "
            f"({format_mean_std(best_oos_row['oos_f1_mean'], best_oos_row['oos_f1_std'])})."
        ),
        figure_entries=figure_entries,
        source_artifacts=[_ARTIFACTS["oos_summary_table"]],
        related_figure_types=["oos_metrics_comparison", "model_comparison_oos_f1"],
        scopes=["aggregate"],
    )
    threshold_claim = build_structured_claim(
        claim_id="threshold_analysis",
        claim_text=(
            f"{best_display} achieved the highest AUROC ({format_ratio(best_thresh['auroc'])}), "
            "indicating the strongest threshold-independent OOS discrimination."
        ),
        figure_entries=figure_entries,
        source_artifacts=[_ARTIFACTS["oos_threshold_comparison"]],
        related_figure_types=["oos_roc_comparison", "oos_pr_comparison"],
        scopes=["analysis"],
    )
    false_accept_claim = build_structured_claim(
        claim_id="false_accept_patterns",
        claim_text=(
            "False accepts remain the dominant OOS failure pattern, "
            "with a small set of intents repeatedly capturing OOS examples."
        ),
        figure_entries=figure_entries,
        source_artifacts=[_ARTIFACTS["oos_false_accept_comparison"]],
        related_figure_types=["oos_error_comparison"],
        scopes=["analysis"],
    )
    claims = [aggregate_claim, threshold_claim, false_accept_claim]

    md_parts = [
        "## OOS Detection Results",
        "",
        "### Aggregate OOS Metrics",
        "",
        _scope_note("aggregate"),
        "",
        agg_oos_table,
        "",
        aggregate_claim["claim_text"],
        "",
    ]
    _append_related_figure_note(md_parts, aggregate_claim)
    md_parts.extend(
        [
            "### OOS Threshold Analysis",
            "",
            _scope_note("representative"),
            "",
            thresh_table,
            "",
            f"{threshold_claim['claim_text']} ({_ARTIFACTS['oos_threshold_comparison']}). "
            "The explicit OOS class probability method substantially outperforms the "
            "maximum softmax probability (MSP) baseline across all models.",
            "",
        ]
    )
    _append_related_figure_note(md_parts, threshold_claim)
    md_parts.extend(
        [
            "### OOS Distribution Challenge",
            "",
            "The test split contains ~18% OOS examples versus ~1.6% in training, "
            "creating a severe distribution shift that challenges all models. "
            "All models show low OOS recall, indicating difficulty detecting OOS examples "
            "despite reasonable OOS precision.",
            "",
            "### False-Accept Patterns",
            "",
            *fa_lines,
            "",
            f"{false_accept_claim['claim_text']} ({_ARTIFACTS['oos_false_accept_comparison']}).",
            "",
        ]
    )
    _append_related_figure_note(md_parts, false_accept_claim)
    if roc_comp_path:
        md_parts.extend([f"![OOS ROC Comparison]({roc_comp_path})", ""])

    md = "\n".join(md_parts)

    # Build metadata
    agg_oos_meta = {
        str(mid): {
            "oos_precision_mean": oos_by[str(mid)]["oos_precision_mean"],
            "oos_precision_std": oos_by[str(mid)]["oos_precision_std"],
            "oos_recall_mean": oos_by[str(mid)]["oos_recall_mean"],
            "oos_recall_std": oos_by[str(mid)]["oos_recall_std"],
            "oos_f1_mean": oos_by[str(mid)]["oos_f1_mean"],
            "oos_f1_std": oos_by[str(mid)]["oos_f1_std"],
        }
        for mid in ModelID
    }
    thresh_meta = {
        str(mid): {k: thresh_by[str(mid)][k] for k in ("auroc", "aupr", "fpr_at_95tpr", "fpr_at_90tpr")}
        for mid in ModelID
    }
    msp_meta = {
        str(mid): {
            "msp_auroc": thresh_by[str(mid)]["msp_auroc"],
            "msp_aupr": thresh_by[str(mid)]["msp_aupr"],
        }
        for mid in ModelID
    }
    fa_meta = {str(mid): fa["models"][str(mid)]["top_capturing_intents"][:5] for mid in ModelID}

    metadata = _meta(
        aggregate_oos_metrics=agg_oos_meta,
        threshold_metrics=thresh_meta,
        msp_baseline=msp_meta,
        best_oos_model={"model_id": best_oos_id, "rationale": f"Highest AUROC ({format_ratio(best_thresh['auroc'])})"},
        claims=claims,
        false_accept_summary=fa_meta,
        figure_references=figure_refs,
        source_artifacts=sources,
    )
    return md, metadata


# ---------------------------------------------------------------------------
# Section J: Error analysis discussion
# ---------------------------------------------------------------------------


def generate_error_analysis() -> SectionOutput:
    """Spec section J / J2."""
    tax = _load("error_taxonomy_summary")
    cal = _load("calibration_summary")
    cross = _load("cross_model_error_comparison")
    worst = _load("worst_classes_comparison")
    pairs = _load("most_confused_pairs_table")
    eas = _load("error_analysis_summary")
    thresh = _load("oos_threshold_comparison")
    figure_entries = load_figure_manifest_entries()

    sources = [
        _ARTIFACTS["error_taxonomy_summary"],
        _ARTIFACTS["calibration_summary"],
        _ARTIFACTS["cross_model_error_comparison"],
        _ARTIFACTS["worst_classes_comparison"],
        _ARTIFACTS["most_confused_pairs_table"],
        _ARTIFACTS["error_analysis_summary"],
        _ARTIFACTS["oos_threshold_comparison"],
        _ARTIFACTS["figure_manifest"],
    ]

    tax_by = _rows_by_model(tax["rows"])

    # Dominant error category per model
    cat_keys = [
        "oos_as_inscope",
        "inscope_as_oos",
        "near_semantic_confusion",
        "cross_domain_confusion",
        "short_query_ambiguity",
    ]
    dominant: dict[str, str] = {}
    for mid in ModelID:
        r = tax_by[str(mid)]
        dominant[str(mid)] = max(cat_keys, key=lambda c: r.get(f"{c}_count", 0))

    # Best/worst calibrated
    best_cal = min(cal["rows"], key=lambda r: r["ece"])
    worst_cal = max(cal["rows"], key=lambda r: r["ece"])

    # Best OOS by AUROC
    best_oos = _best_row(thresh["rows"], "auroc")

    # Cross-model overlap
    cats = cross["categories"]

    # Persistent confusions across models
    intent_models: dict[str, set[str]] = {}
    for row in pairs["rows"]:
        intent_models.setdefault(row["predicted_label_name"], set()).add(row["model_id"])
    persistent = sorted(
        [i for i, ms in intent_models.items() if len(ms) >= 2],
        key=lambda i: len(intent_models[i]),
        reverse=True,
    )

    # Length finding
    length = eas.get("length", {})
    length_sources = list(length.get("source_artifacts", [_ARTIFACTS["error_analysis_summary"]]))

    shared_worst_fmt = ", ".join(f"`{c}`" for c in worst["shared_worst_classes"])
    persistent_fmt = ", ".join(f"`{i}`" for i in persistent[:6])

    taxonomy_claim = build_structured_claim(
        claim_id="error_taxonomy",
        claim_text="All models share `oos_as_inscope` as the dominant error category.",
        figure_entries=figure_entries,
        source_artifacts=[_ARTIFACTS["error_taxonomy_summary"]],
        related_figure_types=["error_taxonomy"],
        scopes=["analysis"],
    )
    calibration_claim = build_structured_claim(
        claim_id="calibration_confidence",
        claim_text=(
            f"Best-calibrated model: {best_cal['model_name']} (ECE = {format_ratio(best_cal['ece'])}); "
            f"worst-calibrated: {worst_cal['model_name']} (ECE = {format_ratio(worst_cal['ece'])})."
        ),
        figure_entries=figure_entries,
        source_artifacts=[_ARTIFACTS["calibration_summary"]],
        related_figure_types=["calibration_comparison", "reliability_diagram", "confidence_histogram"],
        scopes=["analysis"],
    )
    oos_claim = build_structured_claim(
        claim_id="oos_deep_dive",
        claim_text=(
            f"Best OOS detector by AUROC: {ModelID(best_oos['model_id']).display_name} "
            f"({format_ratio(best_oos['auroc'])}); the main failure mode remains `oos_as_inscope`."
        ),
        figure_entries=figure_entries,
        source_artifacts=[_ARTIFACTS["oos_threshold_comparison"], _ARTIFACTS["oos_false_accept_comparison"]],
        related_figure_types=["oos_error_comparison", "oos_error_breakdown", "oos_roc_comparison"],
        scopes=["analysis"],
    )
    overlap_claim = build_structured_claim(
        claim_id="cross_model_overlap",
        claim_text=(
            f"{cats['all_wrong']['count']:,} examples "
            f"({format_ratio(cats['all_wrong']['fraction'])}) are wrong for all models, "
            f"while {cats['model_specific_error']['count']:,} "
            f"({format_ratio(cats['model_specific_error']['fraction'])}) "
            "are model-specific."
        ),
        figure_entries=figure_entries,
        source_artifacts=[_ARTIFACTS["cross_model_error_comparison"]],
        related_figure_types=["cross_model_error_overlap"],
        scopes=["analysis"],
    )
    worst_classes_claim = build_structured_claim(
        claim_id="worst_classes",
        claim_text=f"Shared worst classes across models: {shared_worst_fmt}.",
        figure_entries=figure_entries,
        source_artifacts=[_ARTIFACTS["worst_classes_comparison"]],
        related_figure_types=["worst_classes_heatmap"],
        scopes=["analysis"],
    )
    confused_pairs_claim = build_structured_claim(
        claim_id="persistent_confused_pairs",
        claim_text=f"Persistent OOS-capturing intents across multiple models: {persistent_fmt}.",
        figure_entries=figure_entries,
        source_artifacts=[_ARTIFACTS["most_confused_pairs_table"]],
        related_figure_types=["top_confused_pairs"],
        scopes=["representative"],
    )
    length_claim = build_structured_claim(
        claim_id="length_scope_slices",
        claim_text=(
            "Short-query accuracy remains lower than desired across models, "
            "reinforcing the short-query ambiguity pattern."
        ),
        figure_entries=figure_entries,
        source_artifacts=length_sources,
        related_figure_types=["length_slice", "frequency_slice"],
        scopes=["analysis"],
    )
    claims = [
        taxonomy_claim,
        calibration_claim,
        oos_claim,
        overlap_claim,
        worst_classes_claim,
        confused_pairs_claim,
        length_claim,
    ]

    md_parts = [
        "## Error Analysis Discussion",
        "",
        "### Error Taxonomy",
        "",
        *[
            f"- **{mid.display_name}**: dominant error category is `{dominant[str(mid)]}` "
            f"({tax_by[str(mid)][f'{dominant[str(mid)]}_count']} errors, "
            f"{format_ratio(tax_by[str(mid)][f'{dominant[str(mid)]}_fraction'])} of all errors) "
            f"({_ARTIFACTS['error_taxonomy_summary']})"
            for mid in ModelID
        ],
        "",
        "All models share `oos_as_inscope` as the dominant error category, indicating "
        "that OOS false accepts are the primary failure mode across architectures.",
        "",
    ]
    _append_related_figure_note(md_parts, taxonomy_claim)
    md_parts.extend(
        [
            "### Calibration and Confidence",
            "",
            f"Best-calibrated model: {best_cal['model_name']} (ECE = {format_ratio(best_cal['ece'])}). "
            f"Worst-calibrated: {worst_cal['model_name']} (ECE = {format_ratio(worst_cal['ece'])}) "
            f"({_ARTIFACTS['calibration_summary']}).",
            "",
        ]
    )
    _append_related_figure_note(md_parts, calibration_claim)
    md_parts.extend(
        [
            "### OOS Detection Deep Dive",
            "",
            f"Best OOS detector by AUROC: {ModelID(best_oos['model_id']).display_name} "
            f"(AUROC = {format_ratio(best_oos['auroc'])}) ({_ARTIFACTS['oos_threshold_comparison']}). "
            f"The main failure mode across all models is `oos_as_inscope` (false accepts).",
            "",
        ]
    )
    _append_related_figure_note(md_parts, oos_claim)
    md_parts.extend(
        [
            "### Cross-Model Error Overlap",
            "",
            f"Of {cross['total_test_examples']:,} test examples:",
            f"- All models correct: {cats['all_correct']['count']:,} ({format_ratio(cats['all_correct']['fraction'])})",
            f"- All models wrong: {cats['all_wrong']['count']:,} ({format_ratio(cats['all_wrong']['fraction'])})",
            f"- Model-specific errors: {cats['model_specific_error']['count']:,} "
            f"({format_ratio(cats['model_specific_error']['fraction'])})",
            f"- Partial overlap: {cats['partial_error']['count']:,} "
            f"({format_ratio(cats['partial_error']['fraction'])})",
            "",
            f"Of universally wrong examples, {format_ratio(cross['all_wrong_agreement'])} predict the same "
            f"incorrect class ({_ARTIFACTS['cross_model_error_comparison']}). "
            "See also `outputs/shared/analysis/universally_misclassified_examples.csv`.",
            "",
        ]
    )
    _append_related_figure_note(md_parts, overlap_claim)
    md_parts.extend(
        [
            "### Worst-Class Analysis",
            "",
            f"Classes consistently worst across all models (shared): {shared_worst_fmt} "
            f"({_ARTIFACTS['worst_classes_comparison']}).",
            "",
            "Model-specific worst classes:",
            *[
                f"- **{mid.display_name}**: {', '.join(f'`{c}`' for c in worst['unique_per_model'].get(str(mid), []))}"
                for mid in ModelID
            ],
            "",
        ]
    )
    _append_related_figure_note(md_parts, worst_classes_claim)
    md_parts.extend(
        [
            "### Confused Pairs (OOS False-Accept Targets)",
            "",
            f"Intents that persistently capture OOS examples across 2+ models: "
            f"{persistent_fmt} ({_ARTIFACTS['most_confused_pairs_table']}). "
            "These span multiple domains (travel, food, finance), suggesting OOS queries are topically diverse.",
            "",
        ]
    )
    _append_related_figure_note(md_parts, confused_pairs_claim)
    md_parts.extend(
        [
            "### Length and In-Scope / OOS Slices",
            "",
            "The scope comparison separates in-scope classes from the supervised OOS class; "
            "it does not measure training frequency. Historical frequency_slice filenames are retained.",
            "",
            "Short-query accuracy per model (representative run):",
            *[
                f"- **{mid.display_name}**: {format_ratio(length.get(str(mid), {}).get('short_accuracy', 0))}"
                for mid in ModelID
                if str(mid) in length
            ],
            "",
            f"({_ARTIFACTS['error_analysis_summary']})",
            "",
        ]
    )
    _append_related_figure_note(md_parts, length_claim)

    md = "\n".join(md_parts)

    confusion_fig = next(
        (f for f in figure_entries if f["figure_type"] == "confusion_matrix"),
        None,
    )
    if confusion_fig:
        fn = ModelID(confusion_fig.get("model_name", "mlp")).display_name
        md += "\n" + "\n".join(
            [
                "### Confusion Matrix",
                "",
                f"![Test-set confusion matrix across 151 intent classes including OOS for {fn}. "
                f"Data: representative run, CLINC150 test set.]({confusion_fig['figure_path']})",
                "",
            ]
        )

    metadata = _meta(
        claims=claims,
        taxonomy_summary={
            str(mid): {
                "dominant_category": dominant[str(mid)],
                "oos_as_inscope_count": tax_by[str(mid)]["oos_as_inscope_count"],
            }
            for mid in ModelID
        },
        calibration_finding={
            "best_model": best_cal["model_id"],
            "best_ece": best_cal["ece"],
            "worst_model": worst_cal["model_id"],
            "worst_ece": worst_cal["ece"],
        },
        oos_finding={
            "best_model": best_oos["model_id"],
            "best_auroc": best_oos["auroc"],
            "main_failure_mode": "oos_as_inscope",
        },
        cross_model_overlap={
            "all_correct": cats["all_correct"],
            "all_wrong": cats["all_wrong"],
            "model_specific": cats["model_specific_error"],
        },
        worst_classes_finding={
            "shared": worst["shared_worst_classes"],
            "model_specific": worst["unique_per_model"],
        },
        confused_pairs_finding={"persistent_intents": persistent[:6]},
        length_finding={
            str(mid): length.get(str(mid), {}).get("short_accuracy") for mid in ModelID if str(mid) in length
        },
        source_artifacts=sources,
    )
    return md, metadata


# ---------------------------------------------------------------------------
# Section K: Representative examples
# ---------------------------------------------------------------------------


def _select_examples(all_examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select 15-25 diverse examples ensuring category and model coverage.

    Spec K requires at least 3 OOS false accepts, 3 semantic confusions,
    3 cross-domain confusions, 3 short-query ambiguity examples, and
    at least 2 examples from each model.
    """
    target_categories: list[tuple[str, int]] = [
        ("oos_as_inscope", 3),
        ("near_semantic_confusion", 3),
        ("cross_domain_confusion", 3),
        ("short_query_ambiguity", 3),
    ]
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for cat, minimum in target_categories:
        candidates = sorted(
            [e for e in all_examples if e["primary_category"] == cat],
            key=lambda x: x["max_confidence"],
            reverse=True,
        )
        count = 0
        for ex in candidates:
            if count >= minimum:
                break
            key = (ex["text"], ex["model_id"])
            if key not in seen:
                seen.add(key)
                selected.append(ex)
                count += 1

    for mid in ModelID:
        model_count = sum(1 for ex in selected if ex["model_id"] == str(mid))
        if model_count >= 2:
            continue
        candidates = sorted(
            [e for e in all_examples if e["model_id"] == str(mid)],
            key=lambda x: x["max_confidence"],
            reverse=True,
        )
        for ex in candidates:
            if model_count >= 2:
                break
            key = (ex["text"], ex["model_id"])
            if key not in seen:
                seen.add(key)
                selected.append(ex)
                model_count += 1

    remaining = sorted(
        [e for e in all_examples if (e["text"], e["model_id"]) not in seen],
        key=lambda x: x["max_confidence"],
        reverse=True,
    )
    for ex in remaining:
        if len(selected) >= 25:
            break
        seen.add((ex["text"], ex["model_id"]))
        selected.append(ex)

    cat_order = {c: i for i, (c, _) in enumerate(target_categories)}
    selected.sort(key=lambda x: (cat_order.get(x["primary_category"], 99), -x["max_confidence"]))
    return selected[:25]


def generate_representative_examples() -> SectionOutput:
    """Spec section K / K2."""
    curated = _load("curated_report_examples")
    source = _ARTIFACTS["curated_report_examples"]
    figure_entries = load_figure_manifest_entries()
    all_examples: list[dict[str, Any]] = curated["examples"]
    selected = _select_examples(all_examples)

    table_rows = [
        [
            ex["text"][:60] + ("..." if len(ex["text"]) > 60 else ""),
            ex["true_label_name"],
            ex["predicted_label_name"],
            ModelID(ex["model_id"]).display_name,
            format_ratio(ex["max_confidence"]),
            ex["primary_category"],
            ex["annotation_tag"],
        ]
        for ex in selected
    ]
    example_table = build_markdown_table(
        ["Text", "True Label", "Predicted", "Model", "Confidence", "Category", "Annotation"],
        table_rows,
    )

    criteria = (
        "Examples selected to cover key error patterns: at least 3 OOS false accepts, "
        "3 semantic confusions, 3 cross-domain confusions, 3 short-query ambiguity, "
        "and 2+ examples per model. Sorted by category then confidence."
    )

    selected_counts: dict[str, int] = {}
    for example in selected:
        category = example["primary_category"]
        selected_counts[category] = selected_counts.get(category, 0) + 1

    oos_claim = build_structured_claim(
        claim_id="oos_false_accept_examples",
        claim_text=f"Selected examples include {selected_counts.get('oos_as_inscope', 0)} OOS false-accept cases.",
        figure_entries=figure_entries,
        source_artifacts=[source],
        related_figure_types=["oos_error_breakdown", "oos_error_comparison"],
        scopes=["analysis"],
    )
    semantic_claim = build_structured_claim(
        claim_id="semantic_confusion_examples",
        claim_text=(
            f"Selected examples include {selected_counts.get('near_semantic_confusion', 0)} "
            "within-domain semantic-confusion cases."
        ),
        figure_entries=figure_entries,
        source_artifacts=[source],
        related_figure_types=["top_confused_pairs", "confusion_matrix"],
        scopes=["representative"],
    )
    cross_domain_claim = build_structured_claim(
        claim_id="cross_domain_examples",
        claim_text=(
            f"Selected examples include {selected_counts.get('cross_domain_confusion', 0)} "
            "cross-domain confusion cases."
        ),
        figure_entries=figure_entries,
        source_artifacts=[source],
        related_figure_types=["error_summary", "cross_model_error_overlap"],
        scopes=["representative", "analysis"],
    )
    short_query_claim = build_structured_claim(
        claim_id="short_query_examples",
        claim_text=(
            f"Selected examples include {selected_counts.get('short_query_ambiguity', 0)} short-query ambiguity cases."
        ),
        figure_entries=figure_entries,
        source_artifacts=[source],
        related_figure_types=["length_slice"],
        scopes=["analysis"],
    )
    claims = [oos_claim, semantic_claim, cross_domain_claim, short_query_claim]

    md_parts = [
        "## Representative Examples",
        "",
        _scope_note("representative"),
        "",
        criteria,
        "",
        "### Coverage in Selected Examples",
        "",
        f"- {oos_claim['claim_text']}",
        f"- {semantic_claim['claim_text']}",
        f"- {cross_domain_claim['claim_text']}",
        f"- {short_query_claim['claim_text']}",
        "",
    ]
    for claim in claims:
        _append_related_figure_note(md_parts, claim)
    md_parts.extend(
        [
            example_table,
            "",
            f"Total curated examples: {curated['total_curated']}; selected for report: {len(selected)} ({source}).",
            "",
        ]
    )
    md = "\n".join(md_parts)

    metadata = _meta(
        claims=claims,
        selection_criteria=criteria,
        total_curated_count=curated["total_curated"],
        selected_count=len(selected),
        examples=[
            {
                "text": ex["text"],
                "true_label_name": ex["true_label_name"],
                "predicted_label_name": ex["predicted_label_name"],
                "model_id": ex["model_id"],
                "max_confidence": ex["max_confidence"],
                "primary_category": ex["primary_category"],
                "annotation": ex["annotation_tag"],
            }
            for ex in selected
        ],
        data_scope="representative",
        source_artifact=source,
    )
    return md, metadata


# ---------------------------------------------------------------------------
# Section L: Key findings
# ---------------------------------------------------------------------------


def generate_key_findings() -> SectionOutput:
    """Spec section L / L2."""
    eas = _load("error_analysis_summary")
    eff = _load("efficiency_summary_table")
    figure_entries = load_figure_manifest_entries()

    sources = [
        _ARTIFACTS["error_analysis_summary"],
        _ARTIFACTS["model_comparison_aggregate"],
        _ARTIFACTS["calibration_summary"],
        _ARTIFACTS["oos_threshold_comparison"],
        _ARTIFACTS["efficiency_summary_table"],
        _ARTIFACTS["cross_model_error_comparison"],
    ]

    hl = eas["headline"]
    cal_find = eas["calibration"]
    oos_find = eas["oos_detection"]
    conf_find = eas["confusion"]
    arch_find = eas["architecture_comparison"]
    len_find = eas["length"]

    eff_by = _rows_by_model(eff["rows"])
    length_source_artifacts = list(len_find.get("source_artifacts", [_ARTIFACTS["error_analysis_summary"]]))

    findings: list[dict[str, Any]] = [
        {
            "finding_id": "headline",
            "category": "headline",
            "claim": (
                f"{ModelID(hl['best_model']).display_name} achieved the best aggregate test macro F1 of "
                f"{format_mean_std(hl['test_macro_f1_mean'], hl['test_macro_f1_std'])}"
            ),
            "metric_value": hl["test_macro_f1_mean"],
            "metric_std": hl["test_macro_f1_std"],
            "source_artifact": hl["source_artifact"],
            "source_artifacts": [hl["source_artifact"]],
            "claim_id": "headline",
            "claim_text": (
                f"{ModelID(hl['best_model']).display_name} achieved the best aggregate test macro F1 of "
                f"{format_mean_std(hl['test_macro_f1_mean'], hl['test_macro_f1_std'])}"
            ),
            "related_figure_types": ["model_comparison_test_macro_f1"],
            "related_figure_paths": build_structured_claim(
                claim_id="headline",
                claim_text="",
                figure_entries=figure_entries,
                source_artifacts=[hl["source_artifact"]],
                related_figure_types=["model_comparison_test_macro_f1"],
                scopes=["aggregate"],
            )["related_figure_paths"],
        },
        {
            "finding_id": "calibration",
            "category": "calibration",
            "claim": (
                f"Best calibrated: {cal_find['best_calibrated']} (ECE = {format_ratio(cal_find['best_ece'])}). "
                f"Worst: {cal_find['worst_calibrated']} (ECE = {format_ratio(cal_find['worst_ece'])})"
            ),
            "metric_value": cal_find["best_ece"],
            "metric_std": None,
            "source_artifact": cal_find["source_artifact"],
            "source_artifacts": [cal_find["source_artifact"]],
            "claim_id": "calibration",
            "claim_text": (
                f"Best calibrated: {cal_find['best_calibrated']} (ECE = {format_ratio(cal_find['best_ece'])}). "
                f"Worst: {cal_find['worst_calibrated']} (ECE = {format_ratio(cal_find['worst_ece'])})"
            ),
            "related_figure_types": ["calibration_comparison"],
            "related_figure_paths": build_structured_claim(
                claim_id="calibration",
                claim_text="",
                figure_entries=figure_entries,
                source_artifacts=[cal_find["source_artifact"]],
                related_figure_types=["calibration_comparison"],
                scopes=["analysis"],
            )["related_figure_paths"],
        },
        {
            "finding_id": "oos_detection",
            "category": "oos_detection",
            "claim": (
                f"{ModelID(oos_find['best_model']).display_name} achieved the best OOS detection "
                f"with AUROC = {format_ratio(oos_find['best_auroc'])} and AUPR = {format_ratio(oos_find['best_aupr'])}"
            ),
            "metric_value": oos_find["best_auroc"],
            "metric_std": None,
            "source_artifact": oos_find["source_artifact"],
            "source_artifacts": [oos_find["source_artifact"]],
            "claim_id": "oos_detection",
            "claim_text": (
                f"{ModelID(oos_find['best_model']).display_name} achieved the best OOS detection "
                f"with AUROC = {format_ratio(oos_find['best_auroc'])} and AUPR = {format_ratio(oos_find['best_aupr'])}"
            ),
            "related_figure_types": ["oos_roc_comparison", "oos_pr_comparison"],
            "related_figure_paths": build_structured_claim(
                claim_id="oos_detection",
                claim_text="",
                figure_entries=figure_entries,
                source_artifacts=[oos_find["source_artifact"]],
                related_figure_types=["oos_roc_comparison", "oos_pr_comparison"],
                scopes=["analysis"],
            )["related_figure_paths"],
        },
        {
            "finding_id": "confusion",
            "category": "confusion",
            "claim": (
                "Dominant error category across all models: `oos_as_inscope`. "
                "Dominant per model: "
                + ", ".join(
                    f"{ModelID(k).display_name}: {v}" for k, v in conf_find["dominant_error_categories"].items()
                )
            ),
            "metric_value": None,
            "metric_std": None,
            "source_artifact": conf_find["source_artifact"],
            "source_artifacts": [conf_find["source_artifact"]],
            "claim_id": "confusion",
            "claim_text": (
                "Dominant error category across all models: `oos_as_inscope`. Dominant per model: "
                + ", ".join(
                    f"{ModelID(k).display_name}: {v}" for k, v in conf_find["dominant_error_categories"].items()
                )
            ),
            "related_figure_types": ["error_taxonomy", "top_confused_pairs"],
            "related_figure_paths": build_structured_claim(
                claim_id="confusion",
                claim_text="",
                figure_entries=figure_entries,
                source_artifacts=[conf_find["source_artifact"]],
                related_figure_types=["error_taxonomy", "top_confused_pairs"],
                scopes=["analysis", "representative"],
            )["related_figure_paths"],
        },
        {
            "finding_id": "architecture",
            "category": "architecture",
            "claim": (
                f"{arch_find['all_wrong_count']} examples ({format_ratio(arch_find['all_wrong_fraction'])}) "
                f"misclassified by all models; {arch_find['model_specific_count']} "
                f"({format_ratio(arch_find['model_specific_fraction'])}) unique to one model. "
                f"Agreement on wrong class: {format_ratio(arch_find['all_wrong_agreement'])}"
            ),
            "metric_value": arch_find["all_wrong_fraction"],
            "metric_std": None,
            "source_artifact": arch_find["source_artifact"],
            "source_artifacts": [arch_find["source_artifact"]],
            "claim_id": "architecture",
            "claim_text": (
                f"{arch_find['all_wrong_count']} examples ({format_ratio(arch_find['all_wrong_fraction'])}) "
                f"misclassified by all models; {arch_find['model_specific_count']} "
                f"({format_ratio(arch_find['model_specific_fraction'])}) unique to one model. "
                f"Agreement on wrong class: {format_ratio(arch_find['all_wrong_agreement'])}"
            ),
            "related_figure_types": ["cross_model_error_overlap"],
            "related_figure_paths": build_structured_claim(
                claim_id="architecture",
                claim_text="",
                figure_entries=figure_entries,
                source_artifacts=[arch_find["source_artifact"]],
                related_figure_types=["cross_model_error_overlap"],
                scopes=["analysis"],
            )["related_figure_paths"],
        },
        {
            "finding_id": "length",
            "category": "length",
            "claim": (
                "Short-query accuracy: "
                + ", ".join(
                    f"{ModelID(k).display_name} {format_ratio(v['short_accuracy'])}"
                    for k, v in len_find.items()
                    if isinstance(v, dict) and "short_accuracy" in v
                )
            ),
            "metric_value": None,
            "metric_std": None,
            "source_artifact": _ARTIFACTS["error_analysis_summary"],
            "source_artifacts": length_source_artifacts,
            "claim_id": "length",
            "claim_text": (
                "Short-query accuracy: "
                + ", ".join(
                    f"{ModelID(k).display_name} {format_ratio(v['short_accuracy'])}"
                    for k, v in len_find.items()
                    if isinstance(v, dict) and "short_accuracy" in v
                )
            ),
            "related_figure_types": ["length_slice"],
            "related_figure_paths": build_structured_claim(
                claim_id="length",
                claim_text="",
                figure_entries=figure_entries,
                source_artifacts=length_source_artifacts,
                related_figure_types=["length_slice"],
                scopes=["analysis"],
            )["related_figure_paths"],
        },
        {
            "finding_id": "efficiency",
            "category": "efficiency",
            "claim": (
                "Parameter counts: "
                + ", ".join(
                    f"{mid.display_name} {format_param_count(eff_by[str(mid)]['parameter_count'])}" for mid in ModelID
                )
                + ". Inference throughput: "
                + ", ".join(
                    f"{mid.display_name} {eff_by[str(mid)]['inference_examples_per_sec_mean']:.0f} ex/s"
                    for mid in ModelID
                )
            ),
            "metric_value": None,
            "metric_std": None,
            "source_artifact": _ARTIFACTS["efficiency_summary_table"],
            "source_artifacts": [_ARTIFACTS["efficiency_summary_table"]],
            "claim_id": "efficiency",
            "claim_text": (
                "Parameter counts: "
                + ", ".join(
                    f"{mid.display_name} {format_param_count(eff_by[str(mid)]['parameter_count'])}" for mid in ModelID
                )
                + ". Inference throughput: "
                + ", ".join(
                    f"{mid.display_name} {eff_by[str(mid)]['inference_examples_per_sec_mean']:.0f} ex/s"
                    for mid in ModelID
                )
            ),
            "related_figure_types": ["model_efficiency_comparison"],
            "related_figure_paths": build_structured_claim(
                claim_id="efficiency",
                claim_text="",
                figure_entries=figure_entries,
                source_artifacts=[_ARTIFACTS["efficiency_summary_table"]],
                related_figure_types=["model_efficiency_comparison"],
                scopes=["aggregate"],
            )["related_figure_paths"],
        },
        {
            "finding_id": "recommendation",
            "category": "recommendation",
            "claim": eas["top_recommendation"],
            "metric_value": None,
            "metric_std": None,
            "source_artifact": _ARTIFACTS["error_analysis_summary"],
            "source_artifacts": [_ARTIFACTS["error_analysis_summary"]],
            "claim_id": "recommendation",
            "claim_text": eas["top_recommendation"],
            "related_figure_types": ["oos_error_comparison", "top_confused_pairs"],
            "related_figure_paths": build_structured_claim(
                claim_id="recommendation",
                claim_text="",
                figure_entries=figure_entries,
                source_artifacts=[_ARTIFACTS["error_analysis_summary"]],
                related_figure_types=["oos_error_comparison", "top_confused_pairs"],
                scopes=["analysis", "representative"],
            )["related_figure_paths"],
        },
    ]

    md_parts = [
        "## Key Findings",
        "",
        "With only 3 repeated runs, differences between models may not be statistically meaningful. "
        "Where models have overlapping mean +/- std ranges, this is noted rather than declaring one superior.",
        "",
    ]
    for idx, finding in enumerate(findings, start=1):
        md_parts.append(
            f"{idx}. **{finding['category'].replace('_', ' ').title()}**: "
            f"{finding['claim']} ({finding['source_artifact']})"
        )
        note = render_related_figure_note(finding["related_figure_paths"])
        if note is not None:
            md_parts.append(note)
        md_parts.append("")
    md = "\n".join(md_parts)

    metadata = _meta(findings=findings, source_artifacts=sources)
    return md, metadata


# ---------------------------------------------------------------------------
# Section M: Limitations
# ---------------------------------------------------------------------------

_ANALYSIS_LIMITATIONS: list[str] = [
    "Qualitative analysis uses a single representative run per model, not all seeds.",
    "CLINC150 is balanced; real-world class distributions may differ significantly.",
    "No interpretability analysis (attention, saliency) was performed.",
    "Post-hoc calibration (temperature scaling) was not applied.",
    "Length slices use whitespace tokenization; scope slices compare supervised in-scope and OOS classes, "
    "not training frequency.",
]

_PROJECT_LIMITATIONS: list[str] = [
    "Single dataset only (CLINC150); results may not generalize to other intent-classification benchmarks.",
    "No pretrained word embeddings (GloVe, word2vec) used; embeddings trained from scratch.",
    "Whitespace tokenizer rather than subword tokenization (BPE, WordPiece).",
    "No transformer-based models compared (BERT, DistilBERT, etc.).",
    "3 repeated runs provide limited statistical power; 5+ runs would strengthen variance estimates.",
    "OOS training data is sparse relative to test distribution (250 train vs 1,000 test OOS examples).",
    "No cross-dataset validation or domain-transfer evaluation.",
    "Model checkpoints not committed to repository; reproduction requires retraining.",
]


def generate_limitations() -> SectionOutput:
    """Spec section M / M2."""
    source = _ARTIFACTS["error_analysis_notes"]

    analysis_bullets = [f"- {lim}" for lim in _ANALYSIS_LIMITATIONS]
    project_bullets = [f"- {lim}" for lim in _PROJECT_LIMITATIONS]

    md = "\n".join(
        [
            "## Limitations",
            "",
            "### Analysis-Level Limitations",
            "",
            *analysis_bullets,
            "",
            "### Project-Level Limitations",
            "",
            *project_bullets,
            "",
        ]
    )

    metadata = _meta(
        analysis_limitations=_ANALYSIS_LIMITATIONS,
        project_limitations=_PROJECT_LIMITATIONS,
        source_artifacts=[source],
    )
    return md, metadata


# ---------------------------------------------------------------------------
# Section N: Future improvements
# ---------------------------------------------------------------------------

_HIGH_PRIORITY: list[dict[str, str]] = [
    {
        "improvement": "Collect more OOS training examples to reduce false accepts",
        "rationale": (
            "OOS false accepts are the dominant error across all models; "
            "more OOS training data directly addresses the distribution mismatch."
        ),
    },
    {
        "improvement": "Merge or relabel persistently confused intent pairs within the same domain",
        "rationale": ("Several intent pairs share near-identical semantics and consistently confuse all models."),
    },
    {
        "improvement": "Apply confidence thresholding in deployment to flag uncertain predictions",
        "rationale": (
            "Many errors occur at high confidence; a deployment threshold "
            "could redirect uncertain queries to human review."
        ),
    },
    {
        "improvement": "Evaluate temperature scaling for post-hoc calibration improvement",
        "rationale": (
            "Calibration varies significantly across models; "
            "temperature scaling could improve reliability without retraining."
        ),
    },
]

_MEDIUM_PRIORITY: list[dict[str, str]] = [
    {
        "improvement": "Add pretrained word embeddings (GloVe, word2vec) to Text CNN and BiLSTM",
        "rationale": ("Pretrained embeddings could improve generalization especially for rare words and OOS queries."),
    },
    {
        "improvement": "Systematic hyperparameter tuning (dropout, hidden size, learning rate)",
        "rationale": ("Current configs use limited grid search; broader exploration may improve all models."),
    },
    {
        "improvement": "Add parameter-count vs accuracy Pareto analysis",
        "rationale": ("Quantify the efficiency-accuracy trade-off to guide model selection for deployment."),
    },
    {
        "improvement": "Add threshold analysis for OOS detection with operating-point selection",
        "rationale": ("Enable tunable precision-recall trade-off for OOS detection in production."),
    },
]

_LOWER_PRIORITY: list[dict[str, str]] = [
    {
        "improvement": "Compare against transformer-based models (DistilBERT, BERT-base)",
        "rationale": "Establish an upper-bound reference for the lightweight models evaluated.",
    },
    {
        "improvement": "Evaluate on additional intent-classification datasets",
        "rationale": "Validate whether findings generalize beyond CLINC150.",
    },
    {
        "improvement": "Subword tokenization (BPE, WordPiece) for better OOV handling",
        "rationale": "Reduce OOV rates and improve generalization to unseen vocabulary.",
    },
    {
        "improvement": "Multi-task learning combining intent classification and OOS detection",
        "rationale": ("Joint training may improve OOS discrimination by explicitly modeling the boundary."),
    },
    {
        "improvement": "Bootstrap confidence intervals for more rigorous statistical comparison",
        "rationale": ("3 repeated runs provide limited statistical power; bootstrapping would strengthen claims."),
    },
]


def generate_future_improvements() -> SectionOutput:
    """Spec section N / N2."""
    source = _ARTIFACTS["error_analysis_summary"]

    def _fmt_items(items: list[dict[str, str]]) -> list[str]:
        return [f"- **{it['improvement']}**: {it['rationale']}" for it in items]

    md = "\n".join(
        [
            "## Future Improvements",
            "",
            "### High Priority",
            "",
            *_fmt_items(_HIGH_PRIORITY),
            "",
            "### Medium Priority",
            "",
            *_fmt_items(_MEDIUM_PRIORITY),
            "",
            "### Lower Priority / Future Work",
            "",
            *_fmt_items(_LOWER_PRIORITY),
            "",
            f"({source})",
            "",
        ]
    )

    metadata = _meta(
        high_priority=_HIGH_PRIORITY,
        medium_priority=_MEDIUM_PRIORITY,
        lower_priority=_LOWER_PRIORITY,
        source_artifacts=[source],
    )
    return md, metadata


# ---------------------------------------------------------------------------
# Section O: Reproducibility
# ---------------------------------------------------------------------------


def generate_reproducibility() -> SectionOutput:
    """Spec section O / O2."""
    proto = _load("evaluation_protocol")
    eff = _load("efficiency_summary_table")
    sources = [_ARTIFACTS["evaluation_protocol"], _ARTIFACTS["efficiency_summary_table"]]

    pc = proto["protocol_config"]
    eff_by = _rows_by_model(eff["rows"])

    python_version = ">=3.11"
    uv_lock_exists = (PROJECT_ROOT / "uv.lock").exists()

    approx_runtime: dict[str, float] = {}
    for mid in ModelID:
        per_run_s = eff_by[str(mid)]["training_time_seconds_mean"]
        approx_runtime[str(mid)] = round(per_run_s * pc["run_count"] / 60, 1)

    steps = [
        "Clone https://github.com/alexiwoh/clinc150-project into a separate checkout; "
        "generated outputs are replaced in place.",
        "Install locked dependencies from the repository root with `uv sync --frozen`.",
        "Run frozen final evaluations, then tracking, report figures, error analysis and report generation.",
        "For source reconstruction, rebuild dataset summaries and train-only preprocessing before evaluation.",
        "Run offline tests with `uv run --frozen pytest -q`; generated-artifact checks require trained checkpoints.",
        "Verify outputs exist under `outputs/` with the expected directory structure",
    ]

    nondeterminism = [
        "MPS backend nondeterminism on Apple Silicon (PyTorch does not guarantee deterministic MPS operations).",
        "DataLoaders use num_workers=0 and the recorded per-run shuffle seed.",
        "Exact numeric reproduction across different hardware is not guaranteed.",
    ]

    md = "\n".join(
        [
            "## Reproducibility",
            "",
            "### Environment Setup",
            "",
            f"- Python version: {python_version}",
            "- Install from the repository root: `uv sync --frozen`",
            "- Secondary route: activate a virtual environment, then `python -m pip install -r requirements.txt`. "
            "The portable export includes `-e .` to install the project.",
            f"- Pinned dependency versions available: {'`uv.lock`' if uv_lock_exists else '`requirements.txt`'}",
            "- PyTorch with MPS support for Apple Silicon (optional; CPU fallback available)",
            "",
            "### Dataset Acquisition",
            "",
            "CLINC150 is loaded via the HuggingFace `datasets` library (`clinc/clinc_oos`, `plus` subset). "
            "The configured revision is pinned and future runs record ordered split hashes. "
            "The first uncached download requires network access; official split membership is preserved.",
            "",
            "### Frozen Final Evaluations and Report",
            "",
            "```bash",
            "uv run --frozen python scripts/run_repeated_evaluation.py --model all --run-count 3",
            "uv run --frozen python scripts/run_experiment_tracking.py",
            "uv run --frozen python scripts/run_report_figures.py",
            "uv run --frozen python scripts/run_error_analysis.py",
            "uv run --frozen python scripts/run_report_generation.py",
            "```",
            "",
            f"Seeds: {pc['seed_list']}",
            "",
            "To rebuild preprocessing from source first run `uv run --frozen python scripts/explore_dataset.py` "
            "and `uv run --frozen python scripts/run_preprocessing.py`. The evaluation commands reuse existing "
            "frozen hyperparameters; they do not retune. "
            "See the repository README for explicit tuning-source selection.",
            "",
            "### Expected Output Structure",
            "",
            "```",
            "outputs/",
            "  mlp/           # MLP model artifacts (tuning, final_runs, aggregate, figures, analysis)",
            "  text_cnn/      # Text CNN model artifacts",
            "  bilstm/        # BiLSTM model artifacts",
            "  shared/        # Cross-model comparisons, figures, analysis, report",
            "```",
            "",
            "### Approximate Runtime",
            "",
            *[
                f"- **{mid.display_name}** (3 runs): ~{approx_runtime[str(mid)]} min "
                f"(training only; {format_time_seconds(eff_by[str(mid)]['training_time_seconds_mean'])} s/run)"
                for mid in ModelID
            ],
            "",
            "Hardware, device, source commit, input hashes and effective settings are recorded per run. "
            "These times cover the training loop including validation/checkpoint writes, excluding setup, "
            "preprocessing and tuning. Batched evaluation ms/example is throughput, not single-query service latency.",
            "",
            "### Nondeterminism Notes",
            "",
            *[f"- {note}" for note in nondeterminism],
            "",
            "### Reproduction Checklist",
            "",
            *[f"{i}. {step}" for i, step in enumerate(steps, 1)],
            "",
        ]
    )

    metadata = _meta(
        python_version=python_version,
        install_command="uv sync --frozen",
        pinned_versions_available=uv_lock_exists,
        dataset_source="clinc/clinc_oos (plus subset)",
        pipeline_command="uv run --frozen python scripts/run_repeated_evaluation.py --model all --run-count 3",
        report_command="uv run --frozen python scripts/run_report_generation.py",
        seed_list=pc["seed_list"],
        run_count=pc["run_count"],
        approximate_runtime_minutes=approx_runtime,
        hardware_context="See each current run_metadata.json for the actual machine and device",
        nondeterminism_notes=nondeterminism,
        reproduction_steps=steps,
        source_artifacts=sources,
    )
    return md, metadata
