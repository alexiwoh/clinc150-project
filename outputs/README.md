# Experiment artifacts

Current final results use `mlp/`, `text_cnn/`, and `bilstm/`: each model has frozen settings, current run ledgers, per-run predictions, aggregate metrics, and representative diagnostics. `shared/` contains the cross-model summaries, figure manifest, analysis, and [full report](shared/report/full_report_draft.md).

`reports/`, `logs/`, and `figures/` preserve earlier single-run experiments and original tuning artifacts. Their metrics have a different experiment scope from the current repeated-run summaries. Earlier metadata does not establish the source, raw split identity, or effective initialization behavior retroactively. The original tuning CSVs remain available for deliberate frozen-config extraction.

`../outputs_small_archive/` is historical. Model checkpoints are intentionally ignored and require training. Use the current ledgers to determine active run membership; directories alone do not establish it.
