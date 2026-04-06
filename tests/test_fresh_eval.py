"""Fresh-process evaluation smoke test: checkpoint loads and evaluates without re-training."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from src.config import MLPBaselineConfig
from src.models.mlp import MLPClassifier
from src.utils import save_checkpoint


@pytest.fixture
def mock_eval_env(tmp_path: Path) -> dict:
    """Set up a mock environment with checkpoint and preprocessing artifacts."""
    input_dim = 50
    num_classes = 5
    config = MLPBaselineConfig(hidden_dim=32, batch_size=8)

    model = MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        dropout_rate=config.dropout_rate,
        activation=config.activation,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    ckpt_path = tmp_path / "test_checkpoint.pt"
    save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        epoch=5,
        best_metric=0.85,
        config=config.to_dict(),
        artifact_refs={"vectorizer": "mock"},
    )

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    label_names = [f"intent_{i}" for i in range(num_classes)]
    label_to_id = {name: i for i, name in enumerate(label_names)}
    id_to_label = {i: name for i, name in enumerate(label_names)}

    (artifacts_dir / "label_to_id.json").write_text(json.dumps(label_to_id))
    (artifacts_dir / "id_to_label.json").write_text(json.dumps({str(k): v for k, v in id_to_label.items()}))

    return {
        "checkpoint_path": ckpt_path,
        "config": config,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "label_names": label_names,
        "artifacts_dir": artifacts_dir,
        "tmp_path": tmp_path,
    }


def test_fresh_eval_loads_and_evaluates(mock_eval_env: dict) -> None:
    """Confirm evaluate_mlp_baseline works from checkpoint without re-training."""
    from src.evaluate import evaluate_mlp_baseline

    env = mock_eval_env
    n_test = 20
    rng = np.random.RandomState(0)
    test_features = csr_matrix(rng.randn(n_test, env["input_dim"]).astype(np.float32))
    test_labels = list(rng.randint(0, env["num_classes"], n_test))
    test_texts = [f"test sentence {i}" for i in range(n_test)]

    class FakeDataset:
        def __getitem__(self, key):
            return {"text": test_texts, "intent": test_labels}

    class FakeVectorizer:
        def transform(self, texts):
            return test_features

    class FakeCLINCDataset:
        @classmethod
        def load(cls, config):
            return cls()

        def __getitem__(self, split):
            return {"text": test_texts, "intent": test_labels}

    fake_artifacts = {
        "vectorizer": FakeVectorizer(),
        "summary": {"tfidf_fitted_feature_dim": env["input_dim"]},
    }

    reports_dir = env["tmp_path"] / "reports"

    with (
        patch("src.evaluate.load_preprocessing_artifacts", return_value=fake_artifacts),
        patch("src.evaluate.CLINCDataset", FakeCLINCDataset),
        patch("src.evaluate.ARTIFACTS_DIR", env["artifacts_dir"]),
        patch("src.evaluate.REPORTS_DIR", reports_dir),
        patch("src.evaluate.NUM_CLASSES", env["num_classes"]),
    ):
        results = evaluate_mlp_baseline(env["checkpoint_path"], env["config"])

    assert "test_accuracy" in results
    assert "test_macro_f1" in results
    assert "oos_precision" in results
    assert "confusion_matrix" in results
    assert "top_confusions" in results
    assert "top_errors" in results
    assert "inference_latency" in results
    assert results["inference_latency"]["avg_ms_per_example"] >= 0

    assert (reports_dir / "mlp_test_metrics.json").exists()
    assert (reports_dir / "mlp_confusion_matrix.csv").exists()
    assert (reports_dir / "mlp_top_errors.json").exists()
    assert (reports_dir / "mlp_comparison_row.json").exists()

    errors_data = json.loads((reports_dir / "mlp_top_errors.json").read_text())
    assert "selection_rule" in errors_data
