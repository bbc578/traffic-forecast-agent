from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from traffic_agent.analysis.explainability import explain_node_forecast, simple_occlusion_importance


def test_simple_occlusion_importance() -> None:
    window = np.ones((3, 2, 1), dtype=np.float32)

    def predict_fn(x: np.ndarray) -> np.ndarray:
        return np.array([x.sum()])

    result = simple_occlusion_importance(window, predict_fn)
    assert len(result["time_steps"]) == 3
    assert len(result["nodes"]) == 2


def test_explain_node_forecast(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    y_true = np.ones((2, 3, 2), dtype=np.float32) * 50
    y_pred = y_true.copy()
    np.savez_compressed(
        run_dir / "predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        history_mean=np.ones((2, 2), dtype=np.float32) * 50,
        node_ids=np.array(["node_0", "node_1"]),
    )
    (run_dir / "metrics.json").write_text(json.dumps({"model_name": "last_value"}), encoding="utf-8")
    result = explain_node_forecast(run_dir, "node_0")
    assert result["node_id"] == "node_0"
