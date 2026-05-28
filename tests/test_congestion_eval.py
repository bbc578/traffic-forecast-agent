from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from traffic_agent.analysis.congestion_eval import run_congestion_subset_eval


def test_congestion_subset_eval_outputs_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps({"model_name": "last_value", "horizon": 3}), encoding="utf-8")
    y_true = np.array([[[20.0, 50.0], [22.0, 51.0], [21.0, 49.0]]], dtype=np.float32)
    y_pred = y_true + 1
    np.savez_compressed(run_dir / "predictions.npz", y_true=y_true, y_pred=y_pred, history_mean=np.array([[50.0, 50.0]]))
    output = run_congestion_subset_eval(tmp_path / "outputs", tmp_path / "subsets.csv")
    assert output.exists()
    assert output.with_suffix(".md").exists()
