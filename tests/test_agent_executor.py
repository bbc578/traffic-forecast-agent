from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from traffic_agent.agent.executor import AgentExecutor


def test_agent_executor_runs_and_saves_trace(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps({"model_name": "last_value", "mae": 1.0, "rmse": 2.0, "mape": 3.0}),
        encoding="utf-8",
    )
    np.savez_compressed(
        run_dir / "predictions.npz",
        y_true=np.ones((2, 3, 2), dtype=np.float32),
        y_pred=np.ones((2, 3, 2), dtype=np.float32),
        history_mean=np.ones((2, 2), dtype=np.float32),
        node_ids=np.array(["node_0", "node_1"]),
    )
    response = AgentExecutor(outputs_dir=str(tmp_path)).run("未来哪里最堵？", run_name="run")
    assert response.trace_id is not None
    assert response.tools_used == ["get_top_congestion_nodes"]
