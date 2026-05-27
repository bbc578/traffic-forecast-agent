from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from traffic_agent.agent.rule_based_agent import RuleBasedTrafficAgent
from traffic_agent.agent.tools import get_latest_metrics


def _create_run(base: Path) -> Path:
    run_dir = base / "demo_run"
    run_dir.mkdir(parents=True)
    metrics = {
        "model_name": "historical_average",
        "dataset_path": "demo",
        "input_steps": 12,
        "horizon": 3,
        "mae": 1.0,
        "rmse": 2.0,
        "mape": 3.0,
        "created_at": "2026-01-01T00:00:00Z",
        "is_synthetic_data": True,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    y_true = np.ones((2, 3, 4), dtype=np.float32) * 50
    y_pred = y_true.copy()
    y_pred[:, :, 1] = 20
    np.savez_compressed(
        run_dir / "predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        history_mean=np.ones((2, 4), dtype=np.float32) * 50,
        node_ids=np.array(["a", "b", "c", "d"]),
    )
    return run_dir


def test_get_latest_metrics(tmp_path: Path) -> None:
    run_dir = _create_run(tmp_path)
    metrics = get_latest_metrics(str(run_dir))
    assert metrics["model_name"] == "historical_average"


def test_missing_run_dir_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Run directory not found"):
        get_latest_metrics(str(tmp_path / "missing"))


def test_rule_based_agent_congestion_and_report(tmp_path: Path) -> None:
    _create_run(tmp_path)
    agent = RuleBasedTrafficAgent(outputs_dir=str(tmp_path))
    congestion = agent.query("未来哪里最堵？", run_name="demo_run")
    assert congestion.tool_used == "get_top_congestion_nodes"
    assert congestion.data
    report = agent.query("生成日报", run_name="demo_run")
    assert report.tool_used == "generate_daily_report"
    assert "不可直接用于真实交通管控决策" in report.answer
