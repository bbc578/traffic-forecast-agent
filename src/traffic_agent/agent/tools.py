from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from traffic_agent.analysis.anomaly import detect_residual_anomalies
from traffic_agent.analysis.congestion import identify_congestion_risk
from traffic_agent.analysis.reports import generate_markdown_report


def _run_path(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: {path}. Train a model first.")
    return path


def get_latest_metrics(run_dir: str) -> dict[str, Any]:
    """Read metrics.json from a run directory."""
    path = _run_path(run_dir)
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in run directory: {path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_predictions(run_dir: str | Path) -> dict[str, Any]:
    path = _run_path(run_dir)
    predictions_path = path / "predictions.npz"
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions.npz not found in run directory: {path}")
    with np.load(predictions_path, allow_pickle=False) as loaded:
        return {
            "y_true": loaded["y_true"],
            "y_pred": loaded["y_pred"],
            "history_mean": loaded["history_mean"] if "history_mean" in loaded else None,
            "node_ids": [str(item) for item in loaded["node_ids"].tolist()],
        }


def get_top_congestion_nodes(run_dir: str, k: int = 5) -> list[dict[str, Any]]:
    """Return Top-K congestion risk nodes based on saved predictions."""
    predictions = _load_predictions(run_dir)
    return identify_congestion_risk(
        predictions["y_pred"],
        node_ids=predictions["node_ids"],
        recent_history_mean=predictions["history_mean"],
        top_k=k,
    )


def get_model_summary(run_dir: str) -> str:
    """Return a concise model performance summary from saved metrics."""
    metrics = get_latest_metrics(run_dir)
    return (
        f"模型 {metrics.get('model_name')} 在测试集上的指标为："
        f"MAE={metrics.get('mae'):.4f}, "
        f"RMSE={metrics.get('rmse'):.4f}, "
        f"MAPE={metrics.get('mape'):.4f}%。"
    )


def get_anomalies(run_dir: str) -> list[dict[str, Any]]:
    """Return residual z-score anomaly hints from saved predictions."""
    predictions = _load_predictions(run_dir)
    return detect_residual_anomalies(
        predictions["y_true"], predictions["y_pred"], node_ids=predictions["node_ids"]
    )


def generate_daily_report(run_dir: str) -> str:
    """Generate a markdown daily report from one run."""
    metrics = get_latest_metrics(run_dir)
    congestion_nodes = get_top_congestion_nodes(run_dir, k=5)
    anomalies = get_anomalies(run_dir)
    return generate_markdown_report(metrics, congestion_nodes, anomalies)


def compare_runs(outputs_dir: str) -> list[dict[str, Any]]:
    """Compare available run metrics under an outputs directory."""
    base = Path(outputs_dir)
    if not base.exists():
        raise FileNotFoundError(f"Outputs directory not found: {base}")
    rows = []
    for metrics_path in sorted(base.glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(
            {
                "run_name": metrics_path.parent.name,
                "model_name": metrics.get("model_name"),
                "horizon": metrics.get("horizon"),
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
                "mape": metrics.get("mape"),
                "created_at": metrics.get("created_at"),
            }
        )
    return rows
