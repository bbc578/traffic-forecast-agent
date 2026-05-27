from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from traffic_agent.analysis.congestion import identify_congestion_risk
from traffic_agent.analysis.error_analysis import load_predictions


def simple_occlusion_importance(
    input_window: np.ndarray,
    predict_fn,
    baseline_value: float = 0.0,
) -> dict[str, list[dict[str, float]]]:
    """Approximate importance by replacing one time step or node and measuring prediction change."""
    original = predict_fn(input_window)
    time_rows = []
    for step in range(input_window.shape[0]):
        occluded = input_window.copy()
        occluded[step] = baseline_value
        delta = float(np.mean(np.abs(original - predict_fn(occluded))))
        time_rows.append({"time_step": float(step), "importance": delta})
    node_rows = []
    for node in range(input_window.shape[1]):
        occluded = input_window.copy()
        occluded[:, node, :] = baseline_value
        delta = float(np.mean(np.abs(original - predict_fn(occluded))))
        node_rows.append({"node_index": float(node), "importance": delta})
    return {"time_steps": time_rows, "nodes": node_rows}


def explain_node_forecast(run_dir: str | Path, node_id: str, horizon_step: int = 1) -> dict[str, Any]:
    """Explain one saved forecast with heuristic context, not causal attribution."""
    path = Path(run_dir)
    predictions = load_predictions(path)
    node_ids = [str(item) for item in predictions["node_ids"].tolist()]
    if node_id not in node_ids:
        raise ValueError(f"node_id={node_id} not found. Available example: {node_ids[:3]}")
    node_idx = node_ids.index(node_id)
    h_idx = max(0, horizon_step - 1)
    y_pred = predictions["y_pred"]
    y_true = predictions.get("y_true")
    history_mean = predictions.get("history_mean")
    risk = identify_congestion_risk(
        y_pred,
        node_ids=node_ids,
        recent_history_mean=history_mean,
        top_k=len(node_ids),
    )
    risk_item = next((item for item in risk if item["node_id"] == node_id), None)
    metrics_path = path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    predicted_value = float(y_pred[0, h_idx, node_idx])
    true_value = float(y_true[0, h_idx, node_idx]) if y_true is not None else None
    error = abs(true_value - predicted_value) if true_value is not None else None
    return {
        "node_id": node_id,
        "horizon_step": horizon_step,
        "predicted_speed": predicted_value,
        "true_speed": true_value,
        "absolute_error": error,
        "recent_history_mean": float(history_mean[0, node_idx]) if history_mean is not None else None,
        "risk_reason": risk_item["reason"] if risk_item else "not in top risk nodes",
        "model_name": metrics.get("model_name"),
        "limitations": [
            "This is a heuristic explanation from saved predictions.",
            "Occlusion-style and residual explanations are not causal traffic evidence.",
        ],
    }
