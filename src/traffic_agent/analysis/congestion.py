from __future__ import annotations

from collections.abc import Sequence

import numpy as np

VALID_RISK_LEVELS = {"low", "medium", "high"}


def _as_horizon_node(predicted_speed: np.ndarray) -> np.ndarray:
    arr = np.asarray(predicted_speed, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2:
        raise ValueError("predicted_speed must have shape [horizon, nodes] or [batch, horizon, nodes].")
    return arr


def identify_congestion_risk(
    predicted_speed: np.ndarray,
    node_ids: Sequence[str] | None = None,
    recent_history_mean: np.ndarray | None = None,
    speed_threshold: float = 35.0,
    drop_ratio_threshold: float = 0.25,
    top_k: int = 5,
) -> list[dict[str, object]]:
    """Rank nodes by congestion risk from predicted speed and optional recent history."""
    speed = _as_horizon_node(predicted_speed)
    num_nodes = speed.shape[1]
    ids = list(node_ids) if node_ids is not None else [f"node_{i}" for i in range(num_nodes)]
    if len(ids) != num_nodes:
        raise ValueError(f"node_ids length {len(ids)} does not match num_nodes {num_nodes}.")

    min_speed = np.nanmin(speed, axis=0)
    threshold_component = np.clip((speed_threshold - min_speed) / max(speed_threshold, 1e-6), 0, 1)

    drop_component = np.zeros(num_nodes, dtype=np.float32)
    if recent_history_mean is not None:
        history = np.asarray(recent_history_mean, dtype=np.float32)
        if history.ndim == 2:
            history = history.reshape(-1, history.shape[-1]).mean(axis=0)
        if history.ndim != 1 or history.shape[0] != num_nodes:
            raise ValueError("recent_history_mean must have shape [nodes] or [batch, nodes].")
        drop_ratio = (history - min_speed) / np.maximum(np.abs(history), 1e-6)
        drop_component = np.clip(drop_ratio / drop_ratio_threshold, 0, 1)

    risk_score = 0.7 * threshold_component + 0.3 * drop_component
    order = np.argsort(-risk_score)[:top_k]
    results: list[dict[str, object]] = []
    for idx in order:
        score = float(risk_score[idx])
        if min_speed[idx] < speed_threshold * 0.8 or score >= 0.75:
            level = "high"
        elif min_speed[idx] < speed_threshold or score >= 0.35:
            level = "medium"
        else:
            level = "low"

        reasons = []
        if min_speed[idx] < speed_threshold:
            reasons.append(f"predicted speed below {speed_threshold:.1f}")
        if drop_component[idx] > 0:
            reasons.append("predicted speed drops from recent history")
        reason = "; ".join(reasons) if reasons else "relative risk is low in the selected horizon"
        results.append(
            {
                "node_id": ids[int(idx)],
                "risk_score": round(score, 4),
                "predicted_min_speed": round(float(min_speed[idx]), 4),
                "risk_level": level,
                "reason": reason,
            }
        )
    return results
