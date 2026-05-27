from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def detect_residual_anomalies(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    node_ids: Sequence[str] | None = None,
    z_threshold: float = 3.0,
    max_items: int = 20,
) -> list[dict[str, object]]:
    """Detect large residual anomalies with a simple z-score heuristic."""
    true = np.asarray(y_true, dtype=np.float32)
    pred = np.asarray(y_pred, dtype=np.float32)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {true.shape} and {pred.shape}.")
    if true.ndim != 3:
        raise ValueError("Expected y_true and y_pred shape [batch, horizon, nodes].")

    residual = np.abs(true - pred)
    mean = residual.mean()
    std = residual.std() + 1e-6
    z_scores = (residual - mean) / std
    batch_idx, horizon_idx, node_idx = np.where(z_scores >= z_threshold)
    ids = list(node_ids) if node_ids is not None else [f"node_{i}" for i in range(true.shape[-1])]

    anomalies = []
    for b, h, n in zip(batch_idx, horizon_idx, node_idx, strict=False):
        anomalies.append(
            {
                "node_id": ids[int(n)],
                "sample_index": int(b),
                "time_step": int(h),
                "anomaly_score": round(float(z_scores[b, h, n]), 4),
                "absolute_error": round(float(residual[b, h, n]), 4),
            }
        )
    anomalies.sort(key=lambda item: float(item["anomaly_score"]), reverse=True)
    return anomalies[:max_items]
