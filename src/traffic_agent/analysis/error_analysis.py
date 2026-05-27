from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traffic_agent.training.metrics import mae, rmse


def load_predictions(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"predictions.npz not found: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def error_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Compute MAE/RMSE for each forecast horizon step."""
    rows = []
    for step in range(y_true.shape[1]):
        rows.append(
            {
                "horizon_step": step + 1,
                "mae": mae(y_true[:, step], y_pred[:, step]),
                "rmse": rmse(y_true[:, step], y_pred[:, step]),
            }
        )
    return pd.DataFrame(rows)


def error_by_node(y_true: np.ndarray, y_pred: np.ndarray, node_ids: list[str] | None = None) -> pd.DataFrame:
    """Compute node-wise errors."""
    ids = node_ids or [f"node_{idx}" for idx in range(y_true.shape[-1])]
    rows = []
    for idx, node_id in enumerate(ids):
        rows.append(
            {
                "node_id": node_id,
                "mae": mae(y_true[:, :, idx], y_pred[:, :, idx]),
                "rmse": rmse(y_true[:, :, idx], y_pred[:, :, idx]),
            }
        )
    return pd.DataFrame(rows).sort_values("mae", ascending=False)


def error_by_time_of_day(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: list[str] | None = None,
    steps_per_day: int = 288,
) -> pd.DataFrame:
    """Approximate time-of-day error buckets from timestamps or sample index."""
    sample_error = np.abs(y_true - y_pred).mean(axis=(1, 2))
    if timestamps:
        parsed = pd.to_datetime(pd.Series(timestamps), errors="coerce")
        hour = parsed.dt.hour.fillna(np.arange(len(sample_error)) % steps_per_day // 12).astype(int)
    else:
        hour = pd.Series((np.arange(len(sample_error)) % steps_per_day) // 12)
    return pd.DataFrame({"hour": hour[: len(sample_error)], "mae": sample_error}).groupby("hour", as_index=False).mean()


def residual_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Summarize residual distribution."""
    residual = (y_pred - y_true).reshape(-1)
    return {
        "mean": float(np.mean(residual)),
        "std": float(np.std(residual)),
        "p05": float(np.percentile(residual, 5)),
        "p50": float(np.percentile(residual, 50)),
        "p95": float(np.percentile(residual, 95)),
    }


def worst_case_segments(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    node_ids: list[str] | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Return sample/horizon/node cells with the largest absolute errors."""
    ids = node_ids or [f"node_{idx}" for idx in range(y_true.shape[-1])]
    errors = np.abs(y_true - y_pred)
    flat_indices = np.argsort(errors.reshape(-1))[::-1][:top_k]
    rows = []
    for flat in flat_indices:
        sample, horizon, node = np.unravel_index(flat, errors.shape)
        rows.append(
            {
                "sample_index": int(sample),
                "horizon_step": int(horizon + 1),
                "node_id": ids[int(node)],
                "absolute_error": float(errors[sample, horizon, node]),
                "y_true": float(y_true[sample, horizon, node]),
                "y_pred": float(y_pred[sample, horizon, node]),
            }
        )
    return rows


def compare_models_by_horizon(run_dirs: list[str | Path]) -> pd.DataFrame:
    """Compare multiple run directories by horizon step."""
    frames = []
    for run_dir in run_dirs:
        payload = load_predictions(run_dir)
        frame = error_by_horizon(payload["y_true"], payload["y_pred"])
        frame["run_name"] = Path(run_dir).name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
