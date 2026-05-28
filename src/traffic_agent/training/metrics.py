from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def mae(y_true: Any, y_pred: Any) -> float:
    """Mean absolute error."""
    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true: Any, y_pred: Any) -> float:
    """Root mean squared error."""
    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mape(y_true: Any, y_pred: Any, epsilon: float = 1.0) -> float:
    """Mean absolute percentage error with a denominator floor.

    Traffic speed data can contain zero or near-zero readings. A small numerical epsilon makes MAPE
    explode into unusable percentages, so the default denominator floor is 1 speed unit.
    """
    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    denominator = np.maximum(np.abs(true), epsilon)
    return float(np.mean(np.abs((true - pred) / denominator)) * 100.0)


def _masked_arrays(
    y_true: Any,
    y_pred: Any,
    mask_value: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    mask = np.isfinite(true) & np.isfinite(pred)
    if mask_value is not None:
        mask &= true != mask_value
    if not np.any(mask):
        return np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64)
    return true[mask], pred[mask]


def masked_mae(y_true: Any, y_pred: Any, mask_value: float | None = None) -> float:
    """MAE ignoring NaN/Inf and optional sentinel values."""
    true, pred = _masked_arrays(y_true, y_pred, mask_value)
    return mae(true, pred)


def masked_rmse(y_true: Any, y_pred: Any, mask_value: float | None = None) -> float:
    """RMSE ignoring NaN/Inf and optional sentinel values."""
    true, pred = _masked_arrays(y_true, y_pred, mask_value)
    return rmse(true, pred)


def masked_mape(
    y_true: Any,
    y_pred: Any,
    mask_value: float | None = None,
    epsilon: float = 1.0,
) -> float:
    """MAPE ignoring NaN/Inf, optional sentinel values, and low-speed denominators."""
    true, pred = _masked_arrays(y_true, y_pred, mask_value)
    keep = np.abs(true) > epsilon
    if not np.any(keep):
        return 0.0
    return mape(true[keep], pred[keep], epsilon=epsilon)
