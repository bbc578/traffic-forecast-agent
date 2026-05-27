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


def mape(y_true: Any, y_pred: Any, epsilon: float = 1e-5) -> float:
    """Mean absolute percentage error with epsilon protection."""
    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    denominator = np.maximum(np.abs(true), epsilon)
    return float(np.mean(np.abs((true - pred) / denominator)) * 100.0)
