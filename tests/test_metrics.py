from __future__ import annotations

import numpy as np

from traffic_agent.training.metrics import mae, mape, rmse


def test_metrics_simple_arrays() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 4.0])
    assert mae(y_true, y_pred) == 2.0 / 3.0
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(2.0 / 3.0))
    assert np.isclose(mape(y_true, y_pred), ((1.0 + 0.0 + 1.0 / 3.0) / 3.0) * 100)


def test_mape_handles_zero_true() -> None:
    value = mape(np.array([0.0, 1.0]), np.array([0.5, 1.0]))
    assert np.isfinite(value)
