from __future__ import annotations

import numpy as np

from traffic_agent.training.metrics import mae, mape, masked_mae, masked_mape, masked_rmse, rmse


def test_metrics_simple_arrays() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 4.0])
    assert mae(y_true, y_pred) == 2.0 / 3.0
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(2.0 / 3.0))
    assert np.isclose(mape(y_true, y_pred), ((1.0 + 0.0 + 1.0 / 3.0) / 3.0) * 100)


def test_mape_handles_zero_true() -> None:
    value = mape(np.array([0.0, 1.0]), np.array([0.5, 1.0]))
    assert np.isfinite(value)


def test_masked_metrics_ignore_nan_and_mask_value() -> None:
    y_true = np.array([1.0, np.nan, 0.0, -1.0])
    y_pred = np.array([2.0, 3.0, 10.0, 99.0])
    assert masked_mae(y_true, y_pred, mask_value=-1.0) == 5.5
    assert np.isfinite(masked_rmse(y_true, y_pred, mask_value=-1.0))
    assert np.isfinite(masked_mape(y_true, y_pred, mask_value=-1.0))
