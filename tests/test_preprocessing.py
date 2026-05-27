from __future__ import annotations

import numpy as np

from traffic_agent.data.preprocessing import preprocess_data, split_time_ordered


def test_time_ordered_split_ratios() -> None:
    data = np.arange(100 * 2 * 2, dtype=np.float32).reshape(100, 2, 2)
    train, val, test, indices = split_time_ordered(data, train_ratio=0.7, val_ratio=0.1)
    assert train.shape[0] == 70
    assert val.shape[0] == 10
    assert test.shape[0] == 20
    assert indices == (70, 80)
    assert np.array_equal(train[-1], data[69])
    assert np.array_equal(val[0], data[70])


def test_scaler_fit_only_on_train() -> None:
    train = np.ones((70, 2, 1), dtype=np.float32) * 10
    val = np.ones((10, 2, 1), dtype=np.float32) * 1000
    test = np.ones((20, 2, 1), dtype=np.float32) * 1000
    data = np.concatenate([train, val, test], axis=0)
    result = preprocess_data(data, train_ratio=0.7, val_ratio=0.1)
    assert np.isclose(result.scaler.mean_[0], 10.0)


def test_missing_values_filled() -> None:
    data = np.random.default_rng(0).normal(size=(100, 3, 2)).astype(np.float32)
    data[0:5, :, :] = np.nan
    data[20, 1, 0] = np.nan
    result = preprocess_data(data)
    assert not np.isnan(result.train).any()
    assert not np.isnan(result.val).any()
    assert not np.isnan(result.test).any()
