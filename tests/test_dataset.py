from __future__ import annotations

import numpy as np

from traffic_agent.data.dataset import TrafficWindowDataset


def test_sliding_window_count() -> None:
    data = np.zeros((30, 4, 2), dtype=np.float32)
    dataset = TrafficWindowDataset(data, input_steps=12, horizon=3)
    assert len(dataset) == 30 - 12 - 3 + 1


def test_window_shapes() -> None:
    data = np.zeros((30, 4, 2), dtype=np.float32)
    dataset = TrafficWindowDataset(data, input_steps=12, horizon=3)
    x, y = dataset[0]
    assert tuple(x.shape) == (12, 4, 2)
    assert tuple(y.shape) == (3, 4)


def test_horizon_setting() -> None:
    data = np.zeros((40, 4, 2), dtype=np.float32)
    dataset = TrafficWindowDataset(data, input_steps=12, horizon=6)
    _, y = dataset[0]
    assert tuple(y.shape) == (6, 4)
