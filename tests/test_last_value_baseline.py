from __future__ import annotations

import torch

from traffic_agent.models.last_value import LastValueBaseline


def test_last_value_baseline_repeats_last_speed() -> None:
    x = torch.zeros(2, 4, 3, 2)
    x[:, -1, :, 0] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pred = LastValueBaseline(horizon=3)(x)
    assert pred.shape == (2, 3, 3)
    assert torch.allclose(pred[:, 0], x[:, -1, :, 0])
