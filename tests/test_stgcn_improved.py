from __future__ import annotations

import torch

from traffic_agent.models.stgcn_block import STGCNImproved


def test_stgcn_improved_shape() -> None:
    model = STGCNImproved(5, 2, 3, torch.eye(5), hidden_size=8, num_layers=1)
    out = model(torch.randn(4, 12, 5, 2))
    assert out.shape == (4, 3, 5)
