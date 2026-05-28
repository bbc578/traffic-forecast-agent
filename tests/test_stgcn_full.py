from __future__ import annotations

import torch

from traffic_agent.models.stgcn_full import STGCNFull


def test_stgcn_full_forward_shape_cpu() -> None:
    model = STGCNFull(5, 1, 3, torch.eye(5), hidden_channels=8, num_blocks=1)
    out = model(torch.randn(2, 12, 5, 1))
    assert out.shape == (2, 3, 5)
