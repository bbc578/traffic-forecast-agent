from __future__ import annotations

import torch

from traffic_agent.models.graph_wavenet_full import GraphWaveNetFull


def test_graph_wavenet_full_forward_shape_cpu() -> None:
    model = GraphWaveNetFull(
        num_nodes=5,
        num_features=1,
        horizon=3,
        adjacency=torch.eye(5),
        residual_channels=8,
        dilation_channels=8,
        skip_channels=16,
        end_channels=32,
        blocks=1,
        layers=1,
    )
    out = model(torch.randn(2, 12, 5, 1))
    assert out.shape == (2, 3, 5)
