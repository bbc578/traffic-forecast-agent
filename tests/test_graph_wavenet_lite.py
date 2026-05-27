from __future__ import annotations

import torch

from traffic_agent.models.graph_wavenet_lite import GraphWaveNetLite


def test_graph_wavenet_lite_shape() -> None:
    model = GraphWaveNetLite(5, 2, 6, torch.eye(5), hidden_size=8, num_layers=2)
    out = model(torch.randn(4, 12, 5, 2))
    assert out.shape == (4, 6, 5)
