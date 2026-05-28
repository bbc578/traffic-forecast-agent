from __future__ import annotations

import torch

from traffic_agent.config import ProjectConfig
from traffic_agent.models.model_factory import SUPPORTED_MODELS, build_model


def test_model_factory_builds_full_models() -> None:
    config = ProjectConfig.model_validate(
        {
            "data": {"path": "demo", "horizon": 3},
            "training": {},
            "model": {"hidden_size": 8, "graph_wavenet_full": {"blocks": 1, "layers": 1, "skip_channels": 16, "end_channels": 32}},
            "analysis": {},
            "outputs": {},
        }
    )
    assert "graph_wavenet_full" in SUPPORTED_MODELS
    assert "stgcn_full" in SUPPORTED_MODELS
    for name in ["graph_wavenet_full", "stgcn_full"]:
        model = build_model(name, config, 5, 1, torch.eye(5), 0)
        assert model(torch.randn(2, 12, 5, 1)).shape == (2, 3, 5)
