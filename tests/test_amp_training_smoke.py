from __future__ import annotations

import pytest
import torch

from traffic_agent.models.graph_wavenet_full import GraphWaveNetFull


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_amp_training_smoke_cuda() -> None:
    model = GraphWaveNetFull(5, 1, 3, torch.eye(5, device="cuda"), blocks=1, layers=1).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(2, 12, 5, 1, device="cuda")
    y = torch.randn(2, 3, 5, device="cuda")
    with torch.amp.autocast("cuda"):
        loss = torch.nn.functional.l1_loss(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    assert torch.isfinite(loss)
