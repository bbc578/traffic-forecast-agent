from __future__ import annotations

import torch
from torch import nn


class LastValueBaseline(nn.Module):
    """Repeat the latest observed speed as the future forecast."""

    def __init__(self, horizon: int = 3, target_feature_index: int = 0) -> None:
        super().__init__()
        self.horizon = horizon
        self.target_feature_index = target_feature_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last_speed = x[:, -1, :, self.target_feature_index]
        return last_speed.unsqueeze(1).repeat(1, self.horizon, 1)
