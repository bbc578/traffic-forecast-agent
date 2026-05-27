from __future__ import annotations

import torch
from torch import nn


class HistoricalAverageBaseline(nn.Module):
    """Predict future speed by repeating the historical average speed."""

    def __init__(self, horizon: int = 3, target_feature_index: int = 0) -> None:
        super().__init__()
        self.horizon = horizon
        self.target_feature_index = target_feature_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        speed = x[..., self.target_feature_index]
        mean_speed = speed.mean(dim=1)
        return mean_speed.unsqueeze(1).repeat(1, self.horizon, 1)
