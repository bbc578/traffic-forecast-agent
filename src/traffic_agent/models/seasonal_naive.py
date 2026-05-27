from __future__ import annotations

import torch
from torch import nn


class SeasonalNaiveBaseline(nn.Module):
    """Use same-time-of-day values when available, otherwise fall back to historical average."""

    def __init__(
        self,
        horizon: int = 3,
        target_feature_index: int = 0,
        steps_per_day: int = 288,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.target_feature_index = target_feature_index
        self.steps_per_day = steps_per_day
        self.used_fallback = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        speed = x[..., self.target_feature_index]
        if x.shape[1] > self.steps_per_day:
            seasonal = speed[:, -self.steps_per_day : -self.steps_per_day + self.horizon, :]
            if seasonal.shape[1] == self.horizon:
                self.used_fallback = False
                return seasonal
        self.used_fallback = True
        mean_speed = speed.mean(dim=1)
        return mean_speed.unsqueeze(1).repeat(1, self.horizon, 1)
