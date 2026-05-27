from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset for sliding-window traffic forecasting."""

    def __init__(
        self,
        data: np.ndarray,
        input_steps: int = 12,
        horizon: int = 3,
        target_feature_index: int = 0,
    ) -> None:
        if data.ndim != 3:
            raise ValueError(f"Expected data shape [T, N, F], got {data.shape}.")
        if horizon not in {3, 6, 12}:
            raise ValueError("horizon must be one of 3, 6, or 12.")
        if input_steps <= 0:
            raise ValueError("input_steps must be positive.")
        if data.shape[0] < input_steps + horizon:
            raise ValueError("Data split is too short for the requested input_steps and horizon.")

        self.data = data.astype(np.float32)
        self.input_steps = input_steps
        self.horizon = horizon
        self.target_feature_index = target_feature_index
        self.num_samples = data.shape[0] - input_steps - horizon + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Sample index out of range: {idx}")
        start = idx
        x_end = start + self.input_steps
        y_end = x_end + self.horizon
        x = self.data[start:x_end]
        y = self.data[x_end:y_end, :, self.target_feature_index]
        return torch.from_numpy(x), torch.from_numpy(y)
