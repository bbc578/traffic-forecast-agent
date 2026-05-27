from __future__ import annotations

import torch
from torch import nn


class GRUForecaster(nn.Module):
    """Temporal-only GRU baseline without explicit graph structure."""

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        horizon: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=num_nodes * num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Linear(hidden_size, horizon * num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        flattened = x.reshape(batch_size, x.shape[1], -1)
        output, _ = self.gru(flattened)
        return self.head(output[:, -1]).reshape(batch_size, self.horizon, self.num_nodes)
