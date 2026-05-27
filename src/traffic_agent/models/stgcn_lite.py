from __future__ import annotations

import torch
from torch import nn


class STGCNLite(nn.Module):
    """STGCN-inspired demo model with adjacency message passing and GRU."""

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        horizon: int,
        adjacency: torch.Tensor,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        adj = adjacency.float()
        adj = adj + torch.eye(adj.shape[0], dtype=adj.dtype, device=adj.device)
        degree = adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("normalized_adjacency", adj / degree)
        self.node_projection = nn.Linear(num_features, hidden_size)
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=num_nodes * hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Linear(hidden_size, horizon * num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, input_steps, _, _ = x.shape
        messages = torch.einsum("ij,btjf->btif", self.normalized_adjacency, x)
        hidden_nodes = torch.relu(self.node_projection(messages))
        temporal_input = hidden_nodes.reshape(batch_size, input_steps, -1)
        output, _ = self.gru(temporal_input)
        last = output[:, -1, :]
        predictions = self.head(last)
        return predictions.reshape(batch_size, self.horizon, self.num_nodes)
