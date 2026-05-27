from __future__ import annotations

import torch
from torch import nn


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    adj = adjacency.float()
    adj = adj + torch.eye(adj.shape[0], dtype=adj.dtype, device=adj.device)
    degree = adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return adj / degree


class STGCNBlock(nn.Module):
    """Educational temporal-gated + graph-message-passing block."""

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.filter_conv = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0))
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0))
        self.graph_projection = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = x
        temporal = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        temporal = temporal.permute(0, 2, 3, 1)
        graph = torch.einsum("ij,btjc->btic", adjacency, temporal)
        graph = self.graph_projection(graph)
        graph = self.norm(graph)
        graph = self.dropout(graph).permute(0, 3, 1, 2)
        return torch.relu(graph + residual)


class STGCNImproved(nn.Module):
    """STGCN-inspired model with gated temporal convolution, graph mixing, residuals, and norm."""

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        horizon: int,
        adjacency: torch.Tensor,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.register_buffer("adjacency", normalize_adjacency(adjacency))
        self.input_projection = nn.Linear(num_features, hidden_size)
        self.blocks = nn.ModuleList([STGCNBlock(hidden_size, dropout) for _ in range(max(1, num_layers))])
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x).permute(0, 3, 1, 2)
        for block in self.blocks:
            hidden = block(hidden, self.adjacency)
        pooled = hidden.mean(dim=2).permute(0, 2, 1)
        return self.head(pooled).permute(0, 2, 1)
