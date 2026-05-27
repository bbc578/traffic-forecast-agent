from __future__ import annotations

import torch
from torch import nn

from traffic_agent.models.stgcn_block import normalize_adjacency


class DilatedTemporalGraphBlock(nn.Module):
    """Small Graph WaveNet-inspired block with dilated temporal convolution and graph mixing."""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        padding = (dilation, 0)
        self.filter_conv = nn.Conv2d(channels, channels, kernel_size=(2, 1), dilation=(dilation, 1), padding=padding)
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=(2, 1), dilation=(dilation, 1), padding=padding)
        self.graph_projection = nn.Linear(channels, channels)
        self.skip_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        filtered = self.filter_conv(x)[..., : x.shape[2], :]
        gated = self.gate_conv(x)[..., : x.shape[2], :]
        temporal = torch.tanh(filtered) * torch.sigmoid(gated)
        graph_input = temporal.permute(0, 2, 3, 1)
        graph = torch.einsum("ij,btjc->btic", adjacency, graph_input)
        graph = self.graph_projection(graph).permute(0, 3, 1, 2)
        graph = self.dropout(graph)
        return torch.relu(graph + residual), self.skip_projection(temporal)


class GraphWaveNetLite(nn.Module):
    """Educational Graph WaveNet-inspired model with static and optional adaptive adjacency."""

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        horizon: int,
        adjacency: torch.Tensor,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        adaptive_adjacency: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.adaptive_adjacency = adaptive_adjacency
        self.register_buffer("static_adjacency", normalize_adjacency(adjacency))
        self.input_projection = nn.Linear(num_features, hidden_size)
        self.blocks = nn.ModuleList(
            [DilatedTemporalGraphBlock(hidden_size, dilation=2**i, dropout=dropout) for i in range(max(1, num_layers))]
        )
        if adaptive_adjacency:
            embed_dim = min(16, hidden_size)
            self.node_embedding_source = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
            self.node_embedding_target = nn.Parameter(torch.randn(embed_dim, num_nodes) * 0.1)
        self.output = nn.Linear(hidden_size, horizon)

    def current_adjacency(self) -> torch.Tensor:
        if not self.adaptive_adjacency:
            return self.static_adjacency
        adaptive = torch.softmax(torch.relu(self.node_embedding_source @ self.node_embedding_target), dim=1)
        return 0.5 * self.static_adjacency + 0.5 * adaptive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x).permute(0, 3, 1, 2)
        skip_total = 0.0
        adjacency = self.current_adjacency()
        for block in self.blocks:
            hidden, skip = block(hidden, adjacency)
            skip_total = skip_total + skip
        pooled = torch.relu(skip_total).mean(dim=2).permute(0, 2, 1)
        return self.output(pooled).permute(0, 2, 1)
