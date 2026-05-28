from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from traffic_agent.models.graph_wavenet_full import row_normalize


class TemporalGLU(nn.Module):
    """GLU temporal convolution for [B, C, N, T] tensors."""

    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv2d(channels, channels * 2, kernel_size=(1, kernel_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0, 0, 0))
        left, gate = self.conv(x).chunk(2, dim=1)
        return self.dropout(left * torch.sigmoid(gate))


class ChebGraphConv(nn.Module):
    """Chebyshev-style K-order graph convolution over normalized adjacency."""

    def __init__(self, channels: int, cheb_order: int) -> None:
        super().__init__()
        self.cheb_order = cheb_order
        self.proj = nn.Conv2d(channels * cheb_order, channels, kernel_size=1)

    @staticmethod
    def nconv(x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcnt,nm->bcmt", x, support)

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        terms = [x]
        if self.cheb_order > 1:
            terms.append(self.nconv(x, support))
        for order in range(2, self.cheb_order):
            terms.append(2 * self.nconv(terms[-1], support) - terms[-2])
        return self.proj(torch.cat(terms[: self.cheb_order], dim=1))


class STConvBlock(nn.Module):
    """TemporalConv -> GraphConv -> TemporalConv block with residual and layer norm."""

    def __init__(self, channels: int, temporal_kernel_size: int, cheb_order: int, dropout: float) -> None:
        super().__init__()
        self.temp1 = TemporalGLU(channels, temporal_kernel_size, dropout)
        self.graph = ChebGraphConv(channels, cheb_order)
        self.temp2 = TemporalGLU(channels, temporal_kernel_size, dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.temp1(x)
        x = F.relu(self.graph(x, support))
        x = self.temp2(x)
        x = x + residual[..., -x.size(3) :]
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class STGCNFull(nn.Module):
    """Paper-inspired STGCN with ST-Conv blocks and Chebyshev-style graph convolution.

    This is not an official reproduction. It keeps the core STGCN idea: temporal gated convolution,
    K-order graph convolution, residual paths, normalization, dropout, and horizon projection.
    Input shape is [B, T, N, F], output shape is [B, H, N].
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        horizon: int,
        adjacency: torch.Tensor,
        hidden_channels: int = 64,
        temporal_kernel_size: int = 3,
        cheb_order: int = 3,
        num_blocks: int = 3,
        dropout: float = 0.3,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.register_buffer("support", row_normalize(adjacency + torch.eye(num_nodes, device=adjacency.device)))
        self.input_projection = nn.Conv2d(num_features, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [STConvBlock(hidden_channels, temporal_kernel_size, cheb_order, dropout) for _ in range(num_blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden_channels) if use_layer_norm else nn.Identity()
        self.output = nn.Linear(hidden_channels, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 2, 1)
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x, self.support)
        pooled = x.mean(dim=3).permute(0, 2, 1)
        pooled = self.final_norm(pooled)
        return self.output(pooled).permute(0, 2, 1)
