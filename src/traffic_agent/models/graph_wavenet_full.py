from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def row_normalize(adjacency: torch.Tensor) -> torch.Tensor:
    adj = adjacency.float()
    return adj / adj.sum(dim=1, keepdim=True).clamp_min(1e-6)


class CausalConv2d(nn.Module):
    """Causal temporal convolution for tensors shaped [B, C, N, T]."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0, 0, 0))
        return self.conv(x)


class DiffusionGraphConv(nn.Module):
    """Multi-support diffusion graph convolution used by GraphWaveNetFull."""

    def __init__(self, channels: int, support_count: int, gcn_depth: int, dropout: float) -> None:
        super().__init__()
        self.gcn_depth = gcn_depth
        self.mlp = nn.Conv2d(channels * (support_count * gcn_depth + 1), channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def nconv(x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcnt,nm->bcmt", x, support)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        outputs = [x]
        for support in supports:
            x1 = self.nconv(x, support)
            outputs.append(x1)
            for _ in range(2, self.gcn_depth + 1):
                x1 = self.nconv(x1, support)
                outputs.append(x1)
        return self.dropout(self.mlp(torch.cat(outputs, dim=1)))


class GraphWaveNetFull(nn.Module):
    """Paper-inspired Graph WaveNet with dilated gated temporal conv and diffusion graph conv.

    This is a portfolio implementation inspired by Graph WaveNet. It includes the main structural
    ideas but is not an official reproduction of the paper's training protocol or reported results.
    Input shape is [B, T, N, F], output shape is [B, H, N].
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        horizon: int,
        adjacency: torch.Tensor,
        residual_channels: int = 64,
        dilation_channels: int = 64,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
        gcn_depth: int = 2,
        dropout: float = 0.3,
        use_adaptive_adj: bool = True,
        node_embedding_dim: int = 10,
        add_identity_support: bool = True,
        add_transpose_support: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.use_adaptive_adj = use_adaptive_adj
        base_adj = row_normalize(adjacency)
        supports = [base_adj]
        if add_transpose_support:
            supports.append(row_normalize(adjacency.T))
        if add_identity_support:
            supports.append(torch.eye(num_nodes, dtype=base_adj.dtype, device=base_adj.device))
        self.register_buffer("base_supports", torch.stack(supports))

        if use_adaptive_adj:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, node_embedding_dim) * 0.1)
            self.nodevec2 = nn.Parameter(torch.randn(node_embedding_dim, num_nodes) * 0.1)

        self.start_conv = nn.Conv2d(num_features, residual_channels, kernel_size=1)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        support_count = len(supports) + int(use_adaptive_adj)
        for block in range(blocks):
            for layer in range(layers):
                dilation = 2**layer
                self.filter_convs.append(CausalConv2d(residual_channels, dilation_channels, kernel_size, dilation))
                self.gate_convs.append(CausalConv2d(residual_channels, dilation_channels, kernel_size, dilation))
                self.residual_convs.append(nn.Conv2d(dilation_channels, residual_channels, kernel_size=1))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=1))
                self.graph_convs.append(DiffusionGraphConv(dilation_channels, support_count, gcn_depth, dropout))
                self.bn.append(nn.BatchNorm2d(residual_channels))

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=1)
        self.end_conv_2 = nn.Conv2d(end_channels, horizon, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def _supports(self) -> list[torch.Tensor]:
        supports = [support for support in self.base_supports]
        if self.use_adaptive_adj:
            adaptive = torch.softmax(torch.relu(self.nodevec1 @ self.nodevec2), dim=1)
            supports.append(adaptive)
        return supports

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 2, 1)
        x = self.start_conv(x)
        skip: torch.Tensor | None = None
        supports = self._supports()
        for filter_conv, gate_conv, residual_conv, skip_conv, graph_conv, bn in zip(
            self.filter_convs,
            self.gate_convs,
            self.residual_convs,
            self.skip_convs,
            self.graph_convs,
            self.bn,
            strict=False,
        ):
            residual = x
            gated = torch.tanh(filter_conv(x)) * torch.sigmoid(gate_conv(x))
            graph = graph_conv(gated, supports)
            skip_part = skip_conv(gated)
            skip = skip_part if skip is None else skip[..., -skip_part.size(3) :] + skip_part
            x = residual_conv(graph)
            x = x + residual[..., -x.size(3) :]
            x = bn(x)
        assert skip is not None
        out = F.relu(skip)
        out = self.dropout(F.relu(self.end_conv_1(out)))
        out = self.end_conv_2(out)
        return out[..., -1].contiguous()
