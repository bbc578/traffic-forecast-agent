from __future__ import annotations

import torch
from torch import nn

from traffic_agent.config import ProjectConfig
from traffic_agent.models.baseline import HistoricalAverageBaseline
from traffic_agent.models.graph_wavenet_full import GraphWaveNetFull
from traffic_agent.models.graph_wavenet_lite import GraphWaveNetLite
from traffic_agent.models.gru import GRUForecaster
from traffic_agent.models.last_value import LastValueBaseline
from traffic_agent.models.lstm import LSTMForecaster
from traffic_agent.models.seasonal_naive import SeasonalNaiveBaseline
from traffic_agent.models.stgcn_block import STGCNImproved
from traffic_agent.models.stgcn_full import STGCNFull
from traffic_agent.models.stgcn_lite import STGCNLite

SUPPORTED_MODELS = {
    "historical_average",
    "last_value",
    "seasonal_naive",
    "lstm",
    "gru",
    "stgcn_lite",
    "stgcn_improved",
    "stgcn_full",
    "graph_wavenet_lite",
    "graph_wavenet_full",
}
NON_TRAINABLE_MODELS = {"historical_average", "last_value", "seasonal_naive"}


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_model(
    model_name: str,
    config: ProjectConfig,
    num_nodes: int,
    num_features: int,
    adjacency: torch.Tensor,
    target_feature_index: int,
) -> nn.Module:
    """Build a forecasting model from a stable registry of names."""
    if model_name == "historical_average":
        return HistoricalAverageBaseline(config.data.horizon, target_feature_index)
    if model_name == "last_value":
        return LastValueBaseline(config.data.horizon, target_feature_index)
    if model_name == "seasonal_naive":
        return SeasonalNaiveBaseline(config.data.horizon, target_feature_index)
    if model_name == "lstm":
        return LSTMForecaster(
            num_nodes,
            num_features,
            config.data.horizon,
            config.model.hidden_size,
            config.model.num_layers,
            config.model.dropout,
        )
    if model_name == "gru":
        return GRUForecaster(
            num_nodes,
            num_features,
            config.data.horizon,
            config.model.hidden_size,
            config.model.num_layers,
            config.model.dropout,
        )
    if model_name == "stgcn_lite":
        return STGCNLite(
            num_nodes,
            num_features,
            config.data.horizon,
            adjacency,
            config.model.hidden_size,
            config.model.num_layers,
            config.model.dropout,
        )
    if model_name == "stgcn_improved":
        return STGCNImproved(
            num_nodes,
            num_features,
            config.data.horizon,
            adjacency,
            config.model.hidden_size,
            max(1, config.model.num_layers),
            config.model.dropout,
        )
    if model_name == "graph_wavenet_lite":
        return GraphWaveNetLite(
            num_nodes,
            num_features,
            config.data.horizon,
            adjacency,
            config.model.hidden_size,
            max(1, config.model.num_layers),
            config.model.dropout,
            adaptive_adjacency=True,
        )
    if model_name == "graph_wavenet_full":
        params = {
            "residual_channels": 64,
            "dilation_channels": 64,
            "skip_channels": 256,
            "end_channels": 512,
            "kernel_size": 2,
            "blocks": 4,
            "layers": 2,
            "gcn_depth": 2,
            "dropout": config.model.dropout,
            "use_adaptive_adj": True,
            "node_embedding_dim": 10,
            "add_identity_support": True,
            "add_transpose_support": True,
        } | config.model.graph_wavenet_full
        return GraphWaveNetFull(
            num_nodes=num_nodes,
            num_features=num_features,
            horizon=config.data.horizon,
            adjacency=adjacency,
            **params,
        )
    if model_name == "stgcn_full":
        params = {
            "hidden_channels": config.model.hidden_size,
            "temporal_kernel_size": 3,
            "cheb_order": 3,
            "num_blocks": 3,
            "dropout": config.model.dropout,
            "use_layer_norm": True,
        } | config.model.stgcn_full
        return STGCNFull(
            num_nodes=num_nodes,
            num_features=num_features,
            horizon=config.data.horizon,
            adjacency=adjacency,
            **params,
        )
    raise ValueError(f"Unsupported model: {model_name}. Choose from {sorted(SUPPORTED_MODELS)}.")
