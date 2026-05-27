from __future__ import annotations

import torch
from torch import nn

from traffic_agent.config import ProjectConfig
from traffic_agent.models.baseline import HistoricalAverageBaseline
from traffic_agent.models.graph_wavenet_lite import GraphWaveNetLite
from traffic_agent.models.gru import GRUForecaster
from traffic_agent.models.last_value import LastValueBaseline
from traffic_agent.models.lstm import LSTMForecaster
from traffic_agent.models.seasonal_naive import SeasonalNaiveBaseline
from traffic_agent.models.stgcn_block import STGCNImproved
from traffic_agent.models.stgcn_lite import STGCNLite

SUPPORTED_MODELS = {
    "historical_average",
    "last_value",
    "seasonal_naive",
    "lstm",
    "gru",
    "stgcn_lite",
    "stgcn_improved",
    "graph_wavenet_lite",
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
    raise ValueError(f"Unsupported model: {model_name}. Choose from {sorted(SUPPORTED_MODELS)}.")
