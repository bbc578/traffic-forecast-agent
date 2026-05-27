from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_agent.data.loader import load_traffic_npz
from traffic_agent.data.prepare_real_dataset import build_correlation_adjacency, prepare_real_dataset


def test_prepare_real_dataset_from_hdf_and_dcrnn_adj(tmp_path: Path) -> None:
    traffic_path = tmp_path / "traffic.h5"
    adj_path = tmp_path / "adj_mx.pkl"
    output = tmp_path / "prepared.npz"
    frame = pd.DataFrame(
        np.arange(20, dtype=np.float32).reshape(10, 2),
        index=pd.date_range("2024-01-01", periods=10, freq="5min"),
        columns=["s1", "s2"],
    )
    frame.to_hdf(traffic_path, key="df")
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    with adj_path.open("wb") as f:
        pickle.dump((["s1", "s2"], {"s1": 0, "s2": 1}, adjacency), f)

    prepare_real_dataset("metr-la", traffic_path, output, adj_file=adj_path)
    data, loaded_adj, metadata = load_traffic_npz(output)
    assert data.shape == (10, 2, 1)
    assert loaded_adj.shape == (2, 2)
    assert metadata["feature_names"] == ["speed"]
    assert metadata["is_synthetic_data"] is False
    assert metadata["dataset_name"] == "metr-la"


def test_correlation_adjacency_uses_train_segment_shape() -> None:
    data = np.random.default_rng(0).normal(size=(50, 4, 1)).astype(np.float32)
    adjacency = build_correlation_adjacency(data, train_ratio=0.6)
    assert adjacency.shape == (4, 4)
    assert np.allclose(np.diag(adjacency), 0.0)
