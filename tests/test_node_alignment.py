from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_agent.data.loader import load_traffic_npz
from traffic_agent.data.prepare_real_dataset import prepare_real_dataset


def test_prepare_real_dataset_reorders_h5_columns_to_adjacency_order(tmp_path: Path) -> None:
    traffic_path = tmp_path / "traffic.h5"
    adj_path = tmp_path / "adj.pkl"
    output = tmp_path / "prepared.npz"
    frame = pd.DataFrame(
        {
            "C": [30.0, 31.0],
            "A": [10.0, 11.0],
            "B": [20.0, 21.0],
        }
    )
    frame.to_hdf(traffic_path, key="df")
    adjacency = np.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 0.0, 4.0],
            [5.0, 6.0, 0.0],
        ],
        dtype=np.float32,
    )
    with adj_path.open("wb") as f:
        pickle.dump((["A", "B", "C"], {"A": 0, "B": 1, "C": 2}, adjacency), f)

    prepare_real_dataset("metr-la", traffic_path, output, adj_file=adj_path)
    data, aligned_adj, metadata = load_traffic_npz(output)
    assert metadata["node_ids"] == ["A", "B", "C"]
    assert data[0, :, 0].tolist() == [10.0, 20.0, 30.0]
    assert np.allclose(aligned_adj, adjacency)
    report = json.loads((tmp_path / "node_alignment_report.json").read_text(encoding="utf-8"))
    assert report["aligned_node_count"] == 3
    assert report["alignment_strategy"] == "intersection_reordered_to_adjacency_sensor_ids"
