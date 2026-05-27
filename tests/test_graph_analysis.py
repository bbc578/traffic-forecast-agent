from __future__ import annotations

import numpy as np

from traffic_agent.analysis.graph_analysis import adjacency_stats, compare_adjacency_matrices, top_connected_nodes


def test_graph_analysis_stats() -> None:
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    stats = adjacency_stats(adjacency)
    assert stats["num_nodes"] == 3
    assert stats["num_edges"] == 4
    assert top_connected_nodes(adjacency, ["a", "b", "c"], k=1)[0]["node_id"] == "b"
    assert "physical" in compare_adjacency_matrices(physical=adjacency)
