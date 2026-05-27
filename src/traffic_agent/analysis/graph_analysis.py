from __future__ import annotations

from typing import Any

import numpy as np


def adjacency_stats(adjacency: np.ndarray) -> dict[str, Any]:
    """Compute simple graph statistics for an adjacency matrix."""
    adj = np.asarray(adjacency)
    off_diag = adj.copy()
    np.fill_diagonal(off_diag, 0)
    edges = int(np.count_nonzero(off_diag))
    num_nodes = int(adj.shape[0])
    degree = np.count_nonzero(off_diag, axis=1)
    return {
        "num_nodes": num_nodes,
        "num_edges": edges,
        "density": float(edges / max(num_nodes * (num_nodes - 1), 1)),
        "average_degree": float(degree.mean()),
        "isolated_nodes": np.where(degree == 0)[0].astype(int).tolist(),
    }


def top_connected_nodes(adjacency: np.ndarray, node_ids: list[str] | None = None, k: int = 10) -> list[dict[str, Any]]:
    """Return nodes with the highest unweighted degree."""
    adj = np.asarray(adjacency)
    ids = node_ids or [f"node_{idx}" for idx in range(adj.shape[0])]
    degree = np.count_nonzero(adj, axis=1)
    order = np.argsort(-degree)[:k]
    return [{"node_id": ids[int(idx)], "degree": int(degree[idx])} for idx in order]


def correlation_lag_analysis(
    data: np.ndarray,
    adjacency: np.ndarray,
    max_lag: int = 6,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Estimate lagged neighbor correlations as an exploratory congestion propagation signal."""
    speed = np.asarray(data)[..., 0]
    rows = []
    src_nodes, dst_nodes = np.where(np.asarray(adjacency) > 0)
    for src, dst in zip(src_nodes, dst_nodes, strict=False):
        best = {"source": int(src), "target": int(dst), "lag": 0, "correlation": 0.0}
        for lag in range(1, max_lag + 1):
            a = speed[:-lag, src]
            b = speed[lag:, dst]
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            if abs(corr) > abs(best["correlation"]):
                best = {"source": int(src), "target": int(dst), "lag": lag, "correlation": corr}
        rows.append(best)
    rows.sort(key=lambda item: abs(float(item["correlation"])), reverse=True)
    return rows[:top_k]


def compare_adjacency_matrices(
    physical: np.ndarray | None = None,
    correlation: np.ndarray | None = None,
    adaptive: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare available adjacency matrices with the same summary stats."""
    result = {}
    for name, matrix in {"physical": physical, "correlation": correlation, "adaptive": adaptive}.items():
        if matrix is not None:
            result[name] = adjacency_stats(matrix)
    return result
