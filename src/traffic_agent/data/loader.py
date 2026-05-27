from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _decode_array(values: np.ndarray) -> list[str]:
    return [str(item) for item in values.tolist()]


def load_traffic_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a traffic `.npz` file and validate basic schema constraints."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Traffic data file not found: {data_path}. "
            "Generate demo data or provide a local .npz file with the expected schema."
        )

    with np.load(data_path, allow_pickle=False) as loaded:
        required = {"data", "adjacency", "feature_names", "node_ids"}
        missing = required.difference(loaded.files)
        if missing:
            raise ValueError(f"Traffic data file is missing required fields: {sorted(missing)}")

        data = loaded["data"].astype(np.float32)
        adjacency = loaded["adjacency"].astype(np.float32)
        feature_names = _decode_array(loaded["feature_names"])
        node_ids = _decode_array(loaded["node_ids"])
        is_synthetic = bool(loaded["is_synthetic_data"]) if "is_synthetic_data" in loaded else False
        timestamps = _decode_array(loaded["timestamps"]) if "timestamps" in loaded else []
        dataset_name = str(loaded["dataset_name"]) if "dataset_name" in loaded else data_path.stem
        adjacency_type = str(loaded["adjacency_type"]) if "adjacency_type" in loaded else "physical"

    if data.ndim != 3:
        raise ValueError(f"`data` must be 3-dimensional [T, N, F], got shape {data.shape}.")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"`adjacency` must be a 2D square matrix [N, N], got shape {adjacency.shape}."
        )
    if data.shape[1] != adjacency.shape[0]:
        raise ValueError(
            "`data` node dimension and `adjacency` size must match, "
            f"got data N={data.shape[1]} and adjacency N={adjacency.shape[0]}."
        )
    if len(feature_names) != data.shape[2]:
        raise ValueError(
            f"`feature_names` length must match feature dimension, got {len(feature_names)} "
            f"names for F={data.shape[2]}."
        )
    if len(node_ids) != data.shape[1]:
        raise ValueError(
            f"`node_ids` length must match node dimension, got {len(node_ids)} IDs "
            f"for N={data.shape[1]}."
        )

    metadata: dict[str, Any] = {
        "feature_names": feature_names,
        "node_ids": node_ids,
        "is_synthetic_data": is_synthetic,
        "timestamps": timestamps,
        "dataset_name": dataset_name,
        "adjacency_type": adjacency_type,
        "path": str(data_path),
    }
    return data, adjacency, metadata
