from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_hdf_traffic(path: str | Path) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Read a METR-LA/PEMS-BAY style HDF5 traffic file."""
    try:
        try:
            frame = pd.read_hdf(path)
        except ValueError:
            frame = pd.read_hdf(path, key="df")
    except ImportError as exc:
        raise ImportError(
            "Reading .h5 traffic files requires PyTables. Install dependencies with "
            "`pip install -r requirements.txt` or `pip install tables`."
        ) from exc
    except (ValueError, OSError) as exc:
        raise ValueError(f"Could not read HDF5 traffic file {path}: {exc}") from exc

    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame in {path}, got {type(frame).__name__}.")
    timestamps = [str(item) for item in frame.index.to_list()]
    node_ids = [str(col) for col in frame.columns.to_list()]
    values = frame.to_numpy(dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected traffic frame shape [T, N], got {values.shape}.")
    data = values[:, :, None]
    return data, ["speed"], node_ids, timestamps


def read_dcrnn_adjacency(path: str | Path) -> tuple[list[str], dict[str, int], np.ndarray]:
    """Read DCRNN `adj_mx.pkl` files."""
    with Path(path).open("rb") as f:
        payload: Any = pickle.load(f, encoding="latin1")
    if not isinstance(payload, tuple) or len(payload) < 3:
        raise ValueError("Expected DCRNN adj_mx.pkl tuple: (sensor_ids, sensor_id_to_ind, adj_mx).")
    sensor_ids, sensor_id_to_ind, adjacency = payload[:3]
    return [str(item) for item in sensor_ids], dict(sensor_id_to_ind), np.asarray(adjacency, dtype=np.float32)


def build_correlation_adjacency(data: np.ndarray, train_ratio: float = 0.7, threshold: float = 0.3) -> np.ndarray:
    """Build a correlation graph from the train time segment only to avoid leakage."""
    train_end = max(2, int(data.shape[0] * train_ratio))
    speed = data[:train_end, :, 0]
    corr = np.corrcoef(np.nan_to_num(speed, nan=np.nanmean(speed)), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0).astype(np.float32)
    adjacency = np.where(np.abs(corr) >= threshold, np.abs(corr), 0.0).astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def prepare_real_dataset(
    dataset: str,
    traffic_file: str | Path,
    output: str | Path,
    adj_file: str | Path | None = None,
    build_correlation_adj: bool = False,
    train_ratio: float = 0.7,
) -> Path:
    """Convert real traffic data to the project's unified `.npz` schema."""
    traffic_path = Path(traffic_file)
    if not traffic_path.exists():
        raise FileNotFoundError(f"Traffic file not found: {traffic_path}")
    data, feature_names, node_ids, timestamps = read_hdf_traffic(traffic_path)

    adjacency_type = "physical"
    if adj_file:
        adj_path = Path(adj_file)
        if not adj_path.exists():
            raise FileNotFoundError(f"Adjacency file not found: {adj_path}")
        adj_node_ids, _, adjacency = read_dcrnn_adjacency(adj_path)
        if len(adj_node_ids) == data.shape[1] and adj_node_ids != node_ids:
            node_ids = adj_node_ids
    elif build_correlation_adj:
        adjacency = build_correlation_adjacency(data, train_ratio=train_ratio)
        adjacency_type = "correlation"
    else:
        raise ValueError("Provide --adj-file or set --build-correlation-adj true.")

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Adjacency must be square [N, N], got {adjacency.shape}.")
    if data.shape[1] != adjacency.shape[0]:
        raise ValueError(
            f"Traffic node count ({data.shape[1]}) and adjacency size ({adjacency.shape[0]}) do not match."
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        data=data.astype(np.float32),
        adjacency=adjacency.astype(np.float32),
        feature_names=np.array(feature_names),
        node_ids=np.array(node_ids),
        timestamps=np.array(timestamps),
        dataset_name=np.array(dataset),
        adjacency_type=np.array(adjacency_type),
        is_synthetic_data=np.array(False),
    )
    missing_ratio = float(np.isnan(data).mean())
    time_range = f"{timestamps[0]} -> {timestamps[-1]}" if timestamps else "unknown"
    print(f"dataset_name={dataset}")
    print(f"data_shape={data.shape}")
    print(f"adjacency_shape={adjacency.shape}")
    print(f"time_range={time_range}")
    print(f"missing_ratio={missing_ratio:.6f}")
    print(f"adjacency_type={adjacency_type}")
    print(f"output_path={output_path}")
    return output_path


def _str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare METR-LA/PEMS-BAY style real traffic data.")
    parser.add_argument("--dataset", required=True, choices=["metr-la", "pems-bay"])
    parser.add_argument("--traffic-file", required=True)
    parser.add_argument("--adj-file", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-correlation-adj", default="false")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    args = parser.parse_args()
    prepare_real_dataset(
        dataset=args.dataset,
        traffic_file=args.traffic_file,
        adj_file=args.adj_file,
        output=args.output,
        build_correlation_adj=_str_to_bool(args.build_correlation_adj),
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
