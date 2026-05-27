from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_adjacency(num_nodes: int) -> np.ndarray:
    """Create a simple weighted road-like adjacency matrix."""
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for offset, weight in ((1, 1.0), (2, 0.45)):
            j = (i + offset) % num_nodes
            adjacency[i, j] = weight
            adjacency[j, i] = weight
    return adjacency


def generate_synthetic_traffic(
    timesteps: int = 7 * 24 * 12,
    num_nodes: int = 20,
    seed: int = 42,
    missing_rate: float = 0.01,
    event_count: int = 8,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Generate synthetic traffic speed and flow data for workflow demos."""
    rng = np.random.default_rng(seed)
    adjacency = build_adjacency(num_nodes)
    steps_per_day = 24 * 12
    t = np.arange(timesteps)
    hour = (t % steps_per_day) / 12.0

    morning_peak = np.exp(-0.5 * ((hour - 8.0) / 1.3) ** 2)
    evening_peak = np.exp(-0.5 * ((hour - 18.0) / 1.6) ** 2)
    peak_pressure = morning_peak + 1.15 * evening_peak

    node_bias = rng.normal(0.0, 3.0, size=num_nodes)
    base_speed = 62.0 + node_bias
    speed = np.zeros((timesteps, num_nodes), dtype=np.float32)
    flow = np.zeros((timesteps, num_nodes), dtype=np.float32)

    shared_noise = rng.normal(0.0, 1.8, size=(timesteps, 1))
    node_noise = rng.normal(0.0, 2.2, size=(timesteps, num_nodes))
    speed[:, :] = base_speed[None, :] - 19.0 * peak_pressure[:, None] + shared_noise + node_noise

    normalized_adj = adjacency / np.maximum(adjacency.sum(axis=1, keepdims=True), 1e-6)
    for _ in range(2):
        speed = 0.78 * speed + 0.22 * (speed @ normalized_adj.T)

    for _ in range(event_count):
        start = int(rng.integers(0, max(1, timesteps - 18)))
        duration = int(rng.integers(6, 18))
        center = int(rng.integers(0, num_nodes))
        affected = [center]
        affected.extend(np.flatnonzero(adjacency[center] > 0).tolist()[:2])
        drop = float(rng.uniform(14.0, 28.0))
        speed[start : start + duration, affected] -= drop

    speed = np.clip(speed, 5.0, 85.0)
    flow_noise = rng.normal(0.0, 30.0, size=(timesteps, num_nodes))
    flow[:, :] = 520.0 + 18.0 * (70.0 - speed) + 180.0 * peak_pressure[:, None] + flow_noise
    flow = np.clip(flow, 40.0, 1800.0)

    data = np.stack([speed, flow], axis=-1).astype(np.float32)
    missing_mask = rng.random(size=data.shape) < missing_rate
    data[missing_mask] = np.nan

    feature_names = ["speed", "flow"]
    node_ids = [f"sensor_{i:03d}" for i in range(num_nodes)]
    return data, adjacency, feature_names, node_ids


def save_synthetic_dataset(output: str | Path, seed: int = 42) -> None:
    """Generate and save the default synthetic traffic dataset."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data, adjacency, feature_names, node_ids = generate_synthetic_traffic(seed=seed)
    np.savez_compressed(
        output_path,
        data=data,
        adjacency=adjacency,
        feature_names=np.array(feature_names),
        node_ids=np.array(node_ids),
        is_synthetic_data=np.array(True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic demo traffic data.")
    parser.add_argument("--output", default="data/sample/synthetic_traffic.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    save_synthetic_dataset(args.output, seed=args.seed)
    print(
        "Saved synthetic demo data to "
        f"{args.output}. This data is for workflow demonstration only."
    )


if __name__ == "__main__":
    main()
