from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from traffic_agent.data.loader import load_traffic_npz


def generate_data_card(data: str | Path, output: str | Path) -> Path:
    """Generate a Markdown data card for a prepared `.npz` dataset."""
    values, adjacency, metadata = load_traffic_npz(data)
    timestamps = metadata.get("timestamps", [])
    time_range = f"{timestamps[0]} -> {timestamps[-1]}" if timestamps else "unknown"
    missing_ratio = float(np.isnan(values).mean())
    text = f"""# Data Card: {metadata.get("dataset_name")}

## Dataset
- Synthetic: {metadata.get("is_synthetic_data")}
- Path: {metadata.get("path")}
- Time range: {time_range}

## Shape
- Data: {values.shape}
- Adjacency: {adjacency.shape}
- Nodes: {values.shape[1]}
- Features: {values.shape[2]}
- Feature names: {metadata.get("feature_names")}

## Quality
- Missing ratio: {missing_ratio:.6f}
- Adjacency type: {metadata.get("adjacency_type")}

## Preprocessing
Training uses time-ordered splits, forward/backward fill, train-only scaler fit, and no random time shuffle.

## Usage Limits
Synthetic data is for workflow demonstration only. Real data provenance and license must be checked by the user.
"""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a data card.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(generate_data_card(args.data, args.output))


if __name__ == "__main__":
    main()
