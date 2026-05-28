from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate_model_card(run_dir: str | Path, output: str | Path) -> Path:
    """Generate a run-level model card."""
    run_path = Path(run_dir)
    metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    text = f"""# Model Card: {metrics.get("model_name")}

## Inputs and Outputs
- Input: `[batch, input_steps, num_nodes, num_features]`
- Output: `[batch, horizon, num_nodes]` predicted speed

## Graph Structure
- Adjacency type: {metrics.get("adjacency_type")}

## Parameters
- Trainable parameters: {metrics.get("parameter_count")}

## Training Configuration
- Horizon: {metrics.get("horizon")}
- Seed: {metrics.get("seed")}
- Dataset: {metrics.get("dataset_name")}
- Synthetic: {metrics.get("is_synthetic_data")}

## Metrics
- MAE: {metrics.get("mae")}
- RMSE: {metrics.get("rmse")}
- MAPE: {metrics.get("mape")}
- Masked MAE: {metrics.get("masked_mae")}
- Masked RMSE: {metrics.get("masked_rmse")}
- Masked MAPE: {metrics.get("masked_mape")}

## Intended Use
Offline learning, model comparison, and portfolio demonstration.

## Out-of-Scope Use
Real traffic control, signal timing decisions, or claims of city-wide operational readiness.

## Known Limitations
Metrics are split-specific and must not be generalized beyond the prepared dataset and protocol.
"""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a model card.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(generate_model_card(args.run_dir, args.output))


if __name__ == "__main__":
    main()
