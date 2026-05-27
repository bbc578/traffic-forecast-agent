from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_run_summary(run_dir: str | Path) -> dict[str, object]:
    """Load metrics and prediction metadata for a completed run."""
    path = Path(run_dir)
    metrics_path = path / "metrics.json"
    predictions_path = path / "predictions.npz"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in run directory: {path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions.npz not found in run directory: {path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {"run_dir": str(path), "metrics": metrics, "predictions_path": str(predictions_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a saved run evaluation summary.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    summary = load_run_summary(args.run_dir)
    metrics = summary["metrics"]
    print(f"Run: {summary['run_dir']}")
    print(f"Model: {metrics.get('model_name')}")
    print(f"Dataset: {metrics.get('dataset_path')}")
    print(f"Horizon: {metrics.get('horizon')}")
    print(f"MAE: {metrics.get('mae')}")
    print(f"RMSE: {metrics.get('rmse')}")
    print(f"MAPE: {metrics.get('mape')}")


if __name__ == "__main__":
    main()
