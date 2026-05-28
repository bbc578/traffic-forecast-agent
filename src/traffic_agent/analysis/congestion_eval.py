from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traffic_agent.training.metrics import masked_mape, mae, rmse


def _subset_masks(y_true: np.ndarray, history_mean: np.ndarray | None) -> dict[str, np.ndarray]:
    low_speed = y_true < 35
    normal_flow = (y_true >= 45) & (np.var(y_true, axis=1, keepdims=True) < 4.0)
    high_variance = np.var(y_true, axis=1, keepdims=True) >= 16.0
    high_variance = np.repeat(high_variance, y_true.shape[1], axis=1)
    if history_mean is not None:
        drop = (history_mean[:, None, :] - y_true) / np.maximum(np.abs(history_mean[:, None, :]), 1.0)
        speed_drop = drop > 0.25
    else:
        speed_drop = np.zeros_like(y_true, dtype=bool)
    return {
        "low_speed": low_speed,
        "speed_drop": speed_drop,
        "high_variance": high_variance,
        "normal_flow": normal_flow,
    }


def evaluate_run_subsets(run_dir: str | Path) -> list[dict[str, Any]]:
    run_path = Path(run_dir)
    metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    with np.load(run_path / "predictions.npz", allow_pickle=False) as loaded:
        y_true = loaded["y_true"]
        y_pred = loaded["y_pred"]
        history_mean = loaded["history_mean"] if "history_mean" in loaded else None
    rows = []
    for subset, mask in _subset_masks(y_true, history_mean).items():
        count = int(mask.sum())
        if count == 0:
            rows.append({"subset": subset, "sample_count": 0, "model_name": metrics.get("model_name")})
            continue
        rows.append(
            {
                "run_name": run_path.name,
                "model_name": metrics.get("model_name"),
                "dataset_name": metrics.get("dataset_name"),
                "horizon": metrics.get("horizon"),
                "subset": subset,
                "sample_count": count,
                "mae": mae(y_true[mask], y_pred[mask]),
                "rmse": rmse(y_true[mask], y_pred[mask]),
                "masked_mape": masked_mape(y_true[mask], y_pred[mask]),
            }
        )
    return rows


def run_congestion_subset_eval(runs_dir: str | Path, output: str | Path) -> Path:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(Path(runs_dir).glob("*/metrics.json")):
        pred_path = metrics_path.parent / "predictions.npz"
        if pred_path.exists():
            rows.extend(evaluate_run_subsets(metrics_path.parent))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    table = frame.to_csv(index=False) if not frame.empty else "No eligible runs found."
    output_path.with_suffix(".md").write_text(
        "# Congestion Subset Evaluation\n\n"
        "Subsets are derived from true future speeds and recent history. "
        "Report whether graph models improve low_speed or speed_drop subsets; do not assume they do.\n\n"
        "```csv\n" + table + "\n```",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on congestion-related subsets.")
    parser.add_argument("--runs-dir", default="outputs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(run_congestion_subset_eval(args.runs_dir, args.output))


if __name__ == "__main__":
    main()
