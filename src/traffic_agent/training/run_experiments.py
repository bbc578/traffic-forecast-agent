from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from traffic_agent.config import load_config
from traffic_agent.training.train import train_model


def _markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No runs completed."
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def run_experiments(
    config_path: str,
    models: list[str],
    horizons: list[int],
    seeds: list[int],
    output: str,
) -> Path:
    """Run model/horizon/seed loops and save a summary CSV plus Markdown table."""
    rows = []
    base_config = load_config(config_path)
    for model_name in models:
        for horizon in horizons:
            for seed in seeds:
                run_suffix = f"h{horizon}_seed{seed}"
                run_dir = train_model(config_path, model_name, horizon=horizon, seed=seed, run_suffix=run_suffix)
                metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                rows.append(
                    {
                        "run_name": run_dir.name,
                        "model_name": metrics["model_name"],
                        "dataset_name": metrics["dataset_name"],
                        "is_synthetic_data": metrics["is_synthetic_data"],
                        "horizon": metrics["horizon"],
                        "seed": metrics["seed"],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "mape": metrics["mape"],
                        "masked_mae": metrics["masked_mae"],
                        "adjacency_type": metrics["adjacency_type"],
                    }
                )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    md_path = output_path.with_suffix(".md")
    label = "Synthetic demo experiment" if base_config.data.path.endswith("synthetic_traffic.npz") else "Experiment"
    md_path.write_text(f"# {label}\n\n{_markdown_table(rows)}\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible forecasting experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_experiments(args.config, args.models, args.horizons, args.seeds, args.output)


if __name__ == "__main__":
    main()
