from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from traffic_agent.config import load_config
from traffic_agent.training.train import train_model


def model_group(model_name: str) -> str:
    if model_name in {"last_value", "historical_average", "seasonal_naive"}:
        return "naive_baseline"
    if model_name in {"lstm", "gru"}:
        return "temporal_only"
    if model_name in {"stgcn_lite", "stgcn_improved", "graph_wavenet_lite"}:
        return "lite_graph"
    if model_name in {"stgcn_full", "graph_wavenet_full"}:
        return "paper_inspired_full_graph"
    return "other"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No runs completed."
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.to_dict(orient="records"):
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _row_from_metrics(run_name: str, metrics: dict[str, Any], status: str = "success") -> dict[str, Any]:
    return {
        "status": status,
        "run_name": run_name,
        "dataset_name": metrics.get("dataset_name"),
        "model_name": metrics.get("model_name"),
        "model_group": model_group(str(metrics.get("model_name"))),
        "horizon": metrics.get("horizon"),
        "seed": metrics.get("seed"),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "mape": metrics.get("mape"),
        "masked_mae": metrics.get("masked_mae"),
        "masked_rmse": metrics.get("masked_rmse"),
        "masked_mape": metrics.get("masked_mape"),
        "best_epoch": metrics.get("best_epoch"),
        "train_seconds": metrics.get("train_seconds"),
        "parameter_count": metrics.get("parameter_count"),
        "is_synthetic_data": metrics.get("is_synthetic_data"),
    }


def run_experiments(
    config_path: str,
    models: list[str],
    horizons: list[int],
    seeds: list[int],
    output: str,
) -> Path:
    """Run model/horizon/seed loops and save raw plus aggregate summaries."""
    rows: list[dict[str, Any]] = []
    config = load_config(config_path)
    base_run_name = config.outputs.run_name
    for model_name in models:
        for horizon in horizons:
            for seed in seeds:
                run_suffix = f"h{horizon}_s{seed}"
                try:
                    run_dir = train_model(
                        config_path,
                        model_name,
                        horizon=horizon,
                        seed=seed,
                        run_suffix=run_suffix,
                    )
                    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                    rows.append(_row_from_metrics(run_dir.name, metrics))
                except Exception as exc:
                    rows.append(
                        {
                            "status": "failed",
                            "run_name": f"{base_run_name}_{model_name}_{run_suffix}",
                            "dataset_name": Path(config.data.path).stem,
                            "model_name": model_name,
                            "model_group": model_group(model_name),
                            "horizon": horizon,
                            "seed": seed,
                            "error": str(exc),
                        }
                    )
                    print(f"WARNING: experiment failed for {model_name} h={horizon} seed={seed}: {exc}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    output_path.with_suffix(".md").write_text(f"# Experiment Summary\n\n{_markdown_table(frame)}\n", encoding="utf-8")

    success = frame[frame["status"] == "success"].copy()
    if not success.empty:
        agg = (
            success.groupby(["dataset_name", "model_name", "model_group", "horizon"], dropna=False)
            .agg(
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                masked_mape_mean=("masked_mape", "mean"),
                masked_mape_std=("masked_mape", "std"),
                train_seconds_mean=("train_seconds", "mean"),
                parameter_count=("parameter_count", "max"),
            )
            .reset_index()
        )
    else:
        agg = pd.DataFrame()
    agg_path = output_path.with_name(f"{output_path.stem}_agg.csv")
    agg.to_csv(agg_path, index=False)
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
