from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from traffic_agent.training.train import train_model

ADAPTIVE_MODELS = {"graph_wavenet_lite", "graph_wavenet_full"}
GRAPH_MODE_MAP = {
    "identity": "identity",
    "physical": "physical_graph",
    "correlation": "correlation_graph",
    "adaptive": "adaptive_graph",
}


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No ablation runs completed."
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.to_dict(orient="records"):
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def run_ablation(
    config: str,
    output: str,
    models: list[str],
    horizons: list[int],
    seeds: list[int],
    graph_types: list[str],
) -> Path:
    """Run identity/physical/correlation/adaptive graph ablations."""
    rows: list[dict[str, Any]] = []
    for model_name in models:
        for graph_type in graph_types:
            if graph_type == "adaptive" and model_name not in ADAPTIVE_MODELS:
                rows.append({"status": "skipped", "model_name": model_name, "graph_type": graph_type})
                continue
            for horizon in horizons:
                for seed in seeds:
                    try:
                        run_dir = train_model(
                            config,
                            model_name,
                            horizon=horizon,
                            seed=seed,
                            run_suffix=f"{graph_type}_h{horizon}_s{seed}",
                            adjacency_mode=GRAPH_MODE_MAP[graph_type],
                        )
                        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                        rows.append(
                            {
                                "status": "success",
                                "run_name": run_dir.name,
                                "model_name": model_name,
                                "graph_type": graph_type,
                                "horizon": horizon,
                                "seed": seed,
                                "mae": metrics["mae"],
                                "rmse": metrics["rmse"],
                                "masked_mape": metrics["masked_mape"],
                                "is_synthetic_data": metrics["is_synthetic_data"],
                            }
                        )
                    except Exception as exc:
                        rows.append(
                            {
                                "status": "failed",
                                "model_name": model_name,
                                "graph_type": graph_type,
                                "horizon": horizon,
                                "seed": seed,
                                "error": str(exc),
                            }
                        )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    output_path.with_suffix(".md").write_text(
        "# Graph Ablation Summary\n\n"
        "Compare identity, physical, correlation, and adaptive graph settings. "
        "Do not claim graph utility unless results beat identity consistently.\n\n"
        f"{_markdown_table(frame)}\n",
        encoding="utf-8",
    )
    success = frame[frame["status"] == "success"].copy()
    if not success.empty:
        agg = (
            success.groupby(["model_name", "graph_type", "horizon"], dropna=False)
            .agg(
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                masked_mape_mean=("masked_mape", "mean"),
                masked_mape_std=("masked_mape", "std"),
            )
            .reset_index()
        )
    else:
        agg = pd.DataFrame()
    agg.to_csv(output_path.with_name(f"{output_path.stem}_agg.csv"), index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run graph structure ablations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--graph-types", nargs="+", choices=sorted(GRAPH_MODE_MAP), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_ablation(args.config, args.output, args.models, args.horizons, args.seeds, args.graph_types)


if __name__ == "__main__":
    main()
