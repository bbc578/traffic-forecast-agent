from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from traffic_agent.training.train import train_model


ABLATION_PLAN = {
    "no_graph": ("stgcn_improved", "no_graph"),
    "physical_graph": ("stgcn_improved", "physical_graph"),
    "correlation_graph": ("stgcn_improved", "correlation_graph"),
    "adaptive_graph": ("graph_wavenet_lite", "adaptive_graph"),
    "temporal_only_gru": ("gru", "temporal_only"),
    "spatial_temporal": ("graph_wavenet_lite", "physical_graph"),
}


def run_ablation(config: str, output: str, variants: list[str] | None = None) -> Path:
    """Run graph/temporal ablations and summarize metrics."""
    selected = variants or list(ABLATION_PLAN)
    rows = []
    for variant in selected:
        model_name, adjacency_mode = ABLATION_PLAN[variant]
        run_dir = train_model(config, model_name, run_suffix=variant, adjacency_mode=adjacency_mode)
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "variant": variant,
                "run_name": run_dir.name,
                "model_name": model_name,
                "adjacency_mode": adjacency_mode,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "is_synthetic_data": metrics["is_synthetic_data"],
            }
        )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run graph and temporal ablations.")
    parser.add_argument("--config", default="configs/ablation.yaml")
    parser.add_argument("--output", default="experiments/results/ablation_summary.csv")
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()
    run_ablation(args.config, args.output, args.variants)


if __name__ == "__main__":
    main()
