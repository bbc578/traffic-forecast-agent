from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.express as px

from traffic_agent.agent.tools import compare_runs, get_top_congestion_nodes
from traffic_agent.analysis.error_analysis import error_by_horizon, error_by_time_of_day


def export_figures(run_dir: str, output_dir: str) -> Path:
    """Export common portfolio figures. Requires kaleido for PNG writing."""
    run_path = Path(run_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(run_path / "predictions.npz", allow_pickle=False) as loaded:
        y_true = loaded["y_true"]
        y_pred = loaded["y_pred"]
        timestamps = [str(item) for item in loaded["timestamps"].tolist()] if "timestamps" in loaded else None

    px.line(y=[y_true[:100, 0, 0], y_pred[:100, 0, 0]]).write_image(out / "prediction_curve_node0.png")
    px.bar(error_by_horizon(y_true, y_pred), x="horizon_step", y="mae").write_image(out / "error_by_horizon.png")
    px.bar(error_by_time_of_day(y_true, y_pred, timestamps), x="hour", y="mae").write_image(
        out / "error_by_time_of_day.png"
    )
    px.bar(get_top_congestion_nodes(str(run_path), k=10), x="node_id", y="risk_score").write_image(
        out / "congestion_topk.png"
    )
    comparison = compare_runs(str(run_path.parent))
    if comparison:
        px.bar(comparison, x="model_name", y="mae", color="horizon").write_image(out / "model_comparison.png")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dashboard figures for a run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(export_figures(args.run_dir, args.output_dir))


if __name__ == "__main__":
    main()
