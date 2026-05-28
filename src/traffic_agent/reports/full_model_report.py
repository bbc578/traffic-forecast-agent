from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _load_metrics(runs_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(runs_dir).glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append(metrics | {"run_name": path.parent.name})
    return pd.DataFrame(rows)


def _csv_block(frame: pd.DataFrame) -> str:
    return "```csv\n" + (frame.to_csv(index=False) if not frame.empty else "no rows\n") + "```"


def generate_full_model_report(runs_dir: str | Path, results_dir: str | Path, output: str | Path) -> Path:
    """Generate a truthful full-model experiment report from local artifacts."""
    metrics = _load_metrics(runs_dir)
    results_path = Path(results_dir)
    last_value_best = None
    if not metrics.empty and "last_value" in set(metrics["model_name"]):
        last_value_best = metrics[metrics["model_name"] == "last_value"].sort_values("mae").iloc[0].to_dict()
    full_rows = metrics[metrics["model_name"].isin(["stgcn_full", "graph_wavenet_full"])] if not metrics.empty else pd.DataFrame()
    lite_rows = metrics[
        metrics["model_name"].isin(["stgcn_improved", "graph_wavenet_lite", "stgcn_lite"])
    ] if not metrics.empty else pd.DataFrame()

    if last_value_best is None:
        executive = "LastValue baseline is missing, so no deep-model superiority claim can be made."
    else:
        better = metrics[metrics["mae"] < last_value_best["mae"]] if "mae" in metrics else pd.DataFrame()
        executive = (
            "At least one model beats LastValue on MAE in local artifacts."
            if not better.empty
            else "LastValue remains the strongest MAE baseline in local artifacts."
        )

    report = [
        "# METR-LA Full Model Report",
        "",
        "## Executive Summary",
        executive,
        "",
        "This report is generated from local artifacts only. It does not claim SOTA or official paper reproduction.",
        "",
        "## Dataset",
        _csv_block(metrics[["run_name", "dataset_name", "is_synthetic_data", "num_nodes", "horizon"]] if not metrics.empty else metrics),
        "",
        "## Models",
        "Naive baselines, temporal-only models, lite graph models, and paper-inspired full graph models are separated.",
        "",
        "## Results",
        _csv_block(
            metrics[
                [
                    "run_name",
                    "model_name",
                    "horizon",
                    "seed",
                    "mae",
                    "rmse",
                    "masked_mape",
                    "parameter_count",
                    "train_seconds",
                ]
            ]
            if not metrics.empty
            else metrics
        ),
        "",
        "## Baseline Discussion",
        "LastValue is a strong short-term traffic baseline because speed has strong local persistence.",
        "",
        "## Full vs Lite",
        _csv_block(pd.concat([full_rows, lite_rows], ignore_index=True) if not metrics.empty else pd.DataFrame()),
        "",
        "## Graph Ablation",
        "Ablation files found: " + ", ".join(str(path) for path in sorted(results_path.glob("*ablation*.csv"))),
        "",
        "## Congestion Subset",
        "Congestion subset files found: " + ", ".join(str(path) for path in sorted(results_path.glob("*congestion_subset*.csv"))),
        "",
        "## Error Diagnostics",
        "Use Dashboard Error Diagnostics and `error_analysis.py` outputs for horizon/node/speed-bin failure analysis.",
        "",
        "## Agent Analysis",
        "The Agent uses registered read-only tools and saves traces. It cannot control traffic systems.",
        "",
        "## Limitations",
        "- These models are paper-inspired, not official reproductions.",
        "- No SOTA claim is made.",
        "- No real traffic control use is supported.",
        "- Single-dataset conclusions do not generalize to all cities.",
        "- If full models do not beat LastValue, state that directly.",
        "",
        "## Resume-safe Bullets",
        "- If full models beat LastValue: report the exact generated metrics and protocol.",
        "- If they do not: emphasize rigorous baselines, diagnostics, and honest failure analysis.",
    ]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(report), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full model experiment report.")
    parser.add_argument("--runs-dir", default="outputs")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(generate_full_model_report(args.runs_dir, args.results_dir, args.output))


if __name__ == "__main__":
    main()
