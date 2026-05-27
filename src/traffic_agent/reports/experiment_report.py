from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_metrics(runs_dir: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8")) | {"run_name": path.parent.name}
        for path in sorted(Path(runs_dir).glob("*/metrics.json"))
    ]


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No completed runs found."
    columns = ["run_name", "model_name", "dataset_name", "is_synthetic_data", "horizon", "mae", "rmse", "mape"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def generate_experiment_report(runs_dir: str | Path, output: str | Path) -> Path:
    """Generate a Markdown experiment report from saved metrics."""
    rows = _load_metrics(runs_dir)
    synthetic = all(bool(row.get("is_synthetic_data")) for row in rows) if rows else True
    text = "\n".join(
        [
            "# Traffic Forecasting Experiment Report",
            "",
            "## Purpose",
            "Evaluate traffic forecasting baselines and graph-aware models under a time-ordered split.",
            "",
            "## Dataset",
            "Synthetic smoke-test data only." if synthetic else "At least one run is marked as real-data based.",
            "",
            "## Task Definition",
            "Predict future node-level traffic speed from a fixed historical window.",
            "",
            "## Metrics Table",
            _table(rows),
            "",
            "## Baseline Comparison",
            "LastValue must be treated as a strong short-term baseline. "
            "If deep models do not beat it, do not claim model superiority.",
            "",
            "## Ablation and Error Diagnostics",
            "Use `run_ablation.py` and Dashboard Error Diagnostics to inspect horizon, node, and time-of-day failures.",
            "",
            "## Agent and Dashboard",
            "The agent reads saved artifacts through registered tools and stores traces. "
            "It is not a traffic-control agent.",
            "",
            "## Limitations",
            "- No SOTA or industrial deployment claim is made.",
            "- Synthetic results are workflow validation only.",
            "- Explanations are heuristic, not causal evidence.",
            "",
            "## Next Steps",
            "Prepare METR-LA/PEMS-BAY, rerun experiments, and fill real metrics only after "
            "code-generated results exist.",
        ]
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment report markdown.")
    parser.add_argument("--runs-dir", default="outputs")
    parser.add_argument("--output", default="experiments/reports/experiment_report.md")
    args = parser.parse_args()
    print(generate_experiment_report(args.runs_dir, args.output))


if __name__ == "__main__":
    main()
