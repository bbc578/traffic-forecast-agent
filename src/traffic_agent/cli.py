from __future__ import annotations

import argparse
import subprocess
import sys

from traffic_agent.app.export_figures import export_figures
from traffic_agent.data.generate_synthetic import save_synthetic_dataset
from traffic_agent.data.prepare_real_dataset import prepare_real_dataset
from traffic_agent.reports.experiment_report import generate_experiment_report
from traffic_agent.training.run_ablation import run_ablation
from traffic_agent.training.run_experiments import run_experiments
from traffic_agent.training.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="traffic-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-synthetic")
    gen.add_argument("--output", default="data/sample/synthetic_traffic.npz")
    gen.add_argument("--seed", type=int, default=42)

    prep = sub.add_parser("prepare-real")
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--traffic-file", required=True)
    prep.add_argument("--adj-file", default=None)
    prep.add_argument("--output", required=True)
    prep.add_argument("--build-correlation-adj", action="store_true")

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--model", required=True)

    exp = sub.add_parser("run-experiments")
    exp.add_argument("--config", required=True)
    exp.add_argument("--models", nargs="+", required=True)
    exp.add_argument("--horizons", nargs="+", type=int, required=True)
    exp.add_argument("--seeds", nargs="+", type=int, required=True)
    exp.add_argument("--output", required=True)

    abl = sub.add_parser("ablation")
    abl.add_argument("--config", default="configs/ablation.yaml")
    abl.add_argument("--output", default="experiments/results/ablation_summary.csv")
    abl.add_argument("--models", nargs="+", default=["stgcn_full", "graph_wavenet_full"])
    abl.add_argument("--horizons", nargs="+", type=int, default=[3])
    abl.add_argument("--seeds", nargs="+", type=int, default=[42])
    abl.add_argument("--graph-types", nargs="+", default=["identity", "physical"])

    rep = sub.add_parser("report")
    rep.add_argument("--runs-dir", default="outputs")
    rep.add_argument("--output", default="experiments/reports/experiment_report.md")

    figs = sub.add_parser("export-figures")
    figs.add_argument("--run-dir", required=True)
    figs.add_argument("--output-dir", required=True)

    sub.add_parser("app")
    sub.add_parser("api")
    args = parser.parse_args()

    if args.command == "generate-synthetic":
        save_synthetic_dataset(args.output, args.seed)
    elif args.command == "prepare-real":
        prepare_real_dataset(args.dataset, args.traffic_file, args.output, args.adj_file, args.build_correlation_adj)
    elif args.command == "train":
        train_model(args.config, args.model)
    elif args.command == "run-experiments":
        run_experiments(args.config, args.models, args.horizons, args.seeds, args.output)
    elif args.command == "ablation":
        run_ablation(args.config, args.output, args.models, args.horizons, args.seeds, args.graph_types)
    elif args.command == "report":
        generate_experiment_report(args.runs_dir, args.output)
    elif args.command == "export-figures":
        export_figures(args.run_dir, args.output_dir)
    elif args.command == "app":
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "src/traffic_agent/app/streamlit_app.py"],
            check=False,
        )
    elif args.command == "api":
        subprocess.run([sys.executable, "-m", "uvicorn", "traffic_agent.api.main:app", "--reload"], check=False)


if __name__ == "__main__":
    main()
