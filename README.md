# Urban Traffic Forecasting & Congestion Analysis Agent

中文名：城市路网交通态势预测与拥堵分析智能体系统

This is a portfolio project for smart-transportation students. It combines traffic forecasting
algorithms, congestion-risk analysis, an analysis-oriented Dashboard, and a traceable tool-calling
Agent. It is not a real traffic-control system and must not be used for live signal decisions.

## What Makes This More Than a Demo

- Real dataset pipeline for METR-LA / PEMS-BAY style HDF5 + DCRNN adjacency files.
- Strong baselines: LastValue, HistoricalAverage, SeasonalNaive, LSTM, GRU.
- Graph-aware models: STGCNImproved and GraphWaveNetLite, both educational simplified versions.
- Experiment scripts for model/horizon/seed loops and graph ablations.
- Error diagnostics by horizon, node, time of day, residuals, and worst cases.
- Agent framework with Tool Registry, Planner, Executor, Trace, and structured answers.
- Analysis Dashboard with narrative tabs instead of isolated tables.

Synthetic data is included only as a quick-start and CI smoke-test path. Synthetic metrics do not
prove real traffic forecasting performance.

## Quick Start with Synthetic Data

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m traffic_agent.data.generate_synthetic --output data/sample/synthetic_traffic.npz
python -m traffic_agent.training.train --config configs/demo.yaml --model last_value
python -m traffic_agent.training.train --config configs/demo.yaml --model historical_average
python -m traffic_agent.training.train --config configs/demo.yaml --model lstm
python -m traffic_agent.training.train --config configs/demo.yaml --model gru
python -m traffic_agent.training.train --config configs/demo.yaml --model stgcn_improved
python -m traffic_agent.training.train --config configs/demo.yaml --model graph_wavenet_lite
```

Run a synthetic smoke experiment:

```bash
python -m traffic_agent.training.run_experiments \
  --config configs/demo.yaml \
  --models last_value historical_average lstm gru stgcn_improved graph_wavenet_lite \
  --horizons 3 6 12 \
  --seeds 42 \
  --output experiments/results/demo_experiment_summary.csv
```

## Real Dataset Workflow

The repository does not include large real datasets. Prepare files locally:

```text
data/raw/METR-LA/metr-la.h5
data/raw/METR-LA/adj_mx.pkl
data/raw/PEMS-BAY/pems-bay.h5
data/raw/PEMS-BAY/adj_mx.pkl
```

Convert METR-LA:

```bash
python -m traffic_agent.data.prepare_real_dataset \
  --dataset metr-la \
  --traffic-file data/raw/METR-LA/metr-la.h5 \
  --adj-file data/raw/METR-LA/adj_mx.pkl \
  --output data/processed/metr_la.npz
```

Convert PEMS-BAY:

```bash
python -m traffic_agent.data.prepare_real_dataset \
  --dataset pems-bay \
  --traffic-file data/raw/PEMS-BAY/pems-bay.h5 \
  --adj-file data/raw/PEMS-BAY/adj_mx.pkl \
  --output data/processed/pems_bay.npz
```

If no physical adjacency exists, you may use:

```bash
--build-correlation-adj true
```

This uses only the training time segment to compute correlations, but it is not road topology.
It is recorded as `adjacency_type=correlation`.

Train real configs:

```bash
python -m traffic_agent.training.train --config configs/metr_la.yaml --model last_value
python -m traffic_agent.training.train --config configs/metr_la.yaml --model stgcn_improved
python -m traffic_agent.training.train --config configs/pems_bay.yaml --model graph_wavenet_lite
```

If the real `.npz` file is missing, training fails clearly and will not fall back to synthetic data.

## Algorithms

- LastValue: repeats the latest observed speed; strong for short-term traffic.
- HistoricalAverage: repeats the input-window mean speed.
- SeasonalNaive: uses same-time-of-day history when available, otherwise falls back honestly.
- LSTM / GRU: temporal-only neural baselines, no explicit graph structure.
- STGCNImproved: gated temporal convolution + adjacency message passing + residual + normalization.
- GraphWaveNetLite: dilated temporal convolution + static adjacency + optional adaptive adjacency.

STGCNImproved and GraphWaveNetLite are educational simplified implementations, not faithful paper
reproductions and not SOTA claims.

## Experiment Protocol

- Time-ordered train/validation/test split, no random time shuffle.
- Fit scaler only on training data.
- Evaluate horizons `3/6/12` for 15/30/60 minutes.
- Run multiple seeds for real reports.
- Always include LastValue and HistoricalAverage.
- Inspect horizon, node, time-of-day, and worst-case errors.
- Use ablations to test graph value:

```bash
python -m traffic_agent.training.run_ablation \
  --config configs/ablation.yaml \
  --output experiments/results/ablation_summary.csv
```

## Results

Do not fill fake real-data results. If you only ran synthetic quick-start, label results as
`is_synthetic_data=true`.

Each run saves:

- `metrics.json`
- `predictions.npz`
- `config.yaml`
- `run_summary.md`
- `model.pt` for trainable models

Large outputs are ignored by Git.

## Dashboard

```bash
streamlit run src/traffic_agent/app/streamlit_app.py
```

Tabs:

- Overview
- Data Explorer
- Network View
- Forecast Explorer
- Model Comparison
- Error Diagnostics
- Congestion Risk
- Agent Console
- Report

Export figures:

```bash
python -m traffic_agent.app.export_figures \
  --run-dir outputs/demo_run_stgcn_improved \
  --output-dir experiments/figures/demo_run_stgcn_improved
```

## FastAPI

```bash
uvicorn traffic_agent.api.main:app --reload
```

Endpoints include `/health`, `/runs`, `/runs/{run_name}/metrics`, `/runs/{run_name}/congestion`,
and `/agent/query`.

## Agent Design

The default Agent is deterministic and offline:

- Tool Registry defines read-only tools and schemas.
- Planner maps Chinese queries to tool plans.
- Executor only calls registered tools.
- Trace saves plan, inputs, output summaries, data sources, answer, and limitations.
- Optional LLM wrapper can be extended later but is not required for CI or quick start.

The Agent never controls traffic signals and never invents data outside saved artifacts.

## Reports

```bash
python -m traffic_agent.reports.experiment_report \
  --runs-dir outputs \
  --output experiments/reports/experiment_report.md

python -m traffic_agent.reports.model_card \
  --run-dir outputs/demo_run_stgcn_improved \
  --output experiments/reports/model_card_demo_run.md

python -m traffic_agent.reports.data_card \
  --data data/processed/metr_la.npz \
  --output experiments/reports/data_card_metr_la.md
```

## Tests and CI

```bash
pytest
ruff check .
```

CI installs dependencies, generates synthetic data, runs a LastValue smoke train, then runs ruff and
pytest. CI does not require real datasets.

## Documentation

- `docs/algorithm_principles.md`
- `docs/experiment_protocol.md`
- `docs/agent_design.md`
- `docs/visualization_design.md`
- `docs/resume_usage.md`

## Resume Usage

Synthetic-only wording:

> Built a smart traffic forecasting engineering demo using synthetic traffic sensor data to validate
> data preprocessing, model training, congestion-risk analysis, agent querying, and visualization workflow.

Real-data wording after you actually run METR-LA / PEMS-BAY:

> Built an urban traffic speed forecasting and congestion-risk analysis system on METR-LA / PEMS-BAY,
> comparing LastValue, HA, LSTM, GRU, STGCNImproved, and GraphWaveNetLite across horizons, nodes,
> time-of-day buckets, and graph-structure ablations.

Do not write fake improvement percentages, “SOTA”, “industrial-grade”, or “directly usable for real
traffic control.”

## Limitations

- Synthetic results are workflow validation only.
- Real datasets must be downloaded and prepared by the user.
- Graph models are simplified educational implementations.
- Correlation adjacency is not physical road topology.
- Error explanations are heuristic, not causal evidence.
- Dashboard/API are portfolio-local tools, not production deployment.
