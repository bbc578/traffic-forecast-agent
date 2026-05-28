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
- STGCNFull: paper-inspired ST-Conv stack with temporal GLU, Chebyshev-style graph convolution,
  residuals, layer normalization, and dropout.
- GraphWaveNetFull: paper-inspired dilated causal temporal stack with gated activations,
  multi-support diffusion graph convolution, skip/end convolutions, and adaptive adjacency.

STGCNImproved and GraphWaveNetLite are lite/educational graph models for explanation and debugging.
STGCNFull and GraphWaveNetFull are paper-inspired full models for stronger experiments. None of
them is an official faithful reproduction unless the original training protocol and evaluation
setting are strictly matched.

## Why Full Models?

Lite models are useful for understanding graph message passing and temporal modules. Full models are
added because the current METR-LA result shows LastValue is very strong for 15-minute prediction.
More complete structures, longer training, multi-horizon runs, and multi-seed evaluation are needed
before making any claim about graph-model value.

Model hierarchy:

- Naive baselines: LastValue, HistoricalAverage, SeasonalNaive
- Temporal-only models: LSTM, GRU
- Lite graph models: STGCNLite, STGCNImproved, GraphWaveNetLite
- Paper-inspired full graph models: STGCNFull, GraphWaveNetFull

## RTX 5090 Training

The training script supports CUDA auto-detection, AMP, gradient clipping, early stopping, train logs,
best checkpoint saving, and longer runs. A 5090 is not required for quick smoke tests, but it is
appropriate for full METR-LA multi-horizon/multi-seed experiments.

```bash
python -m traffic_agent.training.train --config configs/metr_la_5090.yaml --model graph_wavenet_full
python -m traffic_agent.training.train --config configs/metr_la_5090.yaml --model stgcn_full
```

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

Current METR-LA physical-adjacency result snapshot, generated locally and not claimed as SOTA:

| model | MAE | RMSE | MAPE | masked MAPE |
| --- | ---: | ---: | ---: | ---: |
| last_value | 3.5117 | 8.5862 | 45.8909 | 8.1707% |
| graph_wavenet_lite | 3.8526 | 8.4696 | 69.3887 | 8.3610% |
| stgcn_improved | 4.1708 | 8.2505 | 103.7481 | 8.6860% |
| historical_average | 5.1282 | 11.3924 | 112.0198 | 11.4164% |
| gru | 6.0862 | 11.3899 | 193.5160 | 10.9495% |

Interpretation:

- LastValue is best on MAE and masked MAPE, showing strong short-term speed persistence.
- STGCNImproved is best on RMSE, suggesting graph-temporal structure may reduce some large errors.
- GraphWaveNetLite is close to LastValue but does not beat it on MAE.
- GRU needs further tuning or implementation diagnosis.
- Raw MAPE is not a primary conclusion metric because low-speed samples can dominate it.

Each run saves:

- `metrics.json`
- `predictions.npz`
- `config.yaml`
- `run_summary.md`
- `model.pt` for trainable models

For METR-LA style speed data, ordinary MAPE can be misleading when true speeds are near zero. The
training script records a `mape_denominator_floor` and also saves masked metrics. Prefer MAE, RMSE,
and masked MAPE in reports.

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

Full-model report:

```bash
python -m traffic_agent.reports.full_model_report \
  --runs-dir outputs \
  --results-dir experiments/results \
  --output experiments/reports/metr_la_full_model_report.md
```

## Next Full-model Experiments

Prepare METR-LA:

```bash
python -m traffic_agent.data.prepare_real_dataset \
  --dataset metr-la \
  --traffic-file data/raw/METR-LA/metr-la.h5 \
  --adj-file data/raw/METR-LA/adj_mx.pkl \
  --output data/processed/metr_la.npz
```

Run full model experiments:

```bash
python -m traffic_agent.training.run_experiments \
  --config configs/metr_la_5090.yaml \
  --models last_value historical_average seasonal_naive gru lstm stgcn_improved graph_wavenet_lite stgcn_full graph_wavenet_full \
  --horizons 3 6 12 \
  --seeds 42 43 44 \
  --output experiments/results/metr_la_5090_full_summary.csv
```

Run graph ablation:

```bash
python -m traffic_agent.training.run_ablation \
  --config configs/metr_la_5090.yaml \
  --models stgcn_full graph_wavenet_full graph_wavenet_lite stgcn_improved \
  --horizons 3 6 12 \
  --seeds 42 43 44 \
  --graph-types identity physical correlation adaptive \
  --output experiments/results/metr_la_5090_ablation_summary.csv
```

Run congestion subset evaluation:

```bash
python -m traffic_agent.analysis.congestion_eval \
  --runs-dir outputs \
  --output experiments/results/metr_la_congestion_subset_eval.csv
```

Prohibited claims:

- “Achieved SOTA”
- “Official paper reproduction”
- “Graph models significantly outperform baseline”
- “Can be used for real traffic control”
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
