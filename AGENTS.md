# AGENTS.md

## Core Rules

- Do not invent experiment results.
- Do not present synthetic results as real traffic results.
- Do not write SOTA, industrial-grade, or real-traffic-control claims.
- Agent features must not claim they can control signals or replace traffic authorities.
- Real data configs must fail clearly if prepared real `.npz` files are missing.

## Development Rules

- New models require tests, docs, and README updates.
- Training logic changes must check for time-series leakage.
- Scalers must fit on train data only.
- Correlation adjacency must be built from train time only and labeled as correlation.
- Dashboard charts need interpretation text.
- Large raw datasets, model weights, and predictions should not be committed.

## Commands to Run After Changes

```bash
ruff check .
pytest
python -m traffic_agent.data.generate_synthetic --output data/sample/synthetic_traffic.npz
python -m traffic_agent.training.train --config configs/demo.yaml --model last_value
```

If a command fails or cannot be run because of environment limits, state that in the final summary.

## Useful Training Commands

```bash
python -m traffic_agent.training.train --config configs/demo.yaml --model last_value
python -m traffic_agent.training.train --config configs/demo.yaml --model gru
python -m traffic_agent.training.train --config configs/demo.yaml --model stgcn_improved
python -m traffic_agent.training.run_experiments \
  --config configs/demo.yaml \
  --models last_value historical_average lstm gru stgcn_improved graph_wavenet_lite \
  --horizons 3 6 12 \
  --seeds 42 \
  --output experiments/results/demo_experiment_summary.csv
```
