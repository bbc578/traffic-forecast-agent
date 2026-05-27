# Experiment Protocol

## Data Preparation

Synthetic data is only for quick start, CI, and smoke tests.

Real METR-LA / PEMS-BAY style data must be prepared explicitly:

```bash
python -m traffic_agent.data.prepare_real_dataset \
  --dataset metr-la \
  --traffic-file data/raw/METR-LA/metr-la.h5 \
  --adj-file data/raw/METR-LA/adj_mx.pkl \
  --output data/processed/metr_la.npz
```

If no physical adjacency is available, `--build-correlation-adj true` can build a train-only
correlation graph. This is not road topology and must be reported as `adjacency_type=correlation`.

## Split

Use time-ordered train/validation/test splits. Default is 70% / 10% / 20%. Do not randomly shuffle
timestamps before split.

## Horizons

Evaluate `horizon=3,6,12` for future 15, 30, and 60 minutes at 5-minute resolution.

## Seeds

Use multiple seeds for neural models when reporting real results. Synthetic smoke tests may use a
single seed.

## Baselines

Always include LastValue and HistoricalAverage. SeasonalNaive is useful when enough same-time-of-day
history exists. LSTM/GRU isolate temporal modeling without graph structure.

## Ablation

Run graph ablations:

- no_graph / identity adjacency;
- physical_graph;
- correlation_graph;
- adaptive_graph;
- temporal_only;
- spatial_temporal.

## Results Format

Each run writes `metrics.json`, `predictions.npz`, `config.yaml`, and `run_summary.md`. Large files
should not be committed.

## Missing Real Data

If real files are absent, run only synthetic smoke tests. Do not present synthetic metrics as
METR-LA or PEMS-BAY results.

## Resume Usage

Separate synthetic workflow validation from real dataset experiments. Only report real metrics after
the code generates them from prepared real data.
