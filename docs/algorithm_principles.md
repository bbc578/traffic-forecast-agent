# Algorithm Principles

## Problem Definition

The road network is represented as a graph `G=(V,E,A)`, where `V` is the set of sensor nodes,
`E` is the set of road-neighbor relations, and `A` is an adjacency matrix. At time `t`, traffic
state is `X_t in R^(N x F)`, where `N` is the number of nodes and `F` is the number of features
such as speed and flow.

Given an input window `X_{t-T+1:t}`, the forecasting task predicts future speed
`Y_{t+1:t+H}` for all nodes. In the default demo, `T=12` means the past 60 minutes at 5-minute
resolution, and `H=3/6/12` means future 15/30/60 minutes.

## Time Split and Leakage

Traffic forecasting is a time-series problem. Randomly shuffling timestamps can leak future
traffic patterns into training and make metrics unrealistically optimistic. This project uses
time-ordered train/validation/test splits and fits scalers only on the training segment.

## Why Baselines Matter

Strong baselines prevent overclaiming:

- LastValue repeats the most recent observed speed. It is strong for short-term traffic.
- HistoricalAverage averages the input window.
- SeasonalNaive reuses same-time-of-day history when the input window is long enough.
- LSTM/GRU are temporal-only neural baselines.
- STGCNImproved and GraphWaveNetLite test whether explicit graph structure helps.

If a deep model does not beat LastValue, the correct conclusion is not “the model is effective”;
it is that the setup, features, graph, horizon, or training protocol needs more diagnosis.

## Why Graph Structure Matters

Neighboring road segments can show correlated congestion propagation. A graph model mixes
neighbor node states before or during temporal modeling. This can help when adjacency captures
meaningful upstream/downstream relations. However, graph edges are not causal proof; a correlation
graph is only a data-driven approximation, not physical topology.

## STGCNImproved

The implemented STGCNImproved is educational and simplified:

- temporal gated convolution captures local time patterns;
- graph message passing mixes neighbor states with `A`;
- residual connections preserve stable signals;
- layer normalization and dropout improve training stability.

It is STGCN-inspired, not a faithful reproduction of the original paper.

## GraphWaveNetLite

GraphWaveNetLite is inspired by Graph WaveNet:

- dilated temporal convolution expands temporal receptive field;
- static adjacency injects known graph support;
- optional adaptive adjacency learns node-to-node similarity;
- skip connections help gradients and preserve multi-scale features.

It is an educational simplification, not a complete Graph WaveNet reproduction.

## Metrics

- MAE: average absolute speed error.
- RMSE: penalizes large errors more heavily.
- MAPE: relative error percentage with a denominator floor; still sensitive when true speeds are low.
- Masked metrics: ignore NaN/Inf, optional sentinel values, and low-speed denominators for masked MAPE.

MAPE should be interpreted carefully for low-speed or zero-speed values. For real METR-LA reports,
prefer MAE, RMSE, and masked MAPE over raw MAPE.

## Reading Experiments

Do not rely on a single total metric. Inspect:

- horizon-wise error;
- node-wise error;
- time-of-day error;
- worst cases;
- whether graph models beat temporal-only and LastValue baselines.
