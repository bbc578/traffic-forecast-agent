# Visualization Design

The Dashboard is designed as an analysis narrative:

1. Overview: verify dataset type, model, horizon, and headline metrics.
2. Data Explorer: inspect raw speed patterns, missing ratio, and feature metadata.
3. Network View: inspect adjacency density and connected nodes.
4. Forecast Explorer: compare true and predicted curves for a selected node and horizon step.
5. Model Comparison: compare baselines and neural models, with LastValue highlighted conceptually.
6. Error Diagnostics: inspect horizon, node, time-of-day, residual, and worst-case errors.
7. Congestion Risk: show risk nodes and reasons without claiming accident detection.
8. Agent Console: show answer, intent, tools, trace id, data, and limitations.
9. Report: generate markdown summaries.

Each chart includes a short interpretation note. Synthetic data always displays a warning.
