from __future__ import annotations

import numpy as np

from traffic_agent.analysis.error_analysis import (
    error_by_horizon,
    error_by_node,
    residual_distribution,
    worst_case_segments,
)


def test_error_analysis_outputs() -> None:
    y_true = np.ones((4, 3, 2), dtype=np.float32)
    y_pred = y_true + 1
    assert len(error_by_horizon(y_true, y_pred)) == 3
    assert len(error_by_node(y_true, y_pred, ["a", "b"])) == 2
    assert residual_distribution(y_true, y_pred)["mean"] == 1.0
    assert worst_case_segments(y_true, y_pred, ["a", "b"], top_k=2)[0]["absolute_error"] == 1.0
