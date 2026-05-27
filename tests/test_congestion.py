from __future__ import annotations

from traffic_agent.analysis.congestion import VALID_RISK_LEVELS, identify_congestion_risk


def test_low_speed_node_high_risk() -> None:
    predicted = [[[50.0, 20.0, 55.0], [52.0, 18.0, 53.0], [51.0, 19.0, 54.0]]]
    results = identify_congestion_risk(predicted, node_ids=["a", "b", "c"], top_k=2)
    assert results[0]["node_id"] == "b"
    assert results[0]["risk_level"] == "high"


def test_top_k_count() -> None:
    predicted = [[50.0, 20.0, 30.0, 60.0]]
    results = identify_congestion_risk(predicted, top_k=3)
    assert len(results) == 3


def test_risk_level_valid() -> None:
    predicted = [[50.0, 20.0, 30.0, 60.0]]
    results = identify_congestion_risk(predicted, top_k=4)
    assert all(item["risk_level"] in VALID_RISK_LEVELS for item in results)
