from __future__ import annotations

from traffic_agent.agent.executor import AgentExecutor


def test_agent_full_model_missing_results_returns_guidance(tmp_path) -> None:
    response = AgentExecutor(outputs_dir=str(tmp_path)).run("完整模型有没有超过 LastValue？")
    assert response.intent in {"compare_full_vs_lite", "compare_against_last_value"}
    assert response.data is not None
