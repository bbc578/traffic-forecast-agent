from __future__ import annotations

from traffic_agent.agent.planner import RuleBasedPlanner


def test_agent_planner_refuses_control() -> None:
    plan = RuleBasedPlanner().plan("直接控制红绿灯")
    assert plan.intent == "safety_refusal"
    assert plan.refusal_reason


def test_agent_planner_selects_compare() -> None:
    plan = RuleBasedPlanner().plan("哪个模型效果最好？")
    assert plan.selected_tools == ["compare_models"]
