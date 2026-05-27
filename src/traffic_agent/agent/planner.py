from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentPlan(BaseModel):
    intent: str
    selected_tools: list[str] = Field(default_factory=list)
    arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    clarification_needed: bool = False
    refusal_reason: str | None = None


class RuleBasedPlanner:
    """Deterministic planner for Chinese traffic-analysis queries."""

    safety_terms = ["控制红绿灯", "下发控制", "替代交管", "真实交通信号控制", "直接控制"]

    def plan(self, query: str, run_name: str | None = None, outputs_dir: str = "outputs") -> AgentPlan:
        text = query.strip().lower()
        args: dict[str, Any] = {"outputs_dir": outputs_dir, "run_name": run_name}
        if any(term in text for term in self.safety_terms):
            return AgentPlan(
                intent="safety_refusal",
                confidence=0.99,
                refusal_reason="本项目仅用于学习和离线分析，不可直接用于真实交通管控决策。",
            )
        if "哪个模型" in text or "比较" in text or "对比" in text or "horizon" in text or "图结构" in text:
            return AgentPlan(
                intent="compare_models",
                selected_tools=["compare_models"],
                arguments=args,
                confidence=0.85,
            )
        if "哪里最堵" in text or "拥堵" in text or "风险" in text:
            return AgentPlan(
                intent="congestion_risk",
                selected_tools=["get_top_congestion_nodes"],
                arguments=args,
                confidence=0.9,
            )
        if "为什么" in text or "解释" in text or "失败案例" in text:
            if "node_" in text:
                args["node_id"] = text[text.find("node_") :].split()[0].strip("，。,.")
            return AgentPlan(
                intent="explain_forecast",
                selected_tools=["explain_node_forecast"],
                arguments=args,
                confidence=0.78,
            )
        if "时间段" in text or "几点" in text:
            return AgentPlan(
                intent="error_by_time",
                selected_tools=["get_error_by_time_of_day"],
                arguments=args,
                confidence=0.82,
            )
        if "节点" in text and "误差" in text:
            return AgentPlan(
                intent="error_by_node",
                selected_tools=["get_error_by_node"],
                arguments=args,
                confidence=0.82,
            )
        if "日报" in text:
            return AgentPlan(
                intent="daily_report",
                selected_tools=["generate_daily_report"],
                arguments=args,
                confidence=0.9,
            )
        if "实验报告" in text:
            return AgentPlan(
                intent="experiment_report",
                selected_tools=["generate_experiment_report"],
                arguments=args,
                confidence=0.88,
            )
        if "数据卡" in text:
            return AgentPlan(intent="data_card", selected_tools=["get_data_card"], arguments=args, confidence=0.8)
        if "模型" in text or "指标" in text or "效果" in text:
            return AgentPlan(
                intent="model_metrics",
                selected_tools=["get_latest_metrics"],
                arguments=args,
                confidence=0.82,
            )
        if "dashboard" in text or "怎么看" in text:
            return AgentPlan(
                intent="visualization_help",
                selected_tools=["get_visualization_recommendations"],
                arguments=args,
                confidence=0.78,
            )
        return AgentPlan(intent="unknown", confidence=0.2, clarification_needed=True)
