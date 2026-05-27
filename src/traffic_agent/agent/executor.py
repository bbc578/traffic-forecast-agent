from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from traffic_agent.agent.planner import AgentPlan, RuleBasedPlanner
from traffic_agent.agent.tool_registry import default_tool_registry
from traffic_agent.agent.trace import save_trace, summarize_output


class StructuredAgentAnswer(BaseModel):
    answer: str
    intent: str
    confidence: float
    tools_used: list[str] = Field(default_factory=list)
    data: dict[str, Any] | list[Any] | None = None
    trace_id: str | None = None
    limitations: list[str] = Field(default_factory=list)
    suggested_followups: list[str] = Field(default_factory=list)


class AgentExecutor:
    """Execute planned read-only traffic-analysis tool calls and persist trace."""

    def __init__(self, outputs_dir: str = "outputs") -> None:
        self.outputs_dir = outputs_dir
        self.registry = default_tool_registry()
        self.planner = RuleBasedPlanner()

    def run(self, query: str, run_name: str | None = None) -> StructuredAgentAnswer:
        plan = self.planner.plan(query, run_name=run_name, outputs_dir=self.outputs_dir)
        limitations = ["Agent uses saved offline artifacts only; it does not control traffic systems."]
        if plan.refusal_reason:
            return StructuredAgentAnswer(
                answer=plan.refusal_reason,
                intent=plan.intent,
                confidence=plan.confidence,
                limitations=limitations,
                suggested_followups=["可以询问模型指标、拥堵风险、误差诊断或实验报告。"],
            )
        if plan.clarification_needed:
            return StructuredAgentAnswer(
                answer="我无法确定你的分析意图。可尝试：哪个模型效果最好？未来哪里最堵？哪些时间段误差最大？",
                intent=plan.intent,
                confidence=plan.confidence,
                limitations=limitations,
            )

        tool_outputs: list[Any] = []
        tools_called = []
        for tool_name in plan.selected_tools:
            if tool_name not in self.registry:
                raise ValueError(f"Planner selected unregistered tool: {tool_name}")
            spec = self.registry[tool_name]
            if not spec.read_only:
                raise PermissionError(f"Tool is not read-only and cannot be executed by this agent: {tool_name}")
            output = spec.function(**plan.arguments)
            tool_outputs.append(output)
            tools_called.append(tool_name)

        answer = self._compose_answer(plan, tool_outputs)
        trace_run_dir = self._trace_run_dir(run_name)
        trace_id = save_trace(
            trace_run_dir,
            {
                "user_query": query,
                "normalized_query": query.strip().lower(),
                "plan": plan.model_dump(),
                "tools_called": tools_called,
                "tool_inputs": plan.arguments,
                "tool_outputs_summary": [summarize_output(output) for output in tool_outputs],
                "data_sources": [str(Path(trace_run_dir) / "metrics.json")],
                "answer": answer,
                "limitations": limitations,
            },
        )
        data = tool_outputs[0] if len(tool_outputs) == 1 else {"outputs": tool_outputs}
        return StructuredAgentAnswer(
            answer=answer,
            intent=plan.intent,
            confidence=plan.confidence,
            tools_used=tools_called,
            data=data,
            trace_id=trace_id,
            limitations=limitations,
            suggested_followups=["查看误差诊断", "生成实验报告", "比较 LastValue baseline"],
        )

    def _trace_run_dir(self, run_name: str | None) -> str:
        if run_name:
            return str(Path(self.outputs_dir) / run_name)
        candidates = sorted(Path(self.outputs_dir).glob("*/metrics.json"))
        return str(candidates[-1].parent) if candidates else self.outputs_dir

    @staticmethod
    def _compose_answer(plan: AgentPlan, outputs: list[Any]) -> str:
        if not outputs:
            return "没有可用工具输出。"
        if plan.intent == "compare_models":
            return "已基于 outputs 中保存的 metrics.json 比较模型；请优先检查 LastValue baseline 是否被深度模型稳定超过。"
        if plan.intent == "congestion_risk":
            return "已根据保存预测计算拥堵风险 Top-K。该结果是离线风险提示，不是事故或管控决策。"
        if plan.intent == "explain_forecast":
            return "已生成节点预测解释。该解释基于残差、历史趋势和风险规则，不是因果解释。"
        if plan.intent in {"daily_report", "experiment_report"}:
            return str(outputs[0])
        return "已完成工具调用，结果来自本地 run artifacts。"
