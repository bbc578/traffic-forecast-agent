from __future__ import annotations

from pathlib import Path

from traffic_agent.agent.schemas import AgentResponse
from traffic_agent.agent.tools import (
    compare_runs,
    generate_daily_report,
    get_anomalies,
    get_model_summary,
    get_top_congestion_nodes,
)


class RuleBasedTrafficAgent:
    """Small Chinese rule-based agent for querying saved traffic run artifacts."""

    def __init__(self, outputs_dir: str = "outputs") -> None:
        self.outputs_dir = Path(outputs_dir)

    def _resolve_run_dir(self, run_name: str | None) -> Path:
        if run_name:
            run_dir = self.outputs_dir / run_name
            if not run_dir.exists():
                raise FileNotFoundError(f"Run not found: {run_name}. Check outputs directory.")
            return run_dir
        candidates = sorted(self.outputs_dir.glob("*/metrics.json"))
        if not candidates:
            raise FileNotFoundError("No completed runs found under outputs. Train a model first.")
        return candidates[-1].parent

    def query(self, query: str, run_name: str | None = None) -> AgentResponse:
        """Route a natural-language query to local analysis tools."""
        normalized = query.strip().lower()
        try:
            if any(word in normalized for word in ["哪里最堵", "拥堵", "最堵", "风险"]):
                run_dir = self._resolve_run_dir(run_name)
                data = get_top_congestion_nodes(str(run_dir), k=5)
                return AgentResponse(
                    answer="已根据保存的预测结果计算 Top 拥堵风险节点。",
                    tool_used="get_top_congestion_nodes",
                    data=data,
                )
            if any(word in normalized for word in ["效果", "指标", "mae", "rmse", "mape", "表现"]):
                run_dir = self._resolve_run_dir(run_name)
                return AgentResponse(
                    answer=get_model_summary(str(run_dir)),
                    tool_used="get_model_summary",
                    data=None,
                )
            if any(word in normalized for word in ["日报", "报告", "总结"]):
                run_dir = self._resolve_run_dir(run_name)
                report = generate_daily_report(str(run_dir))
                return AgentResponse(
                    answer=report,
                    tool_used="generate_daily_report",
                    data={"markdown": report},
                )
            if any(word in normalized for word in ["比较", "对比", "哪个模型"]):
                data = compare_runs(str(self.outputs_dir))
                return AgentResponse(
                    answer="已读取 outputs 目录下各 run 的实际指标进行比较。",
                    tool_used="compare_runs",
                    data=data,
                )
            if any(word in normalized for word in ["异常", "异常点"]):
                run_dir = self._resolve_run_dir(run_name)
                data = get_anomalies(str(run_dir))
                return AgentResponse(
                    answer="已基于预测残差 z-score 规则查询异常提示。",
                    tool_used="get_anomalies",
                    data=data,
                )
        except FileNotFoundError as exc:
            return AgentResponse(answer=str(exc), tool_used="error", data=None)

        examples = "可尝试提问：未来哪里最堵？模型效果怎么样？生成日报。比较模型。有哪些异常？"
        return AgentResponse(answer=f"暂时无法识别该问题。{examples}", tool_used="none", data=None)
