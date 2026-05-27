from __future__ import annotations

import os

from traffic_agent.agent.executor import AgentExecutor, StructuredAgentAnswer


class OptionalLLMAgent:
    """Placeholder optional LLM layer that safely falls back to deterministic planning."""

    def __init__(self, outputs_dir: str = "outputs", enable_llm_agent: bool = False) -> None:
        self.outputs_dir = outputs_dir
        self.enable_llm_agent = enable_llm_agent and bool(os.getenv("OPENAI_API_KEY"))
        self.fallback = AgentExecutor(outputs_dir=outputs_dir)

    def run(self, query: str, run_name: str | None = None) -> StructuredAgentAnswer:
        if not self.enable_llm_agent:
            return self.fallback.run(query, run_name=run_name)
        # The project intentionally keeps CI and quick start independent from external LLM APIs.
        # A production extension would map LLM JSON plans onto the same Tool Registry.
        return self.fallback.run(query, run_name=run_name)
