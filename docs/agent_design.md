# Agent Design

The project uses a deterministic offline traffic-analysis agent by default.

## Architecture

1. Tool Registry: declares tool name, description, Pydantic schemas, read-only flag, data requirement,
   safety notes, and function.
2. Planner: maps Chinese user queries to an `AgentPlan`.
3. Executor: executes only registered read-only tools.
4. Trace: stores query, plan, tools, tool inputs, output summaries, data sources, answer, limitations.
5. Structured Answer: returns answer, intent, confidence, tools, data, trace id, limitations, and follow-ups.

## Safety Boundary

The agent cannot control traffic signals, issue real traffic policies, or replace traffic authorities.
It only reads saved artifacts and produces offline analysis.

## Optional LLM Layer

`llm_agent_optional.py` is intentionally a safe fallback wrapper. If `OPENAI_API_KEY` and a future
config flag are present, it can be extended to map LLM JSON plans onto the same Tool Registry. The LLM
would select tools and phrase explanations; it would not predict traffic or control roads.
