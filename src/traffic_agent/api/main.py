from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from traffic_agent.agent.rule_based_agent import RuleBasedTrafficAgent
from traffic_agent.agent.tools import get_latest_metrics, get_top_congestion_nodes

OUTPUTS_DIR = Path("outputs")
app = FastAPI(title="Urban Traffic Forecasting & Congestion Analysis Agent")


class AgentQueryRequest(BaseModel):
    query: str
    run_name: str | None = None


class AgentQueryResponse(BaseModel):
    answer: str
    tool_used: str
    data: dict[str, Any] | list[dict[str, Any]] | None = None


def _run_dir(run_name: str) -> Path:
    path = OUTPUTS_DIR / run_name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_name}")
    return path


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runs")
def list_runs() -> dict[str, list[str]]:
    if not OUTPUTS_DIR.exists():
        return {"runs": []}
    runs = sorted(path.name for path in OUTPUTS_DIR.iterdir() if (path / "metrics.json").exists())
    return {"runs": runs}


@app.get("/runs/{run_name}/metrics")
def run_metrics(run_name: str) -> dict[str, Any]:
    try:
        return get_latest_metrics(str(_run_dir(run_name)))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/runs/{run_name}/congestion")
def run_congestion(run_name: str, k: int = Query(default=5, ge=1, le=50)) -> list[dict[str, Any]]:
    try:
        return get_top_congestion_nodes(str(_run_dir(run_name)), k=k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/agent/query", response_model=AgentQueryResponse)
def agent_query(request: AgentQueryRequest) -> AgentQueryResponse:
    agent = RuleBasedTrafficAgent(outputs_dir=str(OUTPUTS_DIR))
    response = agent.query(request.query, request.run_name)
    if response.tool_used == "error":
        raise HTTPException(status_code=404, detail=response.answer)
    return AgentQueryResponse.model_validate(response.model_dump())
