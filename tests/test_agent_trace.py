from __future__ import annotations

from pathlib import Path

from traffic_agent.agent.trace import save_trace


def test_save_trace_creates_json(tmp_path: Path) -> None:
    trace_id = save_trace(tmp_path, {"answer": "ok"})
    assert (tmp_path / "agent_traces" / f"{trace_id}.json").exists()
