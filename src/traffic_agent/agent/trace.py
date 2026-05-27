from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def summarize_output(value: Any, limit: int = 500) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text[:limit] + ("..." if len(text) > limit else "")


def save_trace(run_dir: str | Path, payload: dict[str, Any]) -> str:
    """Save an agent trace under the selected run directory."""
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S_%f")
    trace_dir = Path(run_dir) / "agent_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_id = f"trace_{timestamp}"
    payload = {**payload, "trace_id": trace_id, "created_at": datetime.now(tz=UTC).isoformat()}
    (trace_dir / f"{trace_id}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return trace_id
