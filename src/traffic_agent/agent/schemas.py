from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AgentResponse(BaseModel):
    answer: str
    tool_used: str
    data: dict[str, Any] | list[dict[str, Any]] | None = None
