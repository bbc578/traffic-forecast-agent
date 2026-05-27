from __future__ import annotations

from traffic_agent.agent.tool_registry import default_tool_registry


def test_tool_registry_contains_read_only_tools() -> None:
    registry = default_tool_registry()
    assert "get_latest_metrics" in registry
    assert all(spec.read_only for spec in registry.values())
