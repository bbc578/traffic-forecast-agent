from __future__ import annotations

import json
from pathlib import Path

from traffic_agent.reports.full_model_report import generate_full_model_report


def test_full_model_report_handles_missing_full_models(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "last"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({"model_name": "last_value", "mae": 1.0, "horizon": 3, "is_synthetic_data": False}),
        encoding="utf-8",
    )
    output = generate_full_model_report(tmp_path / "outputs", tmp_path / "results", tmp_path / "report.md")
    assert output.exists()
    assert "LastValue" in output.read_text(encoding="utf-8")
