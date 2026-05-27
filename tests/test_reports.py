from __future__ import annotations

import json
from pathlib import Path

from traffic_agent.reports.experiment_report import generate_experiment_report
from traffic_agent.reports.model_card import generate_model_card


def test_reports_generate_markdown(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_name": "run",
                "model_name": "last_value",
                "dataset_name": "synthetic_demo",
                "is_synthetic_data": True,
                "horizon": 3,
                "mae": 1.0,
                "rmse": 2.0,
                "mape": 3.0,
            }
        ),
        encoding="utf-8",
    )
    report = generate_experiment_report(tmp_path / "outputs", tmp_path / "report.md")
    card = generate_model_card(run_dir, tmp_path / "card.md")
    assert report.exists()
    assert card.exists()
