from __future__ import annotations

import json
from pathlib import Path

from traffic_agent.training import run_experiments


def test_run_experiments_summarizes_metrics(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "demo.yaml"
    config.write_text(
        "seed: 42\n"
        "data:\n  path: data/sample/synthetic_traffic.npz\n  input_steps: 12\n  horizon: 3\n"
        "training: {}\nmodel: {}\nanalysis: {}\noutputs: {}\n",
        encoding="utf-8",
    )

    def fake_train_model(*_, **__) -> Path:
        run_dir = tmp_path / "outputs" / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "model_name": "last_value",
                    "dataset_name": "synthetic_demo",
                    "is_synthetic_data": True,
                    "horizon": 3,
                    "seed": 42,
                    "mae": 1.0,
                    "rmse": 2.0,
                    "mape": 3.0,
                    "masked_mae": 1.0,
                    "adjacency_type": "physical",
                }
            ),
            encoding="utf-8",
        )
        return run_dir

    monkeypatch.setattr(run_experiments, "train_model", fake_train_model)
    output = tmp_path / "summary.csv"
    run_experiments.run_experiments(str(config), ["last_value"], [3], [42], str(output))
    assert output.exists()
    assert output.with_suffix(".md").exists()
