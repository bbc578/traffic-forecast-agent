from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_train_log_expected_columns(tmp_path: Path) -> None:
    path = tmp_path / "train_log.csv"
    pd.DataFrame(
        [
            {
                "epoch": 1,
                "train_loss": 1.0,
                "val_loss": 1.1,
                "learning_rate": 0.001,
                "epoch_seconds": 0.5,
                "gpu_memory_allocated_mb": 0.0,
                "gpu_memory_reserved_mb": 0.0,
            }
        ]
    ).to_csv(path, index=False)
    columns = set(pd.read_csv(path).columns)
    assert {"epoch", "train_loss", "val_loss", "learning_rate", "epoch_seconds"}.issubset(columns)
