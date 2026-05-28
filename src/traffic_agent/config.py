from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    path: str
    input_steps: int = 12
    horizon: int = 3
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    target_feature: str = "speed"


class TrainingConfig(BaseModel):
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int | None = None
    gradient_clip_norm: float | None = None
    use_amp: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    compile_model: bool = False
    device: str = "auto"
    loss: str = "masked_mae"


class ModelConfig(BaseModel):
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    graph_wavenet_full: dict[str, Any] = Field(default_factory=dict)
    stgcn_full: dict[str, Any] = Field(default_factory=dict)


class AnalysisConfig(BaseModel):
    congestion_speed_threshold: float = 35.0
    top_k: int = 5


class OutputsConfig(BaseModel):
    dir: str = "outputs"
    run_name: str = "demo_run"
    allow_overwrite: bool = False


class ProjectConfig(BaseModel):
    seed: int = 42
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a YAML project configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return ProjectConfig.model_validate(raw)


def save_config(config: ProjectConfig, path: str | Path) -> None:
    """Write a validated configuration as YAML."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f, sort_keys=False, allow_unicode=True)
