from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from traffic_agent.agent.tools import (
    compare_runs,
    generate_daily_report,
    get_latest_metrics,
    get_top_congestion_nodes,
)
from traffic_agent.analysis.error_analysis import (
    error_by_horizon,
    error_by_node,
    error_by_time_of_day,
    load_predictions,
)
from traffic_agent.analysis.explainability import explain_node_forecast


class ToolInput(BaseModel):
    run_dir: str | None = None
    outputs_dir: str = "outputs"
    run_name: str | None = None
    node_id: str | None = None
    horizon_step: int = 1
    k: int = 5


class ToolOutput(BaseModel):
    result: Any


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    read_only: bool
    requires_real_data: bool
    safety_notes: str
    function: Callable[..., Any]

    model_config = {"arbitrary_types_allowed": True}


def _resolve_run_dir(args: ToolInput) -> str:
    if args.run_dir:
        return args.run_dir
    if args.run_name:
        return str(Path(args.outputs_dir) / args.run_name)
    candidates = sorted(Path(args.outputs_dir).glob("*/metrics.json"))
    if not candidates:
        raise FileNotFoundError(f"No runs found in {args.outputs_dir}.")
    return str(candidates[-1].parent)


def list_runs(outputs_dir: str = "outputs", **_: Any) -> list[dict[str, Any]]:
    base = Path(outputs_dir)
    return [{"run_name": path.parent.name, "metrics_path": str(path)} for path in sorted(base.glob("*/metrics.json"))]


def get_run_metadata(**kwargs: Any) -> dict[str, Any]:
    metrics = get_latest_metrics(_resolve_run_dir(ToolInput(**kwargs)))
    keys = ["dataset_name", "is_synthetic_data", "model_name", "horizon", "num_nodes", "num_features", "adjacency_type"]
    return {key: metrics.get(key) for key in keys}


def compare_models(outputs_dir: str = "outputs", **_: Any) -> list[dict[str, Any]]:
    return compare_runs(outputs_dir)


def get_prediction_curve(**kwargs: Any) -> dict[str, Any]:
    args = ToolInput(**kwargs)
    payload = load_predictions(_resolve_run_dir(args))
    node_ids = [str(item) for item in payload["node_ids"].tolist()]
    node_id = args.node_id or node_ids[0]
    node_idx = node_ids.index(node_id)
    return {
        "node_id": node_id,
        "y_true": payload["y_true"][:100, :, node_idx].reshape(-1).round(4).tolist(),
        "y_pred": payload["y_pred"][:100, :, node_idx].reshape(-1).round(4).tolist(),
    }


def get_error_by_horizon(**kwargs: Any) -> list[dict[str, Any]]:
    payload = load_predictions(_resolve_run_dir(ToolInput(**kwargs)))
    return error_by_horizon(payload["y_true"], payload["y_pred"]).to_dict(orient="records")


def get_error_by_node(**kwargs: Any) -> list[dict[str, Any]]:
    payload = load_predictions(_resolve_run_dir(ToolInput(**kwargs)))
    node_ids = [str(item) for item in payload["node_ids"].tolist()]
    return error_by_node(payload["y_true"], payload["y_pred"], node_ids).head(20).to_dict(orient="records")


def get_error_by_time_of_day(**kwargs: Any) -> list[dict[str, Any]]:
    payload = load_predictions(_resolve_run_dir(ToolInput(**kwargs)))
    timestamps = [str(item) for item in payload["timestamps"].tolist()] if "timestamps" in payload else None
    return error_by_time_of_day(payload["y_true"], payload["y_pred"], timestamps).to_dict(orient="records")


def get_data_card(**kwargs: Any) -> dict[str, Any]:
    metrics = get_latest_metrics(_resolve_run_dir(ToolInput(**kwargs)))
    return {
        "dataset_name": metrics.get("dataset_name"),
        "is_synthetic_data": metrics.get("is_synthetic_data"),
        "num_nodes": metrics.get("num_nodes"),
        "num_timesteps": metrics.get("num_timesteps"),
        "num_features": metrics.get("num_features"),
        "train_time_range": metrics.get("train_time_range"),
        "val_time_range": metrics.get("val_time_range"),
        "test_time_range": metrics.get("test_time_range"),
    }


def get_model_card(**kwargs: Any) -> dict[str, Any]:
    metrics = get_latest_metrics(_resolve_run_dir(ToolInput(**kwargs)))
    return {
        "model_name": metrics.get("model_name"),
        "parameter_count": metrics.get("parameter_count"),
        "adjacency_type": metrics.get("adjacency_type"),
        "metrics": {
            key: metrics.get(key)
            for key in ["mae", "rmse", "mape", "masked_mae", "masked_rmse", "masked_mape"]
        },
    }


def get_visualization_recommendations(**_: Any) -> list[str]:
    return [
        "Start with Overview to verify dataset type and horizon.",
        "Use Forecast Explorer for node-level prediction and residual curves.",
        "Use Error Diagnostics before making any claim about model quality.",
        "For synthetic runs, treat every figure as workflow validation only.",
    ]


def generate_experiment_report(outputs_dir: str = "outputs", **_: Any) -> str:
    rows = compare_runs(outputs_dir)
    if not rows:
        return "No completed runs found."
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _daily_report(**kwargs: Any) -> str:
    return generate_daily_report(_resolve_run_dir(ToolInput(**kwargs)))


def _congestion(**kwargs: Any) -> list[dict[str, Any]]:
    args = ToolInput(**kwargs)
    return get_top_congestion_nodes(_resolve_run_dir(args), k=args.k)


def _metrics(**kwargs: Any) -> dict[str, Any]:
    return get_latest_metrics(_resolve_run_dir(ToolInput(**kwargs)))


def _explain(**kwargs: Any) -> dict[str, Any]:
    args = ToolInput(**kwargs)
    run_dir = _resolve_run_dir(args)
    node_id = args.node_id
    if node_id is None:
        payload = load_predictions(run_dir)
        node_id = str(payload["node_ids"].tolist()[0])
    return explain_node_forecast(run_dir, node_id=node_id, horizon_step=args.horizon_step)


def default_tool_registry() -> dict[str, ToolSpec]:
    common = {"input_schema": ToolInput, "output_schema": ToolOutput, "read_only": True, "requires_real_data": False}
    specs = [
        ("list_runs", "List completed output runs.", list_runs),
        ("get_run_metadata", "Read run metadata.", get_run_metadata),
        ("get_latest_metrics", "Read metrics from one run.", _metrics),
        ("compare_models", "Compare saved run metrics.", compare_models),
        ("get_prediction_curve", "Return prediction curve data for one node.", get_prediction_curve),
        ("get_top_congestion_nodes", "Rank congestion risk nodes.", _congestion),
        ("get_error_by_horizon", "Compute horizon-wise errors.", get_error_by_horizon),
        ("get_error_by_node", "Compute node-wise errors.", get_error_by_node),
        ("get_error_by_time_of_day", "Compute time-of-day error buckets.", get_error_by_time_of_day),
        ("explain_node_forecast", "Explain a node forecast heuristically.", _explain),
        ("generate_daily_report", "Generate run daily report.", _daily_report),
        ("generate_experiment_report", "Generate experiment summary report.", generate_experiment_report),
        ("get_data_card", "Return dataset metadata card.", get_data_card),
        ("get_model_card", "Return model card fields.", get_model_card),
        (
            "get_visualization_recommendations",
            "Recommend dashboard interpretation steps.",
            get_visualization_recommendations,
        ),
    ]
    return {
        name: ToolSpec(
            name=name,
            description=description,
            function=func,
            safety_notes="Read-only analysis tool. Does not control traffic systems.",
            **common,
        )
        for name, description, func in specs
    }


def tool_registry_as_json() -> str:
    registry = default_tool_registry()
    serializable = {name: spec.model_dump(exclude={"function"}) for name, spec in registry.items()}
    return json.dumps(serializable, ensure_ascii=False, default=str)
