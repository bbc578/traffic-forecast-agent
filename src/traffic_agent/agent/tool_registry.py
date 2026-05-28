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
    worst_case_segments,
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


def compare_against_last_value(outputs_dir: str = "outputs", **_: Any) -> dict[str, Any]:
    rows = compare_runs(outputs_dir)
    last = [row for row in rows if row.get("model_name") == "last_value"]
    if not last:
        return {
            "status": "missing",
            "message": "缺少 last_value 实验结果。",
            "run_command": "python -m traffic_agent.training.train --config configs/metr_la_5090.yaml --model last_value",
        }
    best_last = min(last, key=lambda row: row.get("mae", float("inf")))
    comparisons = []
    for row in rows:
        if row.get("model_name") != "last_value" and row.get("mae") is not None:
            comparisons.append(
                {
                    "model_name": row.get("model_name"),
                    "run_name": row.get("run_name"),
                    "mae_delta_vs_last_value": row["mae"] - best_last["mae"],
                    "beats_last_value_mae": row["mae"] < best_last["mae"],
                }
            )
    return {"last_value": best_last, "comparisons": comparisons}


def compare_full_vs_lite(outputs_dir: str = "outputs", **_: Any) -> dict[str, Any]:
    rows = compare_runs(outputs_dir)
    wanted = {"graph_wavenet_full", "graph_wavenet_lite", "stgcn_full", "stgcn_improved"}
    available = [row for row in rows if row.get("model_name") in wanted]
    missing = sorted(wanted.difference({row.get("model_name") for row in available}))
    return {
        "available": available,
        "missing": missing,
        "run_command": (
            "python -m traffic_agent.training.run_experiments --config configs/metr_la_5090.yaml "
            "--models stgcn_improved graph_wavenet_lite stgcn_full graph_wavenet_full "
            "--horizons 3 6 12 --seeds 42 43 44 "
            "--output experiments/results/metr_la_5090_full_summary.csv"
        ),
    }


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


def get_failure_cases(**kwargs: Any) -> list[dict[str, Any]]:
    args = ToolInput(**kwargs)
    payload = load_predictions(_resolve_run_dir(args))
    node_ids = [str(item) for item in payload["node_ids"].tolist()]
    return worst_case_segments(payload["y_true"], payload["y_pred"], node_ids, top_k=args.k)


def explain_why_last_value_is_strong(**_: Any) -> str:
    return (
        "短时交通速度具有很强的时间惯性，未来 15 分钟常接近最近观测值。"
        "因此 LastValue 是必须击败的强 baseline；复杂模型未超过它时不能声称更有效。"
    )


def explain_raw_mape_issue(**_: Any) -> str:
    return (
        "Raw MAPE 在真实速度接近 0 时分母过小，会产生很大的百分比。"
        "本项目报告 MAE、RMSE 和 masked MAPE，并记录 mape_denominator_floor。"
    )


def get_horizon_analysis(outputs_dir: str = "outputs", **_: Any) -> list[dict[str, Any]]:
    rows = compare_runs(outputs_dir)
    return sorted(rows, key=lambda row: (row.get("horizon") or 0, row.get("model_name") or ""))


def get_ablation_summary(outputs_dir: str = "experiments/results", **_: Any) -> dict[str, Any]:
    path = Path(outputs_dir)
    files = sorted(path.glob("*ablation*.csv"))
    if not files:
        return {"status": "missing", "message": "缺少图结构消融结果。"}
    return {"files": [str(file) for file in files]}


def get_congestion_subset_eval(outputs_dir: str = "experiments/results", **_: Any) -> dict[str, Any]:
    path = Path(outputs_dir)
    files = sorted(path.glob("*congestion_subset*.csv"))
    if not files:
        return {"status": "missing", "message": "缺少拥堵子集评估结果。"}
    return {"files": [str(file) for file in files]}


def generate_resume_bullets(outputs_dir: str = "outputs", **_: Any) -> list[str]:
    comparison = compare_against_last_value(outputs_dir)
    if comparison.get("status") == "missing":
        return ["完成交通预测工程框架搭建；真实模型对比需要先运行 LastValue baseline。"]
    any_beats = any(item["beats_last_value_mae"] for item in comparison["comparisons"])
    if any_beats:
        return [
            "在 METR-LA 真实交通数据上完成多模型时空预测对比，并用 MAE/RMSE/masked MAPE 评估。",
            "对 LastValue、GRU、STGCN/GraphWaveNet 系列模型进行 horizon 与图结构消融分析。",
        ]
    return [
        "在 METR-LA 真实交通数据和物理路网拓扑上完成多模型预测实验，发现 LastValue 在短时 MAE 上仍是强 baseline。",
        "实现 full graph models、图结构消融和拥堵子集评估，用误差诊断解释复杂模型未稳定超过 baseline 的原因。",
    ]


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
        ("compare_against_last_value", "Compare all models against LastValue baseline.", compare_against_last_value),
        ("compare_full_vs_lite", "Compare full graph models against lite graph models.", compare_full_vs_lite),
        ("get_horizon_analysis", "Summarize saved results by horizon.", get_horizon_analysis),
        ("get_ablation_summary", "Locate graph ablation result files.", get_ablation_summary),
        ("get_congestion_subset_eval", "Locate congestion subset evaluation files.", get_congestion_subset_eval),
        ("get_failure_cases", "Return worst forecast cases.", get_failure_cases),
        ("explain_why_last_value_is_strong", "Explain why LastValue is a strong traffic baseline.", explain_why_last_value_is_strong),
        ("explain_raw_mape_issue", "Explain raw MAPE failure mode.", explain_raw_mape_issue),
        ("generate_resume_bullets", "Generate resume-safe bullets from local results.", generate_resume_bullets),
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
