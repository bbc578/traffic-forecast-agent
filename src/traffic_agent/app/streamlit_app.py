from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from traffic_agent.agent.executor import AgentExecutor
from traffic_agent.agent.tools import generate_daily_report, get_top_congestion_nodes
from traffic_agent.analysis.error_analysis import (
    error_by_horizon,
    error_by_node,
    error_by_time_of_day,
    residual_distribution,
    worst_case_segments,
)
from traffic_agent.analysis.graph_analysis import adjacency_stats, top_connected_nodes
from traffic_agent.data.loader import load_traffic_npz

OUTPUTS_DIR = Path("outputs")


def list_runs() -> list[str]:
    return sorted(path.name for path in OUTPUTS_DIR.glob("*/metrics.json")) if OUTPUTS_DIR.exists() else []


def load_metrics(run_name: str) -> dict[str, Any]:
    return json.loads((OUTPUTS_DIR / run_name / "metrics.json").read_text(encoding="utf-8"))


def load_predictions(run_name: str) -> dict[str, Any]:
    with np.load(OUTPUTS_DIR / run_name / "predictions.npz", allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def metric_card(label: str, value: Any) -> None:
    st.metric(label, f"{float(value):.4f}" if isinstance(value, int | float) else value)


def show_empty_state() -> None:
    st.info("尚未发现 run。先运行 synthetic smoke train，或准备真实数据后训练。")
    st.code(
        "python -m traffic_agent.data.generate_synthetic --output data/sample/synthetic_traffic.npz\n"
        "python -m traffic_agent.training.train --config configs/demo.yaml --model last_value"
    )


def prediction_figure(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> go.Figure:
    max_points = min(240, len(y_true))
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true[:max_points], mode="lines", name="True speed"))
    fig.add_trace(go.Scatter(y=y_pred[:max_points], mode="lines", name="Predicted speed"))
    fig.update_layout(title=title, xaxis_title="Test horizon samples", yaxis_title="Speed")
    return fig


def main() -> None:
    st.set_page_config(page_title="Traffic Forecast Agent", layout="wide")
    st.title("Urban Traffic Forecasting & Congestion Analysis Agent")
    st.caption("Algorithm + diagnostics + agent trace + visualization. Offline portfolio system, not traffic control.")

    runs = list_runs()
    if not runs:
        show_empty_state()
        return

    run_name = st.sidebar.selectbox("Run", runs)
    metrics = load_metrics(run_name)
    predictions = load_predictions(run_name)
    node_ids = [str(item) for item in predictions["node_ids"].tolist()]
    is_synthetic = bool(metrics.get("is_synthetic_data"))
    if is_synthetic:
        st.warning("当前 run 使用 synthetic demo data：只代表流程演示，不代表真实城市交通状况或真实预测效果。")

    tabs = st.tabs(
        [
            "Overview",
            "Data Explorer",
            "Network View",
            "Forecast Explorer",
            "Model Comparison",
            "Error Diagnostics",
            "Congestion Risk",
            "Horizon Analysis",
            "Ablation Analysis",
            "Subset Evaluation",
            "GPU Training Monitor",
            "Agent Console",
            "Report",
        ]
    )

    with tabs[0]:
        st.subheader("Run Overview")
        cols = st.columns(6)
        for col, (label, key) in zip(
            cols,
            [
                ("Dataset", "dataset_name"),
                ("Model", "model_name"),
                ("Horizon", "horizon"),
                ("MAE", "mae"),
                ("RMSE", "rmse"),
                ("MAPE", "mape"),
            ],
            strict=False,
        ):
            with col:
                metric_card(label, metrics.get(key, "unknown"))
        st.json(
            {
                "synthetic": metrics.get("is_synthetic_data"),
                "adjacency_type": metrics.get("adjacency_type"),
                "num_nodes": metrics.get("num_nodes"),
                "num_features": metrics.get("num_features"),
                "train_time_range": metrics.get("train_time_range"),
                "test_time_range": metrics.get("test_time_range"),
            }
        )
        st.caption("解读：先确认数据类型、预测窗口和 split，再比较误差；synthetic run 不能作为真实效果证明。")

    with tabs[1]:
        st.subheader("Data Explorer")
        data_path = metrics.get("dataset_path")
        if data_path and Path(str(data_path)).exists():
            data, _, metadata = load_traffic_npz(str(data_path))
            selected_node = st.selectbox("Node", metadata["node_ids"], key="data_node")
            node_idx = metadata["node_ids"].index(selected_node)
            speed = data[:, node_idx, 0]
            speed_fig = px.line(y=speed[: min(1000, len(speed))], labels={"x": "time", "y": "speed"})
            st.plotly_chart(speed_fig, use_container_width=True)
            st.write({"missing_ratio": float(np.isnan(data).mean()), "feature_names": metadata["feature_names"]})
            st.caption("解读：真实数据实验前先看缺失率、日内模式和异常速度段，避免直接相信总指标。")
        else:
            st.info("本地没有找到 dataset_path，仍可查看已保存的 prediction artifacts。")

    with tabs[2]:
        st.subheader("Network View")
        data_path = metrics.get("dataset_path")
        if data_path and Path(str(data_path)).exists():
            _, adjacency, metadata = load_traffic_npz(str(data_path))
            stats = adjacency_stats(adjacency)
            st.json(stats)
            connected = pd.DataFrame(top_connected_nodes(adjacency, metadata["node_ids"], k=10))
            st.dataframe(connected, use_container_width=True)
            degrees = adjacency.astype(bool).sum(axis=1)
            angles = np.linspace(0, 2 * np.pi, len(degrees))
            fig = px.scatter(
                x=np.cos(angles),
                y=np.sin(angles),
                size=degrees + 1,
                color=degrees,
                hover_name=metadata["node_ids"],
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("解读：没有经纬度时使用 circular layout。节点颜色/大小表示连接度，不代表真实地理位置。")
        else:
            st.info("需要本地数据文件才能展示 adjacency 结构。")

    with tabs[3]:
        st.subheader("Forecast Explorer")
        selected_node = st.selectbox("Node", node_ids, key="forecast_node")
        horizon_step = st.slider("Horizon step", 1, int(metrics.get("horizon", predictions["y_pred"].shape[1])), 1)
        node_idx = node_ids.index(selected_node)
        y_true = predictions["y_true"][:, horizon_step - 1, node_idx]
        y_pred = predictions["y_pred"][:, horizon_step - 1, node_idx]
        forecast_fig = prediction_figure(y_true, y_pred, f"{selected_node} horizon={horizon_step}")
        st.plotly_chart(forecast_fig, use_container_width=True)
        residual_fig = px.line(y=(y_pred - y_true)[:240], labels={"x": "sample", "y": "residual"})
        st.plotly_chart(residual_fig, use_container_width=True)
        st.caption("解读：看预测曲线是否跟随趋势，也要看残差是否集中在高峰或突发下降段。")

    with tabs[4]:
        st.subheader("Model Comparison")
        rows = []
        for candidate in runs:
            row = load_metrics(candidate) | {"run_name": candidate}
            rows.append(row)
        frame = pd.DataFrame(rows)
        columns = ["run_name", "model_name", "horizon", "mae", "rmse", "mape", "is_synthetic_data"]
        st.dataframe(frame[columns], use_container_width=True)
        comparison_fig = px.bar(frame, x="model_name", y="mae", color="horizon", hover_name="run_name")
        st.plotly_chart(comparison_fig, use_container_width=True)
        if "last_value" in set(frame["model_name"]):
            deep_rows = frame[~frame["model_name"].isin(["last_value", "historical_average", "seasonal_naive"])]
            best_deep = deep_rows["mae"].min()
            last_value = frame[frame["model_name"] == "last_value"]["mae"].min()
            if pd.notna(best_deep) and best_deep >= last_value:
                st.warning("深度模型没有超过 LastValue baseline，不能声称模型预测能力优于强 naive baseline。")
        st.caption("解读：交通短时预测必须先看 LastValue；只看深度模型之间的差异是不完整的。")

    with tabs[5]:
        st.subheader("Error Diagnostics")
        y_true = predictions["y_true"]
        y_pred = predictions["y_pred"]
        st.plotly_chart(px.bar(error_by_horizon(y_true, y_pred), x="horizon_step", y="mae"), use_container_width=True)
        node_error = error_by_node(y_true, y_pred, node_ids)
        st.plotly_chart(px.bar(node_error.head(20), x="node_id", y="mae"), use_container_width=True)
        timestamps = [str(item) for item in predictions["timestamps"].tolist()] if "timestamps" in predictions else None
        tod = error_by_time_of_day(y_true, y_pred, timestamps)
        st.plotly_chart(px.bar(tod, x="hour", y="mae"), use_container_width=True)
        st.write("Residual distribution", residual_distribution(y_true, y_pred))
        st.dataframe(pd.DataFrame(worst_case_segments(y_true, y_pred, node_ids, top_k=20)), use_container_width=True)
        st.caption("解读：高误差 horizon、节点和时间段比单个总 MAE 更能说明模型失败模式。")

    with tabs[6]:
        st.subheader("Congestion Risk")
        top_k = st.slider("Top-K", 1, 20, 5)
        congestion = get_top_congestion_nodes(str(OUTPUTS_DIR / run_name), k=top_k)
        st.dataframe(pd.DataFrame(congestion), use_container_width=True)
        st.caption("解读：风险来自预测速度阈值和历史下降幅度，是离线拥堵风险提示，不是事故检测。")

    with tabs[7]:
        st.subheader("Horizon Analysis")
        files = sorted(Path("experiments/results").glob("*summary*.csv"))
        if files:
            selected = st.selectbox("Experiment summary CSV", [str(path) for path in files])
            frame = pd.read_csv(selected)
            st.dataframe(frame, use_container_width=True)
            if {"horizon", "model_name", "mae"}.issubset(frame.columns):
                st.plotly_chart(px.line(frame, x="horizon", y="mae", color="model_name", markers=True), use_container_width=True)
            st.caption("解读：较长 horizon 更能检验时空模型是否捕捉传播动态；不要只看 h=3。")
        else:
            st.info("缺少多 horizon 结果。运行 README 中的 metr_la_5090_full_summary 命令。")

    with tabs[8]:
        st.subheader("Ablation Analysis")
        files = sorted(Path("experiments/results").glob("*ablation*.csv"))
        if files:
            selected = st.selectbox("Ablation CSV", [str(path) for path in files])
            frame = pd.read_csv(selected)
            st.dataframe(frame, use_container_width=True)
            if {"graph_type", "mae", "model_name"}.issubset(frame.columns):
                st.plotly_chart(px.bar(frame, x="graph_type", y="mae", color="model_name", barmode="group"), use_container_width=True)
            st.caption("解读：只有 physical/adaptive 稳定优于 identity 时，才可以说图结构在该设置下有收益。")
        else:
            st.info("缺少图结构消融结果。运行 `python -m traffic_agent.training.run_ablation ...`。")

    with tabs[9]:
        st.subheader("Congestion Subset Evaluation")
        files = sorted(Path("experiments/results").glob("*congestion_subset*.csv"))
        if files:
            selected = st.selectbox("Subset CSV", [str(path) for path in files])
            frame = pd.read_csv(selected)
            st.dataframe(frame, use_container_width=True)
            if {"subset", "mae", "model_name"}.issubset(frame.columns):
                st.plotly_chart(px.bar(frame, x="subset", y="mae", color="model_name", barmode="group"), use_container_width=True)
            st.caption("解读：如果图模型在 speed_drop/low_speed 子集更好，才是拥堵场景价值证据。")
        else:
            st.info("缺少拥堵子集评估结果。")

    with tabs[10]:
        st.subheader("GPU Training Monitor")
        log_path = OUTPUTS_DIR / run_name / "train_log.csv"
        if log_path.exists():
            log = pd.read_csv(log_path)
            st.dataframe(log, use_container_width=True)
            if {"epoch", "train_loss", "val_loss"}.issubset(log.columns):
                melted = log.melt(id_vars=["epoch"], value_vars=["train_loss", "val_loss"])
                st.plotly_chart(px.line(melted, x="epoch", y="value", color="variable"), use_container_width=True)
            st.caption("解读：关注 best_epoch、val_loss 平台期、epoch time 和 GPU memory。")
        else:
            st.info("该 run 没有 train_log.csv。请用新版训练脚本重跑。")

    with tabs[11]:
        st.subheader("Agent Console")
        query = st.text_input("中文问题", value="哪个模型效果最好？")
        if st.button("Run Agent"):
            response = AgentExecutor(outputs_dir=str(OUTPUTS_DIR)).run(query, run_name=run_name)
            st.write(response.answer)
            st.write(
                {
                    "intent": response.intent,
                    "confidence": response.confidence,
                    "tools_used": response.tools_used,
                    "trace_id": response.trace_id,
                }
            )
            st.json(response.data)
            st.info("Limitations: " + " ".join(response.limitations))
        st.caption("解读：Agent 只做工具选择、执行和解释；trace 记录了计划、工具输入输出和数据来源。")

    with tabs[12]:
        st.subheader("Report")
        report = generate_daily_report(str(OUTPUTS_DIR / run_name))
        st.markdown(report)
        st.download_button("Download Markdown", report, file_name=f"{run_name}_report.md")
        st.caption("解读：报告可作为作品集材料草稿，但真实数据指标必须由你自己跑出后再填写。")


if __name__ == "__main__":
    main()
