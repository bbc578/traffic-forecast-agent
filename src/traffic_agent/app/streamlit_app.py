from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from traffic_agent.agent.rule_based_agent import RuleBasedTrafficAgent
from traffic_agent.agent.tools import generate_daily_report, get_top_congestion_nodes

OUTPUTS_DIR = Path("outputs")


def list_runs() -> list[str]:
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(path.name for path in OUTPUTS_DIR.iterdir() if (path / "metrics.json").exists())


def load_metrics(run_name: str) -> dict[str, object]:
    with (OUTPUTS_DIR / run_name / "metrics.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(run_name: str) -> dict[str, object]:
    with np.load(OUTPUTS_DIR / run_name / "predictions.npz", allow_pickle=False) as loaded:
        return {
            "y_true": loaded["y_true"],
            "y_pred": loaded["y_pred"],
            "node_ids": [str(item) for item in loaded["node_ids"].tolist()],
        }


def main() -> None:
    st.set_page_config(page_title="Traffic Forecast Agent", layout="wide")
    st.title("城市路网交通态势预测与拥堵分析智能体系统")
    st.warning(
        "当前项目为学习/实习作品 Demo。若使用 synthetic 数据，结果仅代表流程演示，"
        "不代表真实城市交通状况。"
    )

    runs = list_runs()
    if not runs:
        st.info("尚未发现 run。请先运行训练命令生成 outputs/<run_name>/。")
        st.code("python -m traffic_agent.training.train --config configs/demo.yaml --model historical_average")
        return

    run_name = st.sidebar.selectbox("选择 run", runs)
    metrics = load_metrics(run_name)
    predictions = load_predictions(run_name)

    st.subheader("模型指标")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{float(metrics['mae']):.4f}")
    col2.metric("RMSE", f"{float(metrics['rmse']):.4f}")
    col3.metric("MAPE", f"{float(metrics['mape']):.4f}%")

    st.subheader("预测 vs 真实曲线")
    node_ids = predictions["node_ids"]
    selected_node = st.selectbox("选择传感器节点", node_ids)
    node_index = node_ids.index(selected_node)
    y_true = predictions["y_true"][:, :, node_index].reshape(-1)
    y_pred = predictions["y_pred"][:, :, node_index].reshape(-1)
    max_points = min(240, len(y_true))
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true[:max_points], mode="lines", name="True speed"))
    fig.add_trace(go.Scatter(y=y_pred[:max_points], mode="lines", name="Predicted speed"))
    fig.update_layout(xaxis_title="Test horizon samples", yaxis_title="Speed")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top-K 拥堵风险节点")
    top_k = st.slider("Top-K", min_value=1, max_value=20, value=5)
    congestion = get_top_congestion_nodes(str(OUTPUTS_DIR / run_name), k=top_k)
    st.dataframe(pd.DataFrame(congestion), use_container_width=True)

    st.subheader("Agent 问答")
    query = st.text_input("输入中文问题", value="未来哪里最堵？")
    if st.button("提交问题"):
        agent = RuleBasedTrafficAgent(outputs_dir=str(OUTPUTS_DIR))
        response = agent.query(query, run_name=run_name)
        st.write(response.answer)
        if response.data is not None:
            st.json(response.data)

    st.subheader("日报")
    if st.button("生成日报"):
        st.markdown(generate_daily_report(str(OUTPUTS_DIR / run_name)))


if __name__ == "__main__":
    main()
