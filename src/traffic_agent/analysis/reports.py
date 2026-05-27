from __future__ import annotations

from typing import Any

DISCLAIMER = (
    "本报告基于公开/模拟交通数据和简化模型生成，仅用于学习和实习作品展示，"
    "不可直接用于真实交通管控决策。"
)


def generate_markdown_report(
    metrics: dict[str, Any],
    congestion_nodes: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
) -> str:
    """Generate a Chinese markdown daily report from computed artifacts."""
    lines = [
        "# 城市路网交通态势日报",
        "",
        f"> {DISCLAIMER}",
        "",
        "## 模型指标",
        "",
        f"- 模型：{metrics.get('model_name', 'unknown')}",
        f"- 预测窗口：{metrics.get('horizon', 'unknown')}",
        f"- MAE：{metrics.get('mae', 'unknown')}",
        f"- RMSE：{metrics.get('rmse', 'unknown')}",
        f"- MAPE：{metrics.get('mape', 'unknown')}",
        "",
        "## 拥堵风险 Top 节点",
        "",
    ]
    if congestion_nodes:
        for item in congestion_nodes:
            lines.append(
                "- "
                f"{item['node_id']}：{item['risk_level']}，"
                f"最低预测速度 {item['predicted_min_speed']}，"
                f"原因：{item['reason']}"
            )
    else:
        lines.append("- 当前 run 没有可用的拥堵风险结果。")

    lines.extend(["", "## 异常提示", ""])
    if anomalies:
        for item in anomalies[:10]:
            lines.append(
                "- "
                f"{item['node_id']}，样本 {item['sample_index']}，"
                f"预测步 {item['time_step']}，异常分数 {item['anomaly_score']}"
            )
    else:
        lines.append("- 未检测到超过阈值的残差异常。")

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 拥堵识别基于速度阈值和速度下降幅度，是风险提示，不是事故判断。",
            "- 异常检测使用残差 z-score 启发式规则，不是事故检测模型。",
        ]
    )
    return "\n".join(lines)
