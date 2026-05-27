# Urban Traffic Forecasting & Congestion Analysis Agent

中文名：城市路网交通态势预测与拥堵分析智能体系统

这是一个面向智慧交通方向学生的实习作品项目。项目核心是“预测模型 + 拥堵风险分析 + Dashboard + Rule-based Agent”：使用交通传感器时序数据预测未来速度，基于预测结果识别潜在拥堵路段，并用可视化页面和轻量问答层展示结果。

本项目不是完整工业系统，不用于真实交通管控决策。仓库默认使用 synthetic demo data，仅用于流程演示，不代表真实城市交通状况或真实交通预测效果。

## 项目亮点

- 时序交通预测：基于滑动窗口构造未来 15/30/60 分钟速度预测任务。
- 简化时空图模型：提供 STGCN-inspired / STGCN-lite，用于作品集演示，不是完整论文复现。
- 拥堵风险分析：基于预测速度阈值和速度下降幅度输出 Top-K 风险节点。
- 可视化 Dashboard：用 Streamlit 展示指标、预测曲线、风险表和日报。
- 可解释规则 Agent：不接外部 LLM API，只读取本地 `outputs/` 中实际生成的数据。
- 测试与 CI：提供 pytest、ruff 和 GitHub Actions。

## 快速开始

Python 版本要求：3.11+

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

生成 demo 数据：

```bash
python -m traffic_agent.data.generate_synthetic --output data/sample/synthetic_traffic.npz
```

训练或评估模型：

```bash
python -m traffic_agent.training.train --config configs/demo.yaml --model historical_average
python -m traffic_agent.training.train --config configs/demo.yaml --model lstm
python -m traffic_agent.training.train --config configs/demo.yaml --model stgcn_lite
```

启动 Dashboard：

```bash
streamlit run src/traffic_agent/app/streamlit_app.py
```

启动 FastAPI：

```bash
uvicorn traffic_agent.api.main:app --reload
```

如果你没有安装为 editable 包，可以临时设置：

```bash
export PYTHONPATH=src
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
```

## 数据说明

默认数据文件：`data/sample/synthetic_traffic.npz`

字段：

- `data`: 形状为 `[T, N, F]` 的时序数组，默认 7 天、5 分钟粒度、20 个节点。
- `adjacency`: 形状为 `[N, N]` 的路网邻接矩阵。
- `feature_names`: 特征名，至少包含 `speed` 和 `flow`。
- `node_ids`: 节点 ID。
- `is_synthetic_data`: 是否为合成 demo 数据。

合成数据包含早晚高峰速度下降、速度与流量负相关、相邻节点相关性、少量异常拥堵事件和少量缺失值。它只用于演示工程流程，不代表真实交通预测效果。

如需替换为 PeMS/METR-LA/PEMS-BAY 风格真实数据，请自行准备 `.npz` 文件，并保持上述 schema 一致。仓库默认不包含大型真实数据集，也不依赖必须登录或人工下载的外部资源。

## 模型说明

- Historical Average：使用历史窗口内平均速度作为未来预测，不需要训练。
- LSTM：将所有节点和特征 flatten 成时间序列向量，输出未来 horizon 个时间步所有节点速度。
- STGCN-lite：STGCN-inspired 简化模型，使用邻接矩阵做消息传递，再用 GRU 处理时间维度。它是作品集演示版，不是完整 STGCN 论文复现。

训练脚本会自动检测 CUDA；没有 GPU 时使用 CPU。默认 epoch 较少，便于普通笔记本跑通。

## 指标说明

评估指标由本地训练脚本实际计算并保存到 `outputs/<run_name>/metrics.json`：

- MAE：平均绝对误差。
- RMSE：均方根误差。
- MAPE：平均绝对百分比误差，计算时使用 epsilon 避免除以 0。

这些指标不保证在真实数据上达到某个水平。请使用自己运行得到的结果，不要在简历或页面中填写未经验证的提升百分比。

## Agent 说明

当前 Agent 是 rule-based agent，不接外部 LLM API。它根据中文问题识别意图，然后调用本地工具读取 `outputs/` 中实际存在的指标、预测和拥堵风险结果。

支持示例：

- “未来哪里最堵？”
- “模型效果怎么样？”
- “生成日报”
- “比较模型”
- “有哪些异常？”

Agent 只能作为分析助手，不能直接控制交通信号灯或做真实交通控制决策。后续可以在相同工具接口上扩展为 LLM tool-calling agent。

## 项目结构

```text
traffic-forecast-agent/
├── configs/
├── data/
├── outputs/
├── src/traffic_agent/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── analysis/
│   ├── agent/
│   ├── api/
│   └── app/
├── tests/
└── .github/workflows/
```

## 测试

```bash
pytest
ruff check .
```

GitHub Actions 会安装 `requirements.txt`，运行 `ruff check .` 和 `pytest`。CI 不依赖大型真实数据集。

## 简历写法建议

- 构建城市路网交通态势预测 Demo，基于合成/PeMS 风格传感器数据完成滑动窗口构造、时间顺序划分、标准化和多模型预测评估。
- 实现 Historical Average、LSTM 与 STGCN-lite 三类模型，并使用 MAE/RMSE/MAPE 对未来 15/30/60 分钟速度预测进行评估。
- 设计规则型交通分析 Agent，支持自然语言查询模型指标、拥堵风险节点和自动生成交通运行日报。

请用自己实际运行得到的指标替换简历中的实验结果，不要填写虚假的提升百分比或无法验证的效果描述。

## 局限

- 默认数据是 synthetic demo data，只能验证流程。
- 拥堵识别和异常检测是启发式规则，不是事故检测模型。
- STGCN-lite 是简化模型，不是完整 STGCN 论文复现。
- Dashboard 和 API 面向本地作品展示，不是生产级部署方案。
