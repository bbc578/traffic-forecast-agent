# AGENTS.md

## 项目开发约定

- 使用 Python 3.11+，代码位于 `src/traffic_agent/`。
- 保持模块清晰，避免把训练、分析、API、Dashboard 混在一个文件中。
- 使用类型注解和关键函数 docstring。
- 不要虚构实验指标；所有指标必须由训练或评估代码实际计算生成。
- synthetic demo data 必须明确标注为流程演示数据，不代表真实交通预测效果。
- Agent 只能读取和总结本地输出，不可描述为真实交通控制系统。

## 常用命令

```bash
pytest
ruff check .
python -m traffic_agent.data.generate_synthetic --output data/sample/synthetic_traffic.npz
python -m traffic_agent.training.train --config configs/demo.yaml --model historical_average
python -m traffic_agent.training.train --config configs/demo.yaml --model lstm
python -m traffic_agent.training.train --config configs/demo.yaml --model stgcn_lite
```

## 修改要求

- 修改代码后必须尽量运行相关测试。
- 新增功能必须更新 README。
- 如果某功能因环境限制无法完成，最终总结中必须明确说明，不要假装完成。
