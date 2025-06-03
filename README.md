# 数据仓库与数据挖掘综合实践

本项目基于 AdventureWorksDW 数据仓库，按照 CRISP-DM 流程，完成客户流失预测、季节性产品关联规则挖掘、澳元客户RFM分析等数据挖掘任务。

## 目录结构

```
project/
├── data/        # 数据下载、缓存、处理文件
├── notebooks/   # Jupyter分析与可视化（可选）
├── src/         # 主要Python脚本
├── reports/     # 分析报告、图表
├── README.md    # 项目说明
└── pyproject.toml # 依赖管理
```

## 主要任务
1. 客户流失预测与留存策略
2. 季节性产品关联规则挖掘与促销建议
3. 澳元客户RFM模型群体划分与服务方案

## 环境说明
- Python 3.11
- 依赖通过 uv + pyproject.toml 管理
- SQL Server 2019 (docker)

## 数据库连接信息
- host: localhost
- port: 8433
- username: sa
- password: Alaska2017
- database: AdventureWorksDW2017

## 使用说明
所有核心分析代码位于 src/ 目录下，运行前请确保已安装依赖并正确配置数据库环境。
