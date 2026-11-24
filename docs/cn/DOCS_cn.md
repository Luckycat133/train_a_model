# 灵猫墨韵项目文档

[English](../en/DOCS_en.md) | [中文](./DOCS_cn.md) | [文档总索引](../README.md)

---

## 文档概览

本页汇总灵猫墨韵大语言模型项目现有的中文及相关核心文档入口。项目当前状态、验证入口与权重边界请先阅读根目录 [README.md](../../README.md)。

## 入门与架构

- [快速入门](./getting-started.md) — 克隆仓库、安装依赖与基本运行流程
- [安装指南](./installation_guide.md) — 不同操作系统的安装步骤与环境配置
- [模块架构说明（英文）](../en/architecture.md) — `src/` 目录模块化架构概览
- [最新 LLM 架构调研报告（英文）](../en/research-latest-llm-architecture.md) — 2025–2026 LLM 架构调研

## 数据与实验文档

- [数据文档总览](../data/README.md) — 古典文献数据目录与入口
- [数据获取与预处理指南](../data/getting_started.md) — 训练数据下载与预处理步骤
- [数据格式说明](../data/formats.md) — JSONL 格式与字段规范
- [实验平台使用指南](../experiments/README.md) — 预训练、SFT 微调与评估配置说明
- [实验快速开始指南](../experiments/quickstart.md) — 三步启动训练实验

## 工程报告与最佳实践

- [实验报告与归档说明](../../EXPERIMENT_REPORT.md) — 最终结论、已知缺口与归档条件
- [数据集完整说明](../../DATASET.md) — 数据集构成与来源记录
- [训练与工程实践](../../BEST_PRACTICES.md) — 训练优化与工程实践记录
- [权重目录说明](../../model_weights/README.md) — 权重边界与外部发布要求

## 模块更新日志

- [训练模块更新](../../changelog/cn/for_train_model.md) — 训练相关功能更新记录
- [处理器更新](../../changelog/cn/for_processor.md) — 处理器功能更新历史
- [分词器更新](../../changelog/cn/for_tokenizer.md) — 分词器迭代记录
- [数据模块更新](../../changelog/cn/for_data.md) — 数据处理模块更新历史
- [清理工具更新](../../changelog/cn/for_cleanup.md) — 清理工具功能更新历史

## 文档索引与配置参考

- 完整文档目录见 [docs/README.md](../README.md)。
- 配置文件示例请参阅 [config/config.yaml](../../config/config.yaml)。

## 当前文档目录结构

```text
docs/
├── README.md                              # 文档总索引
├── cn/
│   ├── DOCS_cn.md                         # 本文档（中文导航）
│   ├── getting-started.md                 # 快速入门
│   └── installation_guide.md              # 安装指南
├── en/
│   ├── DOCS_en.md                         # English documentation hub
│   ├── architecture.md                    # Modular src/ architecture
│   ├── getting-started.md                 # Getting Started
│   ├── installation_guide.md              # Installation Guide
│   └── research-latest-llm-architecture.md
├── data/
│   ├── README.md                          # 数据文档总览
│   ├── getting_started.md                 # 数据获取与预处理指南
│   └── formats.md                         # 数据格式说明
└── experiments/
    ├── README.md                          # 实验平台使用指南
    └── quickstart.md                      # 实验快速开始指南
```
