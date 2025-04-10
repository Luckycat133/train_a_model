# My LLM Project | 我的大语言模型项目

[English](./docs/en/README.md) | [中文](./README.md)

## 项目概述

这是一个用于训练和开发大语言模型的项目，包含了完整的数据处理、分词、训练和评估流程。项目采用模块化设计，便于扩展和优化。

## 主要功能

- 数据预处理和清洗
- 自定义分词器训练
- 模型训练和优化
- 性能评估和质量控制

## 项目结构

```
├── changelog/       # 更新日志
├── collection/      # 数据收集
├── config/         # 配置文件
├── dataset/        # 数据集
├── docs/           # 项目文档
├── model_weights/  # 模型权重
├── processors/     # 数据处理器
└── test/           # 测试用例
```

## 快速开始

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 配置环境
请参考 [安装指南](./docs/installation/installation_guide.md)

3. 运行训练
```bash
python train_model.py
```

## 文档

- [项目概览](./docs/project_overview.md)
- [目录结构说明](./docs/standards/directory_structure.md)
- [更新日志](./changelog/)

## 许可证

本项目基于 MIT 许可证开源。
