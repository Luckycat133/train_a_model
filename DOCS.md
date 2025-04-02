# 灵猫墨韵项目文档

## 文档概览

本项目文档提供了灵猫墨韵大语言模型项目的完整指南，包括各个模块的使用方法、训练流程、生成方法等内容。

## 模块指南

- [项目概述](docs/project_overview.md) - 项目整体介绍和架构说明
- [技术架构](docs/technical_architecture.md) - 技术架构详细说明与数据流
- [系统架构设计](docs/design_docs/system_architecture.md) - 系统架构详细设计与模块关系
- [安装指南](docs/installation/installation_guide.md) - 不同系统的安装步骤与环境配置
- [训练模块指南](docs/train_model_guide.md) - 模型训练方法与参数配置
- [生成模块指南](docs/generate_guide.md) - 使用模型进行文本生成
- [处理器指南](docs/processor_guide.md) - 数据预处理工具使用说明
- [分词器指南](docs/tokenizer_guide.md) - 分词器训练与使用方法
- [脚本使用指南](docs/scripts_guide.md) - 辅助脚本功能说明
- [清理工具指南](docs/cleanup_guide.md) - 项目清理与空间优化工具
- [项目维护指南](docs/project_maintenance.md) - 项目体积管理和清理方法

## 最佳实践指南

- [训练优化最佳实践](docs/best_practices/training_optimization.md) - 模型训练优化指南
  - 硬件优化策略
  - 超参数调优指南
  - 训练稳定性优化
  - 性能监控与调优

## 代码详解

- [训练模块代码解析](docs/code_explanation/train_model_code.md) - 训练模块核心代码详解
- [分词器代码解析](docs/code_explanation/tokenizer_code.md) - 分词器模块实现分析
- [处理器代码解析](docs/code_explanation/processor_code.md) - 处理器模块架构解析

## 更新日志

- [训练模块更新](changelog/for_train_model.md) - 训练相关功能更新记录
- [处理器更新](changelog/for_processor.md) - 处理器功能更新历史
- [分词器更新](changelog/for_tokenizer.md) - 分词器迭代记录
- [数据模块更新](changelog/for_data.md) - 数据处理模块更新历史
- [清理工具更新](changelog/for_cleanup.md) - 清理工具功能更新历史

## 文档索引

所有文档的详细目录可在[文档索引](docs/INDEX.md)中查找。

## 配置参考

项目配置文件示例及详细参数说明请参阅[配置示例](config/config.yaml)。

## 文档结构

```
My_LLM/
├── README.md           # 项目主要介绍
├── DOCS.md             # 本文档（导航指南）
├── docs/               # 文档主目录
│   ├── project_overview.md  # 项目详细说明
│   ├── technical_architecture.md # 技术架构说明
│   ├── train_model_guide.md # 训练模块指南
│   ├── generate_guide.md    # 生成模块指南
│   ├── processor_guide.md   # 处理器指南
│   ├── tokenizer_guide.md   # 分词器指南
│   ├── scripts_guide.md     # 脚本说明
│   ├── cleanup_guide.md     # 清理工具指南
│   ├── project_maintenance.md # 项目维护指南
│   ├── design_docs/         # 设计文档目录
│   │   └── system_architecture.md # 系统架构设计
│   ├── installation/        # 安装指南目录
│   │   └── installation_guide.md # 安装指南
│   ├── best_practices/      # 最佳实践指南目录
│   │   └── training_optimization.md # 训练优化指南
│   ├── code_explanation/    # 代码解析目录
│   │   ├── train_model_code.md  # 训练模块代码解析
│   │   ├── tokenizer_code.md    # 分词器代码解析
│   │   └── processor_code.md    # 处理器代码解析
│   └── INDEX.md        # 文档索引
└── changelog/          # 更新日志目录
    ├── for_train_model.md  # 训练模块更新日志
    ├── for_processor.md    # 处理器更新日志
    ├── for_tokenizer.md    # 分词器更新日志
    ├── for_data.md         # 数据模块更新日志
    └── for_cleanup.md      # 清理工具更新日志
```

## 核心文档

- [README.md](./README.md) - 项目概述、快速入门和主要功能介绍
- [docs/project_overview.md](./docs/project_overview.md) - 项目详细说明文档
- [docs/technical_architecture.md](./docs/technical_architecture.md) - 技术架构详解
- [docs/design_docs/system_architecture.md](./docs/design_docs/system_architecture.md) - 系统架构设计
- [docs/installation/installation_guide.md](./docs/installation/installation_guide.md) - 安装指南
- [docs/train_model_guide.md](./docs/train_model_guide.md) - 详细的模型训练指南
- [docs/generate_guide.md](./docs/generate_guide.md) - 文本生成功能使用指南
- [docs/best_practices/training_optimization.md](./docs/best_practices/training_optimization.md) - 训练优化最佳实践
- [docs/INDEX.md](./docs/INDEX.md) - 文档完整索引

## 版本更新日志

- [changelog/for_train_model.md](./changelog/for_train_model.md) - 训练模块更新日志
- [changelog/for_cleanup.md](./changelog/for_cleanup.md) - 清理工具更新日志

## 最近更新

- **[2025-03-18]** 添加详细安装指南，支持各主要操作系统
- **[2025-03-18]** 添加系统架构设计文档，详细说明系统模块关系和数据流
- **[2025-03-18]** 添加训练优化最佳实践指南，提供硬件优化与超参数调优建议
- **[2025-03-18]** 整理文档结构，添加代码解析与技术架构文档
- **[2025-03-16]** 训练系统 v0.8.0 发布，增强稳定性与兼容性

## 额外资源

- 安装依赖: `pip install -r requirements.txt`
- 问题反馈和贡献: 请提交Issue或Pull Request

---

*文档最后更新: 2025年3月18日* 