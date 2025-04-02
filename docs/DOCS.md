# 灵猫墨韵项目文档

## 文档概览

本项目文档提供了灵猫墨韵大语言模型项目的完整指南，包括各个模块的使用方法、训练流程、生成方法等内容。

## 模块指南

- [项目概述](docs/project_overview.md) - 项目整体介绍和架构说明
- [训练模块指南](docs/train_model_guide.md) - 模型训练方法与参数配置
- [生成模块指南](docs/generate_guide.md) - 使用模型进行文本生成
- [处理器指南](docs/processor_guide.md) - 数据预处理工具使用说明
- [分词器指南](docs/tokenizer_guide.md) - 分词器训练与使用方法
- [脚本使用指南](docs/scripts_guide.md) - 辅助脚本功能说明
- [清理工具指南](docs/cleanup_guide.md) - 项目清理与空间优化工具
- [项目维护指南](docs/project_maintenance.md) - 项目体积管理和清理方法

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
│   ├── train_model_guide.md # 训练模块指南
│   ├── generate_guide.md    # 生成模块指南
│   ├── processor_guide.md   # 处理器指南
│   ├── tokenizer_guide.md   # 分词器指南
│   ├── scripts_guide.md     # 脚本说明
│   ├── cleanup_guide.md     # 清理工具指南
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
- [docs/train_model_guide.md](./docs/train_model_guide.md) - 详细的模型训练指南
- [docs/generate_guide.md](./docs/generate_guide.md) - 文本生成功能使用指南
- [docs/cleanup_guide.md](./docs/cleanup_guide.md) - 项目清理与空间优化指南
- [docs/INDEX.md](./docs/INDEX.md) - 文档完整索引

## 版本更新日志

- [changelog/for_train_model.md](./changelog/for_train_model.md) - 训练模块更新日志
- [changelog/for_cleanup.md](./changelog/for_cleanup.md) - 清理工具更新日志
- [changelog/for_processor.md](./changelog/for_processor.md) - 处理器模块更新日志

## 最近更新

- **[2025-03-14]** 处理器系统 v2.3.0 发布，增加分词器训练中断处理机制
- **[2025-03-14]** 清理工具 v0.3.1 发布，完善用户文档和系统集成
- **[2025-03-14]** 清理工具 v0.3.0 发布，增加文件保护和空间分析功能
- **[2025-03-14]** 处理器系统 v2.2.2 发布，添加无参数自动执行完整处理流程功能
- **[2023-03-14]** 训练系统 v0.7.0 发布，增加自动保存恢复功能和夜间低功耗模式

## 额外资源

- 安装依赖: `pip install -r requirements.txt`
- 问题反馈和贡献: 请提交Issue或Pull Request

---

*文档最后更新: 2025年3月14日* 