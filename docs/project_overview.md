# 灵猫墨韵 - 古典中文语言模型

![版本](https://img.shields.io/badge/版本-v0.7.0-blue)
![更新日期](https://img.shields.io/badge/更新日期-2023--03--13-green)

## 项目简介

灵猫墨韵是一个专注于中国古典文学的语言模型，旨在理解和生成古典汉语文本。本项目包含数据处理、模型训练、评估和文本生成等完整流程。

## 快速开始

训练模型（已配置最优参数，自动检测并使用GPU）：

```bash
python train_model.py
```

## 文档指引

详细文档位于 `docs` 目录：

- [完整文档入口](./docs/README.md)
- [训练指南](./docs/TRAINING.md)
- [脚本说明](./docs/SCRIPTS.md)
- [更新日志](./changelog/train_model_v0.7.0.md)

## 主要特性

- **v0.7.0版本更新**：
  - 增大模型参数规模（12层transformer，768维度，12头注意力）
  - 添加梯度累积，支持更大批量的训练
  - 新增Apple Silicon GPU加速支持
  - 学习率预热与余弦退火调度
  - 详细的训练统计图表

## 许可证

本项目采用[MIT许可证](LICENSE)。

---

详细内容请查阅[完整文档](./docs/README.md)。 