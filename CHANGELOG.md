# 更新日志

本项目遵循[语义化版本](https://semver.org/lang/zh-CN/)规范。

## [未发布]

### 新增

- 🚀 **命令行优化**：新增快捷命令
  - `--quick`: 快速测试配置
  - `--preset`: 预设置配置（quick/small/medium/large）
  - `--resume`: 自动继续上次训练
  - `--train`: 简化的数据文件参数
  - `-e, -l, -b, -c`: 常用参数的短别名

- 📊 **预设置配置**：提供4种训练预设置
  - `quick`: 快速测试 (5 epochs, 小模型)
  - `small`: 日常训练 (20 epochs, 小模型)
  - `medium`: 正式训练 (30 epochs, 中型模型)
  - `large`: 高质量模型 (50 epochs, 大型模型)

- 🔧 **智能功能**
  - 自动检测最新checkpoint继续训练
  - 训练前清理旧日志（`--clean`）
  - 友好的中文帮助信息和错误提示
  - 训练前显示配置摘要

### 修复

- 🐛 **核心Bug修复**
  - 修复dataset.py的tokenizer调用逻辑，优先使用encode方法
  - 修复trainer.py的model初始化参数名（use_checkpoint → gradient_checkpointing）
  - 修复autocast调用方式，支持BF16/FP16
  - 修复model forward返回tuple解包问题
  - 修复stats字典访问方式
  - 移除torch.save/load的weights_only参数兼容性
  - DataLoader的num_workers设为0避免多进程问题
  - 移除dataset.py中的self.np实例属性避免序列化问题

### 优化

- ⚡ **性能优化**
  - 使用torch.compile加速训练
  - 优化自动混合精度训练
  - 改进梯度检查点实现

- 📝 **文档优化**
  - 简化README结构
  - 添加详细使用示例
  - 完善故障排除指南

### 移除

- 🗑️ **清理冗余文件**
  - 删除所有临时测试脚本
  - 清理重复的训练脚本
  - 移除未使用的示例代码
  - 删除benchmark和test目录

## [0.8.5] - 2026-04-05

### 新增

- ✨ **Modern模式架构**：RoPE + SwiGLU + GQA支持
- 🔧 **实验平台**：完整的配置管理和验证系统
- 📊 **评估模块**：多样性和LM指标评估
- 🎯 **量化工具**：AWQ量化支持
- 📝 **多语言文档**：中英文双语文档

### 修复

- 修复了ClassicalTokenizer的贪婪最大匹配问题
- 优化了数据加载的内存使用
- 修复了梯度检查点的实现

### 优化

- 实现了内存映射数据集
- 优化了训练循环性能
- 改进了日志记录

## [0.1.0] - 2025-04-23

### 新增

- 初始版本的数据处理脚本
- 基础的配置文件系统
- 日志记录功能
- 命令行参数解析
