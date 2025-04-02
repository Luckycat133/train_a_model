# 灵猫墨韵文档目录

<div align="center">

![灵猫墨韵](https://img.shields.io/badge/🐱_灵猫墨韵-文档目录-8A2BE2)

*一站式导航索引，快速找到您需要的所有文档*

</div>

## 目录

- [💡 用户指南](#用户指南)
  - [🚀 快速入门](#快速入门)
  - [⚙️ 模型训练](#模型训练)
  - [✨ 文本生成](#文本生成)
  - [🧰 工具与维护](#工具与维护)
- [📚 参考文档](#参考文档)
  - [🏗️ 系统架构](#系统架构)
  - [⚙️ 模块参考](#模块参考)
  - [🔍 代码详解](#代码详解)
- [📝 标准与规范](#标准与规范)
- [✅ 最佳实践](#最佳实践)

---

## 📂 用户指南

### 🚀 快速入门

- [📋 项目概述](project_overview.md)
  - 项目背景和目标
  - 主要功能和特点
  - 技术架构概览
  
- [💻 安装指南](user_guides/installation/index.md)
  - [🐧 Linux安装](user_guides/installation/index.md#linux安装)
  - [🍎 macOS安装](user_guides/installation/index.md#macos安装)
  - [🪟 Windows安装](user_guides/installation/index.md#windows安装)
  - [🔧 环境配置](user_guides/installation/index.md#环境配置)
  - [❓ 常见问题](user_guides/installation/index.md#常见问题)

### ⚙️ 模型训练

- [🧠 训练指南](user_guides/training/index.md)
  - [🛠️ 训练环境准备](user_guides/training/index.md#训练环境准备)
  - [📊 数据准备](user_guides/training/index.md#数据准备)
  - [⚙️ 参数配置](user_guides/training/index.md#参数配置)
  - [📈 训练过程监控](user_guides/training/index.md#训练过程监控)
  - [💾 模型保存与加载](user_guides/training/index.md#模型保存与加载)

- [🔄 数据处理](user_guides/training/data_preparation.md)
  - [📝 数据格式要求](user_guides/training/data_preparation.md#数据格式要求)
  - [🧹 数据清洗](user_guides/training/data_preparation.md#数据清洗)
  - [🔄 数据转换](user_guides/training/data_preparation.md#数据转换)
  - [📊 数据统计分析](user_guides/training/data_preparation.md#数据统计分析)

- [🔤 分词器指南](user_guides/training/tokenizer.md)
  - [📚 词表构建](user_guides/training/tokenizer.md#词表构建)
  - [⚙️ 分词策略](user_guides/training/tokenizer.md#分词策略)
  - [🔧 分词器训练](user_guides/training/tokenizer.md#分词器训练)
  - [📊 分词效果评估](user_guides/training/tokenizer.md#分词效果评估)

### ✨ 文本生成

- [✍️ 生成指南](user_guides/generation/index.md)
  - [🚀 快速开始](user_guides/generation/index.md#快速开始)
  - [⚙️ 生成参数详解](user_guides/generation/index.md#生成参数详解)
  - [📝 输入提示技巧](user_guides/generation/index.md#输入提示技巧)
  - [📊 生成结果评估](user_guides/generation/index.md#生成结果评估)

- [📚 格律与风格](user_guides/generation/style_guide.md)
  - [📜 诗词格律](user_guides/generation/style_guide.md#诗词格律)
  - [🎨 文风调整](user_guides/generation/style_guide.md#文风调整)
  - [📝 文体选择](user_guides/generation/style_guide.md#文体选择)

### 🧰 工具与维护

- [🔧 实用脚本](user_guides/tools/scripts.md)
  - [📊 数据统计脚本](user_guides/tools/scripts.md#数据统计脚本)
  - [📈 性能评估脚本](user_guides/tools/scripts.md#性能评估脚本)
  - [🔄 批处理脚本](user_guides/tools/scripts.md#批处理脚本)

- [🧹 系统清理](user_guides/maintenance/cleanup.md)
  - [📊 空间分析](user_guides/maintenance/cleanup.md#空间分析)
  - [🗑️ 安全清理](user_guides/maintenance/cleanup.md#安全清理)
  - [⚙️ 配置选项](user_guides/maintenance/cleanup.md#配置选项)

- [🛠️ 项目维护](user_guides/maintenance/project_maintenance.md)
  - [📝 维护计划](user_guides/maintenance/project_maintenance.md#维护计划)
  - [🔄 版本更新](user_guides/maintenance/project_maintenance.md#版本更新)
  - [🐛 问题修复](user_guides/maintenance/project_maintenance.md#问题修复)
  - [📋 贡献指南](user_guides/maintenance/project_maintenance.md#贡献指南)

## 📚 参考文档

### 🏗️ 系统架构

- [📐 系统设计](reference/architecture/system_design.md)
  - [🏗️ 整体架构](reference/architecture/system_design.md#整体架构)
  - [📊 数据流图](reference/architecture/system_design.md#数据流图)
  - [🔄 模块交互](reference/architecture/system_design.md#模块交互)
  - [🛠️ 技术选型](reference/architecture/system_design.md#技术选型)

- [🧩 核心模块设计](reference/architecture/module_design.md)
  - [🧠 模型架构](reference/architecture/module_design.md#模型架构)
  - [🔄 数据处理流程](reference/architecture/module_design.md#数据处理流程)
  - [⚙️ 推理引擎](reference/architecture/module_design.md#推理引擎)

### ⚙️ 模块参考

- [🧠 训练模块](reference/modules/trainer.md)
  - [⚙️ 配置参数](reference/modules/trainer.md#配置参数)
  - [📊 训练流程](reference/modules/trainer.md#训练流程)
  - [📈 性能指标](reference/modules/trainer.md#性能指标)

- [🔄 处理器模块](reference/modules/processor.md)
  - [📝 数据处理流程](reference/modules/processor.md#数据处理流程)
  - [🧹 清洗规则](reference/modules/processor.md#清洗规则)
  - [🔍 质量检查](reference/modules/processor.md#质量检查)

- [🔤 分词器模块](reference/modules/tokenizer.md)
  - [📚 词表设计](reference/modules/tokenizer.md#词表设计)
  - [⚙️ 分词算法](reference/modules/tokenizer.md#分词算法)
  - [🔄 转换流程](reference/modules/tokenizer.md#转换流程)

### 🔍 代码详解

- [🧠 训练代码详解](reference/modules/train_model_code.md)
- [🔄 处理器代码详解](reference/modules/processor_code.md)
- [🔤 分词器代码详解](reference/modules/tokenizer_code.md)

## 📝 标准与规范

- [📏 文档样式指南](standards/style_guide.md)
  - [📝 文档结构](standards/style_guide.md#文档结构)
  - [🎨 格式规范](standards/style_guide.md#格式规范)
  - [🖼️ 图表使用](standards/style_guide.md#图表使用)

- [📁 文档目录结构](standards/directory_structure.md)
  - [📂 顶层目录](standards/directory_structure.md#顶层目录)
  - [📑 文件命名](standards/directory_structure.md#文件命名)
  - [🔗 链接规则](standards/directory_structure.md#链接规则)

- [📄 文档模板](standards/template.md)
  - [📝 用户指南模板](standards/template.md#用户指南模板)
  - [📚 参考文档模板](standards/template.md#参考文档模板)
  - [✅ 最佳实践模板](standards/template.md#最佳实践模板)

## ✅ 最佳实践

- [⚡ 训练优化](user_guides/best_practices/training_optimization.md)
  - [🖥️ 硬件优化](user_guides/best_practices/training_optimization.md#硬件优化策略)
  - [⚙️ 超参数调优](user_guides/best_practices/training_optimization.md#超参数优化指南)
  - [📊 性能监控](user_guides/best_practices/training_optimization.md#性能监控与调优)

- [🚀 训练模型性能优化](user_guides/best_practices/performance_optimization.md)
  - [💾 内存优化技术](user_guides/best_practices/performance_optimization.md#内存优化技术)
  - [⚙️ 计算优化技术](user_guides/best_practices/performance_optimization.md#计算优化技术)
  - [🔄 自适应训练策略](user_guides/best_practices/performance_optimization.md#自适应训练策略)
  - [🧪 性能测试和评估](user_guides/best_practices/performance_optimization.md#性能测试和评估)
  - [💻 硬件兼容性](user_guides/best_practices/performance_optimization.md#硬件兼容性)
  - [❓ 常见问题与解决方案](user_guides/best_practices/performance_optimization.md#常见问题与解决方案)

- [📈 生成效果优化](user_guides/best_practices/generation_optimization.md)
  - [🎯 目标选择](user_guides/best_practices/generation_optimization.md#目标选择)
  - [📝 提示设计](user_guides/best_practices/generation_optimization.md#提示设计)
  - [⚙️ 参数调整](user_guides/best_practices/generation_optimization.md#参数调整)

---

<div align="center">

[🏠 返回主页](index.md) | [📚 项目概述](project_overview.md) | [🔧 安装指南](user_guides/installation/index.md)

*最后更新时间: 2025-03-21*

</div> 