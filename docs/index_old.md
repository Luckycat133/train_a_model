# 灵猫墨韵项目文档索引

## 快速入门
- [项目概述](project_overview.md)
- [快速开始](project_overview.md#快速开始)
- [系统要求](project_overview.md#系统要求)
- [技术架构概览](technical_architecture.md)
- [安装指南](installation/installation_guide.md)
  - [Linux安装](installation/installation_guide.md#linux安装)
  - [macOS安装](installation/installation_guide.md#macos安装)
  - [Windows安装](installation/installation_guide.md#windows安装)

## 系统设计文档
- [系统架构设计](design_docs/system_architecture.md)
  - [核心模块设计](design_docs/system_architecture.md#核心模块设计)
  - [数据流转全景](design_docs/system_architecture.md#数据流转全景)
  - [性能优化设计](design_docs/system_architecture.md#性能优化设计)
  - [扩展性设计](design_docs/system_architecture.md#扩展性设计)

## 核心模块
- [训练模块](train_model_guide.md)
  - [训练参数配置](train_model_guide.md#参数配置)
  - [模型结构](train_model_guide.md#模型结构)
  - [优化策略](train_model_guide.md#优化策略)
  - [断点续训](train_model_guide.md#断点续训)
  
- [生成模块](generate_guide.md)
  - [文本生成](generate_guide.md#文本生成)
  - [批量生成](generate_guide.md#批量生成)
  - [参数控制](generate_guide.md#参数控制)
  - [输出格式](generate_guide.md#输出格式)

- [处理器](processor_guide.md)
  - [数据预处理](processor_guide.md#数据预处理)
  - [清洗策略](processor_guide.md#清洗策略)
  - [格式转换](processor_guide.md#格式转换)
  - [批处理操作](processor_guide.md#批处理操作)

- [分词器](tokenizer_guide.md)
  - [训练分词器](tokenizer_guide.md#训练分词器)
  - [分词策略](tokenizer_guide.md#分词策略)
  - [特殊标记](tokenizer_guide.md#特殊标记)
  - [词表管理](tokenizer_guide.md#词表管理)

## 最佳实践指南
- [训练优化最佳实践](best_practices/training_optimization.md)
  - [硬件优化策略](best_practices/training_optimization.md#硬件优化策略)
  - [超参数优化](best_practices/training_optimization.md#超参数优化指南)
  - [训练稳定性优化](best_practices/training_optimization.md#训练稳定性优化)
  - [性能监控与调优](best_practices/training_optimization.md#性能监控与调优)
  - [数据集优化](best_practices/training_optimization.md#数据集优化)
  - [常见问题解决](best_practices/training_optimization.md#常见问题与解决方案)

## 代码详解
- [训练模块代码解析](code_explanation/train_model_code.md)
- [分词器代码解析](code_explanation/tokenizer_code.md)
- [处理器代码解析](code_explanation/processor_code.md)

## 工具与脚本
- [辅助脚本](scripts_guide.md)
  - [数据统计](scripts_guide.md#数据统计)
  - [模型评估](scripts_guide.md#模型评估)
  - [环境配置](scripts_guide.md#环境配置)
- [清理工具](cleanup_guide.md)
  - [空间分析](cleanup_guide.md#空间分析)
  - [文件清理](cleanup_guide.md#清理操作)
  - [数据保护](cleanup_guide.md#数据保护功能)
  - [最佳实践](cleanup_guide.md#最佳实践)

## 项目维护
- [项目维护指南](project_maintenance.md)
  - [空间占用分析](project_maintenance.md#主要空间占用来源)
  - [清理脚本使用](project_maintenance.md#清理脚本使用方法)
  - [定期维护建议](project_maintenance.md#定期清理建议)
  - [监控项目大小](project_maintenance.md#监控项目大小)

## 更新与变更
- [训练模块更新记录](../changelog/for_train_model.md)
- [处理器更新记录](../changelog/for_processor.md)
- [分词器更新记录](../changelog/for_tokenizer.md)
- [数据模块更新记录](../changelog/for_data.md)
- [清理工具更新记录](../changelog/for_cleanup.md)

## 项目文档

- [DOCS.md](../DOCS.md) - 文档导航与总览

## 工具与配置

- [.gitignore](../.gitignore) - Git忽略文件配置
- [requirements.txt](../requirements.txt) - 项目依赖包列表

*此索引最后更新时间: 2025年3月18日*
