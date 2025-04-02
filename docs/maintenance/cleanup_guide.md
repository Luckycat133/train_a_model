# 灵猫墨韵清理工具使用指南

本文档提供灵猫墨韵项目清理工具 (cleanup.py) 的详细使用说明。

## 概述

灵猫墨韵清理工具是一个专门设计用来管理项目大小、清理临时文件和优化磁盘空间使用的实用工具。它能帮助您有效管理大型文件、日志和缓存，保持项目高效运行。

## 功能特点

- **空间分析**：详细分析项目空间使用情况，找出占用最大的目录和文件
- **数据保护**：内置保护机制，防止意外删除重要文件和数据
- **智能清理**：根据文件类型、大小和使用情况进行智能清理
- **日志管理**：自动整理和归档日志文件，保持日志目录结构清晰
- **Shell脚本集成**：生成可执行的shell脚本，便于集成到自动化工作流

## 基本用法

```bash
# 分析项目空间使用情况（默认操作）
python cleanup.py

# 执行所有清理操作
python cleanup.py --all

# 模拟清理（不实际删除文件）
python cleanup.py --dry-run --all
```

## 命令行选项

### 基本选项

| 选项 | 说明 |
|------|------|
| `--dry-run` | 模拟运行，不实际删除文件 |
| `--all` | 执行所有清理操作 |
| `--force` | 强制删除文件（跳过部分安全检查，谨慎使用） |

### 功能模块

| 选项 | 说明 |
|------|------|
| `--analyze` | 分析空间使用情况 |
| `--compare` | 比较collection和dataset目录 |
| `--temp` | 清理临时文件 |
| `--logs` | 清理日志文件 |
| `--data` | 清理合并数据文件 |
| `--pycache` | 清理__pycache__目录 |
| `--organize` | 整理日志目录 |
| `--create-script` | 创建shell清理脚本 |
| `--clean-scripts` | 清理旧的shell脚本 |

### 高级参数

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--min-size` | 清理临时文件的最小大小(MB) | 100 |
| `--max-log-size` | 保留日志文件的最大大小(MB) | 50 |
| `--log-keep` | 每个目录保留的日志文件数量 | 5 |

## 使用示例

### 空间分析

```bash
# 分析项目空间
python cleanup.py --analyze

# 比较数据目录差异
python cleanup.py --compare
```

### 清理操作

```bash
# 清理大于500MB的临时文件
python cleanup.py --temp --min-size 500

# 清理日志，只保留最新的3个
python cleanup.py --logs --log-keep 3

# 清理Python缓存文件
python cleanup.py --pycache
```

### 组合操作

```bash
# 清理临时文件和日志
python cleanup.py --temp --logs

# 清理所有内容并生成shell脚本
python cleanup.py --all --create-script

# 模拟清理所有内容
python cleanup.py --dry-run --all
```

## 数据保护功能

清理工具内置了强大的数据保护机制，确保您的核心文件和重要数据不会被意外删除。

### 受保护的文件类型

- **核心代码文件**：如 train_model.py, generate.py, processor.py
- **配置文件**：如 config/config.yaml
- **文档文件**：如 README.md, DOCS.md
- **核心训练数据**：如 train_data_train.jsonl, train_data_test.jsonl
- **分词器数据**：如 tokenizer.json

如需查看完整保护列表，请参考清理工具源码中的 `PROTECTED_FILES` 和 `CORE_DATA_FILES` 变量。

## 最佳实践

### 日常维护

```bash
# 每日清理建议
python cleanup.py --temp --min-size 200 --pycache

# 每周清理建议
python cleanup.py --all --log-keep 10
```

### 磁盘空间紧张时

```bash
# 紧急清理
python cleanup.py --all --min-size 50 --max-log-size 20 --log-keep 3
```

### 与自动化任务集成

```bash
# 创建清理脚本
python cleanup.py --create-script

# 之后可以直接运行
./cleanup.sh
```

## 常见问题解答

### Q: 如何确保不会删除重要文件？

A: 默认情况下，清理工具会保护核心文件和训练数据。建议首次使用时添加 `--dry-run` 参数来查看将被删除的文件，确认无误后再执行实际清理。

### Q: 清理后找不到某些数据文件怎么办？

A: 核心训练数据文件（如train_data_train.jsonl等）默认受到保护，不会被删除。如果有文件被意外删除，请检查备份或重新运行数据处理流程生成。

### Q: 如何监控项目大小变化？

A: 定期运行 `python cleanup.py --analyze` 来跟踪项目大小变化。这个命令会生成详细的空间使用报告。

### Q: 清理工具能否添加到定时任务中？

A: 可以。创建shell脚本后（使用`--create-script`选项），可以将其添加到cron任务或其他调度系统中定期执行。

## 疑难解答

### 清理后空间没有明显减少

- 检查是否有大型数据文件未被清理（使用 `--analyze` 查看）
- 确认清理操作是否成功完成（查看输出日志）
- 尝试增加 `--min-size` 参数值以清理更大的文件

### 出现权限错误

- 确保您有权限删除目标文件
- 检查文件是否被其他进程锁定
- 尝试使用管理员权限运行脚本

### 清理操作太慢

- 先使用 `--analyze` 识别最大文件，然后有针对性地清理
- 对于非常大的目录，可以分部分清理而不是一次性清理所有内容

## 版本历史

有关版本更新和新功能，请参阅[清理工具更新日志](../changelog/for_cleanup.md)。

---

*最后更新: 2025年3月14日* 