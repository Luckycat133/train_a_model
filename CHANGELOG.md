# 更新日志

本项目遵循[语义化版本](https://semver.org/lang/zh-CN/)规范。

## [未发布]

### 新增
- 为 `processor.py`, `tokenizer.py`, `train_model.py`, `cleanup.py` 添加了任务开始前的清理功能，用于删除旧的输出/临时文件，确保运行环境干净。
- 为 `train_model.py` 添加了 `--clean-before-run` 命令行参数，用于在训练前清理日志和图表。
- 为 `cleanup.py` 添加了创建统一清理脚本 (`cleanup.sh`) 的功能。

### 修复
- 修复了 `processor.py` 中的多处缩进和语法错误。
- 修复了 `setup_logging` 函数中日志记录重复的问题。
- 修复了 `processor.py` 命令行参数 `-i` 默认值为目录时缺少结尾斜杠的问题。
- 统一了 `process_batch` 函数的文档字符串格式。

### 移除
- 移除了 `processor.py` 中未使用的 `pandas` 导入。
- 移除了 `processor.py` 脚本末尾冗余的 `try-except` 块。

## [0.1.0] - 2025-04-22

### 新增
- 初始版本的数据处理脚本 `processor.py`。
- 基础的配置文件 `config/config.yaml`。
- 日志记录功能。
- 命令行参数解析。 