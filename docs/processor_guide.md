# 灵猫墨韵处理器模块使用指南

本文档提供灵猫墨韵数据处理模块 (processor.py) 的使用说明。

## 概述

处理器模块负责对原始文本数据进行清洗、过滤和转换，准备用于模型训练的高质量数据集。

## 基本使用

```bash
# 不带参数执行（推荐）- 自动执行完整的处理流程
python processor.py

# 处理单个文件
python processor.py --input_file data/raw_poems.txt --output_file dataset/processed_poems.jsonl

# 处理整个目录
python processor.py --input_dir data/raw/ --output_dir dataset/processed/
```

## 主要功能

1. **文本清洗**：移除非法字符、繁简转换、统一标点符号等
2. **文本过滤**：根据长度、质量分数等过滤文本
3. **格式转换**：支持txt、json、jsonl等多种格式转换
4. **数据拆分**：自动拆分为训练集和验证集

## 主要参数

- `--input_file/--input_dir`：输入文件或目录路径
- `--output_file/--output_dir`：输出文件或目录路径
- `--min_length`：文本最小长度，默认10
- `--max_length`：文本最大长度，默认2000
- `--quality_threshold`：文本质量阈值(0-1)，默认0.7
- `--to_simplified`：是否转换为简体，默认True
- `--split_ratio`：训练集比例，默认0.9
- `--format`：输出格式(txt/json/jsonl)，默认jsonl

## 高级选项

### 自定义处理流程

```bash
# 使用自定义处理器链
python processor.py --input_file data/raw_poems.txt --processors clean,deduplicate,normalize,filter --output_file dataset/custom_processed.jsonl
```

### 数据分析

```bash
# 生成数据分析报告
python processor.py --input_file data/raw_poems.txt --analyze --report_file data_report.html
```

### 并行处理

```bash
# 使用多进程加速处理
python processor.py --input_dir data/large_corpus/ --output_dir dataset/processed/ --workers 8
```

## 使用示例

### 命令行处理

```bash
# 完整处理示例
python processor.py \
  --input_dir data/classical_poems/ \
  --output_dir dataset/processed_poems/ \
  --min_length 20 \
  --max_length 1000 \
  --quality_threshold 0.8 \
  --to_simplified True \
  --split_ratio 0.95 \
  --format jsonl \
  --workers 4
```

### Python API 用法

```python
from processor import TextProcessor

# 创建处理器实例
processor = TextProcessor(
    min_length=20,
    max_length=1000,
    quality_threshold=0.8,
    to_simplified=True
)

# 处理文本
processed_text = processor.process("原始文本内容")

# 处理文件
processor.process_file(
    input_file="data/raw_poems.txt",
    output_file="dataset/processed_poems.jsonl"
)
```

## 常见问题

**Q: 如何保留繁体字不转换为简体？**  
A: 使用 `--to_simplified False` 参数。

**Q: 如何控制输出文件的格式？**  
A: 使用 `--format` 参数，支持txt、json、jsonl格式。

**Q: 数据集太大，处理速度慢怎么办？**  
A: 增加 `--workers` 参数值，利用多核并行处理。

## 提示和建议

1. 对于大型语料，建议先使用 `--analyze` 生成数据报告，了解数据特性
2. 调整质量阈值可以平衡数据量和质量，建议从0.7开始尝试
3. 为获得更好的训练效果，建议数据过滤偏严格，保证质量

---

*最后更新: 2023年3月14日*
