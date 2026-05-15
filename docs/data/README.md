# 灵猫墨韵数据系统概览

本目录包含灵猫墨韵（灵猫墨韵）项目的完整数据处理文档，涵盖数据获取、处理、配置等各个环节。

## 📊 数据系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数据获取流程图                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│   │   数据源     │───▶│   下载工具    │───▶│   原始数据   │        │
│   │ (诗词/古文)  │    │ download_*  │    │   (JSON)     │        │
│   └──────────────┘    └──────────────┘    └──────────────┘        │
│                                                  │                   │
│                                                  ▼                   │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│   │   训练数据   │◀───│   处理器     │◀───│   JSONL     │        │
│   │   (Token)    │    │ processor.py │    │   转换      │        │
│   └──────────────┘    └──────────────┘    └──────────────┘        │
│                                                  │                   │
│                                                  ▼                   │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│   │   模型训练   │◀───│   DataLoader │◀───│   数据集    │        │
│   │  src.trainer │    │  PyTorch    │    │   注册      │        │
│   └──────────────┘    └──────────────┘    └──────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 📦 支持的数据源

### 1. 中文诗词数据

| 数据源 | 来源 | 规模 | 格式 |
|--------|------|------|------|
| 全唐诗 | chinese-poetry | ~50,000首 | JSON |
| 全宋词 | chinese-poetry | ~20,000首 | JSON |
| 楚辞 | chinese-poetry | ~600首 | JSON |
| 元曲 | chinese-poetry | ~5,000首 | JSON |
| 诗经 | chinese-poetry | ~305首 | JSON |

### 2. 古文典籍

| 数据源 | 来源 | 规模 | 格式 |
|--------|------|------|------|
| 古文观止 | 自定义/网络 | ~200篇 | JSON/TXT |
| 论语 | chinese-classics | ~20,000字 | JSON |
| 道德经 | chinese-classics | ~5,000字 | JSON |
| 庄子 | chinese-classics | ~80,000字 | JSON |

### 3. 对话数据（用于SFT）

| 格式 | 描述 | 适用场景 |
|------|------|----------|
| ShareGPT | {conversations: [{role, content}]} | 通用对话 |
| OpenAI | {messages: [{role, content}]} | API数据 |
| Alpaca | {instruction, input, output} | 指令微调 |

## 📁 数据格式说明

### 原始数据格式 (JSON)

```json
{
  "author": "李白",
  "title": "静夜思",
  "paragraphs": [
    "床前明月光",
    "疑是地上霜",
    "举头望明月",
    "低头思故乡"
  ]
}
```

### 处理后格式 (JSONL)

```jsonl
{"text": "床前明月光\n疑是地上霜\n举头望明月\n低头思故乡", "author": "李白", "category": "tang"}
{"text": "...", "author": "...", "category": "..."}
```

### 训练数据格式

```python
{
    "input_ids": [1234, 5678, ...],      # 输入token IDs
    "attention_mask": [1, 1, 1, ...],    # 注意力掩码
    "labels": [-100, -100, 5678, ...],   # 标签（忽略部分为-100）
}
```

## 🚀 快速开始

### 1. 下载诗词数据

```bash
# 使用内置脚本下载唐诗宋词
python processors/download_poetry.py

# 数据将保存到 collection/chinese-poetry/poetry.jsonl
```

### 2. 准备训练数据

```bash
# 方式一：使用默认配置运行
python -m src.run --config config/pretrain.yaml

# 方式二：手动处理数据
python processors/processor.py --input collection/data.json --output dataset/train.jsonl
```

### 3. 加载数据集

```python
from src.data import PretrainDataset, SFTDataset

# 预训练数据
pretrain_ds = PretrainDataset(
    data_paths="dataset/pretrain.jsonl",
    context_length=512
)

# SFT对话数据
sft_ds = SFTDataset(
    data_paths="dataset/sft.jsonl",
    dialogue_format="sharegpt"
)
```

## 📂 目录结构

```
docs/data/
├── README.md          # 本文档 - 数据系统概览
├── acquisition.md     # 数据获取指南
├── processing.md      # 数据处理指南
└── configuration.md   # 数据配置指南

src/data/
├── __init__.py        # 模块导出
├── base_dataset.py    # 基础数据集类
├── pretrain_dataset.py # 预训练数据集
├── sft_dataset.py     # SFT对话数据集
└── packed_dataset.py   # 打包数据集

processors/
├── download_poetry.py # 诗词下载脚本
└── processor.py       # 数据处理工具

collection/             # 原始数据存放
dataset/                # 处理后数据存放
```

## 🔧 核心组件

### 数据集类

| 类名 | 用途 | 特点 |
|------|------|------|
| `BaseDataset` | 基础数据集抽象类 | 提供通用加载接口 |
| `PretrainDataset` | 预训练数据 | 支持滑动窗口 |
| `SFTDataset` | 对话微调数据 | 支持多种对话格式 |
| `StreamingDataset` | 流式数据集 | 内存高效 |
| `WeightedMixingDataset` | 混合数据集 | 支持加权采样 |

### 工厂类

| 类名 | 方法 | 用途 |
|------|------|------|
| `PretrainDatasetFactory` | `create_standard()`, `create_concat()`, `create_streaming()` | 创建预训练数据集 |
| `SFTDatasetFactory` | `create_sharegpt()`, `create_alpaca()`, `create_openai()` | 创建SFT数据集 |

## 📚 扩展阅读

- [数据获取指南](acquisition.md) - 详细的数据下载和导入方法
- [数据处理指南](processing.md) - 数据清洗、过滤和转换
- [数据配置指南](configuration.md) - 配置文件格式和数据集注册

## ⚠️ 注意事项

1. **数据质量**：下载的数据建议进行质量检查，过滤低质量样本
2. **编码格式**：所有文本文件使用UTF-8编码
3. **内存管理**：大数据集建议使用流式加载（streaming=True）
4. **数据备份**：原始数据和处理后数据都建议备份

## 🤝 贡献数据

欢迎贡献更多高质量的古典文学数据集！请参考 [acquisition.md](acquisition.md#添加自定义数据源) 了解如何添加自定义数据源。
