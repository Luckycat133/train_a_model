# 灵猫墨韵数据系统指南

本文档是灵猫墨韵项目数据系统的入口文档，提供数据处理管道的完整概览和快速参考。

## 📋 目录

- [快速开始](#快速开始) - 快速上手数据系统
- [数据流程](#数据流程) - 数据处理完整流程
- [核心组件](#核心组件) - 主要模块和类
- [数据获取](#数据获取) - 下载和导入数据
- [数据处理](#数据处理) - 清洗和转换数据
- [数据集配置](#数据集配置) - 配置和使用数据集
- [代码示例](#代码示例) - 常用代码片段
- [故障排除](#故障排除) - 常见问题解答

## 🚀 快速开始

### 一分钟上手

```bash
# 1. 下载诗词数据
python processors/download_poetry.py

# 2. 查看下载的数据
head -n 5 collection/chinese-poetry/poetry.jsonl

# 3. 在代码中使用
python -c "
from src.data import PretrainDataset
ds = PretrainDataset('collection/chinese-poetry/poetry.jsonl', context_length=128)
print(f'加载了 {len(ds)} 个样本')
"
```

## 📊 数据流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        完整数据流程                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [数据源] ──▶ [下载] ──▶ [原始数据] ──▶ [处理] ──▶ [训练数据]     │
│                                                                     │
│   GitHub              processors/         JSONL       src.data/      │
│   chinese-poetry      download_poetry.py            PretrainDataset│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

详细步骤：

1. 数据获取
   └─▶ processors/download_poetry.py  # 从 GitHub 下载 JSON 数据

2. 格式转换
   └─▶ JSON → JSONL 转换              # 适配训练格式

3. 数据清洗
   └─▶ processors/processor.py        # 清洗、过滤

4. 数据加载
   └─▶ src/data/*_dataset.py         # 加载为训练数据集

5. 模型训练
   └─▶ src/trainer.py                 # 使用 PyTorch DataLoader
```

## 🧩 核心组件

### 数据集类

| 类名 | 文件 | 说明 |
|------|------|------|
| `BaseDataset` | [base_dataset.py](src/data/base_dataset.py) | 基础抽象类 |
| `PretrainDataset` | [pretrain_dataset.py](src/data/pretrain_dataset.py) | 预训练数据 |
| `SFTDataset` | [sft_dataset.py](src/data/sft_dataset.py) | 对话微调数据 |
| `StreamingDataset` | [base_dataset.py](src/data/base_dataset.py) | 流式加载 |
| `WeightedMixingDataset` | [base_dataset.py](src/data/base_dataset.py) | 混合采样 |

### 工具模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 下载脚本 | [download_poetry.py](processors/download_poetry.py) | 诗词下载 |
| 处理工具 | [processor.py](processors/processor.py) | 数据清洗 |

### 配置文件

| 配置 | 路径 | 说明 |
|------|------|------|
| 数据配置 | `config/dataset.yaml` | 数据集配置 |
| 训练配置 | `config/*.yaml` | 训练参数 |

## 📥 数据获取

### 支持的数据源

#### 1. 中文诗词

| 数据源 | 数量 | 下载命令 |
|--------|------|----------|
| 全唐诗 | ~50,000首 | 内置脚本 |
| 全宋词 | ~20,000首 | 内置脚本 |
| 楚辞 | ~67首 | 内置脚本 |

**下载命令：**

```bash
python processors/download_poetry.py
```

**下载完成后：**

```
collection/chinese-poetry/
├── poetry.jsonl        # 合并文件
├── poet.tang.0.jsonl   # 唐诗卷1
├── poet.tang.1.jsonl   # 唐诗卷2
├── poet.song.0.jsonl   # 宋词卷1
└── chuci.jsonl         # 楚辞
```

#### 2. 古文典籍

从公共仓库获取：

```bash
# 论语
wget -O collection/lunyu/lunyu.json \
  https://raw.githubusercontent.com/chinese-classics/lunyu/master/lunyu.json

# 道德经
wget -O collection/dao-de-jing/dao.json \
  https://raw.githubusercontent.com/chinese-classics/dao-de-jing/master/dao.json
```

#### 3. 自定义数据

参考 [数据获取指南](docs/data/acquisition.md#添加自定义数据源) 添加自定义数据源。

## 🔧 数据处理

### 快速处理

```bash
# 清洗数据
python processors/processor.py \
  --input collection/raw.jsonl \
  --output collection/cleaned.jsonl \
  --rules remove_html,normalize_whitespace
```

### 处理规则

| 规则 | 说明 | 用途 |
|------|------|------|
| `remove_html` | 移除 HTML 标签 | 网页数据 |
| `normalize_whitespace` | 规范化空白 | 统一格式 |
| `remove_urls` | 移除 URL | 清理链接 |
| `remove_control_chars` | 移除控制字符 | 修复编码 |
| `filter_quality` | 质量过滤 | 提升质量 |

详细处理方法请参考 [数据处理指南](docs/data/processing.md)。

## ⚙️ 数据集配置

### Python API

```python
from src.data import PretrainDataset, SFTDataset

# 预训练数据
ds = PretrainDataset(
    data_paths="collection/chinese-poetry/poetry.jsonl",
    context_length=512,
    stride=256,
    min_length=10
)

# SFT 数据
sft_ds = SFTDataset(
    data_paths="dataset/sft.jsonl",
    dialogue_format="sharegpt"
)
```

### YAML 配置

```yaml
# config/dataset.yaml
datasets:
  pretrain:
    path: "collection/chinese-poetry/poetry.jsonl"
    context_length: 512
    stride: 256
    min_length: 10
```

详细配置请参考 [数据配置指南](docs/data/configuration.md)。

## 💻 代码示例

### 示例 1：加载预训练数据

```python
from src.data import PretrainDataset
from tokenizer import ClassicalTokenizer

# 初始化分词器
tokenizer = ClassicalTokenizer()

# 创建数据集
dataset = PretrainDataset(
    data_paths="collection/chinese-poetry/poetry.jsonl",
    tokenizer=tokenizer,
    context_length=512,
    stride=256
)

print(f"样本数量: {len(dataset)}")
print(f"示例数据: {dataset[0]}")
```

### 示例 2：加载 SFT 对话数据

```python
from src.data import SFTDataset, ChatTemplate

# 自定义模板
template = ChatTemplate(
    system_prefix="[系统] ",
    user_prefix="[用户] ",
    assistant_prefix="[助手] "
)

# 创建数据集
sft_ds = SFTDataset(
    data_paths="dataset/sft_dialogue.jsonl",
    dialogue_format="sharegpt",
    chat_template=template
)
```

### 示例 3：混合数据集

```python
from src.data import WeightedMixingDataset, PretrainDataset

# 创建多个数据源
poetry_ds = PretrainDataset("dataset/poetry.jsonl", context_length=512)
classical_ds = PretrainDataset("dataset/classical.jsonl", context_length=512)
modern_ds = PretrainDataset("dataset/modern.jsonl", context_length=512)

# 混合加载（权重：诗词60%，古文30%，现代10%）
mixed = WeightedMixingDataset(
    datasets=[(poetry_ds, 0.6), (classical_ds, 0.3), (modern_ds, 0.1)],
    replacement=True,
    seed=42
)
```

### 示例 4：流式加载大数据

```python
from src.data import StreamingDataset
from torch.utils.data import DataLoader

# 流式加载
stream_ds = StreamingDataset(
    data_paths=["data1.jsonl", "data2.jsonl", "data3.jsonl"],
    context_length=1024,
    prefetch_buffer=100
)

# 创建 DataLoader
dataloader = DataLoader(stream_ds, batch_size=32)

for batch in dataloader:
    # 训练步骤...
    pass
```

### 示例 5：训练循环

```python
from torch.utils.data import DataLoader

# 创建 DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 训练循环
for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        # 前向传播
        outputs = model(input_ids)
        loss = loss_fn(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## ❓ 故障排除

### 常见问题

#### 1. 编码错误

**问题：** `UnicodeDecodeError: 'utf-8' codec can't decode byte`

**解决：**

```python
# 检测编码
from processors.processor import detect_encoding
encoding = detect_encoding("data.txt")
print(f"检测到编码: {encoding}")

# 指定编码读取
with open("data.txt", "r", encoding=encoding) as f:
    data = f.read()
```

#### 2. 内存不足

**问题：** 大数据集加载时内存不足

**解决：** 使用流式加载

```python
# 替换为流式加载
stream_ds = StreamingDataset(
    data_paths="large_file.jsonl",
    context_length=512,
    prefetch_buffer=50  # 减小缓冲
)
```

#### 3. 数据格式错误

**问题：** `JSONDecodeError` 或数据解析失败

**解决：**

```python
# 验证数据格式
import json

with open("data.jsonl", "r") as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"第 {i+1} 行格式错误: {e}")
```

#### 4. 数据下载失败

**问题：** GitHub API 速率限制

**解决：**

```bash
# 等待限制重置（1小时后自动恢复）
# 或使用代理
export HTTP_PROXY="http://proxy.example.com:8080"
python processors/download_poetry.py
```

## 📚 相关文档

| 文档 | 说明 |
|------|------|
| [数据系统概览](docs/data/README.md) | 完整架构说明 |
| [数据获取指南](docs/data/acquisition.md) | 下载和导入详细方法 |
| [数据处理指南](docs/data/processing.md) | 清洗和转换详细指南 |
| [数据配置指南](docs/data/configuration.md) | 配置和使用数据集 |
| [API 参考](src/data/) | 代码 API 参考 |

## 🤝 贡献指南

欢迎贡献数据处理代码和数据源！

- 添加新数据源：请参考 [数据获取指南](docs/data/acquisition.md#添加自定义数据源)
- 改进处理工具：请修改 [processors/processor.py](processors/processor.py)
- 添加新数据集类：请参考 [base_dataset.py](src/data/base_dataset.py)

## 📝 更新日志

- **2026-05-15** - 初始版本发布
  - 支持中文诗词数据下载
  - 支持预训练和 SFT 数据加载
  - 提供完整的数据处理管道

---

_本文档最后更新：2026-05-15_
