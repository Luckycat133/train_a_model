# 灵猫墨韵数据配置指南

本文档详细说明如何使用 DatasetRegistry、DatasetLoader 以及数据集配置文件的格式和示例。

## 🗂️ 数据集注册表 (DatasetRegistry)

DatasetRegistry 提供数据集的注册和管理功能，支持通过名称快速加载预定义的数据集。

### 基础用法

```python
from src.data import DatasetRegistry

# 初始化注册表
registry = DatasetRegistry()

# 注册数据集
registry.register("poetry_tang", {
    "path": "collection/chinese-poetry/poetry.jsonl",
    "format": "jsonl",
    "text_field": "text",
    "category": "tang"
})

# 加载数据集配置
config = registry.get("poetry_tang")
print(config)
# {'path': 'collection/chinese-poetry/poetry.jsonl', ...}
```

### 预定义数据集

```python
# 注册预定义的诗词数据集
registry.register_poetry_datasets({
    "poetry_tang": "collection/chinese-poetry/poet.tang.jsonl",
    "poetry_song": "collection/chinese-poetry/poet.song.jsonl",
    "poetry_chuci": "collection/chinese-poetry/chuci.jsonl",
    "poetry_lunyu": "collection/lunyu/lunyu.jsonl",
    "poetry_daojing": "collection/dao-de-jing/dao.jsonl"
})

# 批量加载配置
configs = registry.list_all()
print(f"已注册 {len(configs)} 个数据集")
```

### 高级功能

```python
# 添加数据转换器
def transform_poetry(item):
    """转换诗词数据格式"""
    return {
        "text": item["text"],
        "author": item.get("author", "未知"),
        "title": item.get("title", ""),
        "source": "poetry"
    }

registry.register_with_transform("custom_poetry", {
    "path": "collection/custom.jsonl",
    "transform": transform_poetry
})

# 添加数据验证器
def validate_poetry(item):
    """验证诗词数据"""
    required_fields = ["text"]
    return all(field in item for field in required_fields)

registry.register_with_validation("validated_poetry", {
    "path": "collection/validated.jsonl",
    "validator": validate_poetry
})

# 获取带元数据的数据集信息
info = registry.get_with_info("poetry_tang")
print(f"数据集: {info['name']}")
print(f"路径: {info['config']['path']}")
print(f"已注册: {info.get('registered_at', 'N/A')}")
```

## 📥 数据集加载器 (DatasetLoader)

DatasetLoader 负责将配置转换为可训练的数据集对象。

### 基础加载

```python
from src.data import DatasetLoader, PretrainDataset, SFTDataset

# 创建加载器
loader = DatasetLoader(tokenizer=tokenizer)

# 加载预训练数据集
pretrain_ds = loader.load_pretrain(
    data_paths="dataset/pretrain.jsonl",
    context_length=512,
    stride=256
)

# 加载 SFT 数据集
sft_ds = loader.load_sft(
    data_paths="dataset/sft.jsonl",
    dialogue_format="sharegpt"
)
```

### 高级加载选项

```python
# 流式加载（内存高效）
stream_ds = loader.load_streaming(
    data_paths=["dataset/large_file1.jsonl", "dataset/large_file2.jsonl"],
    context_length=1024,
    prefetch_buffer=100
)

# 加权混合加载
mixed_ds = loader.load_mixed(
    datasets=[
        ("dataset/poetry.jsonl", 0.7),
        ("dataset/classical.jsonl", 0.3)
    ],
    replacement=True,
    seed=42
)

# 拼接加载（文档级）
concat_ds = loader.load_concat(
    data_paths="dataset/documents.jsonl",
    context_length=2048,
    separator_id=2
)
```

### 批量加载

```python
from torch.utils.data import DataLoader

# 创建 DataLoader
dataloader = DataLoader(
    pretrain_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 训练循环
for batch in dataloader:
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    # 训练步骤...
```

## 📄 配置文件格式

### YAML 配置示例

```yaml
# config/dataset.yaml

datasets:
  # 预训练数据集配置
  pretrain:
    poetry:
      path: "collection/chinese-poetry/poetry.jsonl"
      format: "jsonl"
      text_field: "text"
      min_length: 10
      max_length: 50000
    
    classical:
      path: "collection/classical/*.jsonl"
      format: "jsonl"
      text_field: "content"
      filter:
        category: ["lunyu", "dao", "zhuang"]

  # SFT 数据集配置
  sft:
    dialogue:
      path: "dataset/sft_dialogue.jsonl"
      format: "jsonl"
      dialogue_format: "sharegpt"
      max_turns: 10
      drop_incomplete: true

  # 验证集配置
  validation:
    path: "dataset/validation.jsonl"
    split_ratio: 0.1

# 数据加载器配置
loader:
  context_length: 512
  stride: 256
  batch_size: 32
  num_workers: 4
  prefetch_factor: 2
  streaming: false
```

### Python 配置示例

```python
# config/dataset_config.py

DATASET_CONFIG = {
    "pretrain": {
        "poetry": {
            "path": "collection/chinese-poetry/poetry.jsonl",
            "format": "jsonl",
            "text_field": "text",
            "filters": {
                "min_length": 10,
                "max_length": 50000,
                "categories": ["tang", "song", "chuci"]
            }
        },
        "classical": {
            "path": "collection/classical/",
            "format": "jsonl",
            "text_field": "content"
        }
    },
    "sft": {
        "dialogue": {
            "path": "dataset/sft.jsonl",
            "format": "jsonl",
            "dialogue_format": "sharegpt",
            "chat_template": {
                "system_prefix": "系统: ",
                "user_prefix": "用户: ",
                "assistant_prefix": "助手: "
            }
        }
    },
    "loader": {
        "context_length": 512,
        "stride": 256,
        "batch_size": 32,
        "streaming": False
    }
}
```

## 💡 使用示例

### 示例 1：预训练数据加载

```python
from src.data import PretrainDataset, PretrainDatasetFactory
from tokenizer import ClassicalTokenizer

# 初始化分词器
tokenizer = ClassicalTokenizer()

# 方法 1：直接创建
dataset = PretrainDataset(
    data_paths="collection/chinese-poetry/poetry.jsonl",
    tokenizer=tokenizer,
    context_length=512,
    stride=256,
    min_length=10,
    max_length=100000
)

# 方法 2：使用工厂类
dataset = PretrainDatasetFactory.create_standard(
    data_paths="collection/chinese-poetry/poetry.jsonl",
    tokenizer=tokenizer,
    context_length=512,
    stride=256
)

# 方法 3：拼接模式
dataset = PretrainDatasetFactory.create_concat(
    data_paths="collection/chinese-poetry/poetry.jsonl",
    tokenizer=tokenizer,
    context_length=2048
)

print(f"加载了 {len(dataset)} 个训练样本")
```

### 示例 2：SFT 对话数据加载

```python
from src.data import SFTDataset, SFTDatasetFactory, ChatTemplate

# 创建自定义聊天模板
template = ChatTemplate(
    system_prefix="[系统] ",
    user_prefix="[用户] ",
    assistant_prefix="[助手] "
)

# 方法 1：直接创建
sft_dataset = SFTDataset(
    data_paths="dataset/sft_dialogue.jsonl",
    tokenizer=tokenizer,
    context_length=512,
    dialogue_format="sharegpt",
    chat_template=template,
    max_turns=10,
    drop_incomplete=True
)

# 方法 2：使用工厂类
sft_dataset = SFTDatasetFactory.create_sharegpt(
    data_paths="dataset/sft_sharegpt.jsonl",
    tokenizer=tokenizer,
    context_length=512
)

# 方法 3：Alpaca 格式
sft_dataset = SFTDatasetFactory.create_alpaca(
    data_paths="dataset/sft_alpaca.jsonl",
    tokenizer=tokenizer
)

print(f"加载了 {len(sft_dataset)} 个对话样本")
```

### 示例 3：混合数据集

```python
from src.data import WeightedMixingDataset

# 创建各个数据集
poetry_ds = PretrainDataset(
    data_paths="collection/poetry.jsonl",
    tokenizer=tokenizer
)

classical_ds = PretrainDataset(
    data_paths="collection/classical.jsonl",
    tokenizer=tokenizer
)

modern_ds = PretrainDataset(
    data_paths="collection/modern.jsonl",
    tokenizer=tokenizer
)

# 创建加权混合数据集
# 权重: 诗词 60%, 古文 30%, 现代 10%
mixed_dataset = WeightedMixingDataset(
    datasets=[
        (poetry_ds, 0.6),
        (classical_ds, 0.3),
        (modern_ds, 0.1)
    ],
    replacement=True,
    seed=42
)

print(f"混合数据集包含 {len(mixed_dataset)} 个样本")
```

### 示例 4：流式加载大数据集

```python
from src.data import StreamingDataset

# 流式加载，适合超大数据集
stream_ds = StreamingDataset(
    data_paths=[
        "collection/large_corpus_1.jsonl",
        "collection/large_corpus_2.jsonl",
        "collection/large_corpus_3.jsonl"
    ],
    tokenizer=tokenizer,
    context_length=1024,
    prefetch_buffer=200
)

# 迭代使用
for i, sample in enumerate(stream_ds):
    if i >= 10000:  # 处理前 10000 个样本
        break
    # 处理样本...

# 使用迭代器
iterator = stream_ds.get_stream_iterator(batch_size=32)
for batch in iterator:
    # 训练步骤...
    pass
```

### 示例 5：在训练中使用

```python
from torch.utils.data import DataLoader
from src.data import PretrainDataset, SFTTrainingCollator

# 加载数据集
train_ds = PretrainDataset(
    data_paths="dataset/train.jsonl",
    tokenizer=tokenizer,
    context_length=512
)

# 创建 DataLoader
train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# SFT 数据集的 collator
sft_ds = SFTDataset(
    data_paths="dataset/sft.jsonl",
    dialogue_format="sharegpt"
)

sft_collator = SFTTrainingCollator(
    pad_token_id=tokenizer.pad_token_id,
    label_pad_token_id=-100
)

sft_loader = DataLoader(
    sft_ds,
    batch_size=16,
    shuffle=True,
    collate_fn=sft_collator
)
```

### 示例 6：配置驱动加载

```python
import yaml

# 加载配置文件
with open("config/dataset.yaml", "r") as f:
    config = yaml.safe_load(f)

# 根据配置加载
datasets = {}
for name, ds_config in config["datasets"].items():
    if "pretrain" in name:
        ds = PretrainDataset(
            data_paths=ds_config["path"],
            tokenizer=tokenizer,
            context_length=config["loader"]["context_length"],
            **ds_config.get("filters", {})
        )
    elif "sft" in name:
        ds = SFTDataset(
            data_paths=ds_config["path"],
            tokenizer=tokenizer,
            dialogue_format=ds_config.get("dialogue_format", "sharegpt")
        )
    
    datasets[name] = ds

print(f"已加载 {len(datasets)} 个数据集")
```

## 🔧 高级配置

### 自定义数据集类

```python
from src.data import BaseDataset

class CustomPoetryDataset(BaseDataset):
    """自定义诗词数据集类"""
    
    def __init__(self, data_paths, tokenizer=None, context_length=512, **kwargs):
        super().__init__(data_paths, tokenizer, context_length)
        self.poetry_type = kwargs.get("poetry_type", "all")
    
    def _load_single_file(self, path):
        """加载 JSONL 文件"""
        items = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if self.poetry_type == "all" or item.get("category") == self.poetry_type:
                        items.append(item)
        return items
    
    def _process_single_item(self, item):
        """处理单个诗词"""
        text = item.get("text", "")
        tokens = self._tokenize_text(text)
        
        if len(tokens) < 2:
            return None
        
        if len(tokens) <= self.context_length:
            return [{
                "input_ids": tokens + [0] * (self.context_length - len(tokens)),
                "labels": tokens + [-100] * (self.context_length - len(tokens))
            }]
        
        # 滑动窗口
        samples = []
        for i in range(0, len(tokens) - self.context_length, self.context_length // 2):
            samples.append({
                "input_ids": tokens[i:i + self.context_length],
                "labels": tokens[i + 1:i + self.context_length + 1]
            })
        return samples

# 使用自定义数据集
dataset = CustomPoetryDataset(
    data_paths="collection/poetry.jsonl",
    tokenizer=tokenizer,
    poetry_type="tang"
)
```

### 数据集缓存

```python
import pickle

def cache_dataset(dataset, cache_path):
    """缓存数据集"""
    with open(cache_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"数据集已缓存到 {cache_path}")

def load_cached_dataset(cache_path):
    """加载缓存的数据集"""
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

# 使用缓存
try:
    train_ds = load_cached_dataset("dataset/train_cache.pkl")
except FileNotFoundError:
    train_ds = PretrainDataset(...)
    cache_dataset(train_ds, "dataset/train_cache.pkl")
```

## 📊 配置参数参考

### PretrainDataset 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_paths` | str/Path/List | 必需 | 数据文件路径 |
| `tokenizer` | Tokenizer | None | 分词器实例 |
| `context_length` | int | 512 | 最大序列长度 |
| `stride` | int | 256 | 滑动窗口步长 |
| `min_length` | int | 10 | 最小文本长度 |
| `max_length` | int | 100000 | 最大文本长度 |
| `streaming` | bool | False | 是否流式加载 |
| `add_special_tokens` | bool | True | 是否添加特殊 token |

### SFTDataset 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_paths` | str/Path/List | 必需 | 数据文件路径 |
| `tokenizer` | Tokenizer | None | 分词器实例 |
| `context_length` | int | 512 | 最大序列长度 |
| `dialogue_format` | str | "sharegpt" | 对话格式 |
| `chat_template` | ChatTemplate | Default | 聊天模板 |
| `max_turns` | int | None | 最大对话轮数 |
| `drop_incomplete` | bool | False | 是否丢弃不完整对话 |

### WeightedMixingDataset 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `datasets` | List[tuple] | 必需 | (数据集, 权重) 元组列表 |
| `replacement` | bool | True | 是否放回采样 |
| `seed` | int | 42 | 随机种子 |

## ⚠️ 注意事项

1. **路径格式**：Windows 系统使用反斜杠，Unix 系统使用正斜杠
2. **编码格式**：始终使用 UTF-8 编码
3. **内存管理**：大数据集使用流式加载
4. **数据验证**：加载后验证数据完整性

## 📖 相关文档

- [数据系统概览](README.md) - 数据系统整体架构
- [数据获取指南](acquisition.md) - 数据下载和导入
- [数据处理指南](processing.md) - 数据清洗和转换
