# 数据获取与预处理指南

本文档介绍如何获取、下载和预处理灵猫墨韵项目的训练数据。

## 目录

- [获取数据](#获取数据)
- [数据下载步骤](#数据下载步骤)
- [数据预处理流程](#数据预处理流程)
- [快速开始](#快速开始)
- [常见问题](#常见问题)

---

## 获取数据

### 方法一：使用内置脚本获取数据

灵猫墨韵项目提供了内置的数据下载脚本，可以自动获取预处理的数据：

```bash
# 进入项目目录
cd /workspace

# 运行数据下载脚本
python processors/download_poetry.py
```

### 方法二：手动下载数据集

如果需要手动获取数据，可以从以下来源下载：

| 数据源 | 描述 | 下载地址 |
|--------|------|----------|
| 古诗文网 | 古诗词数据 | https://www.gushiwen.org/ |
| 中国哲学书电子化计划 | 古典哲学文献 | https://ctext.org/ |
| GitHub 开放数据集 | 预整理的中文古典文献 | 各开源仓库 |

### 方法三：使用开放 API

```python
# 示例：使用古诗文 API 获取数据
import requests
import json

def fetch_poetry():
    """从古诗文网获取诗词数据"""
    url = "https://api.gushiwen.org/shiwen/rest/v1/shiwen/getShiWen"
    params = {"random": "1", "id": "1"}
    response = requests.get(url, params=params)
    return response.json()
```

---

## 数据下载步骤

### 步骤 1：准备下载目录

```bash
# 创建数据目录
mkdir -p dataset/raw
mkdir -p dataset/processed
mkdir -p dataset/tokenized

# 设置环境变量
export DATA_DIR=/workspace/dataset
```

### 步骤 2：下载原始数据

#### 下载古诗词数据

```bash
# 使用 wget 下载
wget -O dataset/raw/poetry.json https://example.com/poetry.json

# 使用 curl 下载
curl -o dataset/raw/poetry.json https://example.com/poetry.json
```

#### 下载古文经典数据

```bash
# 下载论语
wget -O dataset/raw/lunyu.json https://example.com/lunyu.json

# 下载道德经
wget -O dataset/raw/dao.json https://example.com/dao.json
```

### 步骤 3：验证下载完整性

```bash
# 检查文件大小
ls -lh dataset/raw/

# 验证 JSON 文件格式
python -c "import json; json.load(open('dataset/raw/poetry.json'))"

# 统计下载的数据量
wc -l dataset/raw/*.jsonl
```

---

## 数据预处理流程

### 流程概览

```
原始数据 → 清洗 → 格式转换 → 合并 → 分词 → 训练格式
    ↓        ↓         ↓          ↓        ↓        ↓
  .json    .jsonl   .jsonl     .jsonl   .pt/.bin  .pt
```

### 步骤 1：数据清洗

数据清洗是确保数据质量的关键步骤：

```python
# processors/processor.py 中的清洗函数
from processors.processor import clean_text

# 清洗配置
cleaning_rules = {
    "remove_html": True,           # 移除 HTML 标签
    "normalize_whitespace": True,  # 规范化空白字符
    "remove_control_chars": True,  # 移除控制字符
    "remove_urls": True,           # 移除 URL
    "filter_quality": True,        # 质量过滤
    "min_length": 10,              # 最小文本长度
    "max_symbol_ratio": 0.5,      # 最大符号比例
}

# 清洗文本
cleaned_text = clean_text(raw_text, cleaning_rules)
```

### 步骤 2：格式转换

将不同格式的数据转换为统一的 JSONL 格式：

```python
import json

def convert_to_jsonl(input_file, output_file):
    """将各种格式转换为 JSONL"""
    with open(input_file, 'r', encoding='utf-8') as fin:
        with open(output_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                try:
                    data = json.loads(line.strip())
                    # 确保必要字段存在
                    if 'content' in data or 'text' in data:
                        json.dump(data, fout, ensure_ascii=False)
                        fout.write('\n')
                except json.JSONDecodeError:
                    continue
```

### 步骤 3：数据合并

将多个数据源合并为单一训练文件：

```bash
# 使用 cat 合并 JSONL 文件
cat dataset/processed/poetry.jsonl \
    dataset/processed/classical.jsonl \
    > dataset/processed/merged.jsonl

# 去重处理
python -c "
import json
seen = set()
with open('dataset/processed/merged.jsonl', 'w') as out:
    for line in open('dataset/processed/merged.jsonl'):
        obj = json.loads(line)
        key = obj.get('content', '')
        if key and key not in seen:
            seen.add(key)
            out.write(json.dumps(obj, ensure_ascii=False) + '\n')
"
```

### 步骤 4：分词处理

使用项目提供的分词器进行分词：

```bash
# 准备分词器
python -m src.run --prepare

# 或使用 Python 代码
from tokenizer import ClassicalTokenizer

tokenizer = ClassicalTokenizer()
tokens = tokenizer.encode("床前明月光，疑是地上霜")
print(f"Token count: {len(tokens)}")
```

### 步骤 5：转换为训练格式

```python
# 将数据转换为 PyTorch 可用的格式
import torch
from torch.utils.data import Dataset

class PoetryDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx].get('content', '')
        tokens = self.tokenizer.encode(text)
        
        # 截断或填充到固定长度
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens += [0] * (self.max_length - len(tokens))
        
        return torch.tensor(tokens)
```

---

## 快速开始

### 一键数据准备

使用项目提供的一键脚本准备数据：

```bash
# 赋予执行权限
chmod +x scripts/data_pipeline.sh

# 运行完整数据管道
./scripts/data_pipeline.sh --full

# 仅下载数据
./scripts/data_pipeline.sh --download

# 仅预处理数据
./scripts/data_pipeline.sh --process

# 指定数据源
./scripts/data_pipeline.sh --source poetry --source classical
```

### 使用示例

```bash
# 示例 1：完整流程
cd /workspace
./scripts/data_pipeline.sh --full --workers 4

# 示例 2：仅下载诗词数据
./scripts/data_pipeline.sh --download --type poetry

# 示例 3：仅处理现有数据
./scripts/data_pipeline.sh --process --input dataset/raw --output dataset/processed

# 示例 4：指定自定义配置
./scripts/data_pipeline.sh --config config/data_pipeline.yaml
```

---

## 常见问题

### Q1: 下载速度很慢怎么办？

**解决方案：**

1. 使用国内镜像源
2. 使用代理或 VPN
3. 分段下载大文件

```bash
# 使用axel多线程下载
axel -n 10 -o dataset/raw/ https://example.com/large_file.zip

# 使用curl断点续传
curl -C - -o dataset/raw/large_file.zip https://example.com/large_file.zip
```

### Q2: 数据清洗后数据量大幅减少正常吗？

**答案：** 是正常的。数据清洗会移除以下内容：

- HTML 标签和特殊字符
- 过短或过长的文本
- 重复内容
- 格式不规范的条目

通常清洗后保留 70-80% 的原始数据是正常的。

### Q3: 如何处理编码问题？

**解决方案：**

项目使用 `chardet` 自动检测编码：

```python
from processors.processor import detect_encoding

encoding = detect_encoding("path/to/file.txt")
print(f"Detected encoding: {encoding}")

# 手动指定编码
with open("path/to/file.txt", "r", encoding="utf-8") as f:
    content = f.read()
```

常见编码问题处理：

```bash
# 转换编码
iconv -f GBK -t UTF-8 input.txt > output.txt

# 或使用 Python
python -c "
import codecs
with codecs.open('input.txt', 'r', 'gbk') as f:
    content = f.read()
with open('output.txt', 'w', 'utf-8') as f:
    f.write(content)
"
```

### Q4: 分词器加载失败怎么办？

**解决方案：**

1. 检查分词器文件是否存在
2. 重新生成分词器
3. 使用备用分词器

```bash
# 重新生成tokenizer
python -m src.run --prepare --force

# 使用HuggingFace分词器
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
tokenizer.save_pretrained('dataset/tokenizer')
"
```

### Q5: 内存不足如何处理？

**解决方案：**

1. 使用流式处理模式
2. 分批处理数据
3. 增加交换空间

```python
# 使用流式数据集
from src.data import StreamingDataset

dataset = StreamingDataset(
    data_paths="dataset/processed/merged.jsonl",
    tokenizer=tokenizer,
    context_length=512,
)

# 迭代处理
for batch in dataset.get_stream_iterator(batch_size=32):
    # 处理每个批次
    pass
```

### Q6: 如何验证数据质量？

**质量检查清单：**

```python
def validate_dataset(jsonl_path):
    """验证数据集质量"""
    issues = []
    
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                
                # 检查必要字段
                if 'content' not in data and 'text' not in data:
                    issues.append(f"Line {i}: Missing content field")
                
                # 检查内容长度
                content = data.get('content', data.get('text', ''))
                if len(content) < 5:
                    issues.append(f"Line {i}: Content too short")
                
                # 检查是否为空
                if not content.strip():
                    issues.append(f"Line {i}: Empty content")
                    
            except json.JSONDecodeError:
                issues.append(f"Line {i}: Invalid JSON")
    
    if issues:
        print("数据质量问题：")
        for issue in issues[:10]:  # 只显示前10个问题
            print(f"  - {issue}")
    else:
        print("✓ 数据集验证通过")
    
    return len(issues)
```

### Q7: 如何添加自定义数据源？

**步骤：**

1. 准备数据文件（JSONL 格式）
2. 添加数据源配置
3. 运行预处理

```python
# 1. 创建数据源配置
CUSTOM_SOURCES = {
    "my_poetry": {
        "path": "dataset/custom/poetry.jsonl",
        "weight": 1.0,
        "category": "poetry",
    },
    "my_classical": {
        "path": "dataset/custom/classical.jsonl", 
        "weight": 0.5,
        "category": "classical",
    }
}

# 2. 合并到训练数据
from src.data import WeightedMixingDataset

datasets = [
    (PoetryDataset("dataset/processed/poetry.jsonl"), 1.0),
    (ClassicalDataset("dataset/processed/classical.jsonl"), 0.5),
]

mixed_dataset = WeightedMixingDataset(datasets)
```

---

## 故障排除

### 问题诊断流程

```
1. 检查环境配置
   └─ python --version
   └─ pip list | grep -E "torch|numpy|transformers"

2. 检查数据文件
   └─ ls -la dataset/
   └─ head -n 5 dataset/processed/merged.jsonl

3. 检查权限
   └─ ls -l scripts/data_pipeline.sh
   └─ chmod +x scripts/data_pipeline.sh

4. 查看日志
   └─ tail -n 50 logs/data_pipeline.log
```

### 联系支持

如遇到无法解决的问题，请提交 Issue 并包含：

1. 错误信息完整输出
2. 系统环境信息（Python 版本、操作系统）
3. 已尝试的解决方法

---

## 相关文档

- [数据格式说明](formats.md) - 详细的数据格式规格
- [数据集说明](../../DATASET.md) - 完整数据集文档
- [API 文档](../cn/DOCS_cn.md) - 项目 API 使用说明

---

_最后更新：2026-05-15_
