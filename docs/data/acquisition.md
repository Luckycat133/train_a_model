# 灵猫墨韵数据获取指南

本文档详细说明如何下载和导入各种数据源，包括诗词数据、古文典籍以及如何添加自定义数据源。

## 📥 下载诗词数据

### 使用内置下载脚本

灵猫墨韵项目提供了 `processors/download_poetry.py` 脚本，用于从 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) GitHub 仓库下载中文诗词数据。

#### 基本用法

```bash
python processors/download_poetry.py
```

#### 工作原理

1. 通过 GitHub API 列出仓库中的 JSON 文件
2. 下载并解析每个 JSON 文件
3. 将原始 JSON 格式转换为 JSONL 格式
4. 保存到 `collection/chinese-poetry/` 目录

#### 支持的诗词类型

| 类型 | 文件 | 数量 | 说明 |
|------|------|------|------|
| 唐诗 | poet.tang.*.json | ~30,000首 | 按卷分文件存储 |
| 宋词 | poet.song.*.json | ~20,000首 | 按卷分文件存储 |
| 楚辞 | chuci.json | ~67首 | 屈原等人作品 |
| 元曲 | - | 待添加 | - |

#### 输出文件

```
collection/chinese-poetry/
├── poetry.jsonl           # 所有诗词合并文件
├── poet.tang.0.jsonl     # 单独保存唐诗卷
├── poet.tang.1.jsonl
├── ...
├── poet.song.0.jsonl     # 单独保存宋词卷
└── chuci.jsonl           # 楚辞
```

### 手动下载大词库

对于完整词库，建议直接从 GitHub 下载：

```bash
# 克隆 chinese-poetry 仓库
git clone https://github.com/chinese-poetry/chinese-poetry.git

# 或者下载打包文件
wget https://github.com/chinese-poetry/chinese-poetry/archive/refs/heads/master.zip
unzip master.zip
```

## 📚 下载古文典籍

### 推荐的公开数据集

#### 1. Chinese Classical Literature

从 [chinese-classics](https://github.com/chinese-classics) 获取古籍数据：

```bash
# 论语
wget https://raw.githubusercontent.com/chinese-classics/lunyu/master/lunyu.json

# 道德经
wget https://raw.githubusercontent.com/chinese-classics/dao-de-jing/master/dao.json

# 庄子
wget https://raw.githubusercontent.com/chinese-classics/zhuangzi/master/zhuang.json
```

#### 2. 古文观止

从古文观止语料库获取：

```python
# 示例：从网络获取（需要实现爬虫）
import urllib.request
import json

url = "https://raw.githubusercontent.com/gdut-yy/GUWEN/main/guwen.json"
with urllib.request.urlopen(url) as response:
    data = json.loads(response.read())
```

#### 3. 四库全书子集

对于完整版四库全书数据，请访问 [CBDB](https://github.com/d诸子百家) 或 [ctext](https://ctext.org/)。

### 数据格式转换

下载的古籍数据需要转换为项目标准格式：

```python
import json

# 输入：原始 JSON
# 输出：JSONL 格式

def convert_classical_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if isinstance(data, list):
            for item in data:
                record = {
                    'text': item.get('content', item.get('text', '')),
                    'title': item.get('title', ''),
                    'author': item.get('author', ''),
                    'category': 'classical'
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        else:
            record = {
                'text': data.get('content', data.get('text', '')),
                'title': data.get('title', ''),
                'author': data.get('author', ''),
                'category': 'classical'
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

# 使用
convert_classical_text('lunyu.json', 'dataset/lunyu.jsonl')
```

## 🔧 添加自定义数据源

### 步骤 1：准备数据文件

确保数据文件符合以下格式之一：

#### JSONL 格式（推荐）

```jsonl
{"text": "第一首诗词内容", "author": "李白", "category": "tang"}
{"text": "第二首诗词内容", "author": "杜甫", "category": "tang"}
```

#### JSON 数组格式

```json
[
  {"text": "第一首诗词内容", "author": "李白", "category": "tang"},
  {"text": "第二首诗词内容", "author": "杜甫", "category": "tang"}
]
```

#### 纯文本格式

每行一首诗或一篇文章：

```
第一首诗词内容
第二首诗词内容
```

### 步骤 2：创建下载脚本

建议在 `processors/` 目录下创建专门的下载脚本：

```python
#!/usr/bin/env python3
"""
下载自定义数据源的示例脚本
"""

import json
import urllib.request
from pathlib import Path

# 配置
OUTPUT_DIR = Path(__file__).parent.parent / "collection" / "custom"
DATA_SOURCE_URL = "https://example.com/api/data"

def download_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"正在从 {DATA_SOURCE_URL} 下载数据...")
    
    req = urllib.request.Request(
        DATA_SOURCE_URL,
        headers={"Accept": "application/json"}
    )
    
    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    
    # 转换为 JSONL 格式
    output_path = OUTPUT_DIR / "data.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            # 根据数据源调整字段映射
            record = {
                "text": item.get("content", item.get("text", "")),
                "author": item.get("author", "未知"),
                "category": "custom"
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"下载完成，共 {len(data)} 条记录")
    print(f"保存位置: {output_path}")

if __name__ == "__main__":
    download_data()
```

### 步骤 3：注册数据集

在 `src/data/` 目录下添加数据集支持，或直接在代码中使用：

```python
from src.data import PretrainDataset

# 直接使用自定义数据
dataset = PretrainDataset(
    data_paths="collection/custom/data.jsonl",
    context_length=512
)

print(f"加载了 {len(dataset)} 个样本")
```

### 步骤 4：扩展下载脚本（可选）

如需支持更多数据源，可以修改 `processors/download_poetry.py`：

```python
# 在 download_poetry.py 中添加新数据源

ADDITIONAL_SOURCES = [
    {
        "name": "custom_source_1",
        "repo_path": "custom/data1.json",
        "output_name": "custom_data1.jsonl"
    },
    {
        "name": "custom_source_2",
        "repo_path": "custom/data2.json",
        "output_name": "custom_data2.jsonl"
    }
]

# 在 main() 函数中添加下载逻辑
for source in ADDITIONAL_SOURCES:
    records = download_and_convert(source["repo_path"])
    # 保存...
```

## 📡 数据下载 API

### GitHub API 基础用法

```python
import urllib.request
import base64
import json

GITHUB_API = "https://api.github.com"

def github_api_get(path):
    """发送 GitHub API 请求"""
    url = f"{GITHUB_API}{path}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github.v3+json"}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))

def get_file_content(owner, repo, path):
    """获取仓库中单个文件的内容"""
    api_path = f"/repos/{owner}/{repo}/contents/{path}"
    data = github_api_get(api_path)
    
    if isinstance(data, dict) and "content" in data:
        # 文件内容是 Base64 编码的
        content = base64.b64decode(data["content"]).decode("utf-8")
        return json.loads(content)
    return data

def list_files_in_dir(owner, repo, path=""):
    """列出仓库目录中的所有文件"""
    api_path = f"/repos/{owner}/{repo}/contents/{path}"
    contents = github_api_get(api_path)
    
    files = []
    for item in contents:
        if item["type"] == "file":
            files.append(item["name"])
        elif item["type"] == "dir":
            files.extend(list_files_in_dir(owner, repo, item["path"]))
    return files
```

### 使用 requests 库（备选）

```python
import requests

def download_file(url, output_path):
    """使用 requests 下载文件"""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        f.write(response.content)

def download_github_file(owner, repo, path, token=None):
    """从 GitHub 下载文件（支持私有仓库）"""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{path}"
    
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    
    return response.content.decode("utf-8")
```

## 🔍 数据验证

下载数据后，建议进行验证：

```python
import json
from pathlib import Path

def validate_jsonl(file_path, required_fields=None):
    """验证 JSONL 文件格式"""
    required_fields = required_fields or ['text']
    
    valid_count = 0
    invalid_count = 0
    errors = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                
                # 检查必需字段
                missing = [f for f in required_fields if f not in record]
                if missing:
                    errors.append(f"行 {line_num}: 缺少字段 {missing}")
                    invalid_count += 1
                    continue
                
                # 检查文本非空
                if not record.get('text', '').strip():
                    errors.append(f"行 {line_num}: 文本为空")
                    invalid_count += 1
                    continue
                
                valid_count += 1
                
            except json.JSONDecodeError as e:
                errors.append(f"行 {line_num}: JSON 解析错误 - {e}")
                invalid_count += 1
    
    print(f"验证结果: {valid_count} 有效, {invalid_count} 无效")
    
    if errors:
        print("\n错误详情:")
        for error in errors[:10]:  # 只显示前10个错误
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误")
    
    return valid_count, invalid_count

# 使用
validate_jsonl("collection/chinese-poetry/poetry.jsonl", required_fields=['text'])
```

## ⚠️ 常见问题

### 1. GitHub API 速率限制

GitHub API 有速率限制（60次/小时），遇到限制时可以：

```python
import time

def github_api_with_retry(url, max_retries=3):
    """带重试的 API 请求"""
    for attempt in range(max_retries):
        try:
            # ... 发送请求 ...
            return data
        except RateLimitError:
            wait_time = 60 * (attempt + 1)  # 递增等待时间
            print(f"API 速率限制，等待 {wait_time} 秒...")
            time.sleep(wait_time)
    raise Exception("API 请求失败")
```

### 2. 编码问题

处理中文数据时注意编码：

```python
# 始终指定 UTF-8 编码
with open('file.jsonl', 'r', encoding='utf-8') as f:
    ...

# 检测编码
import chardet
with open('file.txt', 'rb') as f:
    raw = f.read()
    encoding = chardet.detect(raw)['encoding']
```

### 3. 网络超时

设置合理的超时时间：

```python
import socket
socket.setdefaulttimeout(60)  # 60秒超时
```

## 📖 相关文档

- [数据系统概览](README.md) - 数据系统整体架构
- [数据处理指南](processing.md) - 数据清洗和转换
- [数据配置指南](configuration.md) - 数据集配置方法
