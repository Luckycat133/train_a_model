# 灵猫墨韵数据处理指南

本文档详细说明数据清洗、过滤、标准化和格式转换的方法和工具。

## 🧹 数据清洗步骤

### 1. 基本清洗流程

```python
from processors.processor import clean_text, detect_encoding

# 清洗规则配置
cleaning_rules = {
    "remove_html": True,              # 移除 HTML 标签
    "normalize_whitespace": True,     # 规范化空白字符
    "remove_control_chars": True,     # 移除控制字符
    "remove_urls": True,              # 移除 URL
    "filter_quality": True,           # 质量过滤
    "min_length": 10,                 # 最小文本长度
    "max_symbol_ratio": 0.5           # 最大符号比例
}

# 清洗文本
raw_text = "这是原始文本，<p>包含HTML标签</p>和URL https://example.com"
cleaned = clean_text(raw_text, cleaning_rules)
```

### 2. HTML 标签移除

```python
import re

_html_re = re.compile(r'<[^>]+>')

def remove_html_tags(text):
    """移除 HTML 标签"""
    return _html_re.sub('', text)

# 示例
text = "<p>床前明月光，<b>疑是地上霜</b></p>"
cleaned = remove_html_tags(text)
# 结果: "床前明月光，疑是地上霜"
```

### 3. 控制字符过滤

```python
import re

_control_re = re.compile(r'[\x00-\x1F\x7F]')

def remove_control_characters(text):
    """移除控制字符"""
    return _control_re.sub('', text)

# 控制字符列表
# \x00-\x1F: ASCII 控制字符 (0-31)
# \x7F: DEL 字符
```

### 4. URL 移除

```python
import re

_url_re = re.compile(r'https?://\S+')

def remove_urls(text):
    """移除 URL"""
    return _url_re.sub('', text)

# 示例
text = "访问 https://example.com 了解更多信息。"
cleaned = remove_urls(text)
# 结果: "访问  了解更多信息。"
```

## 🔍 数据过滤规则

### 1. 长度过滤

```python
def filter_by_length(text, min_len=10, max_len=100000):
    """按长度过滤文本"""
    length = len(text)
    return min_len <= length <= max_len

# 在数据集加载时应用
from src.data import PretrainDataset

dataset = PretrainDataset(
    data_paths="data.jsonl",
    min_length=10,      # 过滤短于10字符的文本
    max_length=50000,   # 过滤长于50000字符的文本
    context_length=512
)
```

### 2. 质量过滤

```python
import chardet

def calculate_quality_score(text):
    """计算文本质量分数"""
    if not text:
        return 0.0
    
    # 1. 字符多样性
    unique_chars = len(set(text))
    diversity = unique_chars / len(text) if len(text) > 0 else 0
    
    # 2. 标点符号比例
    punctuation_count = sum(1 for c in text if c in '，。！？；：、""''（）')
    punctuation_ratio = punctuation_count / len(text) if len(text) > 0 else 0
    
    # 3. 中文字符比例
    chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    chinese_ratio = chinese_count / len(text) if len(text) > 0 else 0
    
    # 综合分数
    score = (diversity * 0.3 + (1 - punctuation_ratio) * 0.3 + chinese_ratio * 0.4)
    return score

def filter_by_quality(text, threshold=0.5):
    """按质量分数过滤"""
    return calculate_quality_score(text) >= threshold
```

### 3. 符号比例过滤

```python
def filter_symbol_ratio(text, max_ratio=0.5):
    """过滤符号比例过高的文本"""
    symbol_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    ratio = symbol_count / len(text) if len(text) > 0 else 0
    return ratio <= max_ratio

# 示例
text = "《》「」『』【】〔〕（）"  # 全是标点
is_valid = filter_symbol_ratio(text, max_ratio=0.5)
# 结果: False，符号比例过高
```

### 4. 重复内容过滤

```python
def filter_duplicates(texts, similarity_threshold=0.9):
    """过滤重复或高度相似的文本"""
    from difflib import SequenceMatcher
    
    unique_texts = []
    for text in texts:
        is_duplicate = False
        for existing in unique_texts:
            similarity = SequenceMatcher(None, text, existing).ratio()
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_texts.append(text)
    
    return unique_texts

def filter_exact_duplicates(texts):
    """过滤完全重复的文本"""
    seen = set()
    unique = []
    
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    
    return unique
```

### 5. 乱码过滤

```python
def filter_garbled_text(text):
    """过滤可能是乱码的文本"""
    # 1. 检测有效字符比例
    valid_chars = sum(1 for c in text if (
        c.isprintable() or 
        '\u4e00' <= c <= '\u9fff' or  # 中文
        '\u3400' <= c <= '\u4dbf'     # 扩展中文
    ))
    
    valid_ratio = valid_chars / len(text) if len(text) > 0 else 0
    
    if valid_ratio < 0.8:
        return False
    
    # 2. 检测连续异常字符
    abnormal_patterns = [
        r'[\x00-\x08]',  # 控制字符
        r'[�]{3,}',      # 替换字符
    ]
    
    for pattern in abnormal_patterns:
        if re.search(pattern, text):
            return False
    
    return True

# 使用 chardet 验证编码
def validate_encoding(text_bytes):
    """验证文本编码是否正确"""
    result = chardet.detect(text_bytes)
    
    # 置信度阈值
    if result['confidence'] < 0.7:
        return False, "编码检测置信度低"
    
    # 编码类型检查
    encoding = result['encoding'].upper()
    if encoding in ['GB2312', 'GB18030', 'UTF-8', 'UTF-16']:
        return True, encoding
    
    return True, encoding
```

## 📝 文本标准化选项

### 1. 空白字符规范化

```python
import re

def normalize_whitespace(text):
    """规范化空白字符"""
    # 将所有空白字符替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 移除首尾空白
    return text.strip()

# 示例
text = "床前    明月  光\n疑是   地上霜"
normalized = normalize_whitespace(text)
# 结果: "床前 明月 光 疑是 地上霜"
```

### 2. 全角半角转换

```python
def to_halfwidth(text):
    """全角转半角"""
    result = []
    for char in text:
        code = ord(char)
        # 全角字符范围: 0xFF01-0xFF5E
        if 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        result.append(chr(code))
    return ''.join(result)

def to_fullwidth(text):
    """半角转全角"""
    result = []
    for char in text:
        code = ord(char)
        # 半角字符范围: 0x0021-0x007E
        if 0x0021 <= code <= 0x007E:
            code += 0xFEE0
        result.append(chr(code))
    return ''.join(result)

# 示例
text = "ＡＢＣ１２３"  # 全角
halfwidth = to_halfwidth(text)
# 结果: "ABC123"
```

### 3. 繁简转换

```python
# 需要安装 opencc-python-reimplemented
# pip install opencc-python-reimplemented

from opencc import OpenCC

cc = OpenCC('s2t')  # 简体中文转繁体中文

def convert_to_traditional(text):
    """简体转繁体"""
    return cc.convert(text)

def convert_to_simplified(text):
    """繁体转简体"""
    cc = OpenCC('t2s')
    return cc.convert(text)

# 示例
simplified = "床前明月光"
traditional = convert_to_traditional(simplified)
# 结果: "床前明月光的"
```

### 4. 标点符号规范化

```python
def normalize_punctuation(text):
    """规范化标点符号"""
    # 定义标点映射
    punctuation_map = {
        ',': '，',  # 英文逗号转中文逗号
        '.': '。',
        '!': '！',
        '?': '？',
        ':': '：',
        ';': '；',
        '"': '""',  # 英文引号转中文
        "'": '''',
        '(': '（',
        ')': '）',
    }
    
    result = []
    for char in text:
        result.append(punctuation_map.get(char, char))
    
    return ''.join(result)

# 示例
text = "Hello, world! Is this a test?"
normalized = normalize_punctuation(text)
# 结果: "Hello，world！Is this a test？"
```

### 5. 特殊字符处理

```python
def normalize_special_chars(text):
    """处理特殊字符"""
    # 替换常见的特殊空格
    text = text.replace('\u3000', ' ')  # 全角空格
    text = text.replace('\u00A0', ' ')  # 不间断空格
    text = text.replace('\u2002', ' ')  # 半角空格
    
    # 移除零宽字符
    zero_width_chars = [
        '\u200B',  # 零宽空格
        '\u200C',  # 零宽非连接符
        '\u200D',  # 零宽连接符
        '\uFEFF',  # 字节顺序标记
    ]
    
    for char in zero_width_chars:
        text = text.replace(char, '')
    
    return text
```

## 🔄 格式转换工具

### 1. JSONL 转换

```python
import json

def json_to_jsonl(input_file, output_file, text_field='text'):
    """JSON 数组转 JSONL"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if isinstance(data, list):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def jsonl_to_json(input_file, output_file):
    """JSONL 转 JSON 数组"""
    records = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
```

### 2. 文本转 JSONL

```python
def txt_to_jsonl(input_file, output_file, metadata=None):
    """纯文本转 JSONL"""
    metadata = metadata or {}
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            record = {
                'text': line,
                'line_number': line_num,
                **metadata
            }
            
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

def txt_to_jsonl_paragraphs(input_file, output_file, separator='\n\n'):
    """按段落分割文本"""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    paragraphs = content.split(separator)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for para in paragraphs:
            record = {'text': para}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
```

### 3. CSV 转 JSONL

```python
import csv

def csv_to_jsonl(input_file, output_file, text_column='text'):
    """CSV 转 JSONL"""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        reader = csv.DictReader(f_in)
        
        for row in reader:
            if text_column in row:
                record = {text_column: row[text_column]}
                # 添加其他列作为元数据
                for key, value in row.items():
                    if key != text_column:
                        record[key] = value
                
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

def jsonl_to_csv(input_file, output_file):
    """JSONL 转 CSV"""
    records = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if not records:
        return
    
    # 获取所有字段
    fieldnames = set()
    for record in records:
        fieldnames.update(record.keys())
    fieldnames = list(fieldnames)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
```

### 4. 数据集格式转换

```python
def poetry_json_to_jsonl(input_file, output_file):
    """古诗词 JSON 转 JSONL"""
    with open(input_file, 'r', encoding='utf-8') as f:
        poems = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for poem in poems:
            # 合并段落为完整文本
            paragraphs = poem.get('paragraphs', [])
            text = '\n'.join(paragraphs)
            
            record = {
                'text': text,
                'author': poem.get('author', ''),
                'title': poem.get('title', ''),
                'category': poem.get('category', '')
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def jsonl_to_training_format(input_file, output_file, add_labels=True):
    """JSONL 转训练格式"""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            record = json.loads(line.strip())
            
            # 创建训练样本
            training_record = {
                'text': record.get('text', ''),
                'input_ids': record.get('input_ids', []),
                'attention_mask': record.get('attention_mask', [1] * len(record.get('input_ids', [])))
            }
            
            if add_labels:
                training_record['labels'] = record.get('input_ids', [])
            
            f_out.write(json.dumps(training_record, ensure_ascii=False) + '\n')
```

## 🔧 批处理工具

### 1. 批量处理脚本

```python
#!/usr/bin/env python3
"""
processors/processor.py - 数据批处理工具
"""

import json
import re
import chardet
from pathlib import Path
from typing import List, Dict, Generator, Iterable

# 正则表达式（从源码复用）
_html_re = re.compile(r'<[^>]+>')
_url_re = re.compile(r'https?://\S+')
_control_re = re.compile(r'[\x00-\x1F\x7F]')


def clean_text(text: str, rules: Dict) -> str:
    """根据规则清洗文本"""
    result = text
    
    if rules.get("remove_html"):
        result = _html_re.sub("", result)
    
    if rules.get("normalize_whitespace"):
        result = re.sub(r"\s+", " ", result).strip()
    
    if rules.get("remove_control_chars"):
        result = _control_re.sub("", result)
    
    if rules.get("remove_urls"):
        result = _url_re.sub("", result)
    
    if rules.get("filter_quality"):
        min_length = rules.get("min_length", 0)
        max_ratio = rules.get("max_symbol_ratio", 1.0)
        
        if len(result) < min_length:
            return ""
        
        if result:
            symbols = sum(not ch.isalnum() and not ch.isspace() for ch in result)
            if symbols / len(result) > max_ratio:
                return ""
    
    return result


def read_data(file_list: Iterable[str], allowed_exts: List[str], 
              batch_size: int = 100, preview: bool = False) -> Generator[List[str], None, None]:
    """批量读取文件"""
    batch: List[str] = []
    
    for file in file_list:
        if allowed_exts and not any(file.endswith(ext) for ext in allowed_exts):
            continue
        
        encoding = detect_encoding(file)
        
        with open(file, "r", encoding=encoding, errors="strict") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                batch.append(line)
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    
    if batch:
        yield batch


def detect_encoding(file_path: str) -> str:
    """检测文件编码"""
    with open(file_path, "rb") as f:
        raw = f.read()
    
    if not raw:
        return "utf-8"
    
    result = chardet.detect(raw)
    encoding = result.get("encoding") or "utf-8"
    
    if encoding.upper() == "TIS-620":
        encoding = "GBK"
    
    return encoding
```

### 2. 命令行使用

```bash
# 基本用法
python processors/processor.py --help

# 批量处理目录中的文件
python processors/processor.py \
    --input collection/raw/ \
    --output collection/cleaned/ \
    --rules remove_html,normalize_whitespace,remove_urls

# 转换格式
python processors/processor.py \
    --input data/poetry.json \
    --output data/poetry.jsonl \
    --convert json_to_jsonl
```

## 📋 数据处理配置示例

```yaml
# config/data_processing.yaml

processing:
  cleaning:
    remove_html: true
    normalize_whitespace: true
    remove_control_chars: true
    remove_urls: true
  
  filtering:
    min_length: 10
    max_length: 50000
    max_symbol_ratio: 0.5
    min_quality_score: 0.5
    remove_duplicates: true
  
  normalization:
    convert_to_halfwidth: false
    traditional_to_simplified: false
    normalize_punctuation: true
    remove_zero_width_chars: true

output:
  format: "jsonl"
  encoding: "utf-8"
  compression: null
```

## ⚠️ 注意事项

1. **数据备份**：处理前请先备份原始数据
2. **渐进处理**：大规模数据建议分批处理
3. **验证结果**：处理后进行数据质量验证
4. **日志记录**：记录处理过程中的问题

## 📖 相关文档

- [数据系统概览](README.md) - 数据系统整体架构
- [数据获取指南](acquisition.md) - 数据下载和导入
- [数据配置指南](configuration.md) - 数据集配置方法
