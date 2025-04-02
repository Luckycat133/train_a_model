# 灵猫墨韵分词器模块使用指南

本文档提供灵猫墨韵分词器模块 (tokenizer.py) 的使用说明。

## 概述

分词器模块专为古典中文文本设计，将原始文本转换为模型训练和推理所需的token序列。

## 基本使用

```bash
# 训练新分词器
python tokenizer.py --input_files dataset/processed_poems.jsonl --vocab_size 50000 --output_file tokenizer.json

# 使用已有分词器处理文本
python tokenizer.py --text "春江潮水连海平" --tokenizer_file tokenizer.json
```

## 主要功能

1. **分词器训练**：从语料库学习词汇表和分词规则
2. **文本分词**：将文本切分为token序列
3. **序列编码/解码**：token与ID之间的双向转换
4. **特殊token处理**：支持添加和处理特殊token

## 主要参数

### 训练分词器

- `--input_files`：训练数据文件路径，支持多文件(用逗号分隔)
- `--vocab_size`：词汇表大小，默认50000
- `--min_frequency`：最小词频，低于此频率的token将被忽略，默认2
- `--output_file`：分词器保存路径，默认tokenizer.json
- `--special_tokens`：特殊token列表，默认"[PAD],[UNK],[BOS],[EOS],[SEP]"

### 使用分词器

- `--tokenizer_file`：分词器文件路径
- `--text`：要处理的文本
- `--encode_mode`：编码模式(ids/tokens)，默认ids

## 高级选项

### 自定义分词器类型

```bash
# 训练字符级分词器
python tokenizer.py --input_files dataset/poems.jsonl --tokenizer_type char

# 训练词级分词器(使用jieba分词)
python tokenizer.py --input_files dataset/poems.jsonl --tokenizer_type word

# 训练BPE分词器
python tokenizer.py --input_files dataset/poems.jsonl --tokenizer_type bpe
```

### 自定义特殊Token

```bash
# 设置自定义特殊token
python tokenizer.py --input_files dataset/poems.jsonl --special_tokens "[PAD],[UNK],[BOS],[EOS],[MASK],[CLS]"
```

## 使用示例

### 从命令行使用

```bash
# 训练一个词汇量为30000的分词器
python tokenizer.py \
  --input_files dataset/corpus_train.jsonl,dataset/corpus_valid.jsonl \
  --vocab_size 30000 \
  --min_frequency 3 \
  --output_file tokenizer_v2.json \
  --special_tokens "[PAD],[UNK],[BOS],[EOS],[SEP],[MASK]"

# 使用分词器处理文本
python tokenizer.py \
  --tokenizer_file tokenizer_v2.json \
  --text "明月几时有，把酒问青天。" \
  --encode_mode tokens
```

### Python API 用法

```python
from tokenizer import ClassicalTokenizer

# 加载分词器
tokenizer = ClassicalTokenizer('tokenizer.json')

# 分词获取token
tokens = tokenizer.tokenize("明月几时有，把酒问青天。")
print("分词结果:", tokens)

# 编码为ID
ids = tokenizer.encode("明月几时有，把酒问青天。")
print("编码结果:", ids)

# 解码ID为文本
text = tokenizer.decode([101, 2356, 3258, 2417, 5689, 8821, 7705])
print("解码结果:", text)
```

## 常见问题

**Q: 如何查看分词器的词汇表？**  
A: 使用 `tokenizer.get_vocab()` 方法可以获取完整词汇表。

**Q: 中英文混合文本如何处理？**  
A: 默认会对中英文混合文本进行适当处理，英文会按词切分，中文会根据分词器类型处理。

**Q: 如何处理OOV(词表外)的词？**  
A: 未知词会被替换为`[UNK]`标记，可通过设置更大的vocab_size来减少OOV。

## 性能与限制

- 推荐词汇表大小: 30,000-50,000 (根据语料库大小调整)
- 最大序列长度: 512 tokens (可通过max_length参数调整)
- 支持的文件格式: txt, json, jsonl

---

*最后更新: 2023年3月14日*
