# 灵猫墨韵生成模块使用指南

本文档提供灵猫墨韵文本生成模块 (generate.py) 的使用说明。

## 概述

生成模块允许您使用训练好的模型生成古典风格的文本。支持多种生成策略和参数控制。

## 基本使用

```bash
# 基本生成示例
python generate.py --prompt "春江潮水连海平" --max_length 100
```

## 主要参数

- `--model_path`：模型权重路径，默认为"model_weights/best_model_v0.7.0.pt"
- `--tokenizer_path`：分词器路径，默认为"tokenizer.json"
- `--prompt`：生成起始提示文本，如"春江花月夜"
- `--max_length`：生成的最大长度，默认150
- `--temperature`：温度参数(0.1-2.0)，控制生成的随机性，默认0.7
- `--top_k`：保留概率最高的k个词进行采样，默认50
- `--top_p`：使用nucleus sampling的概率阈值，默认0.9
- `--no_repeat_ngram_size`：避免重复的n元组大小，默认3

## 高级选项

### 生成策略

```bash
# 使用beam search生成
python generate.py --prompt "明月几时有" --strategy beam --num_beams 5 --max_length 200

# 使用随机采样生成多样文本
python generate.py --prompt "花间一壶酒" --temperature 1.2 --strategy sample --max_length 100
```

### 批量生成

```bash
# 批量生成多个文本
python generate.py --prompt "明月几时有" --num_return_sequences 5 --max_length 50
```

## 输出控制

```bash
# 保存生成结果到文件
python generate.py --prompt "明月几时有" --output_file generated_poems.txt

# 详细输出模式，显示每步生成的概率分布
python generate.py --prompt "明月几时有" --verbose
```

## 示例输出

以下是使用默认参数生成的示例：

**输入提示**：春江潮水连海平

**生成结果**：
```
春江潮水连海平，海上明月共潮生。
滟滟随波千万里，何处春江无月明。
江流宛转绕芳甸，月照花林皆似霰。
空里流霜不觉飞，汀上白沙看不见。
```

## 常见问题

**Q: 如何使生成的文本更多样化？**  
A: 增加temperature参数值（如0.8-1.5之间）以增加随机性。

**Q: 生成的文本重复怎么办？**  
A: 增加no_repeat_ngram_size参数或降低temperature参数。

**Q: 如何生成更连贯的长文本？**  
A: 使用beam search策略，设置较高的num_beams值（5-10）。

## 高级用法

### 集成到Python代码

```python
from generate import generate_text

# 生成文本
generated = generate_text(
    prompt="明月几时有",
    model_path="model_weights/my_custom_model.pt",
    max_length=200,
    temperature=0.7
)
print(generated)
```

---

*最后更新: 2023年3月14日*
