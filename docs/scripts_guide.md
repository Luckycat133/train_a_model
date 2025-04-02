# 灵猫墨韵 - 脚本使用说明

本文档描述了灵猫墨韵古典文学数据处理系统中各脚本的用法和功能。

## processor.py - 数据处理主脚本

`processor.py` 是灵猫墨韵系统的主处理脚本，整合了数据处理、分词器训练、数据优化和文件管理等功能。

### 命令行参数

```bash
python processor.py [选项]
```

选项说明：

| 参数 | 说明 |
|------|------|
| `--config CONFIG` | 指定配置文件路径，默认为 `config/config.yaml` |
| `--force` | 强制重新处理所有数据，忽略已有处理结果 |
| `--clean` | 清理旧文件和目录，包括临时文件、旧处理文件和备份文件 |
| `--optimize` | 优化数据（合并文件、拆分数据集、移动tokenizer） |
| `--validate` | 验证数据完整性，确保所有必需文件存在且格式正确 |
| `--process` | 处理原始数据，生成处理后的数据文件 |
| `--tokenize` | 训练分词器并将结果保存到主目录 |
| `--all` | 执行所有步骤（处理、分词器训练、优化、验证） |

### 使用示例

1. 查看帮助信息：
   ```bash
   python processor.py --help
   ```

2. 执行所有处理步骤：
   ```bash
   python processor.py --all
   ```

3. 只清理旧文件：
   ```bash
   python processor.py --clean
   ```

4. 处理数据并训练分词器：
   ```bash
   python processor.py --process --tokenize
   ```

5. 优化数据并验证完整性：
   ```bash
   python processor.py --optimize --validate
   ```

### 数据处理流程

完整的数据处理流程包括以下步骤：

1. **处理原始数据**：从原始数据源提取和标准化数据。
2. **训练分词器**：使用处理后的数据训练特定于古典文学的分词器。
3. **优化数据**：合并各类文件、拆分训练/测试/生成集、移动tokenizer到主目录。
4. **验证数据完整性**：确保所有需要的文件都存在且格式正确。
5. **清理旧文件**：删除临时文件、旧的处理文件和备份文件。

每次运行 `processor.py` 时，可以选择执行全部或部分步骤。

## train_model.py - 模型训练脚本

`train_model.py` 用于训练灵猫墨韵语言模型，使用由 `processor.py` 生成的数据。

### 命令行参数

```bash
python train_model.py [选项]
```

常用选项说明：

| 参数 | 说明 |
|------|------|
| `--train_file` | 训练数据文件路径，默认为 `dataset/train_data_train.jsonl` |
| `--test_file` | 测试/验证数据文件路径，默认为 `dataset/train_data_test.jsonl` |
| `--model_dir` | 模型保存目录，默认为 `model_weights` |
| `--epochs` | 训练轮次，默认为 20 |
| `--batch_size` | 批处理大小，默认为 32 |
| `--learning_rate` | 学习率，默认为 5e-5 |

### 使用示例

1. 使用默认参数训练模型：
   ```bash
   python train_model.py
   ```

2. 指定训练参数：
   ```bash
   python train_model.py --epochs 30 --batch_size 64 --learning_rate 1e-4
   ```

## generate.py - 文本生成脚本

`generate.py` 用于使用训练好的模型生成古典风格的文本。

### 命令行参数

```bash
python generate.py [选项]
```

选项说明：

| 参数 | 说明 |
|------|------|
| `--model` | 模型路径，默认为 `model_weights/best_model_v0.4.pt` |
| `--prompt` | 生成提示（必需参数） |
| `--max_length` | 最大生成长度，默认为 100 |
| `--temperature` | 温度参数，控制随机性，默认为 0.7 |

### 使用示例

1. 使用默认模型生成文本：
   ```bash
   python generate.py --prompt "春风又绿江南岸"
   ```

2. 调整生成参数：
   ```bash
   python generate.py --model model_weights/best_model_v0.4.pt --prompt "明月几时有" --max_length 200 --temperature 0.9
   ```

## 文件清理说明

`processor.py` 中的清理功能会执行以下操作：

1. 删除临时目录（temp_data、cleanup_temp）
2. 只保留重要的数据文件（train_data*.jsonl、poems.jsonl等）
3. 每种类型的日志文件只保留最新的10个
4. 删除旧的tokenizer备份文件
5. 删除所有__pycache__目录

要手动执行清理：

```bash
python processor.py --clean
```

## 数据验证说明

数据验证功能会检查：

1. 所有必需文件是否存在
2. 文件内容是否为空
3. JSONL文件格式是否正确

要执行验证：

```bash
python processor.py --validate
``` 