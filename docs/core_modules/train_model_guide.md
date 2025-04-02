# 灵猫墨韵模型训练指南

> 本指南详细介绍如何使用灵猫墨韵系统训练自定义语言模型，包括参数配置、优化策略及进阶技巧。

🏠 [主页](../README.md) > [文档中心](../SUMMARY.md) > [核心模块](../SUMMARY.md#核心模块) > 模型训练指南

## 📌 目录

- [版本信息](#版本信息)
- [准备训练数据](#准备训练数据)
- [基本训练命令](#基本训练命令)
- [高级参数配置](#高级参数配置)
- [从检查点恢复训练](#从检查点恢复训练)
- [训练进度与结果](#训练进度与结果)
- [夜间模式说明](#夜间模式说明)
- [主要参数说明](#主要参数说明)
- [示例命令](#示例命令)
- [常见问题](#常见问题)

## 📋 版本信息

- **当前版本**: v0.8.2
- **更新日期**: 2025-03-18

### 🔹 最新特性

- **自动保存和恢复**: 训练中断后可自动恢复最近检查点
- **夜间低功耗模式**: 在指定时间段自动降低资源使用
- **混合精度训练**: 支持FP16混合精度计算，提升训练速度
- **梯度累积支持**: 突破显存限制，模拟更大批次训练

## 📌 准备训练数据

训练前需要准备好数据集文件，系统支持以下格式:

| 格式 | 文件扩展名 | 数据要求 |
|------|----------|---------|
| JSON Lines | `.jsonl` | 每行一个JSON对象，必须包含`text`字段 |
| 纯文本 | `.txt` | 文档间用特殊分隔符分隔 |

### 🔹 数据示例

**JSONL格式**:
```json
{"text": "这是第一个训练样本"}
{"text": "这是第二个训练样本"}
```

### 🔹 数据集目录结构

建议将数据集放在`dataset/`目录下，标准目录结构如下:

```
dataset/
  ├── train_data_train.jsonl   # 训练集
  └── train_data_valid.jsonl   # 验证集
```

> ⚠️ **注意**: 请确保训练数据已经过预处理，包括文本清洗和规范化，以提高训练质量。

## 📌 基本训练命令

最简单的训练命令无需任何参数:

```bash
$ python train_model.py
```

### 🔹 默认配置

上述命令将使用以下默认参数进行训练:

| 参数 | 默认值 | 说明 |
|------|-------|------|
| 训练数据 | dataset/train_data_train.jsonl | 主训练数据集 |
| 验证数据 | dataset/train_data_valid.jsonl | 可选的验证数据集 |
| 模型保存目录 | model_weights/ | 检查点与最终模型保存位置 |
| 批次大小 | 8 | 单次前向传播的样本数 |
| 训练轮数 | 10 | 完整训练周期数 |
| 上下文长度 | 512 | 序列的最大token数 |

> ✅ **最佳实践**: 对于初次使用，建议先用默认参数运行，了解整个流程后再进行参数调整。

## 📌 高级参数配置

### 🔹 训练数据与保存设置

```bash
$ python train_model.py \
  --train_file dataset/my_train_data.jsonl \
  --test_file dataset/my_valid_data.jsonl \
  --model_save_dir custom_model_dir \
  --tokenizer_path custom_tokenizer.json
```

### 🔹 模型参数设置

```bash
$ python train_model.py \
  --d_model 768 \       # 模型维度
  --nhead 12 \          # 注意力头数
  --num_layers 12 \     # Transformer层数
  --dim_feedforward 3072 \ # 前馈网络维度
  --dropout 0.1         # Dropout率
```

### 🔹 训练控制参数

```bash
$ python train_model.py \
  --batch_size 8 \            # 批次大小 
  --learning_rate 5e-5 \      # 学习率
  --epochs 10 \               # 训练轮数
  --accumulation_steps 4 \    # 梯度累积步数
  --context_length 512 \      # 上下文长度
  --max_grad_norm 1.0 \       # 梯度裁剪最大范数
  --weight_decay 0.01         # 权重衰减系数
```

### 🔹 性能优化选项

```bash
$ python train_model.py \
  --use_amp \            # 启用混合精度训练
  --no_cuda              # 禁用CUDA (使用CPU训练)
```

### 🔹 训练恢复与保存

```bash
$ python train_model.py \
  --resume_from model_weights/checkpoint_epoch5_v0.8.2.pt \ # 从指定检查点恢复
  --auto_resume \                   # 自动从最新检查点恢复训练
  --checkpoint_every 1 \            # 每多少轮保存一次检查点
  --save_on_interrupt               # 中断时保存当前状态
```

### 🔹 电源管理参数

```bash
$ python train_model.py \
  --night_mode \         # 启用夜间低功耗模式
  --no_night_mode        # 禁用夜间低功耗模式
```

## 📌 从检查点恢复训练

训练中断后，有多种方式可以恢复:

### 🔹 自动恢复(推荐)

默认启用自动恢复功能，重新运行相同命令即可:

```bash
$ python train_model.py  # 自动查找并恢复最新检查点
```

> ℹ️ **提示**: 系统会自动寻找最新的检查点文件并恢复训练状态，包括模型权重、优化器状态和学习率调度器。

### 🔹 指定检查点恢复

如果需要从特定检查点恢复训练:

```bash
$ python train_model.py --resume_from model_weights/checkpoint_epoch5_v0.8.2.pt
```

### 🔹 禁用自动恢复

如果想从头开始训练:

```bash
$ python train_model.py --no_auto_resume
```

## 📌 训练进度与结果

训练过程中系统会显示详细的进度信息:

![训练进度示例](../assets/images/screenshots/training_progress.png)

### 🔹 训练输出

训练完成后会生成以下内容:

1. **训练统计图表** - 保存在`model_weights/stats/`目录下
2. **模型检查点** - 每个epoch结束后的模型状态
3. **最终模型** - 训练完成后的完整模型
4. **最佳模型** - 验证集上表现最佳的模型权重

### 🔹 模型文件

模型文件将保存到`model_weights`目录(除非另行指定):

- `checkpoint_epoch{N}_v0.8.2.pt`: 每轮结束保存的检查点
- `best_model_v0.8.2.pt`: 验证集上表现最佳的模型
- `final_model_v0.8.2.pt`: 训练结束后的最终模型

## 📌 夜间模式说明

夜间模式是灵猫墨韵的特色功能，专为长时间训练设计。

### 🔹 工作原理

在指定时间段(默认晚上9点到早上8点)，系统会自动:

1. 降低批次大小，减少内存使用
2. 限制CPU核心数，降低功耗
3. 降低GPU显存占用，减少噪音

### 🔹 控制方式

```bash
# 启用夜间模式(默认)
$ python train_model.py --night_mode

# 禁用夜间模式
$ python train_model.py --no_night_mode
```

## 📌 主要参数说明

所有参数都已设置为最优默认值，一般情况下不需要修改。

### 🔹 数据相关参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| train_file | dataset/train_data_train.jsonl | 训练数据路径 |
| test_file | dataset/train_data_valid.jsonl | 验证数据路径 |
| tokenizer_path | tokenizer.json | 分词器路径 |
| context_length | 512 | 上下文长度 |

### 🔹 模型参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| d_model | 768 | 模型维度 |
| nhead | 12 | 注意力头数 |
| num_layers | 12 | Transformer层数 |
| dim_feedforward | 3072 | 前馈网络维度 |
| dropout | 0.1 | Dropout率 |

### 🔹 训练参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| batch_size | 8 | 批次大小 |
| accumulation_steps | 4 | 梯度累积步数 |
| learning_rate | 5e-5 | 学习率 |
| epochs | 10 | 训练轮数 |
| max_grad_norm | 1.0 | 梯度裁剪最大范数 |
| weight_decay | 0.01 | 权重衰减系数 |

### 🔹 硬件相关参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| use_amp | True | 是否使用混合精度训练 |
| device | auto | 训练设备(自动检测) |

## 📌 示例命令

下面是一些常用场景的命令示例:

### 🔹 基本训练

使用默认参数进行训练:

```bash
$ python train_model.py
```

### 🔹 指定训练轮数

```bash
$ python train_model.py --epochs 20
```

### 🔹 低显存配置

适用于显存较小的GPU:

```bash
$ python train_model.py --batch_size 4 --accumulation_steps 8
```

> ⚠️ **注意**: 有效批量大小 = batch_size × accumulation_steps，此配置等效于batch_size=32

### 🔹 CPU训练

强制使用CPU进行训练:

```bash
$ python train_model.py --no_cuda
```

### 🔹 性能优化

禁用混合精度训练(某些情况下可提高精度):

```bash
$ python train_model.py --no_use_amp
```

## ❓ 常见问题

<details>
<summary><b>训练时显存不足怎么办?</b></summary>

可以通过以下方法减少显存占用:

1. 减小批次大小: `--batch_size 4`
2. 增加梯度累积步数: `--accumulation_steps 8`
3. 减小模型维度: `--d_model 512 --nhead 8 --num_layers 8`
4. 确保启用混合精度训练: `--use_amp`

例如:
```bash
$ python train_model.py --batch_size 2 --accumulation_steps 16 --d_model 512
```
</details>

<details>
<summary><b>如何提前结束训练?</b></summary>

可以安全地按下`Ctrl+C`，系统会捕获中断信号并保存当前检查点后退出。

如果启用了`--save_on_interrupt`选项(默认开启)，即使在批次中间中断，也会保存当前状态。
</details>

<details>
<summary><b>如何调整最适合我数据的参数?</b></summary>

建议遵循以下步骤:

1. 先使用默认参数进行短期训练(1-2个epoch)
2. 观察训练和验证损失曲线
3. 如果验证损失下降缓慢，尝试增大学习率(`--learning_rate 1e-4`)
4. 如果模型过拟合(训练损失远低于验证损失)，增加dropout(`--dropout 0.2`)或减小模型

详细调参指南请参考[训练优化最佳实践](../best_practices/training_optimization.md)
</details>

<details>
<summary><b>训练中断后恢复，之前的训练进度会丢失吗?</b></summary>

不会。系统会自动保存每个epoch的检查点，恢复训练时会:

1. 加载模型权重
2. 恢复优化器状态
3. 恢复学习率调度器
4. 继续从中断点训练

只要使用`--auto_resume`选项(默认启用)，或手动指定`--resume_from`参数。
</details>

## ⏭️ 后续步骤

完成模型训练后，您可能需要:

- [文本生成指南](generate_guide.md) - 使用训练好的模型生成文本
- [训练优化最佳实践](../best_practices/training_optimization.md) - 优化训练效果与效率
- [训练模块代码解析](../code_explanation/train_model_code.md) - 深入了解训练代码实现

---

<div align="center">

[⬅️ 项目概览](../project_overview.md) | [🏠 返回主页](../README.md) | [➡️ 生成模型指南](generate_guide.md)

</div>

*最后更新时间: 2025-03-18* 