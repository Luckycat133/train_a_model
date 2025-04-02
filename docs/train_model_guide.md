# 灵猫墨韵语言模型训练指南

## 版本信息

- 当前版本: v0.7.0
- 更新日期: 2023年3月14日

## 新增特性

- **自动保存和恢复**: 训练中断后可自动恢复最近检查点
- **夜间低功耗模式**: 在晚上9点到早上8点自动降低资源使用
- **优化的词汇表**: 词汇量扩充至50,000，提升模型理解能力
- **改进的性能**: 新增混合精度训练和梯度累积支持

## 准备训练数据

训练前需要准备好数据集文件，支持以下格式:

1. JSON Lines格式 (.jsonl): 每行包含一个JSON对象，必须有'text'字段
2. 纯文本格式 (.txt): 每个文档间用特殊分隔符分隔

示例 (jsonl格式):
```json
{"text": "这是第一个训练样本"}
{"text": "这是第二个训练样本"}
```

数据集建议放在 `dataset/` 目录下，例如:
```
dataset/
  ├── train_data_train.jsonl   # 训练集
  └── train_data_valid.jsonl   # 验证集
```

## 基本训练命令

最简单的训练命令:

```bash
python3 train_model.py
```

这将使用默认参数进行训练。默认配置:
- 训练数据: dataset/train_data_train.jsonl
- 模型保存: model_weights/
- 批次大小: 8
- 训练轮数: 10
- 上下文长度: 512
- 自动恢复训练: 启用
- 夜间模式: 启用

## 高级参数配置

### 训练数据与保存设置
```bash
python3 train_model.py \
  --train_file dataset/my_train_data.jsonl \
  --test_file dataset/my_valid_data.jsonl \
  --model_save_dir custom_model_dir \
  --tokenizer_path custom_tokenizer.json
```

### 模型参数设置
```bash
python3 train_model.py \
  --d_model 768 \       # 模型维度
  --nhead 12 \          # 注意力头数
  --num_layers 12 \     # Transformer层数
  --dim_feedforward 3072 \ # 前馈网络维度
  --dropout 0.1         # Dropout率
```

### 训练控制参数
```bash
python3 train_model.py \
  --batch_size 8 \            # 批次大小 
  --learning_rate 5e-5 \      # 学习率
  --epochs 10 \               # 训练轮数
  --accumulation_steps 4 \    # 梯度累积步数
  --context_length 512 \      # 上下文长度
  --max_grad_norm 1.0 \       # 梯度裁剪最大范数
  --weight_decay 0.01         # 权重衰减系数
```

### 性能优化选项
```bash
python3 train_model.py \
  --use_amp \            # 启用混合精度训练
  --no_cuda              # 禁用CUDA (使用CPU训练)
```

### 训练恢复与保存
```bash
python3 train_model.py \
  --resume_from model_weights/checkpoint_epoch5_v0.7.0.pt \ # 从指定检查点恢复
  --auto_resume \                   # 自动从最新检查点恢复训练
  --checkpoint_every 1 \            # 每多少轮保存一次检查点
  --save_on_interrupt               # 中断时保存当前状态
```

### 电源管理参数
```bash
python3 train_model.py \
  --night_mode \         # 启用夜间低功耗模式
  --no_night_mode        # 禁用夜间低功耗模式
```

## 从检查点恢复训练

如果训练中断，可以使用以下方式恢复:

1. **自动恢复**: 默认启用，重新运行相同命令即可自动找到最新检查点
   ```bash
   python3 train_model.py  # 自动查找并恢复最新检查点
   ```

2. **指定检查点恢复**: 从特定检查点恢复训练
   ```bash
   python3 train_model.py --resume_from model_weights/checkpoint_epoch5_v0.7.0.pt
   ```

3. **禁用自动恢复**: 如果想从头开始训练
   ```bash
   python3 train_model.py --no_auto_resume
   ```

## 训练进度与结果

训练过程中会显示进度条和实时损失值。完成后会:

1. 生成训练统计图表
2. 保存最终模型至 `model_weights/final_model_v0.7.0.pt`
3. 保存最佳模型至 `model_weights/best_model_v0.7.0.pt`

## 夜间模式说明

夜间模式会在晚上9点到早上8点自动启用，具有以下特性:

1. 自动降低批次大小
2. 限制CPU使用核心数
3. 限制GPU内存使用

可以通过 `--no_night_mode` 禁用此功能。

## 主要参数说明

所有参数都已设置为最优默认值，一般情况下不需要修改。如需调整，可参考以下说明：

### 数据相关参数

- `--train_file dataset/train_data_train.jsonl`：训练数据文件路径
- `--test_file dataset/test_data.jsonl`：测试数据文件路径
- `--tokenizer_path tokenizer.json`：分词器路径
- `--context_length 512`：上下文长度

### 模型参数

- `--d_model 768`：模型维度
- `--nhead 12`：注意力头数
- `--num_layers 12`：Transformer层数
- `--dim_feedforward 3072`：前馈网络维度
- `--dropout 0.1`：Dropout率

### 训练参数

- `--batch_size 8`：批次大小
- `--accumulation_steps 4`：梯度累积步数（有效批量大小 = batch_size × accumulation_steps）
- `--learning_rate 5e-5`：学习率
- `--epochs 10`：训练轮数
- `--max_grad_norm 1.0`：梯度裁剪最大范数
- `--weight_decay 0.01`：权重衰减系数

### 设备与加速

- `--no_use_amp`：禁用混合精度训练（默认启用）
- `--no_cuda`：禁用CUDA加速（默认启用，如有）

## 示例命令

1. 基本训练（使用默认参数）：
   ```bash
   python3 train_model.py
   ```

2. 指定训练轮数：
   ```bash
   python3 train_model.py --epochs 20
   ```

3. 指定批量大小和累积步数：
   ```bash
   python3 train_model.py --batch_size 4 --accumulation_steps 8
   ```

4. 强制使用CPU训练：
   ```bash
   python3 train_model.py --no_cuda
   ```

5. 禁用混合精度训练：
   ```bash
   python3 train_model.py --no_use_amp
   ```

## 训练输出与模型保存

训练过程中会输出详细的训练信息，并在每个训练周期结束后保存检查点。模型文件将保存到`model_weights`目录下：

- `checkpoint_epoch{N}_v0.7.0.pt`：每轮结束保存的检查点
- `best_model_v0.7.0.pt`：验证集上表现最佳的模型
- `final_model_v0.7.0.pt`：训练结束后的最终模型

此外，系统还会生成详细的训练统计图表，帮助您分析训练过程。 