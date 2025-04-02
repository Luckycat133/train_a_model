# 灵猫墨韵训练优化最佳实践

本文档提供灵猫墨韵模型训练的优化最佳实践，帮助您在不同硬件环境下高效训练模型并获得最佳效果。

## 硬件优化策略

### NVIDIA GPU 优化

如果您使用 NVIDIA GPU 进行训练，可以应用以下优化策略：

#### 1. 显存优化

```bash
# 最大显存配置 (24GB+ 显存)
python train_model.py --batch_size 32 --context_length 768 --d_model 1024 --use_amp True

# 中等显存配置 (12-16GB 显存)
python train_model.py --batch_size 16 --context_length 512 --d_model 768 --use_amp True --accumulation_steps 2

# 低显存配置 (8GB 显存)
python train_model.py --batch_size 8 --context_length 512 --d_model 768 --use_amp True --accumulation_steps 4

# 极低显存配置 (4-6GB 显存)
python train_model.py --batch_size 4 --context_length 512 --d_model 768 --use_amp True --accumulation_steps 8
```

#### 2. CUDA 优化设置

```python
# 在训练脚本开头添加
import os
import torch

# CUDA 内核优化
torch.backends.cudnn.benchmark = True  # 为固定大小输入优化 CUDNN

# 针对较新的 GPU 架构启用 TF32 (NVIDIA Ampere 及以上)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 设置 CUDA 缓存大小 (MB)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
```

### Apple Silicon 优化

对于使用 Apple M1/M2/M3 芯片的设备：

```bash
# Apple Silicon 优化配置
python train_model.py --device mps --batch_size 16 --use_amp False --accumulation_steps 2

# 注意: Apple Silicon 上目前不支持 PyTorch 的自动混合精度
```

**Apple Silicon 特别说明**：
- 使用 `mps` 设备而不是 `cuda`
- 不支持自动混合精度 (`use_amp=False`)
- 使用梯度累积来模拟更大批次
- 考虑将 CPU 内存限制降低，为 MPS 设备释放更多共享内存

### CPU 训练优化

在仅有 CPU 的环境中训练：

```bash
# 多核 CPU 优化配置
python train_model.py --device cpu --batch_size 8 --accumulation_steps 4
```

**CPU 优化技巧**：
- 启用 PyTorch 的 OpenMP 并行处理
- 设置环境变量：`export OMP_NUM_THREADS=8` (根据核心数调整)
- 考虑减小模型大小 (如 `--d_model 512 --nhead 8 --num_layers 8`)
- 使用梯度累积增大有效批次大小

## 超参数优化指南

### 1. 批次大小与学习率

批次大小和学习率是相互关联的关键参数：

| 批次大小 | 建议学习率范围 | 说明 |
|---------|--------------|------|
| 8 | 1e-5 ~ 3e-5 | 小批量需要较小学习率 |
| 16 | 3e-5 ~ 5e-5 | 中等批量的平衡选择 |
| 32 | 5e-5 ~ 1e-4 | 大批量可使用较大学习率 |
| 64+ | 1e-4 ~ 3e-4 | 超大批量（通过梯度累积） |

**学习率缩放原则**：当使用梯度累积增大有效批次大小时，学习率应近似按批次大小的平方根比例缩放。

```python
# 学习率缩放公式
base_lr = 5e-5  # 基准学习率 (批次大小 16)
base_batch = 16  # 基准批次大小

# 新批次大小
new_batch = 64  # 有效批次大小 (实际批次 × 累积步数)

# 缩放学习率
scaled_lr = base_lr * math.sqrt(new_batch / base_batch)
# 结果约为 1e-4
```

### 2. 模型大小与上下文长度

根据任务复杂度和可用资源选择合适的模型大小：

| 模型配置 | 参数数量 | 适用场景 | 建议上下文长度 |
|---------|---------|---------|--------------|
| 小型 | ~40M | 资源受限/快速训练 | 256-512 |
| 中型 | ~110M | 一般应用/平衡性能 | 512-768 |
| 大型 | ~340M | 高复杂度任务 | 768-1024 |

**配置示例**：

```bash
# 小型模型配置
python train_model.py --d_model 512 --nhead 8 --num_layers 8 --context_length 512

# 中型模型配置 (默认)
python train_model.py --d_model 768 --nhead 12 --num_layers 12 --context_length 768

# 大型模型配置
python train_model.py --d_model 1024 --nhead 16 --num_layers 16 --context_length 1024
```

### 3. 优化器与正则化

**权重衰减设置**：

- 过小的权重衰减可能导致过拟合
- 过大的权重衰减可能阻碍模型学习
- 推荐范围: 0.01 ~ 0.1 (根据数据集大小和模型规模调整)

```bash
# 小数据集/小模型
python train_model.py --weight_decay 0.01

# 大数据集/大模型
python train_model.py --weight_decay 0.1
```

**Dropout 率**：

| 数据集大小 | 建议 Dropout 率 | 说明 |
|-----------|----------------|------|
| 小 (<1GB) | 0.2 ~ 0.3 | 较高的 dropout 防止过拟合 |
| 中 (1-10GB) | 0.1 ~ 0.2 | 中等 dropout 平衡正则化 |
| 大 (>10GB) | 0.05 ~ 0.1 | 较低的 dropout 充分利用数据 |

```bash
python train_model.py --dropout 0.1
```

## 训练稳定性优化

### 1. 梯度裁剪

梯度裁剪对于稳定训练至关重要，特别是在大批次或高学习率时：

```bash
# 保守设置 (高稳定性)
python train_model.py --max_grad_norm 0.5

# 标准设置 (平衡)
python train_model.py --max_grad_norm 1.0

# 激进设置 (可能更快收敛，但风险更高)
python train_model.py --max_grad_norm 2.0
```

### 2. 学习率调度

灵猫墨韵支持余弦退火学习率调度，帮助模型找到更好的局部最优解：

```bash
# 预热期设置为总步数的 10%
# 这在训练脚本中已设置为默认值，无需额外参数
```

手动调整学习率调度:

```python
# 在定制训练脚本中修改
warmup_steps = int(total_steps * 0.1)  # 预热阶段步数

def lr_lambda(current_step):
    if current_step < warmup_steps:
        # 线性预热阶段
        return float(current_step) / float(max(1, warmup_steps))
    else:
        # 余弦退火阶段
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

# 应用学习率调度器
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### 3. 混合精度训练

使用混合精度训练（FP16）可大幅提升训练速度并减少显存使用：

```bash
# 启用混合精度训练 (默认)
python train_model.py --use_amp True

# 如遇到数值不稳定问题，可禁用
python train_model.py --use_amp False
```

**调试混合精度问题**：如果遇到 `nan` 损失或训练不稳定，尝试：

1. 减小学习率
2. 减小批次大小
3. 增加梯度累积步数
4. 作为最后手段，禁用混合精度

## 性能监控与调优

### 1. GPU 利用率监控

使用 `nvidia-smi` 工具监控 GPU 利用率和显存：

```bash
# 间隔 5 秒自动更新
watch -n 5 nvidia-smi
```

**优化目标**：

- GPU 利用率 > 80%: 良好的计算效率
- GPU 显存使用率 > 90%: 有效利用可用显存
- 如果 GPU 利用率低但显存几乎用满: 考虑增加批次大小
- 如果 GPU 利用率高但显存有大量空闲: 已达到计算瓶颈，可增加批次大小

### 2. 训练吞吐量评估

评估训练吞吐量以优化配置：

```bash
# 使用 --benchmark 选项评估不同配置的性能
python train_model.py --benchmark --epochs 1 [其他参数]
```

**典型吞吐量参考**：

| 设备 | 批次大小 | 上下文长度 | 模型大小 | 样本/秒 (近似) |
|------|---------|------------|---------|---------------|
| RTX 3090 | 16 | 512 | 中型 | 20-30 |
| RTX 4090 | 32 | 512 | 中型 | 40-60 |
| A100 | 64 | 512 | 中型 | 80-120 |
| M1 Max | 16 | 512 | 中型 | 8-12 |
| 16核 CPU | 8 | 512 | 中型 | 1-2 |

### 3. 夜间模式优化

利用灵猫墨韵的夜间模式在非工作时段进行更密集的训练：

```bash
# 启用夜间模式 (默认)
python train_model.py --night_mode True
```

**夜间模式行为**：
- 在晚上9点至早上8点自动激活
- 调整 GPU 内存限制和批次大小
- 自动降低功耗

## 数据集优化

### 1. 上下文长度与步长

上下文长度和步长影响训练样本数量和质量：

```python
# 在 LMDataset 类中设置
context_length = 512  # 上下文窗口大小
stride = 256          # 滑动窗口步长
```

**推荐设置**：
- 长文本 (小说/文章): 较大上下文 (768-1024)，步长设为上下文长度的 1/2
- 短文本 (诗词/对话): 较小上下文 (256-512)，步长设为上下文长度的 1/4
- 综合文本: 中等上下文 (512)，步长设为上下文长度的 1/2

### 2. 数据集平衡

对于包含多种体裁的语料库，确保数据平衡：

```python
# 在数据处理阶段平衡不同体裁的样本数量
# 例如，对于 processor.py 中的平衡策略：

# 设置各类型最大样本数限制
genre_limits = {
    "诗": 50000,
    "词": 30000,
    "文": 20000,
    "赋": 10000
}

# 实现平衡采样
balanced_data = []
for genre, limit in genre_limits.items():
    genre_data = [item for item in all_data if item["genre"] == genre]
    if len(genre_data) > limit:
        # 随机采样至目标数量
        balanced_data.extend(random.sample(genre_data, limit))
    else:
        balanced_data.extend(genre_data)
```

## 常见问题与解决方案

### 1. 训练损失不下降

**可能的原因**：
- 学习率过高
- 批次大小不合适
- 数据质量问题
- 优化器配置不当

**解决方案**：
1. 尝试降低学习率 (减半)
2. 增加梯度裁剪强度 (减小 `max_grad_norm`)
3. 检查数据集质量和预处理流程
4. 尝试较小的批次大小

### 2. 过拟合问题

**症状**：训练损失持续下降但验证损失开始上升

**解决方案**：
1. 增加权重衰减 (`--weight_decay 0.1`)
2. 提高 dropout 率 (`--dropout 0.2`)
3. 减小模型规模或减少训练轮数
4. 增加训练数据量或应用数据增强

### 3. 内存/显存溢出

**解决方案**：
1. 减小批次大小
2. 减小上下文长度
3. 减小模型规模
4. 使用梯度累积 (增加 `--accumulation_steps`)
5. 启用混合精度训练 (`--use_amp True`)

## 最终配置推荐

### NVIDIA GPU (高端)

```bash
python train_model.py \
  --batch_size 16 \
  --context_length 768 \
  --d_model 1024 \
  --nhead 16 \
  --num_layers 12 \
  --learning_rate 5e-5 \
  --epochs 10 \
  --accumulation_steps 2 \
  --use_amp True \
  --weight_decay 0.05 \
  --dropout 0.1 \
  --max_grad_norm 1.0 \
  --night_mode True \
  --checkpoint_every 1
```

### 中等配置 (RTX 3060 或类似)

```bash
python train_model.py \
  --batch_size 8 \
  --context_length 512 \
  --d_model 768 \
  --nhead 12 \
  --num_layers 12 \
  --learning_rate 3e-5 \
  --epochs 10 \
  --accumulation_steps 4 \
  --use_amp True \
  --weight_decay 0.05 \
  --dropout 0.1 \
  --max_grad_norm 1.0 \
  --night_mode True
```

### 低资源配置 (集成显卡或 CPU)

```bash
python train_model.py \
  --batch_size 4 \
  --context_length 256 \
  --d_model 512 \
  --nhead 8 \
  --num_layers 8 \
  --learning_rate 1e-5 \
  --epochs 5 \
  --accumulation_steps 8 \
  --use_amp False \
  --weight_decay 0.1 \
  --dropout 0.15 \
  --max_grad_norm 0.5
```

## 结论

优化灵猫墨韵模型训练需要平衡硬件能力、数据特性和模型复杂度。通过适当选择批次大小、学习率、模型大小和优化技术，可以在不同环境下实现高效训练。记住：没有完美的"一刀切"配置，最佳设置取决于您的具体数据和训练目标。

---

*文档最后更新: 2025年3月18日* 