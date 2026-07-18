# 实验平台使用指南

本指南详细介绍如何使用灵猫墨韵项目的实验平台进行各种训练实验，包括预训练、SFT微调和评估。

## 目录结构

```
docs/experiments/
├── README.md         # 本文档 - 总体使用指南
├── quickstart.md     # 快速开始教程
└── (更多文档持续添加中...)
```

## 实验类型概述

### 1. 预训练 (Pretraining)

预训练阶段在大规模语料上训练基础语言模型，学习通用语言表示。

**配置文件**: [config/pretrain.yaml](file:///workspace/config/pretrain.yaml)

**主要特点**:
- 使用大规模文本语料
- 较长的上下文长度 (512-1024)
- 较高的学习率和更多epoch
- 支持现代Transformer架构 (RoPE、SwiGLU、GQA)

**适用场景**:
- 从头训练基础模型
- 继续训练已有的预训练模型
- 扩充模型知识

### 2. 有监督微调 (SFT)

在特定任务数据上进行微调，使模型适应具体应用场景。

**配置文件**: [config/sft.yaml](file:///workspace/config/sft.yaml)

**主要特点**:
- 使用结构化指令数据
- 较短的上下文长度 (256-512)
- 较低的学习率
- 较少的epoch数
- 支持标签平滑

**适用场景**:
- 指令跟随能力训练
- 对话系统微调
- 特定领域适应

### 3. 强化学习 (RL)

使用强化学习方法进一步优化模型输出质量。

**配置文件**: [config/rl.yaml](file:///workspace/config/rl.yaml)

**主要特点**:
- 支持PPO算法
- 奖励模型和参考模型
- KL散度约束
- GAE优势估计

**适用场景**:
- 人类偏好对齐
- 强化学习调优
- 质量提升优化

## 配置系统

### Schema定义

配置系统使用Pydantic进行参数验证，确保配置的合法性和完整性。

主要配置类:

- `ModelConfig` - 模型架构配置
- `TrainingConfig` - 训练超参数配置
- `DatasetConfig` - 数据集配置
- `MemoryConfig` - 内存和性能配置
- `RLConfig` - 强化学习配置

详见: [src/config/schema.py](file:///workspace/src/config/schema.py)

### 验证工具

使用验证器确保配置正确:

```python
from src.config.validator import load_and_validate_config

is_valid, config, warnings = load_and_validate_config(
    "config/pretrain.yaml",
    experiment_type="pretrain",
    strict=True
)
```

详见: [src/config/validator.py](file:///workspace/src/config/validator.py)

## 训练流程

### 基本训练流程

1. **准备数据**
   - 准备训练语料
   - 运行分词器
   - 格式化数据

2. **配置实验**
   - 选择或创建配置文件
   - 验证配置有效性
   - 调整超参数

3. **执行训练**
   - 启动训练脚本
   - 监控训练过程
   - 保存检查点

4. **评估模型**
   - 运行评估脚本
   - 分析指标结果
   - 生成样本

## 高级功能

### 梯度检查点

减少内存占用，以计算换内存:

```yaml
training:
  use_gradient_checkpointing: true
```

### 混合精度训练

加速训练并减少显存:

```yaml
training:
  use_amp: true
```

### 夜间模式

自动调整资源使用:

```yaml
night_mode:
  enabled: true
  start_hour: 21
  end_hour: 8
  batch_divisor: 2
```

### 权重绑定

减少参数量:

```yaml
model:
  use_weight_tying: true
```

## 2025-2026最佳实践

本节介绍2025-2026年大模型训练领域的最新优化技术和最佳实践，帮助您最大化训练效率。

### 核心优化技术

#### 1. HuggingFace Accelerate

Accelerate是HuggingFace提供的分布式训练库，提供开箱即用的多GPU/TPU支持：

```yaml
# 启用Accelerate
training:
  use_accelerate: true
  accelerator_config:
    distributed_type: "multi_gpu"
    mixed_precision: "bf16"
    gradient_accumulation_steps: 4
```

**优势**：
- 自动处理设备放置和分布式训练
- 支持单GPU到数千GPU的线性扩展
- 统一的混合精度接口
- 与HuggingFace生态无缝集成

#### 2. torch.compile

PyTorch 2.0+的编译优化，将Python代码 JIT 编译为优化内核：

```yaml
training:
  use_compile: true
  compile_mode: "default"  # default | reduce-overhead | max-autotune
```

**优势**：
- 内核融合减少内存访问
- 动态形状支持
- 最高可达1.5x加速

#### 3. Gradient Checkpointing

以计算换内存，适用于大模型训练：

```yaml
training:
  use_gradient_checkpointing: true
  checkpoint_every_n_layers: 1
```

**优势**：
- 显存节省50-60%
- 允许更大batch size
- 轻微计算开销（约20-30%）

### 性能对比表格

| 优化技术 | 速度提升 | 显存节省 | 适用场景 |
|---------|---------|---------|---------|
| BF16 AMP | 1.5-2x | 40-50% | 全场景 |
| torch.compile | 1.2-1.5x | - | 推理优化 |
| Fused AdamW | 1.2-1.3x | - | 优化器更新 |
| Gradient Checkpointing | 1x | 50-60% | 大模型 |
| Accelerate | 线性扩展 | - | 多GPU训练 |

### 快速配置示例

```yaml
# 高性能配置（推荐）
training:
  use_amp: true
  amp_dtype: "bf16"
  use_compile: true
  use_gradient_checkpointing: true
  use_fused_adamw: true

# 快速训练配置
training:
  use_amp: true
  amp_dtype: "fp16"
  use_gradient_checkpointing: true

# 多GPU配置
training:
  use_accelerate: true
  accelerator_config:
    distributed_type: "multi_gpu"
    mixed_precision: "bf16"
```

### 推荐硬件配置

| GPU配置 | 推荐优化 | 预期性能 |
|---------|---------|---------|
| RTX 3090/4090 | BF16 + 梯度检查点 | 基础性能 |
| A100 40GB | 全优化开启 | 最佳性能 |
| 多卡A100/H100 | Accelerate | 线性扩展 |

## 实验追踪

### 日志记录

训练日志自动保存到:

```
logs/
├── pretrain/
│   └── events.*
├── sft/
│   └── events.*
└── rl/
    └── events.*
```

### 检查点保存

自动保存模型检查点:

```
model_weights/
├── pretrain/
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_epoch_2.pt
│   └── best_model.pt
└── sft/
    └── ...
```

### 实验配置

建议保存实验配置用于复现:

```bash
cp config/pretrain.yaml logs/pretrain/config.yaml
```

## 常见问题

### Q: 训练过程中显存不足怎么办？

A: 尝试以下方法:
- 启用梯度检查点: `use_gradient_checkpointing: true`
- 减小批量大小
- 启用夜间模式: `night_mode.enabled: true`
- 减少上下文长度

### Q: 如何继续中断的训练？

A: 使用 `resume_from` 参数指定检查点路径，或启用自动恢复:

```bash
python examples/pretrain_example.py --resume model_weights/pretrain/checkpoint_epoch_3.pt
```

### Q: 配置文件验证失败怎么办？

A: 检查:
- 参数值是否在允许范围内
- 数据文件路径是否存在
- YAML格式是否正确

## 相关资源

- [快速开始指南](file:///workspace/docs/experiments/quickstart.md)
- [预训练示例](file:///workspace/examples/pretrain_example.py)
- [SFT示例](file:///workspace/examples/sft_example.py)
- [评估示例](file:///workspace/examples/eval_example.py)
- [配置Schema](file:///workspace/src/config/schema.py)
- [配置验证器](file:///workspace/src/config/validator.py)
