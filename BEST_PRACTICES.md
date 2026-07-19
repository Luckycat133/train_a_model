# 训练配置参考

本指南提供灵猫墨韵项目使用的训练配置参考，帮助用户在不同硬件配置下运行实验。

## 目录

1. [技术概览](#技术概览)
2. [核心优化技术](#核心优化技术)
3. [硬件配置推荐](#硬件配置推荐)
4. [配置示例](#配置示例)
5. [Troubleshooting](#troubleshooting)
6. [性能基准（参考值）](#性能基准参考值)

---

## 技术概览

### 训练技术栈

| 技术 | 版本要求 | 作用 | 说明 |
|------|---------|------|------|
| PyTorch | 2.0+ | 基础框架 | - |
| HuggingFace Accelerate | 0.20+ | 分布式训练 | 可选 |
| torch.compile | 2.0+ | JIT编译 | 实验性 |
| BF16 AMP | CUDA 11.8+ | 混合精度 | 推荐 |
| Fused AdamW | PyTorch 2.0+ | 优化器 | 可选 |
| Gradient Checkpointing | - | 显存优化 | 推荐 |

---

## 核心优化技术

### 1. HuggingFace Accelerate

Accelerate 提供了统一的分布式训练接口。

```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=4,
    log_with="tensorboard"
)

model, optimizer, loader = accelerator.prepare(
    model, optimizer, train_loader
)
```

### 2. torch.compile

PyTorch 2.0 引入的 JIT 编译器。

| 模式 | 适用场景 | 编译时间 | 说明 |
|------|---------|---------|------|
| `default` | 通用场景 | 中等 | 默认选项 |
| `reduce-overhead` | 小batch推理 | 较长 | 减少开销 |
| `max-autotune` | 生产部署 | 最长 | 最大优化 |

### 3. 混合精度训练 (AMP)

| 特性 | BF16 | FP16 |
|------|------|------|
| 指数位 | 8 | 5 |
| 尾数位 | 7 | 10 |
| 动态范围 | 更大 | 受限 |
| 稳定性 | 更好 | 需小心 |
| 硬件支持 | A100+/RTX 30+ | 所有GPU |

### 4. Gradient Checkpointing

通过在前向传播时不保存全部中间激活值来降低显存占用。

### 5. Fused AdamW

融合的 AdamW 优化器，减少内存访问开销。

---

## 硬件配置推荐

### GPU型号对比

| GPU | 显存 | 推荐精度 | 适用场景 |
|-----|------|---------|---------|
| RTX 3090 | 24GB | BF16/FP16 | 小规模实验 |
| RTX 4090 | 24GB | BF16/FP16 | 小规模实验 |
| A100 40GB | 40GB | BF16 | 中等规模 |
| A100 80GB | 80GB | BF16 | 大规模训练 |
| H100 | 80GB | BF16/FP8 | 生产级训练 |

### 配置推荐

#### RTX 3090/4090 (消费级)

```yaml
training:
  batch_size: 8
  use_amp: true
  amp_dtype: "bf16"
  use_gradient_checkpointing: true
  use_compile: false
  accumulation_steps: 4

model:
  num_layers: 12
  d_model: 768
  context_length: 512
```

#### A100 40GB (数据中心)

```yaml
training:
  batch_size: 32
  use_amp: true
  amp_dtype: "bf16"
  use_gradient_checkpointing: true
  use_compile: true
  accumulation_steps: 2

model:
  num_layers: 24
  d_model: 1024
  context_length: 2048
```

---

## Troubleshooting

### 常见问题

#### 1. 显存不足 (CUDA OOM)

按顺序尝试：
1. 启用梯度检查点
2. 减小 batch size
3. 减小模型尺寸
4. 使用混合精度
5. 减小上下文长度

#### 2. torch.compile 编译失败

```python
model = torch.compile(model, mode="reduce-overhead")
model = torch.compile(model, fullgraph=False)
```

#### 3. 训练不稳定/爆炸

```yaml
training:
  use_amp: true
  amp_dtype: "bf16"
  learning_rate: 1.0e-5
  warmup_steps: 2000
  max_grad_norm: 0.5
```

---

## 性能基准（参考值）

以下数值来自公开文献和社区报告，**未经本项目独立验证**。

### 单GPU参考

配置: GPT-2 规模 (12层, 768隐藏维度)

| 配置 | 相对速度 | 说明 |
|------|---------|------|
| FP32 | 1x | 基准 |
| FP16 | ~1.8x | 混合精度 |
| BF16 | ~2x | 推荐 |
| BF16 + Checkpointing | ~1.6x | 显存优化 |
| BF16 + compile | ~2.5x | 需编译 |

---

## 参考资源

- [HuggingFace Accelerate 文档](https://huggingface.co/docs/accelerate)
- [PyTorch 2.0 Compile 指南](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [Mixed Precision Training](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

---

*最后更新: 2026-07-19*
*版本: 0.8.5*