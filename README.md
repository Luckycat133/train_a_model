# 🏮 灵猫墨韵 | Lingmao Moyun

**基于古典中文训练的语言模型实验平台**

> _探索古典中文的深度学习之路，支持现代LLM架构和2025-2026年最佳实践_

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

---

## 🚀 快速开始

### 方式一：使用示例数据训练（推荐新手）

```bash
python -m src.run --quick
```

### 方式二：使用自定义数据

```bash
python -m src.run --train 数据文件.jsonl
```

### 方式三：继续上次训练

```bash
python -m src.run --resume
```

---

## 📖 使用指南

### 基础训练

```bash
# 使用示例数据，20轮训练
python -m src.run

# 指定数据文件和训练轮数
python -m src.run --train 我的数据.jsonl --epochs 50

# 强制使用CPU（没有GPU时）
python -m src.run --train 我的数据.jsonl --no_cuda
```

### 使用预设置

系统提供4种预设置，平衡速度和效果：

| 预设置 | 适用场景 | 参数 |
|--------|---------|------|
| `quick` | 快速测试 | 5轮, 小模型 |
| `small` | 日常训练 | 20轮, 小模型 |
| `medium` | 正式训练 | 30轮, 中型模型 |
| `large` | 高质量模型 | 50轮, 大型模型 |

```bash
python -m src.run --preset small
python -m src.run --preset medium
python -m src.run --preset large
```

### 自定义参数

```bash
# 自定义模型架构
python -m src.run --train 数据.jsonl \
  --d_model 256 \
  --num_layers 8 \
  --nhead 8 \
  --context_length 256

# 自定义训练参数
python -m src.run --train 数据.jsonl \
  --epochs 100 \
  --batch_size 16 \
  --lr 1e-4

# 自定义保存目录
python -m src.run --train 数据.jsonl \
  --save_dir model_weights/我的模型
```

### 常用参数

**模型架构:**
- `-l, --num_layers`: Transformer层数 (默认: 6)
- `--d_model`: 模型维度 (默认: 192)
- `--nhead`: 注意力头数 (默认: 4)
- `-c, --context_length`: 上下文长度 (默认: 128)

**训练参数:**
- `-e, --epochs`: 训练轮数 (默认: 20)
- `-b, --batch_size`: 批次大小 (默认: 8)
- `--lr`: 学习率 (默认: 3e-4)

**控制选项:**
- `--resume`: 继续上次训练
- `--clean`: 训练前清理旧日志
- `--no_cuda`: 强制使用CPU

### 高级用法

```bash
# 使用YAML配置文件
python -m src.run --config config/pretrain.yaml

# 组合使用
python -m src.run \
  --preset small \
  --train 我的数据.jsonl \
  --epochs 100 \
  --save_dir model_weights/实验1
```

---

## 🔧 2025-2026 核心特性

本项目采用现代化的大模型训练技术栈，符合2025-2026年行业最佳实践：

| 特性 | 描述 | 性能提升 |
|------|------|---------|
| **RoPE位置编码** | 更好的长度外推能力 | 长文本处理 |
| **SwiGLU激活** | 更高效的FFN层 | 效果提升 |
| **GQA分组注意力** | 减少KV缓存 | 显存节省 |
| **权重绑定** | 共享嵌入层和输出层 | 减少20%参数 |
| **梯度检查点** | 显存优化 | 30-50% 节省 |
| **torch.compile** | JIT编译优化 | 2-3x 加速 |
| **混合精度训练** | BF16/FP16自动选择 | 1.5-2x 加速 |

---

## 📁 数据格式

训练数据使用JSONL格式，每行一条数据：

```json
{"text": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"}
{"text": "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"}
```

支持字段：`text`, `content`, `body`

---

## 📂 输出

训练完成后，模型保存在 `--save_dir` 指定的目录：

```
model_weights/
├── best_model.pt          # 最佳模型
├── final_model_v*.pt      # 最终模型
├── model_epoch_1.pt       # 第1轮checkpoint
├── model_epoch_2.pt       # 第2轮checkpoint
└── ...
```

---

## ⚙️ 故障排除

**训练太慢?**
```bash
# 启用所有优化
python -m src.run --train 数据.jsonl

# 或者使用更小的模型
python -m src.run --preset quick
```

**显存不足?**
```bash
# 启用夜间模式（自动降低显存占用）
python -m src.run --train 数据.jsonl --night_mode

# 减小批次大小
python -m src.run --train 数据.jsonl --batch_size 2
```

**想从头开始?**
```bash
# 清理旧日志和checkpoint
python -m src.run --train 数据.jsonl --clean
```

---

## 📚 项目结构

```
灵猫墨韵/
├── src/                 # 核心代码
│   ├── config/        # 配置管理
│   ├── data/          # 数据处理模块
│   ├── evaluation/    # 评估模块
│   ├── experiments/  # 实验追踪
│   ├── training/      # 训练器模块
│   ├── config.py      # 配置常量
│   ├── dataset.py     # 数据集类
│   ├── logger.py      # 日志工具
│   ├── model.py       # 模型定义
│   ├── quantization.py# 量化工具
│   ├── run.py         # 命令行入口
│   ├── trainer.py     # 训练循环
│   └── utils.py       # 通用工具
├── config/            # 配置文件
├── dataset/           # 数据集目录
├── logs/              # 训练日志
└── model_weights/     # 模型权重
```

---

## 🔧 配置系统

编辑 `config/config.yaml` 或创建你自己的配置文件。关键参数：

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `d_model` | 192 | 模型维度 |
| `nhead` | 4 | 注意力头数 |
| `num_layers` | 6 | Transformer层数 |
| `context_length` | 128 | 输入序列长度 |
| `batch_size` | 8 | 训练批次大小 |
| `learning_rate` | 3e-4 | 学习率 |

---

## 🎯 下一步

1. 准备你的古典中文数据集
2. 运行快速测试 `python -m src.run --quick`
3. 根据需要调整参数
4. 开始正式训练！

---

## 📜 模块说明

### `src.config`
所有魔法数字提取为命名常量。编辑这里而不是在代码中搜索。

### `src.dataset`
`LMDataset` — 内存映射数据集，高效处理大文件训练。

### `src.model`
`SimpleTransformer` — 支持Modern模式的轻量级Transformer（RoPE + SwiGLU + GQA）。

### `src.trainer`
`train_model()` - 完整的训练循环、评估和checkpoint保存/加载工具。

### `src.run`
CLI入口 via `python -m src.run`。

---

## ⚠️ 已知问题和限制

- ClassicalTokenizer使用贪婪最大匹配；计划使用BPE/WordPiece混合方案
- 大数据集加载受内存限制；mmap有帮助但推荐32GB+内存
- MacOS Metal GPU支持尚未测试

---

## 🤝 贡献和讨论

欢迎提交Issue或PR！这是一个学习项目——欢迎建设性反馈。

---

_最后更新: 2026-05-16_
