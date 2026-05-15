# 快速开始指南

本指南帮助您快速上手灵猫墨韵项目的训练流程。通过三个简单步骤，您即可启动第一个训练实验。

## 前置要求

### 1. 环境准备

确保已安装所有依赖:

```bash
pip install -r requirements.txt
```

主要依赖:
- PyTorch >= 2.0
- PyYAML
- Pydantic
- tqdm

### 2. 数据准备

准备训练语料，格式要求:

**纯文本格式** (预训练):
```
床前明月光，疑是地上霜。
举头望明月，低头思故乡。
...
```

**JSONL格式** (SFT):
```json
{"instruction": "写一首关于春天的诗", "input": "", "output": "春风又绿江南岸..."}
{"instruction": "将下列句子翻译成现代汉语", "input": "床前明月光", "output": "床前有明亮的月光"}
```

### 3. 分词器准备

运行分词器生成词表:

```bash
python tokenizer.py
```

## 快速实验流程

### 方式一：使用配置文件 (推荐)

#### 步骤1：选择配置模板

项目提供三种配置模板:

- **预训练**: [config/pretrain.yaml](file:///workspace/config/pretrain.yaml)
- **SFT微调**: [config/sft.yaml](file:///workspace/config/sft.yaml)
- **强化学习**: [config/rl.yaml](file:///workspace/config/rl.yaml)

#### 步骤2：自定义配置

编辑配置文件:

```yaml
# config/pretrain.yaml

model:
  d_model: 512        # 可根据硬件调整
  nhead: 8
  num_layers: 6

training:
  batch_size: 8       # 根据显存调整
  epochs: 10

dataset:
  train_file: dataset/my_train.txt
  test_file: dataset/my_test.txt

tokenizer:
  path: tokenizer.json
```

#### 步骤3：验证配置

```bash
python -m src.config.validator config/pretrain.yaml
```

#### 步骤4：启动训练

```bash
# 预训练
python examples/pretrain_example.py --config config/pretrain.yaml

# SFT微调
python examples/sft_example.py --config config/sft.yaml --pretrained model_weights/pretrain/best_model.pt

# 评估
python examples/eval_example.py --model model_weights/pretrain/best_model.pt --test-file dataset/test.txt
```

### 方式二：命令行参数覆盖

无需修改配置文件，直接通过命令行调整参数:

```bash
python examples/pretrain_example.py \
    --config config/pretrain.yaml \
    --epochs 5 \
    --batch-size 16 \
    --lr 1e-4
```

### 方式三：编程方式调用

在Python脚本中使用配置系统:

```python
from src.config.validator import load_and_validate_config
from src.trainer import train_model

# 加载并验证配置
is_valid, config, warnings = load_and_validate_config("config/pretrain.yaml")

if is_valid:
    # 转换为训练器参数
    kwargs = config.to_trainer_kwargs()
    
    # 启动训练
    model = train_model(**kwargs)
```

## 常用命令速查

### 预训练命令

```bash
# 基础预训练
python examples/pretrain_example.py --config config/pretrain.yaml

# 调整epoch和batch
python examples/pretrain_example.py --config config/pretrain.yaml --epochs 20 --batch-size 16

# 从检查点恢复
python examples/pretrain_example.py --config config/pretrain.yaml --resume checkpoint.pt

# CPU模式
python examples/pretrain_example.py --config config/pretrain.yaml --device cpu

# 禁用夜间模式
python examples/pretrain_example.py --config config/pretrain.yaml --no-night-mode
```

### SFT微调命令

```bash
# 基础SFT微调
python examples/sft_example.py --config config/sft.yaml --pretrained pretrain_model.pt

# 准备SFT数据
python examples/sft_example.py --prepare-data raw_data.json

# 小数据集快速测试
python examples/sft_example.py --config config/sft.yaml --epochs 1 --batch-size 4
```

### 评估命令

```bash
# 基础评估
python examples/eval_example.py --model model.pt --test-file test.txt

# 评估并生成样本
python examples/eval_example.py --model model.pt --test-file test.txt --generate

# 指定生成提示
python examples/eval_example.py --model model.pt --generate \
    --prompts "床前明月光，" "春风又绿江南岸，"

# 限制评估批次
python examples/eval_example.py --model model.pt --test-file test.txt --max-batches 100

# 保存报告
python examples/eval_example.py --model model.pt --test-file test.txt --output eval_report.json
```

### 配置验证命令

```bash
# 验证配置文件
python -m src.config.validator config/pretrain.yaml

# 指定实验类型
python -m src.config.validator config/pretrain.yaml --type pretrain

# 创建配置模板
python -m src.config.validator --create-template pretrain --output my_config.yaml
```

## 配置模板说明

### 预训练模板

适合从头训练或继续预训练:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 768 | 模型维度 |
| num_layers | 12 | 层数 |
| batch_size | 8 | 批量大小 |
| learning_rate | 5e-5 | 学习率 |
| epochs | 10 | 训练轮数 |
| context_length | 512 | 上下文长度 |

### SFT模板

适合在预训练模型基础上微调:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 512 | 较小模型加速微调 |
| batch_size | 16 | 可适当增大 |
| learning_rate | 2e-5 | 较低学习率 |
| epochs | 3 | 通常3-5轮即可 |
| label_smoothing | 0.05 | 轻微平滑 |

### 强化学习模板

适合偏好对齐和优化:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 4 | RL需要较小batch |
| learning_rate | 1e-5 | 较低学习率 |
| kl_coef | 0.1 | KL惩罚系数 |
| clip_range | 0.2 | PPO截断范围 |

## 常见问题排查

### 问题1：配置文件找不到

```bash
# 确认配置文件存在
ls -la config/*.yaml

# 使用绝对路径
python examples/pretrain_example.py --config /full/path/to/config/pretrain.yaml
```

### 问题2：数据文件不存在

```bash
# 检查数据目录
ls -la dataset/

# 配置文件中的路径相对于工作目录
cd /workspace
python examples/pretrain_example.py
```

### 问题3：验证失败

仔细阅读验证错误信息:

```
Configuration validation failed. Errors found:
  [model] field required
  [training] 1 validation error for TrainingConfig
```

根据提示修复配置。

### 问题4：显存不足

减小模型和批量大小:

```yaml
model:
  d_model: 512
  num_layers: 6

training:
  batch_size: 4
```

### 问题5：训练太慢

- 启用GPU: 确保CUDA可用
- 启用混合精度: `use_amp: true`
- 减少数据加载workers: `num_workers: 2`

## 下一步

- 阅读 [使用指南](file:///workspace/docs/experiments/README.md) 了解更多高级功能
- 查看 [examples/](file:///workspace/examples) 中的完整示例代码
- 了解 [配置Schema](file:///workspace/src/config/schema.py) 详细参数说明
