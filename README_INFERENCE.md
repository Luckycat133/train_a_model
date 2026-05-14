# 灵猫模型推理 (Lingmao Moyun Inference)

本模块提供了灵猫语言模型的推理功能，包括命令行工具和 FastAPI 服务器。

## 功能特性

- **增强的采样策略**：支持温度、Top-K、Top-P 采样和重复惩罚
- **批量推理**：vLLM 风格的批量处理支持
- **KV Cache 加速**：与 FlashAttention 和 MLA 架构完全兼容
- **流式输出**：实时查看生成过程
- **REST API**：FastAPI 风格的推理服务

## 快速开始

### 安装依赖

首先确保已安装项目依赖：

```bash
pip install -r requirements.txt
```

如需使用 API 服务器，还需要安装 API 依赖：

```bash
pip install -r api/requirements.txt
```

## 命令行使用

### 基本用法

```bash
python generate.py --prompt "床前明月光"
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | model_weights/best_model_v0.4.pt | 模型路径 |
| `--tokenizer` | str | tokenizer.json | 分词器路径 |
| `--prompt` | str | (必填) | 生成提示文本 |
| `--max_length` | int | 100 | 最大生成长度 |
| `--max_new_tokens` | int | None | 最大新生成 token 数 |
| `--temperature` | float | 0.7 | 温度参数，控制随机性 |
| `--top_k` | int | 50 | Top-K 采样参数 |
| `--top_p` | float | 0.9 | Top-P (nucleus) 采样参数 |
| `--repetition_penalty` | float | 1.0 | 重复惩罚参数 |
| `--device` | str | auto | 设备 (cuda/cpu) |
| `--stream` | flag | False | 启用流式输出 |

### 示例

#### 普通生成

```bash
python generate.py --prompt "床前明月光" --max_length 50 --temperature 0.7
```

#### 流式输出

```bash
python generate.py --prompt "床前明月光" --stream
```

#### 使用更高的随机性

```bash
python generate.py --prompt "床前明月光" --temperature 1.0 --top_p 0.95
```

## API 服务器使用

### 启动服务器

```bash
cd api
python server.py
```

或使用环境变量配置：

```bash
MODEL_PATH=../model_weights/best_model_v0.4.pt TOKENIZER_PATH=../tokenizer.json PORT=8000 python api/server.py
```

### API 端点

#### 1. 根路径

```http
GET /
```

返回 API 信息和状态。

#### 2. 健康检查

```http
GET /health
```

检查服务健康状态。

#### 3. 模型信息

```http
GET /model/info
```

获取模型和分词器信息。

#### 4. 文本生成

```http
POST /generate
Content-Type: application/json

{
  "prompt": "床前明月光",
  "max_length": 100,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.9,
  "repetition_penalty": 1.0
}
```

**批量生成示例：**

```http
POST /generate
Content-Type: application/json

{
  "prompt": ["床前明月光", "白日依山尽"],
  "max_length": 50,
  "temperature": 0.7
}
```

#### 5. 流式生成

```http
POST /generate/stream
Content-Type: application/json

{
  "prompt": "床前明月光",
  "max_length": 100,
  "temperature": 0.7
}
```

响应格式为 Server-Sent Events (SSE)。

### Python 客户端示例

#### 普通生成

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "prompt": "床前明月光",
    "max_length": 50,
    "temperature": 0.7
})
print(response.json())
```

#### 流式生成

```python
import requests

response = requests.post(
    "http://localhost:8000/generate/stream",
    json={
        "prompt": "床前明月光",
        "max_length": 50,
        "temperature": 0.7
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

## 代码模块使用

### 导入和初始化

```python
from generate import (
    load_model,
    load_tokenizer,
    GenerationConfig,
    generate_text,
    generate_stream
)

# 加载模型和分词器
tokenizer = load_tokenizer("tokenizer.json")
model = load_model("model_weights/best_model_v0.4.pt", device="cuda")
```

### 文本生成

```python
config = GenerationConfig(
    max_length=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.0
)

result = generate_text(model, tokenizer, "床前明月光", config, device="cuda")
print(result)
```

### 批量生成

```python
prompts = ["床前明月光", "白日依山尽", "春眠不觉晓"]
results = generate_text(model, tokenizer, prompts, config, device="cuda")
for result in results:
    print(result)
```

### 流式生成

```python
print("床前明月光", end="", flush=True)
for text in generate_stream(model, tokenizer, "床前明月光", config, device="cuda"):
    print(text, end="", flush=True)
```

## 架构说明

### GenerationConfig

生成参数配置类，包含以下属性：

- `max_length`: 最大生成长度
- `temperature`: 温度参数
- `top_k`: Top-K 采样
- `top_p`: Top-P 采样
- `repetition_penalty`: 重复惩罚
- `use_cache`: 是否使用 KV Cache (默认 True)
- `max_new_tokens`: 最大新生成 token 数
- `stopping_criteria`: 停止条件 token ID 列表

### 采样策略

1. **温度采样 (Temperature)**: 控制随机性，值越高越随机
2. **Top-K 采样**: 仅从概率最高的 K 个 token 中采样
3. **Top-P (Nucleus) 采样**: 从累积概率达到 P 的最小 token 集合中采样
4. **重复惩罚 (Repetition Penalty)**: 减少生成文本中的重复

### KV Cache

系统默认启用 KV Cache，大幅提升长文本生成速度，与 FlashAttention 和 MLA 架构完全兼容。

## 文件结构

```
.
├── generate.py              # 推理主程序
├── api/
│   ├── server.py            # FastAPI 服务器
│   └── requirements.txt     # API 依赖
└── README_INFERENCE.md      # 本文档
```

## 常见问题

### Q: 如何选择合适的温度参数？

A: 一般来说，0.3-0.7 适合确定性较高的任务，0.7-1.2 适合创意性任务。

### Q: 内存不足怎么办？

A: 可以使用 CPU 进行推理，或者减小 `max_length` 参数。如果使用 CUDA，确保显存足够。

### Q: 如何使用自己训练的模型？

A: 使用 `--model` 参数指向您的模型文件，确保模型架构与 SimpleTransformer 兼容。

## 许可证

本项目遵循项目根目录的 LICENSE 文件中的许可证条款。
