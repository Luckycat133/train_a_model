# 灵猫墨韵安装指南

<div align="center">

![安装指南](https://img.shields.io/badge/🔧_安装指南-环境配置-4682B4)

*本文档提供灵猫墨韵模型系统的完整安装步骤和环境配置方法*

</div>

## 📑 目录

- [概述](#概述)
- [系统要求](#系统要求)
- [安装步骤](#安装步骤)
  - [Linux安装](#linux安装)
  - [macOS安装](#macos安装)
  - [Windows安装](#windows安装)
- [环境配置](#环境配置)
  - [GPU配置](#gpu配置)
  - [CPU配置](#cpu配置)
- [验证安装](#验证安装)
- [低资源环境配置](#低资源环境配置)
- [常见问题](#常见问题)

## 概述

灵猫墨韵是一个轻量级的古风文言语言模型系统，专为古典文学文本优化。本指南将帮助您在不同平台上安装并配置灵猫墨韵，以便快速开始使用。

> 💡 **提示**：如果您只需要使用模型而不进行训练，可以使用更轻量的配置方案，参见[低资源环境配置](#低资源环境配置)。

## 系统要求

在开始安装前，请确保您的系统满足以下基本要求：

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Ubuntu 20.04+ / macOS 11+ / Windows 10+ | Ubuntu 22.04 / macOS 12+ / Windows 11 |
| CPU | 4核心 | 8核心或更高 |
| 内存 | 8GB | 16GB或更高 |
| 磁盘空间 | 5GB | 20GB或更高 |
| Python | 3.10+ | 3.10 - 3.11 |
| GPU(可选) | NVIDIA GPU, 4GB VRAM | NVIDIA GPU, 8GB+ VRAM |

## 安装步骤

### Linux安装

1. **安装依赖包**

```bash
sudo apt update
sudo apt install -y build-essential python3-dev python3-pip git
```

2. **克隆代码库**

```bash
git clone https://github.com/username/lingcat-moyun.git
cd lingcat-moyun
```

3. **创建虚拟环境**

```bash
python3 -m venv venv
source venv/bin/activate
```

4. **安装Python依赖**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. **GPU支持(可选)**

如果有NVIDIA GPU，可安装CUDA支持：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### macOS安装

1. **安装Homebrew和依赖**

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.10 git
```

2. **克隆代码库**

```bash
git clone https://github.com/username/lingcat-moyun.git
cd lingcat-moyun
```

3. **创建虚拟环境**

```bash
python3 -m venv venv
source venv/bin/activate
```

4. **安装Python依赖**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. **MPS支持(Apple Silicon可选)**

对于M1/M2 Mac，启用MPS加速：

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

可以将此行添加到`~/.zshrc`或`~/.bash_profile`以永久启用。

### Windows安装

1. **安装Python和Git**

从[Python官网](https://www.python.org/downloads/)下载并安装Python 3.10+
从[Git官网](https://git-scm.com/download/win)下载并安装Git

2. **克隆代码库**

```cmd
git clone https://github.com/username/lingcat-moyun.git
cd lingcat-moyun
```

3. **创建虚拟环境**

```cmd
python -m venv venv
venv\Scripts\activate
```

4. **安装Python依赖**

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

5. **GPU支持(可选)**

如果有NVIDIA GPU:

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 环境配置

### GPU配置

如需启用GPU支持，请确保：

1. 已安装最新的NVIDIA驱动
2. 已安装CUDA工具包(11.8推荐)
3. 修改配置文件中的`device`参数：

```yaml
# config.yaml
training:
  device: "cuda"  # 使用GPU
  precision: "float16"  # 或"bfloat16"以提高性能
```

### CPU配置

对于CPU-only环境：

```yaml
# config.yaml
training:
  device: "cpu"
  precision: "float32"
  num_threads: 4  # 根据CPU核心数调整
```

## 验证安装

安装完成后，可以运行以下命令验证安装：

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python scripts/verify_install.py
```

如果一切正常，将看到验证成功的消息。

## 低资源环境配置

对于仅用于推理的低资源环境，可以使用以下配置：

1. **安装精简版**

```bash
pip install lingcat-moyun-lite  # 轻量版，仅包含推理功能
```

2. **使用量化模型**

```python
from lingcat_moyun import MoYunModel

model = MoYunModel.from_pretrained("moyun-lite-int8")  # 加载量化模型
```

3. **优化内存使用**

```yaml
# config.yaml
generation:
  device: "cpu"
  max_length: 256  # 减小最大长度
  batch_size: 1  # 最小批量
  optimize_memory: true
```

## 常见问题

<details>
<summary><b>安装过程中出现"无法找到CUDA"错误？</b></summary>

如果安装时出现CUDA相关错误：

1. 确认已安装兼容的NVIDIA驱动
2. 验证CUDA工具包是否正确安装：
   ```bash
   nvcc --version
   ```
3. 确保CUDA路径已添加到环境变量
4. 尝试使用特定版本的PyTorch：
   ```bash
   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```
</details>

<details>
<summary><b>如何在没有GPU的环境中优化性能？</b></summary>

在CPU环境中优化性能：

1. 启用Intel MKL加速：
   ```bash
   pip install intel-openmp mkl
   ```
   
2. 使用量化模型减少内存占用
   
3. 调整批处理大小并减少上下文长度
   
4. 考虑使用更小的模型变种（mini或nano版本）
</details>

<details>
<summary><b>模型加载时内存不足如何解决？</b></summary>

如果遇到内存不足问题：

1. 使用`device_map="auto"`启用模型分片：
   ```python
   model = MoYunModel.from_pretrained("moyun-base", device_map="auto")
   ```
   
2. 使用量化模型：
   ```python
   from lingcat_moyun.quantization import load_quantized_model
   model = load_quantized_model("moyun-base", quantization="int8")
   ```
   
3. 降低批处理大小和序列长度
</details>

---

<div align="center">

[📚 文档目录](../../summary.md) | [📝 项目概述](../../project_overview.md) | [🧠 训练指南](../training/index.md)

</div>

*最后更新时间: 2025-03-20* 