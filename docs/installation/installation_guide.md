# 灵猫墨韵安装指南

本文档提供在不同操作系统下安装灵猫墨韵项目的详细步骤和环境配置说明。

## 目录

- [系统要求](#系统要求)
- [安装步骤](#安装步骤)
  - [Linux安装](#linux安装)
  - [macOS安装](#macos安装)
  - [Windows安装](#windows安装)
- [Apple Silicon特殊配置](#apple-silicon特殊配置)
- [CUDA配置](#cuda配置)
- [MPS配置](#mps配置)
- [验证安装](#验证安装)
- [常见问题](#常见问题)

## 系统要求

### 最低配置

- **CPU**: 4核心处理器
- **内存**: 8GB RAM
- **硬盘**: 20GB可用空间
- **Python**: 3.8+

### 推荐配置

- **CPU**: 8核心处理器
- **GPU**: 
  - NVIDIA: RTX 2060或更高
  - Apple: M1/M2/M3系列芯片
- **内存**: 16GB RAM以上
- **硬盘**: 50GB可用空间(SSD)
- **Python**: 3.9+

## 安装步骤

首先，克隆项目仓库：

```bash
git clone https://github.com/yourusername/My_LLM.git
cd My_LLM
```

### Linux安装

1. **安装依赖包**

```bash
# 安装系统依赖
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-pip

# 创建并激活虚拟环境
python3 -m venv llm_env
source llm_env/bin/activate

# 安装Python依赖
pip install --upgrade pip
pip install -r requirements.txt
```

2. **CUDA安装(可选，仅对NVIDIA GPU)**

```bash
# 对于CUDA 11.7
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
```

### macOS安装

1. **安装依赖包**

```bash
# 安装Homebrew (如果尚未安装)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装Python (如果尚未安装)
brew install python

# 创建并激活虚拟环境
python3 -m venv llm_env
source llm_env/bin/activate

# 安装Python依赖
pip install --upgrade pip
pip install -r requirements.txt
```

2. **Apple Silicon特殊配置(M1/M2/M3)**

```bash
# 安装支持MPS的PyTorch版本
pip install torch torchvision
```

### Windows安装

1. **安装依赖包**

```bash
# 创建并激活虚拟环境
python -m venv llm_env
llm_env\Scripts\activate

# 安装Python依赖
pip install --upgrade pip
pip install -r requirements.txt
```

2. **CUDA安装(可选，仅对NVIDIA GPU)**

```bash
# 对于CUDA 11.7
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
```

## Apple Silicon特殊配置

对于Apple M1/M2/M3芯片的Mac设备，需要进行特殊配置以启用MPS加速：

1. **确认安装的PyTorch版本支持MPS**

```bash
# 检查PyTorch版本
python -c "import torch; print(torch.__version__)"

# 确认MPS可用
python -c "import torch; print(torch.backends.mps.is_available())"
```

2. **配置训练脚本使用MPS**

当运行训练脚本时，使用以下参数：

```bash
python train_model.py --device mps
```

## CUDA配置

对于NVIDIA GPU用户，需要确保CUDA环境正确配置：

1. **确认CUDA可用**

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

2. **设置CUDA环境变量(Linux/macOS)**

```bash
export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU
```

3. **设置CUDA环境变量(Windows)**

```bash
set CUDA_VISIBLE_DEVICES=0
```

4. **配置训练脚本使用CUDA**

```bash
python train_model.py --device cuda
```

## MPS配置

对于Apple Silicon设备，使用Metal Performance Shaders (MPS)加速：

1. **确认MPS可用**

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

2. **配置训练脚本使用MPS**

```bash
python train_model.py --device mps
```

3. **性能注意事项**

MPS目前不支持所有PyTorch操作，如果遇到问题可以使用CPU模式：

```bash
python train_model.py --device cpu
```

## 验证安装

安装完成后，可以运行以下命令验证安装是否成功：

```bash
# 激活虚拟环境
# Linux/macOS
source llm_env/bin/activate
# Windows
llm_env\Scripts\activate

# 运行测试脚本
python -c "import torch; import numpy; print('PyTorch版本:', torch.__version__); print('NumPy版本:', numpy.__version__)"

# 测试简单生成
python generate.py --prompt "春江花月夜" --max_length 50
```

如果以上命令执行无误，并成功生成文本，则表示安装成功。

## 常见问题

### 1. 导入PyTorch时出错

**问题**: 导入PyTorch时报错 `ImportError: No module named 'torch'`

**解决方案**: 
- 确认已激活虚拟环境
- 重新安装PyTorch: `pip install torch`

### 2. CUDA不可用

**问题**: `torch.cuda.is_available()` 返回 `False`

**解决方案**:
- 确认安装了支持CUDA的PyTorch版本
- 检查NVIDIA驱动程序是否正确安装
- 运行 `nvidia-smi` 检查GPU是否被系统识别

### 3. MPS不可用

**问题**: Apple Silicon设备上MPS不可用

**解决方案**:
- 确认PyTorch版本>=1.12.0
- 确认macOS版本>=12.3
- 重新安装PyTorch: `pip install --upgrade torch`

### 4. 内存错误

**问题**: 训练时出现内存错误 `CUDA out of memory` 或 `MemoryError`

**解决方案**:
- 减小批次大小: `--batch_size 8`
- 增加梯度累积步数: `--accumulation_steps 4`
- 使用混合精度训练: `--use_amp True`
- 如使用MPS，可能需要多次重启或使用CPU

### 5. 训练太慢

**问题**: 训练速度非常慢

**解决方案**:
- 对于GPU: 启用混合精度 `--use_amp True`
- 调整批次大小到适合的值
- 确认是否使用了正确的设备 (查看训练日志)
- 如果使用CPU，考虑设置 `--batch_size 4 --context_length 256`

---

如果遇到未在此列出的问题，请查阅[项目维护指南](../project_maintenance.md)或提交Issue。

---

*文档最后更新: 2025年3月18日* 