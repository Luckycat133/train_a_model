# 安装指南

[English](./en/installation_guide.md) | [中文](./installation_guide.md)

## 系统要求

- Python 3.8 或更高版本
- 支持CUDA的GPU（推荐用于训练）
- 最少16GB内存（推荐32GB）

## 环境配置

1. 创建虚拟环境
```bash
python -m venv llm_env
source llm_env/bin/activate  # Linux/MacOS
# 或
llm_env\Scripts\activate  # Windows
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 配置设置

1. 复制示例配置文件
```bash
cp config/config.example.yaml config/config.yaml
```

2. 更新配置文件中的设置
- 设置数据路径
- 配置模型参数
- 调整训练设置

## 验证安装

运行以下命令验证安装是否成功：
```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## 常见问题

### CUDA未找到
- 确保正确安装NVIDIA驱动
- 检查CUDA工具包版本兼容性
- 验证PyTorch安装版本与CUDA版本匹配

### 内存问题
- 在配置文件中减小批处理大小
- 关闭不必要的应用程序
- 考虑使用梯度累积

### 包冲突
- 尝试创建新的虚拟环境
- 逐个安装包以识别冲突
- 检查包版本兼容性

## 下一步

- 查看[项目概览](../project_overview.md)
- 检查[目录结构说明](../standards/directory_structure.md)
- 开始基础模型训练