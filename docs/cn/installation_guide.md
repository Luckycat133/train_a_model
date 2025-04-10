# 安装指南

## 系统要求

- Python 3.8 或更高版本
- pip 包管理工具

## 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/your_username/My_LLM.git
cd My_LLM
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境
- 根据您的硬件配置修改 `config/config.yaml` 文件

4. 验证安装
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 常见问题

- 如果遇到依赖冲突，建议使用虚拟环境
- 确保您的CUDA版本与PyTorch版本兼容