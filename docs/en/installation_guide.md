# Installation Guide

[English](./installation_guide.md) | [中文](../cn/installation_guide.md)

---

## System Requirements

- Python 3.8 or higher
- pip package manager

## Installation Steps

1. Clone repository
```bash
git clone https://github.com/your_username/My_LLM.git
cd My_LLM
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure environment
- Modify `../../config/config.yaml` file according to your hardware configuration

4. Verify installation
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## FAQ

- If you encounter dependency conflicts, it's recommended to use a virtual environment
- Make sure your CUDA version is compatible with PyTorch version