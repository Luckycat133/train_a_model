# Installation Guide

[English](./installation_guide.md) | [中文](../installation_guide.md)

## System Requirements

- Python 3.8 or higher
- CUDA compatible GPU (recommended for training)
- 16GB RAM minimum (32GB recommended)

## Environment Setup

1. Create a virtual environment
```bash
python -m venv llm_env
source llm_env/bin/activate  # Linux/MacOS
# or
llm_env\Scripts\activate  # Windows
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the example configuration file
```bash
cp config/config.example.yaml config/config.yaml
```

2. Update the configuration file with your settings
- Set data paths
- Configure model parameters
- Adjust training settings

## Verify Installation

Run the following command to verify the installation:
```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Common Issues

### CUDA Not Found
- Ensure NVIDIA drivers are properly installed
- Check CUDA toolkit version compatibility
- Verify PyTorch installation matches CUDA version

### Memory Issues
- Reduce batch size in config file
- Close unnecessary applications
- Consider using gradient accumulation

### Package Conflicts
- Try creating a fresh virtual environment
- Install packages one by one to identify conflicts
- Check package version compatibility

## Next Steps

- Review the [Project Overview](../project_overview.md)
- Check the [Directory Structure](../standards/directory_structure.md)
- Start with basic model training