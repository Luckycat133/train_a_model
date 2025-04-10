# My LLM Project

[English](./README_en.md) | [Chinese](./README.md)

## Project Overview

This is a project for training and developing large language models, including complete data processing, tokenization, training and evaluation workflows. The project adopts a modular design for easy extension and optimization.

## Key Features

- Data preprocessing and cleaning
- Custom tokenizer training
- Model training and optimization
- Performance evaluation and quality control

## Project Structure

```
├── changelog/       # Change logs
├── collection/      # Data collection
├── config/         # Configuration files
├── dataset/        # Datasets
├── docs/           # Project documentation
├── model_weights/  # Model weights
├── processors/     # Data processors
└── test/           # Test cases
```

## Quick Start

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Setup environment
Refer to [Installation Guide](./docs/en/installation_guide.md) or [安装指南](./docs/cn/installation_guide.md)

3. Run training
```bash
python train_model.py
```

## Documentation

- [Project Overview](./docs/project_overview.md)
- [Directory Structure](./docs/standards/directory_structure.md)
- [Change Log](./changelog/)

## License

This project is open source under MIT License.