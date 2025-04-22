# My LLM Project Documentation

[English](./DOCS_en.md) | [中文](../cn/DOCS_cn.md)

---

## Documentation Overview

This document provides a complete guide to the My LLM project, including module usage, training process, and generation methods.

## Module Guides

- [Project Overview](./project_overview.md) - Project introduction and architecture
- [Technical Architecture](./technical_architecture.md) - Detailed technical architecture and data flow
- [System Architecture Design](./design_docs/system_architecture.md) - System architecture and module relationships
- [Installation Guide](./installation/installation_guide.md) - Installation steps and environment configuration
- [Training Module Guide](./train_model_guide.md) - Model training methods and parameter configuration
- [Generation Module Guide](./generate_guide.md) - Text generation using the model
- [Processor Guide](./processor_guide.md) - Data preprocessing tools
- [Tokenizer Guide](./tokenizer_guide.md) - Tokenizer training and usage
- [Scripts Guide](./scripts_guide.md) - Auxiliary script functions
- [Cleanup Guide](./cleanup_guide.md) - Project cleanup and space optimization
- [Project Maintenance Guide](./project_maintenance.md) - Project volume management

## Best Practices

- [Training Optimization Best Practices](./best_practices/training_optimization.md) - Training optimization guide
  - Hardware optimization strategies
  - Hyperparameter tuning guide
  - Training stability optimization
  - Performance monitoring and tuning

## Code Explanation

- [Training Module Code](./code_explanation/train_model_code.md) - Training module core code
- [Tokenizer Code](./code_explanation/tokenizer_code.md) - Tokenizer module implementation
- [Processor Code](./code_explanation/processor_code.md) - Processor module architecture

## Changelog

- [Training Module Updates](../../changelog/en/for_train_model.md) - Training feature updates
- [Processor Updates](../../changelog/en/for_processor.md) - Processor feature history
- [Tokenizer Updates](../../changelog/en/for_tokenizer.md) - Tokenizer iteration records
- [Data Module Updates](../../changelog/en/for_data.md) - Data processing updates
- [Cleanup Tool Updates](../../changelog/en/for_cleanup.md) - Cleanup tool updates

## Documentation Index

All document details can be found in the [Document Index](./INDEX.md).

## Configuration Reference

Project configuration examples and parameter descriptions can be found in [Configuration Example](../../config/config.yaml).

## Documentation Structure

```
My_LLM/
├── README.md           # Project introduction
├── DOCS.md             # This document (navigation guide)
├── docs/               # Main documentation directory
│   ├── project_overview.md  # Detailed project description
│   ├── technical_architecture.md # Technical architecture
│   ├── train_model_guide.md # Training module guide
│   ├── generate_guide.md    # Generation module guide
│   ├── processor_guide.md   # Processor guide
│   ├── tokenizer_guide.md   # Tokenizer guide
│   ├── scripts_guide.md     # Scripts guide
│   ├── cleanup_guide.md     # Cleanup tool guide
│   ├── project_maintenance.md # Project maintenance
│   ├── design_docs/         # Design documents
│   │   └── system_architecture.md # System architecture
│   ├── installation/        # Installation guides
│   │   └── installation_guide.md # Installation guide
│   ├── best_practices/      # Best practices
│   │   └── training_optimization.md # Training optimization
│   ├── code_explanation/    # Code explanations
│   │   ├── train_model_code.md  # Training code
│   │   ├── tokenizer_code.md    # Tokenizer code
│   │   └── processor_code.md    # Processor code
│   └── INDEX.md        # Document index
└── changelog/          # Changelog directory
    ├── for_train_model.md  # Training updates
    ├── for_processor.md    # Processor updates
    ├── for_tokenizer.md    # Tokenizer updates
    ├── for_data.md         # Data updates
    └── for_cleanup.md      # Cleanup updates
```

## Core Documents

- [README.md](../../README.md) - Project overview and quick start
- [docs/project_overview.md](./project_overview.md) - Detailed project description
- [docs/technical_architecture.md](./technical_architecture.md) - Technical architecture
- [docs/design_docs/system_architecture.md](./design_docs/system_architecture.md) - System architecture
- [docs/installation/installation_guide.md](./installation/installation_guide.md) - Installation guide
- [docs/train_model_guide.md](./train_model_guide.md) - Detailed training guide
- [docs/generate_guide.md](./generate_guide.md) - Text generation guide
- [docs/best_practices/training_optimization.md](./best_practices/training_optimization.md) - Training best practices
- [docs/INDEX.md](./INDEX.md) - Complete document index

## Version Changelog

- [changelog/en/for_train_model.md](../../changelog/en/for_train_model.md) - Training updates
- [changelog/en/for_cleanup.md](../../changelog/en/for_cleanup.md) - Cleanup updates

## Recent Updates

- **[2025-03-18]** Added detailed installation guide for major OS
- **[2025-03-18]** Added system architecture document
- **[2025-03-18]** Added training optimization best practices
- **[2025-03-18]** Organized documentation structure
- **[2025-03-16]** Training system v0.8.0 released

## Additional Resources

- Install dependencies: `pip install -r requirements.txt`
- Feedback and contributions: Submit Issues or Pull Requests

---

*Last updated: March 18, 2025*