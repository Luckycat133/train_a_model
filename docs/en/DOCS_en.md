# Lingmao Moyun (train_a_model) Documentation

[English](./DOCS_en.md) | [中文](../cn/DOCS_cn.md) | [Documentation Index](../README.md)

---

## Documentation Overview

This hub links to the existing English and bilingual documentation in this repository. For current project status, quick verification commands, and model-weight boundaries, start with the root [README.md](../../README.md).

## Getting Started & Architecture

- [Getting Started](./getting-started.md) — Cloning, setup, and basic workflow
- [Installation Guide](./installation_guide.md) — System requirements and environment setup
- [Architecture](./architecture.md) — Refactored modular `src/` overview
- [Research Report: Latest LLM Architecture (2025–2026)](./research-latest-llm-architecture.md) — Architecture research report

## Data & Experiments (Chinese)

- [Data Documentation Overview](../data/README.md) — Classical Chinese training data overview
- [Data Acquisition & Preprocessing](../data/getting_started.md) — Data preparation workflow
- [Data Formats](../data/formats.md) — JSONL schema and field definitions
- [Experiment Platform Guide](../experiments/README.md) — Pretraining, SFT, and evaluation guide
- [Experiment Quickstart](../experiments/quickstart.md) — Quickstart workflow for training runs

## Project Reports & References

- [Experiment Report](../../EXPERIMENT_REPORT.md) — Final findings, known gaps, and archival criteria
- [Dataset Reference](../../DATASET.md) — Full dataset documentation
- [Best Practices](../../BEST_PRACTICES.md) — Training and engineering best practices
- [Model Weights Boundary](../../model_weights/README.md) — Weight tracking and external release requirements

## Module Changelogs

- [Training Module Updates](../../changelog/en/for_train_model.md) — Training feature updates
- [Processor Updates](../../changelog/en/for_processor.md) — Processor feature history
- [Tokenizer Updates](../../changelog/en/for_tokenizer.md) — Tokenizer iteration records
- [Data Module Updates](../../changelog/en/for_data.md) — Data processing updates
- [Cleanup Tool Updates](../../changelog/en/for_cleanup.md) — Cleanup tool updates

## Documentation Index & Configuration

- Browse the complete documentation map in [docs/README.md](../README.md).
- See [config/config.yaml](../../config/config.yaml) for configuration parameters.

## Current Documentation Layout

```text
docs/
├── README.md                              # Complete documentation index
├── en/
│   ├── DOCS_en.md                         # This document (English hub)
│   ├── architecture.md                    # Modular src/ architecture
│   ├── getting-started.md                 # Getting Started
│   ├── installation_guide.md              # Installation Guide
│   └── research-latest-llm-architecture.md
├── cn/
│   ├── DOCS_cn.md                         # 中文文档导航
│   ├── getting-started.md                 # 快速入门
│   └── installation_guide.md              # 安装指南
├── data/
│   ├── README.md                          # Data documentation overview
│   ├── getting_started.md                 # Data acquisition & preprocessing
│   └── formats.md                         # JSONL data formats
└── experiments/
    ├── README.md                          # Experiment platform guide
    └── quickstart.md                      # Experiment quickstart
```
