# My LLM Project | 我的大语言模型项目

[English](./README_en.md) | [Chinese](./README.md)

---

## Project Overview | 项目概述

This is a project for training and developing large language models, including complete data processing, tokenization, training and evaluation workflows. The project adopts a modular design for easy extension and optimization.

---

## Key Features | 主要功能

- Data preprocessing and cleaning
- Custom tokenizer training
- Model training and optimization
- Performance evaluation and quality control

---

## Project Structure | 项目结构

```
├── changelog/       # Change logs
├── collection/      # Data collection
├── config/          # Configuration files
├── dataset/         # Datasets
├── docs/            # Project documentation
├── model_weights/   # Model weights
├── processors/      # Data processors
└── test/            # Test cases
```

### Dataset Collection Instructions

The dataset for this project mainly comes from public ancient book projects (such as Chinese-Poetry), including about 55,000 Tang poems, 260,000 Song poems, 21,000 Song lyrics, and other classical collections. All data has undergone OCR error correction, manual proofreading, punctuation and paragraph optimization, and unified variant character processing standards. Some datasets have also been supplemented with missing content, added syntactic structure, part-of-speech, and semantic annotations. For details on data collection and processing, see [Data-related Changelog](./changelog/en/for_data.md).

You can download the original data from the corresponding open-source repositories based on the dataset source information above, or you can collect and organize datasets suitable for your needs according to the process provided by this project.

**Before starting model training, it is strongly recommended to run the tokenizer-related scripts to prepare the tokenized files.**

---

## Quick Start

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

2. Setup environment

   Refer to [Installation Guide](./docs/en/installation_guide.md) or [安装指南](./docs/cn/installation_guide.md)

3. Run the tokenizer

   ```bash
   python tokenizer.py
   ```

4. Run training

   ```bash
   python train_model.py
   ```

---

## Documentation | 文档

- [Project Overview](./docs/en/project_overview.md)
- [Directory Structure](./docs/en/directory_structure.md)
- [Change Log](./changelog/en/)

---

## License | 许可证

This project is open source under MIT License.

---

## Educational LLM Training Plan | 教育用途的大模型训练计划

This section provides a systematic and reproducible large language model (LLM) training plan for beginners and educators, helping to understand and master the core workflow of LLM development.

### 1. Project Goals and Educational Value

- Practice the full process of data collection, preprocessing, tokenization, model training, and evaluation.
- Develop interdisciplinary skills in data engineering, machine learning, and natural language processing.
- Support classroom teaching, research experiments, and personal skill improvement.

### 2. Dataset Preparation and Collection

1. Refer to the "Dataset Collection Instructions" section to select public ancient book projects or your own text data.
2. Follow the [Data-related Changelog](./changelog/en/for_data.md) for OCR correction, manual proofreading, annotation, etc.
3. It is recommended to organize data in a standardized text format (e.g., one sentence per line, with labels if needed).

### 3. Tokenizer Execution and Data Preprocessing

1. Run the tokenizer script:

   ```bash
   python tokenizer.py
   ```

2. Check the output tokenized files to ensure the results meet expectations.
3. Customize the vocabulary or extend tokenization rules as needed.

### 4. Detailed Model Training Process

1. Configure training parameters (such as model architecture, batch size, learning rate, etc.), see `config/config.yaml` for reference.
2. Start training:

   ```bash
   python train_model.py
   ```

3. Monitor metrics like loss and accuracy during training and adjust parameters as needed.
4. Supports checkpointing and multiple rounds of experiments.

### 5. Evaluation and Optimization Methods

1. Use built-in evaluation scripts or custom test sets to regularly assess model performance.
2. Analyze model outputs and optimize data, model structure, or training strategies accordingly.
3. Record experiment logs for reproducibility and comparison.

### 6. Reproducibility and Extension Suggestions

- Keep scripts and configurations for each step to facilitate reproducibility.
- Try different tokenization methods, model architectures, or data augmentation techniques to explore best practices.
- Summarize experiment results into reports or teaching cases to promote sharing and communication.

For further guidance, please refer to the project documentation or raise questions in the Issues section. Happy learning!