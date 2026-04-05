# 🏮 灵猫墨韵 | Lingmao Moyun

**A personal journey into training large language models from scratch** — starting with Chinese classical poetry.

> _Dataset processing, tokenization, training, and evaluation — all open source. Let's discuss and grow together!_

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

---

## 📁 Project Structure

```
train_a_model/
├── src/                      # Core modules (refactored)
│   ├── __init__.py           # Package entry
│   ├── config.py              # All configuration constants
│   ├── dataset.py             # LMDataset implementation
│   ├── logger.py             # Logging utilities
│   ├── model.py              # SimpleTransformer, PositionalEncoding
│   ├── trainer.py            # Training loop, evaluation, checkpoints
│   ├── run.py                # CLI entry point
│   └── utils.py              # General utilities
├── dataset/                  # Training data
├── processors/               # Data processors
├── model_weights/           # Saved checkpoints
├── test/                    # Test suite
├── config/                  # YAML configs
├── docs/                    # Documentation
├── train_model.py           # Legacy entry (uses src/)
├── generate.py              # Text generation
├── tokenizer.py             # ClassicalTokenizer
├── cleanup.py               # Dataset cleanup
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare tokenizer (recommended before training)
python -m src.run --prepare

# 3. Train model
python -m src.run --config config/default.yaml

# 4. Generate text
python generate.py --checkpoint model_weights/best_model.pt
```

---

## 📜 Modules

### `src.config`
All magic numbers extracted as named constants. Edit here instead of hunting through code.

### `src.dataset`
`LMDataset` — memory-mapped dataset for efficient large-file training.

### `src.model`
`SimpleTransformer` — lightweight transformer with `PositionalEncoding`.

### `src.trainer`
`train_model()`, `evaluate_model()`, checkpoint save/load utilities.

### `src.run`
CLI interface via `python -m src.run`.

---

## 🔧 Configuration

Edit `config/config.yaml` or create your own. Key settings:

| Parameter | Default | Description |
|----------|---------|-------------|
| `d_model` | 256 | Model dimension |
| `nhead` | 8 | Attention heads |
| `num_layers` | 6 | Transformer layers |
| `context_length` | 256 | Input sequence length |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 1e-3 | Optimizer learning rate |

---

## 📚 Documentation

- [English](docs/en/)
- [中文](docs/cn/)

---

## ⚠️ Known Issues & Limitations

- `ClassicalTokenizer` uses greedy max-match; BPE/WordPiece hybrid planned
- Large dataset loading is memory-bound; mmap helps but 32GB+ RAM recommended
- MacOS Metal GPU support not yet tested

---

## 🤝 Contributing & Discussion

Open an issue or PR! This is a learning project — constructive feedback welcome.

---

_Last updated: 2026-04-05_
