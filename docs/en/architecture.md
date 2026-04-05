# Architecture

## Module Overview

The codebase was refactored (2026-04-05) into a modular structure under `src/`.

### src/config.py
Centralizes all magic numbers and hyper-parameters. No magic numbers should appear in other modules.

### src/dataset.py
`LMDataset` wraps raw text files with memory-mapped arrays for efficient random access during training.

### src/model.py
`SimpleTransformer` is a minimal decoder-only transformer. `PositionalEncoding` uses sinusoidal embeddings.

### src/trainer.py
Contains `train_model()` (full training loop), `evaluate_model()`, and checkpoint utilities.

### src/run.py
CLI interface via argparse. Supports `--prepare`, `--train`, `--eval` modes.

## Data Flow

```
Raw JSONL → processor.py → LMDataset → SimpleTransformer → train_model → checkpoint
```

## Migration Notes

The original monolithic `train_model.py` has been split. Old entry points still work:

```bash
# Legacy (still supported)
python train_model.py --config config/default.yaml

# New modular way
python -m src.run --config config/default.yaml
```
