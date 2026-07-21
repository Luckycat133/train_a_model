# Lingmao Moyun — Final Experiment Report

Status: Experimental — preparing for archive  
Last verified: 2026-07-18

## What this repository demonstrates

This repository explores a compact Transformer training pipeline for classical-Chinese-oriented text. It includes data loading, tokenizer integration, training, evaluation, checkpointing, and both legacy and experimental modern model components.

## Reproducible smoke experiment

```bash
python -m pip install -r requirements.txt
python -m src.run --quick --no_cuda
```

The quick command uses the public fixture in `examples/quick_train.jsonl`, one epoch, a 32-dimensional one-layer model, CPU-safe settings, and writes ignored artifacts to `quick_runs/`. It validates plumbing only; it is not a quality benchmark.

## Current evidence

| Item | Status |
|---|---|
| Fixed seed | Not yet enforced across the full trainer |
| Dataset version | Only the public smoke fixture is versioned |
| Loss curve | Not independently verified |
| Held-out metrics | Not available |
| Example generation quality | Not independently verified |
| Weight checksum / external URL | Not published |
| CI | CLI contract and source compilation only |

No performance or quality improvement is claimed without a recorded benchmark.

## Weight inventory

Historical weight files may exist in earlier Git commits. New `.pt`, `.pth`, and `.ckpt` files are ignored. A future external weight release must be documented in `model_weights/README.md`.

History cleanup is intentionally deferred because it requires a complete backup, a reviewed object inventory, credential/device coordination, and a force-push window.

## What was learned

- A large Git repository is a poor distribution channel for model weights.
- A smoke test and a quality evaluation answer different questions.
- CLI examples must be exercised by automation.
- From-scratch pretraining is not justified here without a licensed dataset, compute budget, and fixed evaluation suite.
- Future work should prefer a small, measurable fine-tuning experiment over another unbounded pretraining run.

## Archive exit criteria

- external weight URL and SHA-256 recorded, or weights explicitly declared unavailable;
- full history cleanup completed under a separately approved plan;
- `python -m src.run --quick --no_cuda` verified in an environment with PyTorch;
- final limitations and available metrics recorded without overclaiming.
