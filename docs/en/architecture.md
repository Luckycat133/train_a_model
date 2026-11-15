# Architecture: implementation and acceptance boundaries

Audit date: 2026-11-15. Source baseline: `01a5b3b9faf04ff0a70bfdbbd2ee97f76c5a2c4f`.
This is an experimental repository preparing for archive, not an accepted end-to-end training platform. See the [roadmap](../../ROADMAP.md), [audit](../development/AUDIT_2026-11-15.md), and [validation gates](../development/VALIDATION.md).

## Actual public training path

```text
python -m src.run
  -> argparse + flat YAML defaults + optional quick preset
  -> src.trainer.train_model
  -> src.dataset.LMDataset + src.model.SimpleTransformer
  -> local training/checkpoint helpers (currently blocked)
```

`--train` is a data-path alias, not a training-mode subcommand. `--quick`, `--save_dir` and `--resume` exist. The parser does not expose `--prepare`, `--eval`, SFT, MLA, or all Python-level model options. Consult `--help`; do not invent flags from configuration schemas.

## A separate, unintegrated trainer prototype

`src/training/base_trainer.py`, `pretrain.py`, and `sft_trainer.py` define another trainer family with optional Accelerate integration. The public CLI does not dispatch to it. Subclass constructor arguments and the model-output contract currently disagree with the base class/model. SFT data preparation also needs shift, leakage, padding and tokenizer-interface fixes.

Similarly, the richer `src/data/` implementations must not be assumed to be the datasets used by the public CLI. An implementation in one branch does not establish delivery in another.

## Model contract

`SimpleTransformer` defaults to `mode="modern"`: embeddings, RoPE attention, RMSNorm and SwiGLU; GQA, sliding-window attention, MoE and caching are optional branches. `legacy` uses sinusoidal positional encoding and a TransformerEncoder. It currently lacks an explicit causal mask and is not a verified autoregressive fallback.

Current return values:

| Invocation | Return value |
|---|---|
| default forward | logits `[B,L,V]` |
| `use_cache=True` | `(logits, cache)` |
| `return_aux_loss=True` | `(logits, cache_or_none, auxiliary_loss)` |

The default tensor must not be unpacked as `(outputs, _)`. Neither model branch is numerically accepted: SDPA layout/mask errors and cache position/head-shape problems are documented in the audit. Optional gradient checkpointing and MoE also have blocking contracts.

`MultiHeadLatentAttention` is **not defined in the audited `src/model.py`**. References in `debug_mla.py`, tests, and configuration constants are not an MLA implementation. An MLA design needs an explicit compressed-cache representation, position handling, model routing and full-prefix versus cached-decoding equivalence before acceptance.

## Data and tokenizer boundary

`LMDataset` parses JSONL text and makes shifted windows. It currently consumes `tokenize`/`batch_tokenize` strings where integer IDs are expected; its fallback uses `ord(c) % 30000`. Neither path is an accepted production encoding contract. Concatenation and memory mapping do not by themselves establish document-boundary isolation or bounded-memory streaming.

`ClassicalTokenizer` is a deterministic maximum-match prototype. `train()` is a no-op. `build_tokenizer.py` creates a character vocabulary while phrase matching remains enabled. Loading a JSON vocabulary does not implement that tokenizer format's merges, normalization or special-token policy.

## Configuration boundary

Both `src/config.py` and the `src/config/` package exist. The package initializer attempts imports from `src.config` itself and has fallback constants. Record the actual imported `__file__`; do not assume edits to the standalone module affect execution. The public CLI's flat YAML and experimental nested configuration loaders are separate contracts pending consolidation.

## Checkpoints and verification

The two trainer paths have different checkpoint naming/state conventions. Both contain invalid `torch.save(..., weights_only=True)` calls. The main path's auto-resume discovery and restore order do not establish optimizer/scheduler/scaler/RNG continuity. Inference weight loading and exact training resume are distinct capabilities.

Only source compilation and two CLI tests are covered by the existing CI. The [gates](../development/VALIDATION.md) define the required future behavior; publishing this document does not implement those gates or fix the runtime. A successful smoke run, once obtained, will still not prove language-model quality.
