---
name: lingmao-experiment-closeout
description: Reproduce or close out Lingmao Moyun experiments in train_a_model using the actual CLI, checkpoint provenance and honest evaluation evidence. Use for this preparing-for-archive project's diagnostics, resume requests, reports and weight inventory; not as authorization to restart pretraining or rewrite Git history.
---

# Lingmao Moyun experiment closeout

Start with repository `README.md`, `EXPERIMENT_REPORT.md`, `model_weights/README.md`, `src/run.py`, and `tests/test_cli_contract.py`. The current goal is reproducible engineering evidence and safe closeout. Generic `pytorch-patterns` supports an explicitly requested training change; it must not expand a maintenance task into a new large experiment.

## Choose the evidence tier

1. CLI/config work: use the current environment for `python -m src.run --help` and the CLI contract tests. Inspect `src.config` resolution: this checkout contains both `src/config.py` and a `src/config/` package, so do not assume editing the file changes the imported module.
2. Explicit smoke run: `python -m src.run --quick --no_cuda` uses `examples/quick_train.jsonl`, one epoch and a tiny model, normally under `quick_runs/`. This is actual training, not a free configuration probe. Verify the selected output path and available dependencies first.
3. Resume: inspect target checkpoint, model/tokenizer/config/data compatibility, output directory and trainer restore code. Prefer an exact `--resume_from` for a known checkpoint. `--resume`/auto-resume selects the newest candidate, which may belong to another experiment. The quick preset disables auto-resume by default, but main() processes an explicit `--resume` afterward and turns it back on. Avoid combining a tiny `--quick` model configuration with an unrelated old checkpoint; verify the resolved flags and dimensions before a requested continuation. Do not report resumed optimizer/RNG state unless the checkpoint and restore path actually contain and restore it.
4. Quality: require an independent held-out dataset, split/provenance, model and tokenizer revision, resolved config, seed policy, metrics and representative generated outputs. The quick preset's fixed dimensions do not enforce a full-training random seed; the historical report explicitly lacks that evidence.

## Preserve the experiment

`--clean_before_run` deletes old logs and plots. It is not a normal troubleshooting step. Keep the source data and existing checkpoints; write a requested new run to a distinct output directory, with data/config/tokenizer/checkpoint hashes and terminal status. Never infer quality, convergence or speedup from source compilation, one decreasing loss, a CLI test or an output weight file.

New `.pt`, `.pth`, `.ckpt` and generated weights stay outside Git. Historical weight removal needs its own reviewed backup/object inventory/device coordination and authorization; do not run history rewriting or force-push as part of updating the report. An external release requires its actual URL and SHA-256 plus data/config/metrics/limitations; absence of published weights is an acceptable truthful closeout state.

For documentation-only work verify cited paths and commands without training. For a real run report which tier completed and what remains unverified. Do not claim the archive exit criteria passed merely because README contains the smoke command.
