# Agent entry

Read `README.md`, `EXPERIMENT_REPORT.md`, and `.agents/SKILLS.md` before substantive work. This repository (`Lingmao Moyun` / `train_a_model`) is in **Experimental — Preparing for Archive** status; focus on reproducible engineering evidence and safe closeout rather than starting unbounded pretraining. Preserve existing work, keep changes scoped, and run focused checks proportional to the change.

## Quick Entry Pointers

- **CLI entrypoint**: `src/run.py` (`python -m src.run --help`; smoke path `--quick --no_cuda` writes to `quick_runs/` using `examples/quick_train.jsonl`; explicit checkpoint continuation should prefer `--resume_from` over `--resume`).
- **Config module gotcha**: both `src/config.py` and the `src/config/` package (`src/config/__init__.py`, `schema.py`, `validator.py`) exist—Python resolves the `src/config/` package on `import src.config`, so inspect both before changing defaults.
- **Lightweight verification (CI baseline)**: `python -m compileall -q src` and `pytest -q tests/test_cli_contract.py` (see `.github/workflows/ci.yml`; legacy tests live under `test/`).
- **Evidence & weight boundaries**: never commit new `.pt`/`.pth`/`.ckpt` weights, never rewrite Git history or force-push without explicit approval, and never infer model quality from smoke runs or source compilation (see `model_weights/README.md` and `CHANGELOG.md`).

## Skill Routing

- Default experiment reproduction, checkpoint provenance, report, and archive closeout: `.agents/skills/lingmao-experiment-closeout/SKILL.md`
- Additional curated skills (`pytorch-patterns`, `managing-python-dependencies`, `github-actions`): see `.agents/SKILLS.md`. Load only the Skill matching the current task.
