# Lingmao Moyun — Experiment Status and Closeout Report

Status: **Experimental — Preparing for Archive**  
Source audit: **2026-11-15**, commit `01a5b3b9faf04ff0a70bfdbbd2ee97f76c5a2c4f`  
This date records a source/evidence review, not successful full training verification. The previous report was dated 2026-07-18.

## What this repository contains

A classical-Chinese-oriented Transformer training experiment with data/tokenizer prototypes, a public CLI training path, and a separate experimental trainer family. Modern model components and pretrain/SFT code exist, but interface and numerical blockers prevent acceptance as a complete platform. The audited model file does not define the MLA class referenced by debug/tests.

See the [audit](docs/development/AUDIT_2026-11-15.md), [roadmap](ROADMAP.md), and [gates](docs/development/VALIDATION.md). This documentation change fixes no training code and adds no CI jobs.

## Smoke experiment: acceptance target, not a passing result

```bash
python -m pip install -r requirements.txt
python -m src.run --quick --no_cuda
```

The preset selects the public fixture, one epoch, a 32-dimensional one-layer model and ignored `quick_runs/` output. Current source has tokenizer/AMP/scaler/checkpoint blockers. The complete command was **not executed in a dependency-compliant environment in this audit**. Do not describe it as a demonstrated working pipeline until G1–G3 pass.

## Current evidence

| Item | Evidence and status |
|---|---|
| Lightweight CI | Historical CI run 32663275544 succeeded on the audited commit; compilation and two CLI tests only |
| Local audit probes | Standalone API/tensor reproductions on Python 3.13.5 / PyTorch 2.10.0+cpu; below repository's torch>=2.13 requirement; not repository tests |
| Fixed seed and full reproducibility | Not established across the public trainer and data state |
| Dataset version / licensing | Public smoke fixture tracked; production corpus manifest, rights and split isolation not independently accepted |
| Tokenizer | Prototype and builder exist; ID/string and vocabulary/segmentation contracts are blocked |
| Attention / MLA | Numerical/cache blockers; MLA definition absent from the audited model file |
| True pretrain / SFT / resume | Not accepted; concrete source blockers recorded |
| Loss curves / held-out metrics | No independently verified quality evidence |
| Generation quality | Not independently verified |
| CUDA / MPS / distributed / compile | Not executed in this audit; no performance claim |
| Public weight URL / checksum | No independently verified public release |

Historical source evidence: [CI run](https://github.com/Luckycat133/train_a_model/actions/runs/32663275544). Dependency bot successes are not training evidence. A minimal reproduction of a bad operation is not a full-suite result.

## Weight inventory and data responsibility

Historical weights may exist in earlier Git commits. New `.pt`, `.pth`, and `.ckpt` files remain excluded. External release requires the documentation in `model_weights/README.md`, data rights/version, resolved model/tokenizer configuration, evaluation limits and SHA-256.

History cleanup remains separate: complete backup, reviewed object inventory, collaborator/device coordination and explicit approval before a force-push window. This audit neither rewrites history nor changes repository archival settings.

## Conclusions and next decision

First restore a numerically correct, saveable, reproducible CPU baseline (ROADMAP R1). Existing smoke claims, performance figures and test names are not substitutes for evidence. Without a licensed corpus, compute/token budget and fixed evaluation, another unbounded from-scratch run is not justified by current results.

If development continues, the proposed next experiment is a bounded SFT baseline after the common runtime passes its gates. MLA and other architecture research remain conditional projects, not delivered capabilities.

## Archive exit criteria

- Weights explicitly declared unavailable, or a reviewed external URL and SHA-256 documented.
- R1 engineering smoke verified with exact command, environment, commit and artifacts. Until then this exit criterion remains blocked; recording a failure is not a passing result.
- Final limitations, dataset responsibility and available metrics recorded without overclaiming; no quality claim from smoke alone.
- Required historical weight cleanup completed under its separately approved plan. Until then record the remaining objects and leave this exit criterion open; documentation is not cleanup.

No automatic archive or history rewrite is authorized by this roadmap.
