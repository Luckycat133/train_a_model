# Model weights

Model binaries are intentionally not stored in Git. Historical weights in old commits are outside this release and remain subject to a separate approved backup and cleanup.

## v0.9.1 micro MLA SFT fixture

The [release](https://github.com/Luckycat133/train_a_model/releases/tag/v0.9.1) carries an inference checkpoint, tokenizer and fixed repository-authored synthetic fixtures (Apache-2.0), resolved benchmark configuration, held-out response-token metrics, and `SHA256SUMS`. The v0.9.0 release is source-only; do not associate the newer checkpoint with that tag.

Rebuild from the exact v0.9.1 Git checkout using Python 3.12 and the CPU training dependencies in [CI](../.github/workflows/ci.yml), including PyTorch 2.15.0+cpu:

```sh
python -m benchmarks.micro_mla_sft --out /tmp/micro-mla-a
python -m benchmarks.micro_mla_sft --out /tmp/micro-mla-b
diff -r /tmp/micro-mla-a /tmp/micro-mla-b
(cd /tmp/micro-mla-a && sha256sum -c SHA256SUMS)
```

The bundle's metrics bind the source commit, runtime, fixture hashes, parameter count, pretrain and SFT updates, and before/after token-weighted NLL, perplexity and top-1 accuracy. Compare bytes within the same pinned CPU runtime; the archive does not assert cross-platform bit identity.

Use `torch.load(path, map_location="cpu", weights_only=True)` and the model configuration in the archive. The checkpoint omits optimizer/RNG state; it is not a resume checkpoint, a production pretrained release, a throughput result, or a quality claim. No credentials, network training data or externally generated model responses are used.
