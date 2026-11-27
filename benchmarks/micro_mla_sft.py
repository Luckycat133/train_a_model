"""Run a bounded deterministic MLA pretrain + SFT + held-out CPU benchmark."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SEED = 1729
DIMENSIONS = dict(d_model=32, nhead=4, num_layers=1, dim_feedforward=64,
                  dropout=0.0, use_mla=True, mla_latent_dim=8, mla_num_latent_heads=2)
PRETRAIN = ("甲風入林微光浮影鳥歸巢", "乙雨落庭石階滴水苔生綠", "赤鳥越山晴霧散開霞生光",
            "青舟渡河漁火悠悠岸草長", "星燈映窗夜色悄然人未眠", "雪松照泉寒枝低語霜滿地")
TRAIN = (("風過何方", "穿林"), ("雨至何方", "落庭"), ("赤羽何行", "越山"),
         ("小舟何往", "渡河"), ("星光何照", "映窗"), ("白雪何見", "照泉"))
EVAL = (("庭中聽何聲", "落雨"), ("山下見何色", "青"), ("窗前何物明", "星燈"))
PROMPT = "問：{instruction}；{input}。"
RESPONSE = "答："
BATCH_SIZE = 2
CONTEXT = 32
EPOCHS = 2


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                            for row in rows), encoding="utf-8")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def benchmark(out: Path, source: str, work: Path) -> None:
    # Set before import so container CPU dispatch/thread-pool choices are fixed.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["ATEN_CPU_CAPABILITY"] = "default"
    import torch

    from src.config import VERSION
    from src.model import MultiHeadLatentAttention
    from src.trainer import train_model
    from src.training.sft_trainer import SFTConfig, SFTTrainer
    from tokenizer import ClassicalTokenizer

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False

    if set(p for p, _ in TRAIN) & set(p for p, _ in EVAL):
        raise ValueError("Synthetic training and evaluation prompts overlap")
    tokenizer_path = out / "tokenizer.json"
    # Static character vocabulary over the declared fixture alphabet; it is not fitted on evaluation data.
    sample_alphabet = "".join(PRETRAIN) + "".join(p + r for p, r in TRAIN + EVAL)
    chars = sorted(set(sample_alphabet + "問：；。答N/A"))
    write_json(tokenizer_path, {"vocab": {v: i for i, v in enumerate(
        ["<pad>", "<unk>", "<bos>", "<eos>", *chars])}})
    # Fail if a later tokenizer change makes the fixture silently evaluate on unknown tokens.
    tok = ClassicalTokenizer()
    tok.load(str(tokenizer_path))
    if any(1 in tok.encode(text) for text in list(PRETRAIN) + [p + RESPONSE + r for p, r in TRAIN + EVAL]):
        raise ValueError("Synthetic vocabulary does not cover the benchmark fixture")

    pretrain = out / "pretrain.jsonl"
    train = out / "train.jsonl"
    evaluation = out / "eval.jsonl"
    write_jsonl(pretrain, ({"text": text} for text in PRETRAIN))
    write_jsonl(train, ({"instruction": p, "output": r} for p, r in TRAIN))
    write_jsonl(evaluation, ({"instruction": p, "output": r} for p, r in EVAL))

    train_model(train_file=str(pretrain), tokenizer_path=str(tokenizer_path),
                model_save_dir=str(work / "pretrain"), device=torch.device("cpu"),
                context_length=CONTEXT, batch_size=BATCH_SIZE, epochs=1, learning_rate=0.003,
                seed=SEED, auto_resume=False, night_mode=False, use_amp=False, use_checkpoint=False,
                use_gradient_checkpointing=False, use_compile=False, use_sliding_window=False,
                num_workers=0, **DIMENSIONS)
    pretrained_path = work / "pretrain" / "model_epoch_1.pt"
    pretrain_state = torch.load(pretrained_path, weights_only=True, map_location="cpu")
    pretrain_updates = pretrain_state["stats"]["optimizer_updates"][-1]
    if pretrain_updates < 1:
        raise RuntimeError("Pretraining did not complete an optimizer update")
    cfg = SFTConfig(tokenizer_path=str(tokenizer_path), model_save_dir=str(work / "sft"),
                    context_length=CONTEXT, max_len=CONTEXT, max_prompt_length=24,
                    max_response_length=7, use_amp=False, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    warmup_steps=1, auto_resume=False, learning_rate=0.005,
                    prompt_template=PROMPT, response_template=RESPONSE, **DIMENSIONS)
    trainer = SFTTrainer.from_pretrained(str(pretrained_path), config=cfg)
    if not isinstance(trainer.model.transformer_blocks[0].attention, MultiHeadLatentAttention):
        raise RuntimeError("Expected MLA backend from the pretraining checkpoint")

    def score(current, loader):
        current.model.eval()
        losses, correct, tokens = [], 0, 0
        with torch.inference_mode():
            for batch in loader:
                loss, metrics = current.compute_loss(batch)
                n = metrics["num_tokens"]
                losses.append(float(loss) * n)
                correct += round(metrics["accuracy"] * n)
                tokens += n
        nll = math.fsum(losses) / tokens
        return dict(response_tokens=tokens, correct_tokens=correct,
                    mean_nll=nll, perplexity=math.exp(nll), top1_accuracy=correct / tokens)

    trainer.train_loader, held_out = trainer.create_dataloaders(str(train), str(evaluation), num_workers=0)
    initial = score(trainer, held_out)
    probe = next(iter(held_out))["input_ids"]
    with torch.inference_mode():
        logits_before_train = trainer.model(probe).clone()
    torch.manual_seed(SEED + 1)
    trainer.fit()
    steps = EPOCHS * len(trainer.train_loader)
    saved = Path(cfg.model_save_dir) / f"checkpoint_epoch_{EPOCHS}_step_{steps}.pt"
    state = torch.load(saved, weights_only=True, map_location="cpu")
    if state["step"] != steps:
        raise RuntimeError("SFT did not complete the expected optimizer updates")
    trainer.model.eval()
    with torch.inference_mode():
        logits_after_train = trainer.model(probe)
    if torch.equal(logits_before_train, logits_after_train):
        raise RuntimeError("SFT did not change held-out logits")

    # Publish an inference checkpoint, not environment-specific optimizer/RNG/absolute paths.
    checkpoint = out / "micro-mla-sft.pt"
    buffer = io.BytesIO()
    arch = dict(vocab_size=len(tok.token_to_id), max_len=CONTEXT, **DIMENSIONS)
    torch.save({"trainer_type": "sft", "steps": steps, "model_config": arch,
                "mla_config": state["mla_config"],
                "model_state_dict": OrderedDict(sorted(state["model_state_dict"].items()))}, buffer)
    checkpoint.write_bytes(buffer.getvalue())
    restored = SFTTrainer.from_pretrained(str(checkpoint), config=cfg)
    restored.model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored.model(probe), logits_after_train, rtol=0, atol=0)
    final = score(restored, held_out)

    configuration = dict(architecture=arch, batch_size=BATCH_SIZE, context_length=CONTEXT,
                         max_prompt_length=24, max_response_length=7, prompt_template=PROMPT,
                         response_template=RESPONSE, pretrain_epochs=1, pretrain_learning_rate=0.003,
                         sft_epochs=EPOCHS, sft_learning_rate=0.005, sft_warmup_steps=1,
                         sft_optimizer="AdamW", weight_decay=cfg.weight_decay, sft_seed=SEED + 1,
                         seed=SEED, dtype="float32", intraop_threads=1, interop_threads=1)
    write_json(out / "config.json", configuration)
    files = sorted(out.iterdir())
    write_json(out / "metrics.json", dict(
        schema_version=1, release_version=VERSION, source_commit=source,
        runtime=dict(python=platform.python_version(), torch=str(torch.__version__),
                     os=platform.system(), architecture=platform.machine(), device="cpu"),
        provenance="Repository-authored synthetic fixture; no external/private data. Static fixture alphabet.",
        pretrain_examples=len(PRETRAIN), train_examples=len(TRAIN), held_out_examples=len(EVAL),
        pretrain_optimizer_updates=pretrain_updates, sft_optimizer_updates=steps, model_parameters=sum(p.numel() for p in restored.model.parameters()),
        evaluation_before_sft=initial, evaluation_after_sft=final, reload_logits_exact=True,
        limits="Engineering fixture only, not model quality, throughput or resume-training evidence.",
        sha256={p.name: digest(p) for p in files},
    ))
    (out / "SHA256SUMS").write_text(
        "".join(f"{digest(p)}  {p.name}\n" for p in sorted(out.iterdir())), encoding="ascii"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="New directory for release artifacts")
    args = parser.parse_args()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT, check=False).returncode != 0:
        parser.error("The benchmark requires an unmodified tracked checkout")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="micro-mla-work-") as path:
        where = Path.cwd()
        try:
            os.chdir(path)
            benchmark(output, source, Path(path))
        finally:
            os.chdir(where)


if __name__ == "__main__":
    main()
