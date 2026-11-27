"""Bounded offline SFT prototype gate; not model quality or Accelerate support."""
import json
from pathlib import Path

import pytest
import torch

from src.model import ModernAttention, MultiHeadLatentAttention
from src.trainer import train_model
from src.training.sft_trainer import InstructionResponseDataset, SFTConfig, SFTTrainer
from tokenizer import ClassicalTokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizer.json"


def fixture(tmp_path):
    path = tmp_path / "sft.jsonl"
    rows = [
        {"prompt": "天地", "completion": "玄黃"},
        {"instruction": "問月何時明", "output": "秋夜"},
        {"messages": [{"role": "system", "content": "用文言"},
                      {"role": "user", "content": "日落"},
                      {"role": "assistant", "content": "日暮"},
                      {"role": "user", "content": "其後如何"},
                      {"role": "assistant", "content": "月升長空"}]},
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("use_mla", [False, True], ids=["default", "mla"])
def test_sft_cpu_finetunes_from_real_checkpoint_and_reloads(tmp_path, use_mla):
    torch.set_num_threads(1)
    weights = tmp_path / "pretrain"
    train_model(train_file=str(ROOT / "examples/quick_train.jsonl"),
                tokenizer_path=str(TOKENIZER), device=torch.device("cpu"),
                model_save_dir=str(weights), context_length=32, batch_size=2,
                d_model=32, nhead=4, num_layers=1, dim_feedforward=64, epochs=1,
                auto_resume=False, night_mode=False, seed=42, use_amp=False,
                use_checkpoint=False, use_gradient_checkpointing=False, use_compile=False,
                use_mla=use_mla, mla_latent_dim=8, mla_num_latent_heads=2)
    pretrained_path = weights / "model_epoch_1.pt"
    config = SFTConfig(tokenizer_path=str(TOKENIZER), model_save_dir=str(tmp_path / "sft"),
                       context_length=32, max_prompt_length=24, max_response_length=7,
                       d_model=32, nhead=4, num_layers=1, dim_feedforward=64,
                       use_amp=False, epochs=1, batch_size=2, auto_resume=False,
                       warmup_steps=1, learning_rate=0.01, log_interval=1)
    trainer = SFTTrainer.from_pretrained(str(pretrained_path), config=config)
    initial = torch.load(pretrained_path, weights_only=True)["model_state_dict"]
    assert all(torch.equal(value, initial[key]) for key, value in trainer.model.state_dict().items())
    data_path = fixture(tmp_path)
    dataset = InstructionResponseDataset(data_path, trainer.tokenizer, max_length=32,
                                         max_prompt_length=24, max_response_length=7)
    assert "月升長空" not in dataset.samples[2]["prompt"]
    assert "日暮" in dataset.samples[2]["prompt"]
    for i in range(3):
        item = dataset[i]
        p = item["prompt_length"]
        assert (item["labels"][:p] == -100).all()
        assert item["labels"][-1].item() == dataset.eos_id
    for size in (1, 2, 3):
        loader, _ = trainer.create_dataloaders(str(data_path), batch_size=size, num_workers=0)
        batch = next(iter(loader))
        assert batch["input_ids"].size(0) == size
        assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
        loss, metrics = trainer.compute_loss(batch)
        assert torch.isfinite(loss)
        assert metrics["num_tokens"] == (batch["labels"][:, 1:] != -100).sum()
    unpadded = dataset[2]["input_ids"].unsqueeze(0)
    with torch.no_grad():
        trainer.model.eval()
        modified = unpadded.clone()
        modified[0, -1] = (modified[0, -1] + 1) % len(trainer.tokenizer.token_to_id)
        original_logits = trainer.model(unpadded)
        changed_logits = trainer.model(modified)
        torch.testing.assert_close(original_logits[:, :-1], changed_logits[:, :-1], atol=0, rtol=0)

    trainer.train_loader, _ = trainer.create_dataloaders(str(data_path), num_workers=0)
    trainer.fit()
    output = tmp_path / "sft" / "checkpoint_epoch_1_step_2.pt"
    state = torch.load(output, weights_only=True)
    assert state["trainer_type"] == "sft"
    assert state["step"] == 2
    assert any(not torch.equal(value, initial[key]) for key, value in state["model_state_dict"].items())
    clone = SFTTrainer.from_pretrained(str(output), config=config)
    expected_backend = MultiHeadLatentAttention if use_mla else ModernAttention
    assert isinstance(clone.model.transformer_blocks[0].attention, expected_backend)
    assert isinstance(trainer.model.transformer_blocks[0].attention, expected_backend)
    if use_mla:
        assert clone.model.mla_config == {"use_mla": True, "mla_latent_dim": 8, "mla_num_latent_heads": 2}
    assert all(torch.equal(value, state["model_state_dict"][key]) for key, value in clone.model.state_dict().items())
    with torch.no_grad():
        torch.testing.assert_close(trainer.model.eval()(unpadded), clone.model.eval()(unpadded), rtol=0, atol=0)

    # Exercise actual generation. The hook records the first prompt without changing the model.
    observed_ids = []
    hook = clone.model.register_forward_pre_hook(lambda _model, args: observed_ids.append(args[0].clone()))
    try:
        assert isinstance(clone.generate_response("問月何時明", max_new_tokens=2), str)
    finally:
        hook.remove()
    item = dataset[1]
    assert torch.equal(observed_ids[0][0], item["input_ids"][:item["prompt_length"]])
    assert not clone.model.training


def test_sft_rejects_empty_answer_before_an_update(tmp_path):
    tok = ClassicalTokenizer()
    tok.load(str(TOKENIZER))
    bad = tmp_path / "empty.jsonl"
    bad.write_text('{"prompt":"question","completion":""}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty response"):
        InstructionResponseDataset(bad, tok)[0]


def test_sft_does_not_train_from_implicit_random_initialization(tmp_path):
    trainer = SFTTrainer(config=SFTConfig(model_save_dir=str(tmp_path)))
    with pytest.raises(ValueError, match="explicitly loaded pretrained model"):
        trainer.fit()
