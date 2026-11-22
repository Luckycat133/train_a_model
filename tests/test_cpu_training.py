"""Real CPU training and completed-epoch resume, with no downloaded corpus."""

from pathlib import Path

import pytest
import torch

from src.config import VERSION
from src.trainer import find_latest_checkpoint, train_model


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_training_resumes_exactly_at_epoch_boundary(tmp_path):
    torch.set_num_threads(1)
    full = tmp_path / "full"
    split = tmp_path / "split"
    common = dict(train_file=str(ROOT / "examples/quick_train.jsonl"),
                  tokenizer_path=str(ROOT / "tokenizer.json"), device=torch.device("cpu"),
                  context_length=32, batch_size=2, d_model=32, num_layers=1, nhead=4,
                  dim_feedforward=64, dropout=0.1, learning_rate=0.01, use_amp=False,
                  use_checkpoint=False, use_gradient_checkpointing=False, use_compile=False,
                  night_mode=False, num_workers=0, checkpoint_every=1, accumulation_steps=3)
    continuous = train_model(model_save_dir=str(full), epochs=2, seed=42, auto_resume=False, **common)
    train_model(model_save_dir=str(split), epochs=1, seed=42, auto_resume=False, **common)
    first = split / "model_epoch_1.pt"
    assert find_latest_checkpoint(str(split)) == str(first)
    first_bytes = first.read_bytes()
    first_state = torch.load(first, weights_only=True)
    resumed = train_model(model_save_dir=str(split), epochs=2, seed=9001, auto_resume=True, **common)
    assert first.read_bytes() == first_bytes
    a = torch.load(full / f"final_model_v{VERSION}.pt", weights_only=True)
    b = torch.load(split / f"final_model_v{VERSION}.pt", weights_only=True)
    assert a["epoch"] == b["epoch"] == 2
    for name in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                 "scaler_state_dict", "cpu_rng_state", "resume_fingerprint"):
        assert_equal_tree(a[name], b[name])
    for key in ("losses", "learning_rates", "steps", "tokens", "optimizer_updates"):
        assert a["stats"][key] == b["stats"][key]
    assert a["stats"]["optimizer_updates"][-1] == 6
    assert a["stats"]["tokens"][-1] == 896
    assert all(torch.isfinite(value).all() for value in a["model_state_dict"].values())
    assert any(not torch.equal(value, first_state["model_state_dict"][key])
               for key, value in a["model_state_dict"].items())
    l = torch.arange(32).view(1, -1)
    with torch.no_grad():
        assert torch.equal(continuous.eval()(l), resumed.eval()(l))
    assert not (split / "best_model.pt").exists()


def assert_equal_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            assert_equal_tree(left, right)
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    else:
        assert a == b


def test_invalid_resume_checkpoint_is_not_a_fresh_training_run(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"epoch": 1}, path)
    with pytest.raises(ValueError, match="not a compatible"):
        train_model(
            train_file=str(ROOT / "examples/quick_train.jsonl"),
            tokenizer_path=str(ROOT / "tokenizer.json"), device=torch.device("cpu"),
            model_save_dir=str(tmp_path), resume_from=str(path), context_length=32,
            batch_size=1, d_model=32, num_layers=1, nhead=4, dim_feedforward=64,
            use_checkpoint=False, use_gradient_checkpointing=False, use_compile=False,
            night_mode=False, epochs=2,
        )
    assert not (tmp_path / "model_epoch_1.pt").exists()
