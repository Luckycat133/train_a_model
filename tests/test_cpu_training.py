"""Real CPU training and completed-epoch resume, with no downloaded corpus."""

from pathlib import Path

import pytest
import torch

from src.config import VERSION
from src.model import ModernAttention
from src.trainer import find_latest_checkpoint, train_model


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("use_mla", [False, True], ids=["default", "mla"])
def test_cpu_training_resumes_exactly_at_epoch_boundary(tmp_path, use_mla):
    torch.set_num_threads(1)
    full = tmp_path / "full"
    split = tmp_path / "split"
    common = dict(train_file=str(ROOT / "examples/quick_train.jsonl"),
                  tokenizer_path=str(ROOT / "tokenizer.json"), device=torch.device("cpu"),
                  context_length=32, batch_size=2, d_model=32, num_layers=1, nhead=4,
                  dim_feedforward=64, dropout=0.1, learning_rate=0.01, use_amp=False,
                  use_checkpoint=False, use_gradient_checkpointing=False, use_compile=False,
                  night_mode=False, num_workers=0, checkpoint_every=1, accumulation_steps=3,
                  use_mla=use_mla, mla_latent_dim=8, mla_num_latent_heads=2)
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


@pytest.mark.parametrize("kv_heads", [4, 2])
@pytest.mark.parametrize("window", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_uncached_sdpa_matches_eager_forward_and_gradient(kv_heads, window, masked):
    torch.manual_seed(42)
    options = dict(d_model=32, num_heads=4, num_kv_heads=kv_heads,
                   use_sliding_window=window, window_size=3, attention_dropout=0.0)
    sdpa = ModernAttention(use_flash_attention=True, **options).eval()
    eager = ModernAttention(use_flash_attention=False, **options).eval()
    eager.load_state_dict(sdpa.state_dict())
    x = torch.randn(2, 7, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    mask = None
    if masked:
        mask = torch.ones(2, 7, 7, dtype=torch.bool)
        mask[0, :, 6] = False
        mask[1, 2, :] = False
    a = sdpa(x, attention_mask=mask)[0]
    b = eager(y, attention_mask=mask)[0]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    a.square().sum().backward()
    b.square().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-5)
    if masked:
        assert (a[1, 2] == 0).all()


def test_default_uncached_attention_depends_on_past_but_not_future():
    torch.manual_seed(42)
    a = ModernAttention(d_model=32, num_heads=4, attention_dropout=0.0,
                        use_sliding_window=False, use_flash_attention=True).eval()
    x = torch.randn(1, 7, 32)
    original = a(x)[0]
    modified = x.clone()
    modified[:, 0] = torch.randn(1, 32)
    assert (a(modified)[0][:, 1:] - original[:, 1:]).abs().max() > 1e-5
    modified = x.clone()
    modified[:, -1] = torch.randn(1, 32)
    torch.testing.assert_close(a(modified)[0][:, :-1], original[:, :-1], atol=0, rtol=0)


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_causal_sliding_window_and_boolean_key_mask_keep_only_allowed_past(use_sdpa):
    torch.manual_seed(42)
    a = ModernAttention(d_model=32, num_heads=4, attention_dropout=0.0,
                        use_sliding_window=True, window_size=3, use_flash_attention=use_sdpa).eval()
    x = torch.randn(1, 7, 32)
    mask = torch.ones(1, 7, dtype=torch.bool)
    mask[0, 3] = False
    original = a(x, attention_mask=mask)[0]
    # For query 5 and window 3 the allowed positions are 4 and 5, not 3 or 2.
    ignored = x.clone()
    ignored[:, 2:4] = torch.randn(1, 2, 32)
    torch.testing.assert_close(a(ignored, attention_mask=mask)[0][:, 5], original[:, 5], rtol=0, atol=0)
    allowed = x.clone()
    allowed[:, 4] = torch.randn(1, 32)
    assert (a(allowed, attention_mask=mask)[0][:, 5] - original[:, 5]).abs().max() > 1e-5
