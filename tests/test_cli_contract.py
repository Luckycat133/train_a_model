import argparse
import subprocess
import sys

from src.run import _apply_quick_preset


def test_help_is_available_without_training_dependencies():
    result = subprocess.run(
        [sys.executable, "-m", "src.run", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--quick" in result.stdout
    assert "--train" in result.stdout
    assert "--save_dir" in result.stdout


def test_quick_preset_is_small_and_deterministic():
    args = argparse.Namespace(
        quick=True,
        train_file="dataset/train_data_train.jsonl",
        model_save_dir="model_weights",
        context_length=512,
        d_model=768,
        nhead=12,
        num_layers=12,
        dim_feedforward=3072,
        batch_size=8,
        epochs=10,
        accumulation_steps=4,
        checkpoint_every=1,
        use_amp=True,
        use_checkpoint=True,
        use_compile=True,
        auto_resume=True,
        night_mode=True,
        window_size=512,
    )

    _apply_quick_preset(args)

    assert args.train_file == "examples/quick_train.jsonl"
    assert args.model_save_dir == "quick_runs"
    assert (args.d_model, args.num_layers, args.epochs) == (32, 1, 1)
    assert args.use_amp is False
    assert args.auto_resume is False
