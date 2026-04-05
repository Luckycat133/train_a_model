"""CLI entry point for the Lingmao Moyun training system.

Invoke via ``python -m src.run`` or ``python train_model.py``.
"""

import argparse
import shutil
from pathlib import Path

import yaml

from src.config import (
    DEFAULT_ACCUMULATION_STEPS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_D_MODEL,
    DEFAULT_DIM_FEEDFORWARD,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_NHEAD,
    DEFAULT_NUM_LAYERS,
    DEFAULT_TOKENIZER_PATH,
    DEFAULT_WEIGHT_DECAY,
    MODEL_SAVE_DIR,
    USE_CHECKPOINT,
    USE_COMPILE,
    USE_WEIGHT_TYING,
    USE_SLIDING_WINDOW,
    SWA_WINDOW_SIZE,
    VERSION,
)
from src.logger import get_logger
from src.trainer import train_model

logger = get_logger("LingmaoMoyun")


def _clean_before_run(model_save_dir: str, clean_plots: bool = True) -> None:
    """Delete old log directories and optionally plot files before training."""
    log_dir = Path("logs") / "train_model"
    if log_dir.exists():
        shutil.rmtree(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cleaned log directory: {log_dir}")

    if clean_plots:
        plot_dir = Path(model_save_dir)
        removed = 0
        for pf in plot_dir.glob("training_stats_*.png"):
            try:
                pf.unlink()
                removed += 1
            except OSError:
                pass
        if removed:
            logger.info(f"Removed {removed} old plot file(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lingmao Moyun Language Model Training")

    # Data
    parser.add_argument("--train_file", type=str, default="dataset/train_data_train.jsonl")
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--context_length", type=int, default=DEFAULT_CONTEXT_LENGTH)

    # Model
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--nhead", type=int, default=DEFAULT_NHEAD)
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--dim_feedforward", type=int, default=DEFAULT_DIM_FEEDFORWARD)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)

    # Training
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--checkpoint_every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--model_save_dir", type=str, default=DEFAULT_MODEL_SAVE_DIR)
    parser.add_argument("--accumulation_steps", type=int, default=DEFAULT_ACCUMULATION_STEPS)
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--use_checkpoint", dest="use_checkpoint", action="store_true", default=USE_CHECKPOINT)
    parser.add_argument("--no_use_checkpoint", dest="use_checkpoint", action="store_false")
    parser.add_argument("--use_compile", dest="use_compile", action="store_true", default=USE_COMPILE)
    parser.add_argument("--no_use_compile", dest="use_compile", action="store_false")
    parser.add_argument("--use_weight_tying", dest="use_weight_tying", action="store_true", default=USE_WEIGHT_TYING)
    parser.add_argument("--no_use_weight_tying", dest="use_weight_tying", action="store_false")
    parser.add_argument("--use_sliding_window", dest="use_sliding_window", action="store_true", default=USE_SLIDING_WINDOW)
    parser.add_argument("--no_use_sliding_window", dest="use_sliding_window", action="store_false")
    parser.add_argument("--window_size", type=int, default=SWA_WINDOW_SIZE)

    # Mixed precision / device
    parser.add_argument("--use_amp", dest="use_amp", action="store_true", default=True)
    parser.add_argument("--no_use_amp", dest="use_amp", action="store_false")
    parser.add_argument("--no_cuda", action="store_true")

    # Resume
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--no_auto_resume", dest="auto_resume", action="store_false")

    # Behaviour
    parser.add_argument("--night_mode", dest="night_mode", action="store_true", default=True)
    parser.add_argument("--no_night_mode", dest="night_mode", action="store_false")
    parser.add_argument("--clean_before_run", action="store_true", default=False,
                        help="Delete old logs/plots before starting")

    # Config file - parse first to get config path
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")

    args, extra = parser.parse_known_args()
    config_overrides = {}
    if args.config:
        with open(args.config) as f:
            config_overrides = yaml.safe_load(f) or {}

    # Build arg list from config dict, then re-parse
    config_args = []
    for key, value in config_overrides.items():
        if value is not None:
            config_args.extend(["--" + key, str(value)])
    config_args.extend(extra)

    args = parser.parse_args(config_args)

    if args.no_cuda:
        import torch
        device = torch.device("cpu")
    else:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if args.clean_before_run:
        _clean_before_run(args.model_save_dir)

    train_model(
        train_file=args.train_file,
        test_file=args.test_file,
        model_save_dir=args.model_save_dir,
        tokenizer_path=args.tokenizer_path,
        context_length=args.context_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        use_amp=args.use_amp,
        checkpoint_every=args.checkpoint_every,
        accumulation_steps=args.accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        device=device,
        resume_from=args.resume_from,
        auto_resume=args.auto_resume,
        night_mode=args.night_mode,
        use_checkpoint=args.use_checkpoint,
        use_compile=args.use_compile,
        use_weight_tying=args.use_weight_tying,
        use_sliding_window=args.use_sliding_window,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main()
