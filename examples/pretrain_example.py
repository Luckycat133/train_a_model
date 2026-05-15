"""Pretraining Example Script

This script demonstrates how to run pretraining experiments using the
Lingmao Moyun configuration system.
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.validator import load_and_validate_config
from src.trainer import train_model


def run_pretrain(config_path: str, **override_kwargs):
    """Run pretraining experiment with configuration validation.
    
    Args:
        config_path: Path to the configuration file.
        **override_kwargs: Optional parameter overrides.
    
    Returns:
        Trained model instance.
    """
    print("=" * 60)
    print("Lingmao Moyun Pretraining Experiment")
    print("=" * 60)
    
    is_valid, config, warnings = load_and_validate_config(
        config_path,
        experiment_type="pretrain",
        strict=True
    )
    
    if not is_valid:
        print("\nConfiguration validation failed. Errors found:")
        for category, msgs in warnings.items():
            for msg in msgs:
                print(f"  [{category}] {msg}")
        sys.exit(1)
    
    if warnings and any(w for w in warnings.values()):
        print("\nConfiguration warnings:")
        for category, msgs in warnings.items():
            for msg in msgs:
                print(f"  [{category}] {msg}")
    
    print(f"\nExperiment: {config.experiment_name}")
    print(f"Model: {config.model.d_model}d, {config.model.num_layers}L, {config.model.nhead}head")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Context length: {config.training.context_length}")
    
    trainer_kwargs = config.to_trainer_kwargs()
    trainer_kwargs.update(override_kwargs)
    
    model = train_model(**trainer_kwargs)
    
    print("\n" + "=" * 60)
    print("Pretraining completed!")
    print("=" * 60)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Run pretraining experiment")
    parser.add_argument(
        "--config", "-c",
        default="config/pretrain.yaml",
        help="Path to configuration file"
    )
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--no-night-mode", action="store_true", help="Disable night mode")
    
    args = parser.parse_args()
    
    override_kwargs = {}
    if args.epochs:
        override_kwargs["epochs"] = args.epochs
    if args.batch_size:
        override_kwargs["batch_size"] = args.batch_size
    if args.lr:
        override_kwargs["learning_rate"] = args.lr
    if args.resume:
        override_kwargs["resume_from"] = args.resume
    if args.no_night_mode:
        override_kwargs["night_mode"] = False
    
    if args.device != "auto":
        import torch
        override_kwargs["device"] = torch.device(args.device)
    
    run_pretrain(args.config, **override_kwargs)


if __name__ == "__main__":
    main()
