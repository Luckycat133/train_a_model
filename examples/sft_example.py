"""Supervised Fine-Tuning (SFT) Example Script

This script demonstrates how to run SFT experiments using the
Lingmao Moyun configuration system.
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.validator import load_and_validate_config
from src.trainer import train_model


def run_sft(
    config_path: str,
    pretrained_path: str = None,
    **override_kwargs
):
    """Run SFT experiment with configuration validation.
    
    Args:
        config_path: Path to the SFT configuration file.
        pretrained_path: Optional path to pretrained model weights.
        **override_kwargs: Optional parameter overrides.
    
    Returns:
        Fine-tuned model instance.
    """
    print("=" * 60)
    print("Lingmao Moyun Supervised Fine-Tuning (SFT) Experiment")
    print("=" * 60)
    
    is_valid, config, warnings = load_and_validate_config(
        config_path,
        experiment_type="sft",
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
    print(f"Label smoothing: {config.training.label_smoothing}")
    
    if pretrained_path:
        print(f"\nLoading pretrained weights from: {pretrained_path}")
        override_kwargs["resume_from"] = pretrained_path
    
    trainer_kwargs = config.to_trainer_kwargs()
    trainer_kwargs.update(override_kwargs)
    
    model = train_model(**trainer_kwargs)
    
    print("\n" + "=" * 60)
    print("SFT completed!")
    print("=" * 60)
    
    return model


def prepare_sft_dataset(input_file: str, output_file: str):
    """Prepare SFT dataset from raw data.
    
    This is a placeholder function showing how to structure data for SFT.
    In practice, you would implement the actual data conversion logic here.
    
    Args:
        input_file: Path to raw training data.
        output_file: Path to save the formatted SFT dataset.
    """
    import json
    
    print(f"\nPreparing SFT dataset...")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    
    print("\nExpected JSONL format:")
    print('  {"instruction": "...", "input": "...", "output": "..."}')
    print("\nNote: This is a placeholder. Implement actual data conversion as needed.")


def main():
    parser = argparse.ArgumentParser(description="Run SFT experiment")
    parser.add_argument(
        "--config", "-c",
        default="config/sft.yaml",
        help="Path to SFT configuration file"
    )
    parser.add_argument(
        "--pretrained", "-p",
        default=None,
        help="Path to pretrained model weights"
    )
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--prepare-data", type=str, help="Prepare SFT dataset")
    
    args = parser.parse_args()
    
    if args.prepare_data:
        prepare_sft_dataset(args.prepare_data, "dataset/sft_train.jsonl")
        return
    
    override_kwargs = {}
    if args.epochs:
        override_kwargs["epochs"] = args.epochs
    if args.batch_size:
        override_kwargs["batch_size"] = args.batch_size
    if args.lr:
        override_kwargs["learning_rate"] = args.lr
    
    if args.device != "auto":
        import torch
        override_kwargs["device"] = torch.device(args.device)
    
    run_sft(args.config, args.pretrained, **override_kwargs)


if __name__ == "__main__":
    main()
