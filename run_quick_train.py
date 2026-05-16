#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick training script for Lingmao Moyun demo."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.config.validator import load_and_validate_config
from src.trainer import train_model
from src.logger import get_logger

logger = get_logger("QuickTrain")


def main():
    print("=" * 80)
    print("Lingmao Moyun - Quick Training Demo")
    print("=" * 80)
    
    # Load config
    config_path = "config/quick_test.yaml"
    logger.info(f"Loading config from: {config_path}")
    
    is_valid, config, warnings = load_and_validate_config(
        config_path,
        experiment_type="pretrain",
        strict=False
    )
    
    if not is_valid:
        logger.error("Config validation failed!")
        for cat, msgs in warnings.items():
            for msg in msgs:
                logger.error(f"[{cat}] {msg}")
        return 1
    
    # Print config summary
    print(f"\nModel: {config.model.d_model}d, {config.model.num_layers}L, {config.model.nhead}h")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Context length: {config.training.context_length}")
    print(f"Data: {config.dataset.train_file}")
    print("\n" + "=" * 80 + "\n")
    
    # Convert to trainer kwargs
    trainer_kwargs = config.to_trainer_kwargs()
    
    # Quick overrides for CPU
    trainer_kwargs["device"] = torch.device("cpu")
    trainer_kwargs["night_mode"] = False
    
    # Run training
    logger.info("Starting training...")
    model = train_model(**trainer_kwargs)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    # Try generating something
    try:
        print("\nTesting generation...")
        from src.model import SimpleTransformer
        from tokenizer import ClassicalTokenizer
        
        tokenizer = ClassicalTokenizer()
        prompt = "床前明月光"
        
        # Encode prompt
        prompt_tokens = tokenizer.encode(prompt)
        
        # Generate
        print(f"\nInput: {prompt}")
        print("Generating...", end="", flush=True)
        
        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
            
            # Simple greedy generation
            max_len = 30
            for i in range(max_len):
                outputs, _ = model(input_ids)
                next_token = outputs[0, -1].argmax()
                if next_token.item() == tokenizer.eos_token:
                    break
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                print(".", end="", flush=True)
            
            # Decode
            generated = tokenizer.decode(input_ids[0].tolist())
            print(f"\nOutput: {generated}")
            
    except Exception as e:
        logger.error(f"Generation test failed: {e}")
        import traceback
        traceback.print_exc()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
