#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple training demo for Lingmao Moyun."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.trainer import train_model
from src.logger import get_logger

logger = get_logger("DemoTrain")


def main():
    print("=" * 80)
    print("Lingmao Moyun - Training Demo")
    print("=" * 80)
    
    # Small model config
    train_file = "dataset/sample_train.jsonl"
    test_file = None
    
    # Model params (small for CPU)
    d_model = 128
    nhead = 2
    num_layers = 2
    dim_feedforward = 512
    dropout = 0.1
    
    # Training params
    batch_size = 2
    learning_rate = 1e-4
    epochs = 3
    context_length = 64
    
    print(f"\nModel: {d_model}d, {num_layers}L, {nhead}h")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Context length: {context_length}")
    print(f"Training data: {train_file}")
    print("\n" + "=" * 80 + "\n")
    
    # Train
    try:
        model = train_model(
            train_file=train_file,
            test_file=test_file,
            tokenizer_path="tokenizer.json",
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            context_length=context_length,
            use_amp=False,
            use_compile=False,
            use_gradient_checkpointing=False,
            night_mode=False,
            mode="modern",
            log_stats_interval=10,
        )
        
        print("\n" + "=" * 80)
        print("Training successful!")
        print("=" * 80)
        
        # Try generation
        try:
            print("\nTesting generation...")
            from tokenizer import ClassicalTokenizer
            
            tokenizer = ClassicalTokenizer()
            prompt = "床前明月光"
            print(f"\nPrompt: {prompt}")
            
            prompt_tokens = tokenizer.encode(prompt)
            model.eval()
            
            with torch.no_grad():
                input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
                
                generated = []
                max_steps = 20
                
                print("Generating: ", end="", flush=True)
                
                for i in range(max_steps):
                    outputs, _ = model(input_ids)
                    next_token = outputs[0, -1].argmax(dim=-1).item()
                    
                    if next_token == tokenizer.eos_token:
                        break
                    
                    generated.append(next_token)
                    input_ids = torch.cat(
                        [input_ids, torch.tensor([[next_token]])],
                        dim=1
                    )
                    print(tokenizer.decode([next_token]), end="", flush=True)
                
                full_text = tokenizer.decode(prompt_tokens + generated)
                print(f"\n\nFull output: {full_text}")
                
        except Exception as e:
            logger.warning(f"Generation test skipped: {e}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
