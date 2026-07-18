#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interactive demo for classical Chinese text generation."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from train_usable_model import load_model, generate_text, CharTokenizer


def main():
    print("=" * 80)
    print("灵猫墨韵 - 古典中文文本生成演示")
    print("=" * 80)
    
    # Model paths
    model_dir = "model_weights/usable_model_20ep"
    model_path = os.path.join(model_dir, "best_model.pt")
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first.")
        sys.exit(1)
    
    print(f"Loading model from {model_path}")
    model, tokenizer = load_model(model_path, tokenizer_path)
    print("Model loaded!")
    print(f"Vocab size: {len(tokenizer.token_to_id)}")
    print()
    
    print("Try these examples:")
    examples = [
        "床前明月光，",
        "白日依山尽，",
        "春眠不觉晓，",
        "子曰：学而时习之，",
    ]
    for i, ex in enumerate(examples, 1):
        print(f"{i}. {ex}")
    
    print()
    print("=" * 80)
    print("输入提示 (或 'q' 退出):")
    print("=" * 80)
    
    while True:
        try:
            prompt = input("> ").strip()
            if prompt.lower() in ("q", "quit", "exit"):
                break
            if not prompt:
                continue
            
            # Generate
            output = generate_text(
                model,
                tokenizer,
                prompt,
                max_length=120,
                temperature=0.7
            )
            print()
            print(f"Output: {output}")
            print()
            
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"Error: {e}")
            print()


if __name__ == "__main__":
    main()
