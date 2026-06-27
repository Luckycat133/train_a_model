#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test various generations from trained model."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from train_usable_model import load_model, generate_text


def main():
    print("=" * 80)
    print("测试模型生成效果")
    print("=" * 80)
    
    model_dir = "model_weights/usable_model_20ep"
    model_path = os.path.join(model_dir, "best_model.pt")
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found")
        return
    
    model, tokenizer = load_model(model_path, tokenizer_path)
    print(f"Model loaded, vocab size: {len(tokenizer.token_to_id)}")
    print()
    
    test_prompts = [
        "床前明月光，",
        "白日依山尽，",
        "春眠不觉晓，",
        "红豆生南国，",
        "子曰：学而时习之，",
        "道可道，",
        "北冥有鱼，",
    ]
    
    temperatures = [0.5, 0.7, 1.0]
    
    for temp in temperatures:
        print("\n" + "=" * 80)
        print(f"Temperature: {temp}")
        print("=" * 80)
        
        for prompt in test_prompts:
            try:
                output = generate_text(
                    model,
                    tokenizer,
                    prompt,
                    max_length=80,
                    temperature=temp
                )
                print(f"\nPrompt: {prompt}")
                print(f"Output: {output}")
            except Exception as e:
                print(f"Error for '{prompt}': {e}")
    
    print("\n" + "=" * 80)
    print("模型已保存到:")
    print(f"  - {model_dir}/best_model.pt")
    print(f"  - {model_dir}/final_model.pt")
    print(f"  - {model_dir}/tokenizer.json")
    print(f"  - {model_dir}/checkpoint_epoch_*.pt")
    print()
    print("运行交互式演示: python generate_demo.py")
    print("训练更多epoch: python train_usable_model.py --train --epochs 50")


if __name__ == "__main__":
    main()
