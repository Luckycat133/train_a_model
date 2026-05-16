#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple working training demo for Lingmao Moyun - using character-level tokenization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from src.model import SimpleTransformer
from src.logger import get_logger
from src.dataset import LMDataset

logger = get_logger("SimpleTrain")


def main():
    print("=" * 80)
    print("Lingmao Moyun - Simple Training Demo (Character-Level)")
    print("=" * 80)

    train_file = "dataset/sample_train.jsonl"
    context_length = 64
    batch_size = 2
    d_model = 128
    nhead = 2
    num_layers = 2
    dim_feedforward = 512
    vocab_size = 30000
    max_len = 256

    print(f"\nModel: {d_model}d, {num_layers}L, {nhead}h")
    print(f"Batch size: {batch_size}")
    print(f"Context length: {context_length}")
    print(f"Training data: {train_file}")
    print("\n" + "=" * 80 + "\n")

    # Load dataset (use tokenizer=None for character-level fallback)
    print("Loading dataset (character-level)...")
    dataset = LMDataset(train_file, context_length=context_length, tokenizer=None)
    print(f"Loaded {len(dataset)} samples")

    # Dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    print("\nCreating model...")
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_len=max_len,
        mode="modern",
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop (3 epochs)
    print("\nStarting training...")
    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")
        total_loss = 0.0
        num_batches = 0

        model.train()
        for batch in dataloader:
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]

            optimizer.zero_grad()
            outputs, _ = model(input_ids)
            loss = criterion(outputs.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 10 == 0:
                print(f"  Batch {num_batches}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} complete, Avg Loss: {avg_loss:.4f}")

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    # Test generation
    print("\nTesting generation...")
    model.eval()

    test_text = "床前"
    print(f"Start: {test_text}")

    # Character-level encoding
    input_ids = torch.tensor([[ord(c) % vocab_size for c in test_text]], dtype=torch.long)

    generated = list(test_text)
    max_steps = 20

    with torch.no_grad():
        for i in range(max_steps):
            outputs, _ = model(input_ids)
            next_token_logits = outputs[0, -1]
            next_token_id = torch.argmax(next_token_logits).item()

            # Convert back to character (approximate)
            next_char = chr(next_token_id % 0x10FFFF)
            generated.append(next_char)

            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token_id]], dtype=torch.long)],
                dim=1
            )

            print(f"Step {i+1}: {''.join(generated)}")

    print(f"\nFinal output: {''.join(generated)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
