#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a simple usable tokenizer for training."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tokenizer import ClassicalTokenizer


def build_tokenizer_from_data(data_file):
    """Build tokenizer from sample training data."""
    tokenizer = ClassicalTokenizer()

    # Load sample data to extract vocabulary
    texts = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    text = item.get("text", "") or item.get("content", "")
                    if text:
                        texts.append(text)
                except:
                    pass

    print(f"Loaded {len(texts)} text samples")

    # Build character-level vocab
    vocab = {}
    idx = 0

    # Add special tokens first
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<cls>"]
    for tok in special_tokens:
        vocab[tok] = idx
        idx += 1

    # Add characters from training data
    for text in texts:
        for char in text:
            if char not in vocab:
                vocab[char] = idx
                idx += 1

    print(f"Total vocab size: {len(vocab)}")

    # Set tokenizer state
    tokenizer.token_to_id = vocab
    tokenizer.id_to_token = {v: k for k, v in vocab.items()}

    return tokenizer


if __name__ == "__main__":
    tokenizer = build_tokenizer_from_data("dataset/sample_train.jsonl")

    # Save it
    tokenizer.save("tokenizer.json")
    tokenizer.save("dataset/tokenizer.json")

    print("\nTokenizer created and saved to tokenizer.json and dataset/tokenizer.json")

    # Test it
    test_text = "床前明月光"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"\nTest: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
