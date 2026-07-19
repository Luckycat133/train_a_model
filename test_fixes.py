#!/usr/bin/env python3
"""Test script to verify our bug fixes work correctly."""

import sys
import traceback

def test_model_import():
    """Test that model module can be imported without errors."""
    print("Testing model import...", end=" ")
    from src.model import SimpleTransformer, MoELayer, ModernAttention
    print("✓ Success")

def test_model_forward_backward_compat():
    """Test that SimpleTransformer.forward has backward compatibility."""
    print("\nTesting forward method backward compatibility...", end=" ")
    import torch
    import torch.nn as nn

    from src.model import SimpleTransformer

    vocab_size = 1000
    d_model = 128
    nhead = 4
    num_layers = 2
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )

    # Test default behavior (returns single tensor)
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    outputs = model(input_ids)
    assert isinstance(outputs, torch.Tensor), "Expected single tensor output"
    assert outputs.shape == (batch_size, seq_len, vocab_size), \
        f"Expected shape {(batch_size, seq_len, vocab_size)}, got {outputs.shape}"
    print("✓ Single output works")

    # Test with use_cache=True
    outputs, cache = model(input_ids, use_cache=True)
    assert isinstance(outputs, torch.Tensor), "Expected tensor output"
    assert cache is not None, "Expected cache to be returned"
    print("✓ use_cache=True works")

    # Test with return_aux_loss=True
    outputs, cache, aux_loss = model(input_ids, return_aux_loss=True)
    assert isinstance(outputs, torch.Tensor), "Expected tensor output"
    assert isinstance(aux_loss, torch.Tensor), "Expected aux_loss to be tensor"
    print("✓ return_aux_loss=True works")

def test_moe_layer():
    """Test that MoELayer works correctly."""
    print("\nTesting MoELayer...", end=" ")
    import torch
    from src.model import MoELayer

    d_model = 128
    dim_feedforward = 512
    num_experts = 4
    top_k = 2

    moe = MoELayer(
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        num_experts=num_experts,
        top_k=top_k
    )

    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, d_model)

    output, aux_loss = moe(x)

    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert aux_loss.shape == () or len(aux_loss.shape) == 0, \
        f"Expected scalar loss, got shape {aux_loss.shape}"

    print("✓ Success")

def test_trainer_import():
    """Test that trainer module can be imported without errors."""
    print("\nTesting trainer import...", end=" ")
    from src.trainer import train_model, evaluate_model, save_checkpoint
    print("✓ Success")

def main():
    """Run all tests and print summary."""
    print("=" * 50)
    print("Testing Bug Fixes for Lingmao Moyun")
    print("=" * 50)

    # Run tests
    test_model_import()
    test_model_forward_backward_compat()
    test_moe_layer()
    test_trainer_import()

    # Summary
    print("\n" + "=" * 50)
    print("✓ All tests passed! Bug fixes are working correctly.")
    print("=" * 50)

    return 0

if __name__ == "__main__":
    sys.exit(main())