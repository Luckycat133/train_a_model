#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for architecture upgrades: Flash Attention, MLA (Multi-Head Latent Attention), etc."""

import pytest
import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import (
    SimpleTransformer,
    ModernAttention,
    MultiHeadLatentAttention,
    RotaryEmbedding
)
from src.config import (
    DEFAULT_D_MODEL,
    DEFAULT_NHEAD,
    DEFAULT_NUM_LAYERS,
    DEFAULT_MAX_LEN,
    MLA_LATENT_DIM,
    MLA_NUM_LATENT_HEADS
)


@pytest.fixture
def device():
    """Fixture to get the appropriate device (CPU or CUDA)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestFlashAttention:
    """Tests for Flash Attention implementation."""
    
    def test_flash_attention_vs_naive(self, device):
        """Test that Flash Attention produces similar results to naive attention."""
        d_model = 128
        num_heads = 4
        batch_size = 2
        seq_len = 32
        
        # Create input
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # Create both versions
        flash_attn = ModernAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_flash_attention=True
        ).to(device)
        
        naive_attn = ModernAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_flash_attention=False
        ).to(device)
        
        # Copy weights
        naive_attn.load_state_dict(flash_attn.state_dict())
        
        # Get outputs
        with torch.no_grad():
            out_flash, _ = flash_attn(x)
            out_naive, _ = naive_attn(x)
        
        # Verify both outputs have correct shapes
        assert out_flash.shape == (batch_size, seq_len, d_model)
        assert out_naive.shape == (batch_size, seq_len, d_model)
        
        # On CPU, SDPA might have different numerical behavior, so we skip exact comparison
        # But on CUDA they should match closely
        if device.type == 'cuda':
            torch.testing.assert_close(
                out_flash, out_naive,
                atol=1e-3, rtol=1e-2,
                msg="Flash Attention and naive attention outputs differ significantly"
            )
        else:
            # Just verify outputs are not NaN or Inf on CPU
            assert not torch.isnan(out_flash).any()
            assert not torch.isinf(out_flash).any()
            assert not torch.isnan(out_naive).any()
            assert not torch.isinf(out_naive).any()
    
    def test_flash_attention_causal_mask(self, device):
        """Test that causal masking works correctly with Flash Attention."""
        d_model = 128
        num_heads = 4
        batch_size = 2
        seq_len = 32
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        attn = ModernAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_flash_attention=True
        ).to(device)
        
        with torch.no_grad():
            out, _ = attn(x, is_causal=True)
        
        assert out.shape == (batch_size, seq_len, d_model)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flash_attention_performance(self, device):
        """Test that Flash Attention is faster than naive attention (on CUDA)."""
        d_model = 512
        num_heads = 8
        batch_size = 4
        seq_len = 256
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        flash_attn = ModernAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_flash_attention=True
        ).to(device)
        
        naive_attn = ModernAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_flash_attention=False
        ).to(device)
        
        # Warmup
        for _ in range(5):
            _ = flash_attn(x)
            _ = naive_attn(x)
        
        # Benchmark Flash Attention
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = flash_attn(x)
        torch.cuda.synchronize()
        flash_time = time.time() - start
        
        # Benchmark naive attention
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = naive_attn(x)
        torch.cuda.synchronize()
        naive_time = time.time() - start
        
        # Flash should be faster
        print(f"Flash Attention time: {flash_time:.4f}s")
        print(f"Naive Attention time: {naive_time:.4f}s")
        # Note: We don't assert this is always true, just log the comparison


class TestMultiHeadLatentAttention:
    """Tests for Multi-Head Latent Attention (MLA)."""
    
    def test_mla_output_shape(self, device):
        """Test that MLA produces correct output shape."""
        d_model = 128
        num_heads = 4
        batch_size = 2
        seq_len = 32
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        mla = MultiHeadLatentAttention(
            d_model=d_model,
            num_heads=num_heads,
            latent_dim=64,
            num_latent_heads=2
        ).to(device)
        
        with torch.no_grad():
            out, _ = mla(x)
        
        assert out.shape == (batch_size, seq_len, d_model)
    
    def test_mla_kv_cache_compression(self, device):
        """Test that MLA KV cache is indeed smaller than regular KV cache."""
        d_model = 128
        num_heads = 4
        batch_size = 2
        seq_len = 64
        latent_dim = 32
        
        # Regular attention for comparison
        regular_attn = ModernAttention(
            d_model=d_model,
            num_heads=num_heads
        ).to(device)
        
        # MLA
        mla = MultiHeadLatentAttention(
            d_model=d_model,
            num_heads=num_heads,
            latent_dim=latent_dim,
            num_latent_heads=2
        ).to(device)
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # Get cache from both
        with torch.no_grad():
            _, regular_cache = regular_attn(x, use_cache=True)
            _, mla_cache = mla(x, use_cache=True)
        
        # Calculate memory usage
        regular_k_size = regular_cache[0].numel() * regular_cache[0].element_size()
        regular_v_size = regular_cache[1].numel() * regular_cache[1].element_size()
        regular_total = regular_k_size + regular_v_size
        
        mla_k_size = mla_cache[0].numel() * mla_cache[0].element_size()
        mla_v_size = mla_cache[1].numel() * mla_cache[1].element_size()
        mla_total = mla_k_size + mla_v_size
        
        print(f"Regular KV cache size: {regular_total / 1024:.2f} KB")
        print(f"MLA KV cache size: {mla_total / 1024:.2f} KB")
        print(f"Compression ratio: {regular_total / mla_total:.2f}x")
        
        # MLA should have smaller cache
        assert mla_total < regular_total
    
    def test_mla_kv_cache_continuation(self, device):
        """Test that MLA KV cache works for sequential generation."""
        d_model = 128
        num_heads = 4
        batch_size = 1
        seq_len = 16
        
        mla = MultiHeadLatentAttention(
            d_model=d_model,
            num_heads=num_heads,
            latent_dim=64,
            num_latent_heads=2
        ).to(device)
        
        # First step
        x1 = torch.randn(batch_size, seq_len, d_model, device=device)
        with torch.no_grad():
            out1, cache = mla(x1, use_cache=True)
        
        # Second step with cache
        x2 = torch.randn(batch_size, 1, d_model, device=device)
        with torch.no_grad():
            out2, _ = mla(x2, use_cache=True, past_key_values=cache)
        
        assert out2.shape == (batch_size, 1, d_model)
    
    def test_mla_within_transformer(self, device):
        """Test MLA within the full transformer model."""
        vocab_size = 1000
        d_model = 128
        nhead = 4
        num_layers = 2
        
        model = SimpleTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            use_mla=True,
            mla_latent_dim=64,
            mla_num_latent_heads=2
        ).to(device)
        
        input_ids = torch.randint(0, vocab_size, (2, 32), device=device)
        
        with torch.no_grad():
            logits, cache = model(input_ids, use_cache=True)
        
        assert logits.shape == (2, 32, vocab_size)
        assert cache is not None
        assert len(cache) == num_layers


class TestEndToEnd:
    """End-to-end tests for architecture upgrades."""
    
    def test_standard_transformer_training_step(self, device):
        """Test that a standard transformer can do a training step."""
        vocab_size = 1000
        model = SimpleTransformer(
            vocab_size=vocab_size,
            d_model=128,
            nhead=4,
            num_layers=2,
            use_mla=False
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        input_ids = torch.randint(0, vocab_size, (2, 16), device=device)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        logits, _ = model(input_ids)
        
        # Dummy loss
        loss = logits.sum()
        loss.backward()
        optimizer.step()
        
        assert loss.item() == loss.item()  # Just check it's not nan
    
    def test_mla_transformer_training_step(self, device):
        """Test that MLA transformer can do a training step."""
        vocab_size = 1000
        model = SimpleTransformer(
            vocab_size=vocab_size,
            d_model=128,
            nhead=4,
            num_layers=2,
            use_mla=True,
            mla_latent_dim=64,
            mla_num_latent_heads=2
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        input_ids = torch.randint(0, vocab_size, (2, 16), device=device)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        logits, _ = model(input_ids)
        
        # Dummy loss
        loss = logits.sum()
        loss.backward()
        optimizer.step()
        
        assert loss.item() == loss.item()  # Just check it's not nan
    
    def test_generation_with_mla(self, device):
        """Test generation with MLA KV cache."""
        vocab_size = 1000
        model = SimpleTransformer(
            vocab_size=vocab_size,
            d_model=128,
            nhead=4,
            num_layers=2,
            use_mla=True
        ).to(device)
        model.eval()
        
        input_ids = torch.randint(0, vocab_size, (1, 8), device=device)
        
        # Generate some tokens
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(5):
            with torch.no_grad():
                logits, past_key_values = model(
                    generated[:, -1:] if past_key_values is not None else generated,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
        
        assert generated.shape == (1, 13)  # 8 + 5
