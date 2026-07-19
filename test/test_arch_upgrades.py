#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for architecture upgrades: Flash Attention, GQA, RoPE, etc."""

import pytest
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import (
    SimpleTransformer,
    ModernAttention,
    RotaryEmbedding,
)
from src.config import (
    DEFAULT_D_MODEL,
    DEFAULT_NHEAD,
    DEFAULT_NUM_LAYERS,
    DEFAULT_MAX_LEN,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestFlashAttention:
    def test_flash_attention_vs_naive(self, device):
        d_model = 128
        num_heads = 4
        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        flash_attn = ModernAttention(d_model=d_model, num_heads=num_heads, use_flash_attention=True).to(device)
        naive_attn = ModernAttention(d_model=d_model, num_heads=num_heads, use_flash_attention=False).to(device)
        naive_attn.load_state_dict(flash_attn.state_dict())
        with torch.no_grad():
            out_flash, _ = flash_attn(x)
            out_naive, _ = naive_attn(x)
        assert out_flash.shape == (batch_size, seq_len, d_model)
        assert out_naive.shape == (batch_size, seq_len, d_model)
        if device.type == "cuda":
            torch.testing.assert_close(out_flash, out_naive, atol=1e-3, rtol=1e-2)
        else:
            assert not torch.isnan(out_flash).any()
            assert not torch.isinf(out_flash).any()
            assert not torch.isnan(out_naive).any()
            assert not torch.isinf(out_naive).any()

    def test_flash_attention_causal_mask(self, device):
        d_model = 128
        num_heads = 4
        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        attn = ModernAttention(d_model=d_model, num_heads=num_heads, use_flash_attention=True).to(device)
        with torch.no_grad():
            out, _ = attn(x, is_causal=True)
        assert out.shape == (batch_size, seq_len, d_model)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flash_attention_performance(self, device):
        d_model = 512
        num_heads = 8
        batch_size = 4
        seq_len = 256
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        flash_attn = ModernAttention(d_model=d_model, num_heads=num_heads, use_flash_attention=True).to(device)
        naive_attn = ModernAttention(d_model=d_model, num_heads=num_heads, use_flash_attention=False).to(device)
        for _ in range(5):
            _ = flash_attn(x)
            _ = naive_attn(x)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = flash_attn(x)
        torch.cuda.synchronize()
        flash_time = time.time() - start
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = naive_attn(x)
        torch.cuda.synchronize()
        naive_time = time.time() - start
        print(f"Flash Attention time: {flash_time:.4f}s")
        print(f"Naive Attention time: {naive_time:.4f}s")


class TestEndToEnd:
    def test_standard_transformer_training_step(self, device):
        vocab_size = 1000
        model = SimpleTransformer(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        input_ids = torch.randint(0, vocab_size, (2, 16), device=device)
        model.train()
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
        optimizer.step()
        assert loss.item() == loss.item()

    def test_gqa_transformer_training_step(self, device):
        vocab_size = 1000
        model = SimpleTransformer(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2, num_kv_heads=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        input_ids = torch.randint(0, vocab_size, (2, 16), device=device)
        model.train()
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
        optimizer.step()
        assert loss.item() == loss.item()

    def test_generation_autoregressive(self, device):
        vocab_size = 1000
        model = SimpleTransformer(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2).to(device)
        model.eval()
        input_ids = torch.randint(0, vocab_size, (1, 8), device=device)
        generated = input_ids.clone()
        for _ in range(5):
            with torch.no_grad():
                logits = model(generated)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
        assert generated.shape == (1, 13)