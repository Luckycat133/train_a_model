#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""torch.compile() integration tests. Validates torch.compile() on SimpleTransformer."""

import logging
import sys
from pathlib import Path
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from src.model import SimpleTransformer
except ImportError:
    SimpleTransformer = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class TestCompile:
    """torch.compile() tests"""

    @pytest.fixture
    def model(self):
        return SimpleTransformer(vocab_size=1000, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.0, max_len=128, mode="modern")

    @pytest.fixture
    def sample_input(self):
        return torch.randint(0, 1000, (2, 32))

    def test_compile_available(self):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch required")
        assert hasattr(torch, "compile")

    def test_compile_mode_setting(self, model):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch required")
        compiled = torch.compile(model, mode="reduce-overhead", dynamic=True)
        assert compiled is not None

    def test_compile_backend_setting(self, model):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch required")
        compiled = torch.compile(model, backend="inductor", dynamic=True)
        assert compiled is not None

    def test_compile_dynamic_setting(self, model):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch required")
        compiled = torch.compile(model, dynamic=True)
        assert compiled is not None

    def test_is_compiled_property(self, model):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch required")
        compiled = torch.compile(model)
        assert compiled is not model

    def test_forward_after_compile(self, model, sample_input):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch required")
        compiled = torch.compile(model, mode="reduce-overhead", dynamic=True)
        compiled.eval()
        with torch.no_grad():
            output = compiled(sample_input)
        assert output.shape[-1] == 1000

    def test_decompile(self, model):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch required")
        original_params = sum(p.numel() for p in model.parameters())
        compiled = torch.compile(model)
        compiled_params = sum(p.numel() for p in compiled.parameters())
        assert original_params == compiled_params

    def test_get_compiled_model(self, model):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch required")
        compiled = torch.compile(model, mode="reduce-overhead")
        assert compiled is not None

    def test_output_consistency(self, model, sample_input):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch required")
        model.eval()
        with torch.no_grad():
            output_eager = model(sample_input.clone())
        model2 = SimpleTransformer(vocab_size=1000, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.0, max_len=128, mode="modern")
        model2.eval()
        compiled = torch.compile(model2, mode="reduce-overhead", dynamic=True)
        with torch.no_grad():
            output_compiled = compiled(sample_input)
        assert output_eager.shape == output_compiled.shape

    def test_training_mode(self, model, sample_input):
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch required")
        compiled = torch.compile(model, mode="reduce-overhead")
        compiled.train()
        output = compiled(sample_input)
        loss = output.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in compiled.parameters() if p.requires_grad)
        assert has_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])