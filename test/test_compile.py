#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
torch.compile() 集成测试

测试 torch.compile() 在 SimpleTransformer 模型上的功能:
- 编译模式设置
- 编译后模型前向传播
- 与未编译模型的输出一致性
- 配置选项覆盖
"""

import os
import sys
import time
import logging
from pathlib import Path

import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from src.model import SimpleTransformer
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from src.model import SimpleTransformer
    except ImportError:
        SimpleTransformer = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CompileTest")


class TestCompile:
    """torch.compile() 功能测试"""

    @pytest.fixture
    def model(self):
        """创建测试用的小型模型"""
        model = SimpleTransformer(
            vocab_size=1000,
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.0,
            max_len=128,
            mode="modern",
        )
        return model

    @pytest.fixture
    def sample_input(self):
        """创建测试输入"""
        return torch.randint(0, 1000, (2, 32))

    def test_compile_available(self):
        """测试 torch.compile() 是否可用"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch 2.0+ required for torch.compile()")

        assert hasattr(torch, "compile"), "torch.compile() not available"
        logger.info("torch.compile() is available")

    def test_compile_mode_setting(self, model):
        """测试编译模式设置"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch 2.0+ required for torch.compile()")

        modes = ["reduce-overhead", "max-autotune", "default"]

        for mode in modes:
            model_compiled = model.compile(mode=mode, dynamic=True)
            assert model_compiled._compiled == True
            assert model_compiled._compile_mode == mode
            logger.info(f"Compile mode '{mode}' set successfully")

    def test_compile_backend_setting(self, model):
        """测试编译后端设置"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch 2.0+ required for torch.compile()")

        model.compile(backend="inductor", dynamic=True)
        assert model._compile_backend == "inductor"
        logger.info("Compile backend set successfully")

    def test_compile_dynamic_setting(self, model):
        """测试动态形状设置"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch 2.0+ required for torch.compile()")

        model.compile(dynamic=True)
        assert model._compiled == True
        logger.info("Dynamic compilation enabled")

    def test_is_compiled_property(self, model):
        """测试 is_compiled 属性"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch 2.0+ required for torch.compile()")

        assert model.is_compiled == False
        model.compile()
        assert model.is_compiled == True
        logger.info("is_compiled property works correctly")

    def test_forward_after_compile(self, model, sample_input):
        """测试编译后的前向传播"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch 2.0+ required for torch.compile()")

        model.compile(mode="reduce-overhead", dynamic=True)

        model.eval()
        with torch.no_grad():
            output, _ = model(sample_input)

        assert output.shape == (2, 32, 1000)
        logger.info(f"Forward pass output shape: {output.shape}")

    def test_decompile(self, model):
        """测试解编译功能"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch 2.0+ required for torch.compile()")

        model.compile()
        assert model.is_compiled == True

        model.decompile()
        assert model.is_compiled == False
        logger.info("Decompile works correctly")

    def test_get_compiled_model(self, model):
        """测试获取编译后模型"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch 2.0+ required for torch.compile()")

        model.compile(mode="reduce-overhead")

        compiled = model.get_compiled_model()
        assert compiled is not None
        assert compiled is not model
        logger.info("get_compiled_model returns compiled instance")

    def test_output_consistency(self, model, sample_input):
        """测试编译前后输出一致性"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch 2.0+ required for torch.compile()")

        model.eval()
        with torch.no_grad():
            output_eager, _ = model(sample_input.clone())

        model_compiled = SimpleTransformer(
            vocab_size=1000,
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.0,
            max_len=128,
            mode="modern",
        )
        model_compiled.eval()
        model_compiled.compile(mode="reduce-overhead", dynamic=True)

        with torch.no_grad():
            output_compiled, _ = model_compiled(sample_input)

        max_diff = (output_eager - output_compiled).abs().max().item()
        logger.info(f"Max output difference: {max_diff:.6f}")

        assert max_diff < 1e-5, "Compiled model output differs too much"

    def test_training_mode(self, model, sample_input):
        """测试训练模式下的编译"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch 2.0+ required for torch.compile()")

        model.compile(mode="reduce-overhead")
        model.train()

        output, _ = model(sample_input)
        assert output.requires_grad == False

        loss = output.sum()
        loss.backward()

        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, "Gradients not computed"

        logger.info("Training mode with compilation works")


def run_quick_tests():
    """运行快速测试（不使用 pytest）"""
    logger.info("=" * 60)
    logger.info("Running quick torch.compile() tests...")
    logger.info("=" * 60)

    if not TORCH_AVAILABLE:
        logger.warning("PyTorch 2.0+ required for torch.compile() - skipping tests")
        return

    model = SimpleTransformer(
        vocab_size=1000,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.0,
        max_len=128,
        mode="modern",
    )

    sample_input = torch.randint(0, 1000, (2, 32))

    logger.info("\n1. Test compile mode setting...")
    model.compile(mode="reduce-overhead", dynamic=True)
    assert model._compiled == True
    logger.info("   PASSED")

    logger.info("\n2. Test is_compiled property...")
    assert model.is_compiled == True
    logger.info("   PASSED")

    logger.info("\n3. Test forward pass after compile...")
    model.eval()
    with torch.no_grad():
        output, _ = model(sample_input)
    assert output.shape == (2, 32, 1000)
    logger.info(f"   PASSED - Output shape: {output.shape}")

    logger.info("\n4. Test decompile...")
    model.decompile()
    assert model.is_compiled == False
    logger.info("   PASSED")

    logger.info("\n" + "=" * 60)
    logger.info("All quick tests PASSED!")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_quick_tests()
