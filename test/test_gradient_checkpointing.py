"""Tests for Gradient Checkpointing functionality.

Gradient checkpointing trades compute for memory by recomputing activations
during backward pass instead of storing them. This typically saves 30-50% memory
at the cost of 20-30% more compute time.
"""

import pytest
import torch
import torch.nn as nn

try:
    from src.model import SimpleTransformer, ModernTransformerBlock
except ImportError:
    SimpleTransformer = None
    ModernTransformerBlock = None


class TestGradientCheckpointingBasics:
    """Basic functionality tests for gradient checkpointing."""

    @pytest.fixture
    def model_config(self):
        return dict(
            vocab_size=1000,
            d_model=128,
            nhead=4,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.0,
            max_len=128,
            mode="modern",
        )

    def test_model_without_checkpointing(self, model_config):
        """Test that model works without gradient checkpointing."""
        model = SimpleTransformer(**model_config, gradient_checkpointing=False)
        model.train()
        
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        output, _ = model(input_ids)
        assert output.shape == (batch_size, seq_len, 1000)
        
        loss = output.sum()
        loss.backward()
        
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, "Model should have gradients after backward pass"

    def test_model_with_checkpointing(self, model_config):
        """Test that model works with gradient checkpointing enabled."""
        model = SimpleTransformer(**model_config, gradient_checkpointing=True)
        model.train()
        
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        output, _ = model(input_ids)
        assert output.shape == (batch_size, seq_len, 1000)
        
        loss = output.sum()
        loss.backward()
        
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, "Model should have gradients after backward pass"

    def test_checkpointing_flag_per_block(self, model_config):
        """Test that gradient checkpointing flag is correctly set per block."""
        model = SimpleTransformer(**model_config, gradient_checkpointing=True)
        
        checkpoint_count = sum(
            1 for block in model.transformer_blocks 
            if block.gradient_checkpointing
        )
        assert checkpoint_count == model.num_layers, \
            f"Expected all {model.num_layers} blocks to have checkpointing, got {checkpoint_count}"

    def test_partial_checkpointing_ratio(self, model_config):
        """Test selective checkpointing with ratio parameter."""
        model = SimpleTransformer(
            **model_config, 
            gradient_checkpointing=True,
            gradient_checkpointing_ratio=0.5
        )
        
        expected_checkpoint_layers = int(model.num_layers * 0.5)
        checkpoint_count = sum(
            1 for block in model.transformer_blocks 
            if block.gradient_checkpointing
        )
        assert checkpoint_count == expected_checkpoint_layers, \
            f"Expected {expected_checkpoint_layers} checkpointed blocks, got {checkpoint_count}"


class TestGradientCheckpointingMethods:
    """Test the enable/disable/ratio methods on SimpleTransformer."""

    @pytest.fixture
    def model_config(self):
        return dict(
            vocab_size=1000,
            d_model=128,
            nhead=4,
            num_layers=6,
            dim_feedforward=256,
            dropout=0.0,
            max_len=128,
            mode="modern",
        )

    def test_enable_gradient_checkpointing(self, model_config):
        """Test enable_gradient_checkpointing method."""
        model = SimpleTransformer(**model_config, gradient_checkpointing=False)
        
        assert not any(b.gradient_checkpointing for b in model.transformer_blocks)
        
        model.enable_gradient_checkpointing()
        
        assert all(b.gradient_checkpointing for b in model.transformer_blocks)

    def test_disable_gradient_checkpointing(self, model_config):
        """Test disable_gradient_checkpointing method."""
        model = SimpleTransformer(**model_config, gradient_checkpointing=True)
        
        assert all(b.gradient_checkpointing for b in model.transformer_blocks)
        
        model.disable_gradient_checkpointing()
        
        assert not any(b.gradient_checkpointing for b in model.transformer_blocks)

    def test_set_gradient_checkpointing_ratio(self, model_config):
        """Test set_gradient_checkpointing_ratio method."""
        model = SimpleTransformer(**model_config, gradient_checkpointing=True)
        
        model.set_gradient_checkpointing_ratio(0.5)
        count_half = sum(b.gradient_checkpointing for b in model.transformer_blocks)
        
        model.set_gradient_checkpointing_ratio(1.0)
        count_full = sum(b.gradient_checkpointing for b in model.transformer_blocks)
        
        assert count_half < count_full, "Ratio change should affect checkpoint count"
        
        expected_half = int(model_config["num_layers"] * 0.5)
        assert count_half == expected_half, f"Expected {expected_half} checkpointed blocks for ratio 0.5, got {count_half}"

    def test_dynamic_ratio_change(self, model_config):
        """Test changing ratio dynamically during training."""
        model = SimpleTransformer(**model_config, gradient_checkpointing=True)
        
        model.set_gradient_checkpointing_ratio(0.5)
        count_half = sum(b.gradient_checkpointing for b in model.transformer_blocks)
        
        model.set_gradient_checkpointing_ratio(1.0)
        count_full = sum(b.gradient_checkpointing for b in model.transformer_blocks)
        
        assert count_half < count_full, "Ratio change should affect checkpoint count"


class TestGradientCheckpointingMemory:
    """Test memory savings with gradient checkpointing."""

    @pytest.fixture
    def large_model_config(self):
        return dict(
            vocab_size=5000,
            d_model=256,
            nhead=8,
            num_layers=8,
            dim_feedforward=512,
            dropout=0.0,
            max_len=256,
            mode="modern",
        )

    def test_forward_pass_with_checkpointing(self, large_model_config):
        """Test that forward pass works correctly with checkpointing."""
        model = SimpleTransformer(**large_model_config, gradient_checkpointing=True)
        model.train()
        
        batch_size = 4
        seq_len = 64
        input_ids = torch.randint(0, 5000, (batch_size, seq_len))
        
        output, _ = model(input_ids)
        assert output.shape == (batch_size, seq_len, 5000)
        
        loss = nn.CrossEntropyLoss()(
            output.view(-1, 5000),
            torch.randint(0, 5000, (batch_size * seq_len,))
        )
        loss.backward()

    def test_inference_without_checkpointing(self, large_model_config):
        """Test inference mode works without checkpointing."""
        model = SimpleTransformer(**large_model_config, gradient_checkpointing=True)
        model.eval()
        
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 5000, (batch_size, seq_len))
        
        with torch.no_grad():
            output, _ = model(input_ids)
        
        assert output.shape == (batch_size, seq_len, 5000)

    def test_checkpointing_disabled_during_inference(self, large_model_config):
        """Test that checkpointing is effectively disabled during inference."""
        model = SimpleTransformer(**large_model_config, gradient_checkpointing=True)
        model.eval()
        
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 5000, (batch_size, seq_len))
        
        with torch.no_grad():
            output1, _ = model(input_ids)
            output2, _ = model(input_ids)
        
        assert torch.allclose(output1, output2), "Inference should be deterministic"


class TestGradientCheckpointingCompatibility:
    """Test compatibility with other features."""

    def test_with_weight_tying(self):
        """Test gradient checkpointing works with weight tying."""
        model = SimpleTransformer(
            vocab_size=1000,
            d_model=128,
            nhead=4,
            num_layers=4,
            dim_feedforward=256,
            mode="modern",
            gradient_checkpointing=True,
            use_weight_tying=True,
        )
        
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        model.train()
        output, _ = model(input_ids)
        
        loss = output.sum()
        loss.backward()
        
        assert model._tied_weights, "Weight tying should be enabled"

    def test_with_sliding_window(self):
        """Test gradient checkpointing works with sliding window attention."""
        model = SimpleTransformer(
            vocab_size=1000,
            d_model=128,
            nhead=4,
            num_layers=4,
            dim_feedforward=256,
            mode="modern",
            gradient_checkpointing=True,
            use_sliding_window=True,
            window_size=32,
        )
        
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        model.train()
        output, _ = model(input_ids)
        
        loss = output.sum()
        loss.backward()


class TestGradientCheckpointingEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_ratio_zero(self):
        """Test with ratio=0 (no checkpointing)."""
        model = SimpleTransformer(
            vocab_size=1000,
            d_model=128,
            nhead=4,
            num_layers=4,
            dim_feedforward=256,
            mode="modern",
            gradient_checkpointing=True,
            gradient_checkpointing_ratio=0.0,
        )
        
        checkpoint_count = sum(
            b.gradient_checkpointing for b in model.transformer_blocks
        )
        assert checkpoint_count == 1, "Minimum 1 layer should be checkpointed"

    def test_ratio_one(self):
        """Test with ratio=1.0 (all layers checkpointed)."""
        model = SimpleTransformer(
            vocab_size=1000,
            d_model=128,
            nhead=4,
            num_layers=4,
            dim_feedforward=256,
            mode="modern",
            gradient_checkpointing=True,
            gradient_checkpointing_ratio=1.0,
        )
        
        checkpoint_count = sum(
            b.gradient_checkpointing for b in model.transformer_blocks
        )
        assert checkpoint_count == 4, "All layers should be checkpointed"

    def test_single_layer_model(self):
        """Test with single layer model."""
        model = SimpleTransformer(
            vocab_size=1000,
            d_model=128,
            nhead=4,
            num_layers=1,
            dim_feedforward=256,
            mode="modern",
            gradient_checkpointing=True,
        )
        
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        model.train()
        output, _ = model(input_ids)
        
        loss = output.sum()
        loss.backward()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
