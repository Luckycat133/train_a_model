"""Tests for Gradient Checkpointing functionality."""

import pytest
import torch
import torch.nn as nn

try:
    from src.model import SimpleTransformer, ModernTransformerBlock
except ImportError:
    SimpleTransformer = None
    ModernTransformerBlock = None


class TestGradientCheckpointingBasics:
    @pytest.fixture
    def model_config(self):
        return dict(vocab_size=1000, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.0, max_len=128, mode="modern")

    def test_model_without_checkpointing(self, model_config):
        model = SimpleTransformer(**model_config, use_checkpoint=False)
        model.train()
        input_ids = torch.randint(0, 1000, (2, 32))
        output = model(input_ids)
        assert output.shape == (2, 32, 1000)
        loss = output.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad

    def test_model_with_checkpointing(self, model_config):
        model = SimpleTransformer(**model_config, use_checkpoint=True)
        model.train()
        input_ids = torch.randint(0, 1000, (2, 32))
        output = model(input_ids)
        assert output.shape == (2, 32, 1000)
        loss = output.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad

    def test_checkpointing_flag_persists(self, model_config):
        model = SimpleTransformer(**model_config, use_checkpoint=True)
        assert model.use_checkpoint is True
        model2 = SimpleTransformer(**model_config, use_checkpoint=False)
        assert model2.use_checkpoint is False


class TestGradientCheckpointingMemory:
    @pytest.fixture
    def large_model_config(self):
        return dict(vocab_size=5000, d_model=256, nhead=8, num_layers=8, dim_feedforward=512, dropout=0.0, max_len=256, mode="modern")

    def test_forward_pass_with_checkpointing(self, large_model_config):
        model = SimpleTransformer(**large_model_config, use_checkpoint=True)
        model.train()
        input_ids = torch.randint(0, 5000, (4, 64))
        output = model(input_ids)
        assert output.shape == (4, 64, 5000)
        loss = nn.CrossEntropyLoss()(output.view(-1, 5000), torch.randint(0, 5000, (4 * 64,)))
        loss.backward()

    def test_inference_without_checkpointing(self, large_model_config):
        model = SimpleTransformer(**large_model_config, use_checkpoint=True)
        model.eval()
        input_ids = torch.randint(0, 5000, (2, 32))
        with torch.no_grad():
            output = model(input_ids)
        assert output.shape == (2, 32, 5000)

    def test_checkpointing_deterministic_inference(self, large_model_config):
        model = SimpleTransformer(**large_model_config, use_checkpoint=True)
        model.eval()
        input_ids = torch.randint(0, 5000, (2, 32))
        with torch.no_grad():
            output1 = model(input_ids)
            output2 = model(input_ids)
        assert torch.allclose(output1, output2)


class TestGradientCheckpointingCompatibility:
    def test_with_weight_tying(self):
        model = SimpleTransformer(vocab_size=1000, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, mode="modern", use_checkpoint=True, use_weight_tying=True)
        input_ids = torch.randint(0, 1000, (2, 32))
        model.train()
        output = model(input_ids)
        loss = output.sum()
        loss.backward()
        assert model._tied_weights

    def test_with_sliding_window(self):
        model = SimpleTransformer(vocab_size=1000, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, mode="modern", use_checkpoint=True, use_sliding_window=True, window_size=32)
        input_ids = torch.randint(0, 1000, (2, 64))
        model.train()
        output = model(input_ids)
        loss = output.sum()
        loss.backward()


class TestGradientCheckpointingEdgeCases:
    def test_single_layer_model(self):
        model = SimpleTransformer(vocab_size=1000, d_model=128, nhead=4, num_layers=1, dim_feedforward=256, mode="modern", use_checkpoint=True)
        input_ids = torch.randint(0, 1000, (2, 32))
        model.train()
        output = model(input_ids)
        loss = output.sum()
        loss.backward()

    def test_legacy_mode_with_checkpoint(self):
        model = SimpleTransformer(vocab_size=1000, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, mode="legacy", use_checkpoint=True)
        input_ids = torch.randint(0, 1000, (2, 32))
        model.train()
        output = model(input_ids)
        loss = output.sum()
        loss.backward()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])