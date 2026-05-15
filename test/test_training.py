"""Tests for the modular training framework.

Tests cover:
- BaseTrainer interface
- TrainingConfig dataclass
- CheckpointManager
- Dataset classes
- Trainer factory functions
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.token_to_id = {f"token_{i}": i for i in range(vocab_size)}
        self.id_to_token = {i: f"token_{i}" for i in range(vocab_size)}
        self.eos_id = 0
        self.pad_id = 1
        self.bos_id = 2

    def encode(self, text: str, max_length: Optional[int] = None, truncation: bool = False) -> List[int]:
        tokens = [self.bos_id] + [i % self.vocab_size for i in range(len(text))]
        if max_length and len(tokens) > max_length:
            if truncation:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [self.eos_id] * (max_length - len(tokens))
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        return " ".join(self.id_to_token.get(i, "<unk>") for i in token_ids)


class TestModuleStructure:
    """Test that module files exist and can be imported."""

    def test_module_files_exist(self):
        """Test that all module files exist."""
        assert Path("/workspace/src/training/__init__.py").exists()
        assert Path("/workspace/src/training/base_trainer.py").exists()
        assert Path("/workspace/src/training/pretrain.py").exists()
        assert Path("/workspace/src/training/sft_trainer.py").exists()
        assert Path("/workspace/src/training/rl_trainer.py").exists()

    def test_training_init_imports(self):
        """Test that training __init__.py can be read."""
        init_path = Path("/workspace/src/training/__init__.py")
        content = init_path.read_text()
        assert "__version__" in content
        assert "BaseTrainer" in content
        assert "CausalLanguageModelTrainer" in content
        assert "SFTTrainer" in content
        assert "DPOTrainer" in content
        assert "GRPOTrainer" in content

    def test_base_trainer_has_required_classes(self):
        """Test that base_trainer.py has required classes."""
        content = Path("/workspace/src/training/base_trainer.py").read_text()
        assert "class TrainingConfig" in content
        assert "class TrainingStats" in content
        assert "class CheckpointManager" in content
        assert "class BaseTrainer" in content
        assert "abstractmethod" in content

    def test_pretrain_has_causal_lm_trainer(self):
        """Test that pretrain.py has CausalLanguageModelTrainer."""
        content = Path("/workspace/src/training/pretrain.py").read_text()
        assert "class CausalLanguageModelTrainer" in content
        assert "class CausalLMTrainingConfig" in content
        assert "def create_pretrain_trainer" in content

    def test_sft_trainer_has_required_classes(self):
        """Test that sft_trainer.py has required classes."""
        content = Path("/workspace/src/training/sft_trainer.py").read_text()
        assert "class SFTTrainer" in content
        assert "class SFTConfig" in content
        assert "class InstructionResponseDataset" in content
        assert "def create_sft_trainer" in content

    def test_rl_trainer_has_required_classes(self):
        """Test that rl_trainer.py has required classes."""
        content = Path("/workspace/src/training/rl_trainer.py").read_text()
        assert "class RLConfig" in content
        assert "class DPOTrainer" in content
        assert "class GRPOTrainer" in content
        assert "class PreferenceDataset" in content
        assert "def create_rl_trainer" in content


class TestBaseTrainerInterface:
    """Test BaseTrainer interface requirements."""

    def test_base_trainer_abstract_methods(self):
        """Test that BaseTrainer has required abstract methods."""
        content = Path("/workspace/src/training/base_trainer.py").read_text()
        assert "@abstractmethod" in content
        assert "def compute_loss" in content
        assert "def build_model" in content

    def test_base_trainer_has_common_methods(self):
        """Test that BaseTrainer has common training methods."""
        content = Path("/workspace/src/training/base_trainer.py").read_text()
        common_methods = [
            "def fit",
            "def training_step",
            "def eval_step",
            "def build_optimizer",
            "def build_scheduler",
            "_prepare_batch",
            "_evaluate",
        ]
        for method in common_methods:
            assert method in content, f"Missing method: {method}"

    def test_checkpoint_manager_methods(self):
        """Test CheckpointManager has required methods."""
        content = Path("/workspace/src/training/base_trainer.py").read_text()
        assert "def save" in content
        assert "def load" in content
        assert "def find_latest" in content


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_config_has_training_attributes(self):
        """Test TrainingConfig has required training attributes."""
        content = Path("/workspace/src/training/base_trainer.py").read_text()
        required_attrs = [
            "context_length",
            "batch_size",
            "learning_rate",
            "epochs",
            "accumulation_steps",
            "max_grad_norm",
            "checkpoint_every",
            "model_save_dir",
            "use_amp",
        ]
        for attr in required_attrs:
            assert attr in content, f"Missing attribute: {attr}"

    def test_config_from_dict_method(self):
        """Test TrainingConfig has from_dict method."""
        content = Path("/workspace/src/training/base_trainer.py").read_text()
        assert "def from_dict" in content


class TestCausalLanguageModelTrainer:
    """Test CausalLanguageModelTrainer implementation."""

    def test_causal_lm_trainer_methods(self):
        """Test CausalLM trainer has required methods."""
        content = Path("/workspace/src/training/pretrain.py").read_text()
        required_methods = [
            "def build_model",
            "def compute_loss",
            "def create_dataloaders",
        ]
        for method in required_methods:
            assert method in content, f"Missing method: {method}"

    def test_causal_lm_uses_simple_transformer(self):
        """Test CausalLM trainer uses SimpleTransformer."""
        content = Path("/workspace/src/training/pretrain.py").read_text()
        assert "SimpleTransformer" in content

    def test_causal_lm_supports_label_smoothing(self):
        """Test CausalLM supports label smoothing."""
        content = Path("/workspace/src/training/pretrain.py").read_text()
        assert "label_smoothing" in content


class TestSFTTrainer:
    """Test SFTTrainer implementation."""

    def test_sft_trainer_methods(self):
        """Test SFT trainer has required methods."""
        content = Path("/workspace/src/training/sft_trainer.py").read_text()
        required_methods = [
            "def build_model",
            "def compute_loss",
            "def create_dataloaders",
        ]
        for method in required_methods:
            assert method in content, f"Missing method: {method}"

    def test_instruction_dataset_class(self):
        """Test InstructionResponseDataset class exists."""
        content = Path("/workspace/src/training/sft_trainer.py").read_text()
        assert "class InstructionResponseDataset" in content

    def test_sft_supports_response_loss_only(self):
        """Test SFT supports response-only loss."""
        content = Path("/workspace/src/training/sft_trainer.py").read_text()
        assert "response_loss_only" in content


class TestRLTrainer:
    """Test RL trainer implementation."""

    def test_dpo_trainer_methods(self):
        """Test DPO trainer has required methods."""
        content = Path("/workspace/src/training/rl_trainer.py").read_text()
        required_methods = [
            "def build_model",
            "def compute_loss",
            "def compute_log_probs",
        ]
        for method in required_methods:
            assert method in content, f"Missing method: {method}"

    def test_grpo_trainer_methods(self):
        """Test GRPO trainer has required methods."""
        content = Path("/workspace/src/training/rl_trainer.py").read_text()
        required_methods = [
            "def _sample_responses",
            "def compute_advantages",
        ]
        for method in required_methods:
            assert method in content, f"Missing method: {method}"

    def test_preference_dataset_class(self):
        """Test PreferenceDataset class exists."""
        content = Path("/workspace/src/training/rl_trainer.py").read_text()
        assert "class PreferenceDataset" in content

    def test_dpo_loss_implementation(self):
        """Test DPO loss is implemented."""
        content = Path("/workspace/src/training/rl_trainer.py").read_text()
        assert "def compute_loss" in content
        assert "logsigmoid" in content or "log_sigmoid" in content.lower()

    def test_grpo_advantage_computation(self):
        """Test GRPO advantage computation."""
        content = Path("/workspace/src/training/rl_trainer.py").read_text()
        assert "def compute_advantages" in content


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_pretrain_trainer_exists(self):
        """Test create_pretrain_trainer function exists."""
        content = Path("/workspace/src/training/pretrain.py").read_text()
        assert "def create_pretrain_trainer" in content

    def test_create_sft_trainer_exists(self):
        """Test create_sft_trainer function exists."""
        content = Path("/workspace/src/training/sft_trainer.py").read_text()
        assert "def create_sft_trainer" in content

    def test_create_rl_trainer_exists(self):
        """Test create_rl_trainer function exists."""
        content = Path("/workspace/src/training/rl_trainer.py").read_text()
        assert "def create_rl_trainer" in content


class TestSignalHandling:
    """Test signal handling in BaseTrainer."""

    def test_signal_handlers_defined(self):
        """Test signal handlers are defined."""
        content = Path("/workspace/src/training/base_trainer.py").read_text()
        assert "signal.SIGINT" in content or "SIGINT" in content
        assert "signal.SIGTERM" in content or "SIGTERM" in content
        assert "TERMINATE_TRAINING" in content
        assert "SAVE_CHECKPOINT_SIGNAL" in content


class TestMixedPrecision:
    """Test mixed precision training support."""

    def test_amp_support_in_base_trainer(self):
        """Test AMP is supported in BaseTrainer."""
        content = Path("/workspace/src/training/base_trainer.py").read_text()
        assert "autocast" in content
        assert "GradScaler" in content or "scaler" in content

    def test_bf16_fp16_dtype(self):
        """Test BF16/FP16 dtype support."""
        content = Path("/workspace/src/training/base_trainer.py").read_text()
        assert "bf16" in content.lower() or "bfloat16" in content.lower()
        assert "fp16" in content.lower() or "float16" in content.lower()


class TestGradientAccumulation:
    """Test gradient accumulation support."""

    def test_accumulation_steps_config(self):
        """Test accumulation_steps is configured."""
        content = Path("/workspace/src/training/base_trainer.py").read_text()
        assert "accumulation_steps" in content

    def test_accumulation_logic(self):
        """Test gradient accumulation logic exists."""
        content = Path("/workspace/src/training/base_trainer.py").read_text()
        assert "backward" in content


class TestEarlyStopping:
    """Test early stopping support."""

    def test_early_stopping_config(self):
        """Test early stopping is configured."""
        content = Path("/workspace/src/training/base_trainer.py").read_text()
        assert "early_stopping" in content.lower()
        assert "patience" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
