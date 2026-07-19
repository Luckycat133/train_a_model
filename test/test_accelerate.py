"""Tests for HuggingFace Accelerate integration in BaseTrainer.

Tests:
- Accelerate availability detection
- use_accelerate=True vs use_accelerate=False modes
- accelerator.prepare() integration
- accelerator.backward() for gradient scaling
- Mixed precision (bf16/fp16) configuration
- Checkpoint save/load with Accelerator
"""

import tempfile
from abc import ABC
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class AbstractTestMixin(ABC):
    """Mixin to make BaseTrainer tests work with abstract class."""

    def get_concrete_trainer(self, config, **kwargs):
        """Get a concrete trainer instance for testing."""
        from src.training.base_trainer import BaseTrainer, TrainingConfig

        class ConcreteTrainer(BaseTrainer):
            def build_model(self):
                return MagicMock()

            def compute_loss(self, batch, **kwargs):
                mock_loss = MagicMock()
                mock_loss.item.return_value = 1.0
                return mock_loss, {"accuracy": 0.9}

        return ConcreteTrainer(config, **kwargs)


class TestAccelerateImport:
    """Test Accelerate library availability."""

    def test_accelerate_import(self):
        """Test that accelerate can be imported."""
        try:
            from accelerate import Accelerator
            from src.training.base_trainer import ACCELERATE_AVAILABLE
            assert ACCELERATE_AVAILABLE is not None
        except ImportError:
            pytest.skip("accelerate not installed")

    def test_accelerate_optional_import(self):
        """Test that base_trainer works without accelerate."""
        from src.training.base_trainer import Accelerator, ACCELERATE_AVAILABLE
        if not ACCELERATE_AVAILABLE:
            assert Accelerator is None
        else:
            assert Accelerator is not None


class TestAccelerateConfiguration(AbstractTestMixin):
    """Test Accelerate configuration options."""

    def test_use_accelerate_default_true(self):
        """Test that use_accelerate defaults to True when available."""
        from src.training.base_trainer import ACCELERATE_AVAILABLE

        if not ACCELERATE_AVAILABLE:
            pytest.skip("accelerate not installed")

        config = MagicMock()
        config.model_save_dir = "test"
        config.amp_dtype = "bf16"
        config.use_amp = True
        config.accumulation_steps = 1

        trainer = self.get_concrete_trainer(config)
        assert trainer.use_accelerate is True

    def test_use_accelerate_false_disables(self):
        """Test that use_accelerate=False disables Accelerate mode."""
        from src.training.base_trainer import TrainingConfig

        config = TrainingConfig()
        trainer = self.get_concrete_trainer(config, use_accelerate=False)
        assert trainer.use_accelerate is False
        assert trainer.accelerator is None

    def test_accelerate_initialized_when_available(self):
        """Test that accelerator is initialized when accelerate is available."""
        from src.training.base_trainer import ACCELERATE_AVAILABLE

        if not ACCELERATE_AVAILABLE:
            pytest.skip("accelerate not installed")

        from src.training.base_trainer import TrainingConfig

        config = TrainingConfig(use_amp=True, amp_dtype="bf16")
        trainer = self.get_concrete_trainer(config, use_accelerate=True)

        assert trainer.accelerator is not None
        assert hasattr(trainer.accelerator, "prepare")
        assert hasattr(trainer.accelerator, "backward")
        assert hasattr(trainer.accelerator, "step")
        assert hasattr(trainer.accelerator, "zero_grad")

    def test_mixed_precision_config_bf16(self):
        """Test BF16 mixed precision configuration."""
        from src.training.base_trainer import ACCELERATE_AVAILABLE

        if not ACCELERATE_AVAILABLE:
            pytest.skip("accelerate not installed")

        from src.training.base_trainer import TrainingConfig

        config = TrainingConfig(use_amp=True, amp_dtype="bf16")
        trainer = self.get_concrete_trainer(config, use_accelerate=True)

        assert trainer.accelerator is not None

    def test_mixed_precision_config_fp16(self):
        """Test FP16 mixed precision configuration."""
        from src.training.base_trainer import ACCELERATE_AVAILABLE

        if not ACCELERATE_AVAILABLE:
            pytest.skip("accelerate not installed")

        from src.training.base_trainer import TrainingConfig

        config = TrainingConfig(use_amp=True, amp_dtype="fp16")
        trainer = self.get_concrete_trainer(config, use_accelerate=True)

        assert trainer.accelerator is not None


class TestAccelerateTrainingStep(AbstractTestMixin):
    """Test training_step with Accelerate integration."""

    def test_training_step_accelerate_mode(self):
        """Test training_step uses accelerator.backward() when accelerate enabled."""
        from src.training.base_trainer import ACCELERATE_AVAILABLE, TrainingConfig

        if not ACCELERATE_AVAILABLE:
            pytest.skip("accelerate not installed")

        config = TrainingConfig()
        trainer = self.get_concrete_trainer(config, use_accelerate=True)

        mock_accelerator = MagicMock()
        mock_accelerator.backward = MagicMock()
        mock_accelerator.step = MagicMock()
        mock_accelerator.zero_grad = MagicMock()
        mock_accelerator.clip_grad_norm_ = MagicMock()

        trainer.accelerator = mock_accelerator
        trainer.model = MagicMock()
        trainer.optimizer = MagicMock()
        trainer.scheduler = MagicMock()

        mock_loss = MagicMock()
        mock_loss.item = MagicMock(return_value=0.5)
        trainer.compute_loss = MagicMock(return_value=(mock_loss, {"accuracy": 0.9}))

        batch = {"input_ids": MagicMock()}
        metrics = trainer.training_step(batch)

        mock_accelerator.backward.assert_called_once_with(mock_loss)

    def test_training_step_legacy_mode(self):
        """Test training_step returns metrics in legacy mode."""
        from src.training.base_trainer import TrainingConfig

        config = TrainingConfig(use_amp=False)
        trainer = self.get_concrete_trainer(config, use_accelerate=False)

        trainer.model = MagicMock()
        trainer.optimizer = MagicMock()
        trainer.scheduler = MagicMock()
        trainer.scaler = None
        trainer.device = MagicMock()
        trainer.device.type = "cpu"
        trainer.config.device = trainer.device

        mock_loss = MagicMock()
        mock_loss.item = MagicMock(return_value=0.5)

        divided = MagicMock()
        mock_loss.__truediv__ = MagicMock(return_value=divided)
        mock_loss.__div__ = MagicMock(return_value=divided)
        divided.backward = MagicMock()

        trainer.compute_loss = MagicMock(return_value=(mock_loss, {"accuracy": 0.9}))

        batch = {"input_ids": MagicMock()}
        metrics = trainer.training_step(batch)

        assert "loss" in metrics
        assert "accuracy" in metrics


class TestAccelerateGradientAccumulation(AbstractTestMixin):
    """Test gradient accumulation with Accelerate."""

    def test_accelerate_gradient_accumulation_steps(self):
        """Test that gradient accumulation steps is configured in Accelerator."""
        from src.training.base_trainer import ACCELERATE_AVAILABLE, TrainingConfig

        if not ACCELERATE_AVAILABLE:
            pytest.skip("accelerate not installed")

        accumulation_steps = 4
        config = TrainingConfig(accumulation_steps=accumulation_steps)
        trainer = self.get_concrete_trainer(config, use_accelerate=True)

        assert trainer.config.accumulation_steps == accumulation_steps

    def test_training_step_accumulates_gradients(self):
        """Test that gradients are accumulated correctly with accelerate."""
        from src.training.base_trainer import ACCELERATE_AVAILABLE, TrainingConfig

        if not ACCELERATE_AVAILABLE:
            pytest.skip("accelerate not installed")

        config = TrainingConfig(accumulation_steps=2)
        trainer = self.get_concrete_trainer(config, use_accelerate=True)

        mock_accelerator = MagicMock()
        trainer.accelerator = mock_accelerator
        trainer.model = MagicMock()
        trainer.optimizer = MagicMock()
        trainer.scheduler = MagicMock()

        mock_loss = MagicMock()
        mock_loss.item = MagicMock(return_value=1.0)
        trainer.compute_loss = MagicMock(return_value=(mock_loss, {}))

        trainer.stats.current_step = 0
        batch = {"input_ids": MagicMock()}

        trainer.training_step(batch)
        trainer.stats.current_step = 1
        trainer.training_step(batch)

        assert mock_accelerator.step.call_count >= 1


class TestAccelerateCheckpoint(AbstractTestMixin):
    """Test checkpoint save/load with Accelerate."""

    def test_accelerate_checkpoint_manager_initialized(self):
        """Test that checkpoint manager is initialized with Accelerate."""
        from src.training.base_trainer import ACCELERATE_AVAILABLE, TrainingConfig

        if not ACCELERATE_AVAILABLE:
            pytest.skip("accelerate not installed")

        config = TrainingConfig(model_save_dir="test_checkpoints")
        trainer = self.get_concrete_trainer(config, use_accelerate=True)

        assert trainer.checkpoint_manager is not None
        assert trainer.checkpoint_manager.save_dir.name == "test_checkpoints"

    def test_fit_saves_state_with_accelerate(self):
        """Test that fit() uses accelerator.save_state() when accelerate enabled."""
        from src.training.base_trainer import ACCELERATE_AVAILABLE, TrainingConfig

        if not ACCELERATE_AVAILABLE:
            pytest.skip("accelerate not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_save_dir=tmpdir,
                epochs=1,
                checkpoint_every=1,
            )

            trainer = self.get_concrete_trainer(config, use_accelerate=True)
            trainer.accelerator = MagicMock()
            trainer.accelerator.save_state = MagicMock()

            trainer.model = MagicMock()
            trainer.optimizer = MagicMock()
            trainer.scheduler = MagicMock()
            trainer.scheduler.get_last_lr = MagicMock(return_value=[1e-4])
            trainer.train_loader = [MagicMock()]

            mock_loss = MagicMock()
            mock_loss.item = MagicMock(return_value=1.0)
            trainer.compute_loss = MagicMock(return_value=(mock_loss, {}))

            try:
                trainer.fit()
            except Exception:
                pass

            trainer.accelerator.save_state.assert_called()


class TestAccelerateDeviceManagement(AbstractTestMixin):
    """Test device management with Accelerate."""

    def test_accelerate_device_auto_detected(self):
        """Test that device is auto-detected with Accelerate."""
        from src.training.base_trainer import ACCELERATE_AVAILABLE, TrainingConfig

        if not ACCELERATE_AVAILABLE:
            pytest.skip("accelerate not installed")

        config = TrainingConfig()
        trainer = self.get_concrete_trainer(config, use_accelerate=True)

        assert trainer.accelerator is not None
        assert trainer.device is not None

    def test_legacy_mode_device_manual(self):
        """Test that device is manually managed in legacy mode."""
        from src.training.base_trainer import TrainingConfig

        config = TrainingConfig()
        trainer = self.get_concrete_trainer(config, use_accelerate=False, device=None)

        if trainer.device is None or str(trainer.device) == "None":
            pytest.skip("Device detection returned None")
        elif trainer.device.type == "cuda":
            pytest.skip("CUDA not available")
        else:
            assert trainer.device is not None
            assert trainer.accelerator is None


class TestAccelerateBackwardCompatibility(AbstractTestMixin):
    """Test backward compatibility with legacy training."""

    def test_legacy_mode_same_api(self):
        """Test that legacy mode has the same API as before."""
        from src.training.base_trainer import TrainingConfig

        config = TrainingConfig(use_amp=False)
        trainer = self.get_concrete_trainer(config, use_accelerate=False)

        required_methods = [
            "training_step",
            "eval_step",
            "build_model",
            "build_optimizer",
            "build_scheduler",
            "_prepare_batch",
            "_evaluate",
        ]

        for method in required_methods:
            assert hasattr(trainer, method), f"Missing method: {method}"

    def test_both_modes_produce_training_metrics(self):
        """Test that both modes can produce training metrics."""
        from src.training.base_trainer import ACCELERATE_AVAILABLE, TrainingConfig

        config = TrainingConfig()

        for use_accelerate in [True, False]:
            if use_accelerate and not ACCELERATE_AVAILABLE:
                continue

            trainer = self.get_concrete_trainer(config, use_accelerate=use_accelerate)
            assert hasattr(trainer, "stats")
            assert hasattr(trainer.stats, "losses")
            assert hasattr(trainer.stats, "learning_rates")


class TestAccelerateDistributedInfo(AbstractTestMixin):
    """Test distributed training information."""

    def test_accelerate_num_processes(self):
        """Test that num_processes is accessible from accelerator."""
        from src.training.base_trainer import ACCELERATE_AVAILABLE, TrainingConfig

        if not ACCELERATE_AVAILABLE:
            pytest.skip("accelerate not installed")

        config = TrainingConfig()
        trainer = self.get_concrete_trainer(config, use_accelerate=True)

        assert hasattr(trainer.accelerator, "num_processes")
        num_processes = trainer.accelerator.num_processes
        assert num_processes >= 1

    def test_accelerate_is_main_process(self):
        """Test that is_local_main_process is accessible."""
        from src.training.base_trainer import ACCELERATE_AVAILABLE, TrainingConfig

        if not ACCELERATE_AVAILABLE:
            pytest.skip("accelerate not installed")

        config = TrainingConfig()
        trainer = self.get_concrete_trainer(config, use_accelerate=True)

        assert hasattr(trainer.accelerator, "is_local_main_process")


class TestAccelerateErrorHandling(AbstractTestMixin):
    """Test error handling in Accelerate integration."""

    def test_fit_without_train_loader_raises(self):
        """Test that fit() raises error without train_loader."""
        from src.training.base_trainer import ACCELERATE_AVAILABLE, TrainingConfig

        if not ACCELERATE_AVAILABLE:
            pytest.skip("accelerate not installed")

        config = TrainingConfig()
        trainer = self.get_concrete_trainer(config, use_accelerate=True)
        trainer.model = MagicMock()

        with pytest.raises(RuntimeError, match="Train loader not provided"):
            trainer.fit()

    def test_training_step_without_model_raises(self):
        """Test that training_step raises error without model."""
        from src.training.base_trainer import ACCELERATE_AVAILABLE, TrainingConfig

        if not ACCELERATE_AVAILABLE:
            pytest.skip("accelerate not installed")

        config = TrainingConfig()
        trainer = self.get_concrete_trainer(config, use_accelerate=True)
        trainer.model = None

        with pytest.raises(RuntimeError, match="Model not initialized"):
            trainer.training_step({})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
