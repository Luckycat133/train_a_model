"""Modular training framework for Lingmao Moyun.

This module provides a unified training system supporting multiple training paradigms:

- **Pretraining**: Causal language modeling (CLM) for foundation model training
- **SFT**: Supervised fine-tuning on instruction-response pairs
- **RL**: Reinforcement learning alignment (DPO/GRPO)

All trainers inherit from the BaseTrainer interface and share common functionality
including checkpoint management, mixed precision training, gradient accumulation,
early stopping, and signal handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.base_trainer import BaseTrainer, TrainingConfig, TrainingStats, CheckpointManager
    from src.training.pretrain import CausalLanguageModelTrainer, CausalLMTrainingConfig, create_pretrain_trainer
    from src.training.sft_trainer import SFTTrainer, SFTConfig, InstructionResponseDataset, create_sft_trainer
    from src.training.rl_trainer import RLTrainer, DPOTrainer, GRPOTrainer, RLConfig, PreferenceDataset, create_rl_trainer

__all__ = [
    "BaseTrainer",
    "TrainingConfig",
    "TrainingStats",
    "CheckpointManager",
    "CausalLanguageModelTrainer",
    "CausalLMTrainingConfig",
    "create_pretrain_trainer",
    "SFTTrainer",
    "SFTConfig",
    "InstructionResponseDataset",
    "create_sft_trainer",
    "RLTrainer",
    "DPOTrainer",
    "GRPOTrainer",
    "RLConfig",
    "PreferenceDataset",
    "create_rl_trainer",
]

__version__ = "0.9.0"
