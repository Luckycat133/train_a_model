"""Pretraining trainer for causal language modeling.

This module implements the standard causal language modeling (CLM) pretraining
paradigm, where the model learns to predict the next token given all previous
tokens. This is the foundational pretraining objective used by GPT, LLaMA, and
most modern autoregressive language models.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except Exception:
    torch = None
    nn = None

try:
    from src.config import (
        DEFAULT_ACCUMULATION_STEPS,
        DEFAULT_BATCH_SIZE,
        DEFAULT_CHECKPOINT_EVERY,
        DEFAULT_CONTEXT_LENGTH,
        DEFAULT_D_MODEL,
        DEFAULT_DIM_FEEDFORWARD,
        DEFAULT_DROPOUT,
        DEFAULT_EPOCHS,
        DEFAULT_LEARNING_RATE,
        DEFAULT_MAX_GRAD_NORM,
        DEFAULT_NHEAD,
        DEFAULT_NUM_LAYERS,
        DEFAULT_VOCAB_SIZE,
        DEFAULT_WEIGHT_DECAY,
        GRADIENT_CHECKPOINTING_CHUNKS,
        LABEL_SMOOTHING,
        LOG_DIR,
        LOG_SUBDIR,
        MIN_LR_RATIO,
        MODEL_SAVE_DIR,
        NUM_KV_HEADS,
        SWA_WINDOW_SIZE,
        USE_GRADIENT_CHECKPOINTING,
        USE_SLIDING_WINDOW,
        USE_WEIGHT_TYING,
        WARMUP_STEPS,
        TOTAL_TRAINING_STEPS,
    )
except ImportError:
    DEFAULT_ACCUMULATION_STEPS = 4
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_CHECKPOINT_EVERY = 1
    DEFAULT_CONTEXT_LENGTH = 512
    DEFAULT_D_MODEL = 768
    DEFAULT_DIM_FEEDFORWARD = 3072
    DEFAULT_DROPOUT = 0.1
    DEFAULT_EPOCHS = 10
    DEFAULT_LEARNING_RATE = 5e-5
    DEFAULT_MAX_GRAD_NORM = 1.0
    DEFAULT_NHEAD = 12
    DEFAULT_NUM_LAYERS = 12
    DEFAULT_VOCAB_SIZE = 30000
    DEFAULT_WEIGHT_DECAY = 0.01
    GRADIENT_CHECKPOINTING_CHUNKS = 1
    LABEL_SMOOTHING = 0.1
    LOG_DIR = "logs"
    LOG_SUBDIR = "train_model"
    MIN_LR_RATIO = 0.1
    MODEL_SAVE_DIR = "model_weights"
    NUM_KV_HEADS = None
    SWA_WINDOW_SIZE = 512
    USE_GRADIENT_CHECKPOINTING = True
    USE_SLIDING_WINDOW = False
    USE_WEIGHT_TYING = True
    WARMUP_STEPS = 1000
    TOTAL_TRAINING_STEPS = 10000

try:
    from src.dataset import LMDataset
except ImportError:
    class LMDataset:
        pass

try:
    from src.logger import get_logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    def get_logger(name):
        return logging.getLogger(name)

try:
    from src.model import SimpleTransformer
except ImportError:
    class SimpleTransformer:
        pass

from .base_trainer import BaseTrainer, TrainingConfig, TrainingStats

logger = get_logger("LingmaoMoyun.Pretrain")


class CausalLMTrainingConfig(TrainingConfig):
    """Configuration specific to causal language model pretraining."""

    tokenizer_path: str = "tokenizer.json"
    train_file: Optional[str] = None
    test_file: Optional[str] = None
    stride: int = DEFAULT_CONTEXT_LENGTH // 2
    max_chunks: Optional[int] = None
    use_compile: bool = False
    label_smoothing: float = LABEL_SMOOTHING


class CausalLanguageModelTrainer(BaseTrainer):
    """Trainer for causal language modeling pretraining.

    Implements standard next-token prediction training where the model learns
    to predict each token given all preceding tokens in the sequence.

    Features:
    - Causal (unidirectional) attention mask
    - Next-token prediction loss
    - Optional label smoothing
    - Gradient accumulation for effective larger batch sizes
    - Mixed precision training (BF16/FP16)
    - Gradient checkpointing for memory efficiency
    - Learning rate warmup + cosine decay schedule
    """

    def __init__(
        self,
        config: Optional[CausalLMTrainingConfig] = None,
        model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        eval_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Any] = None,
    ):
        if config is None:
            config = CausalLMTrainingConfig()

        super().__init__(
            config=config,
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
        )

        self.tokenizer = tokenizer
        self._vocab_size = config.vocab_size

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value: int) -> None:
        """Set the vocabulary size and rebuild model if needed."""
        if value != self._vocab_size:
            self._vocab_size = value
            if self.model is not None:
                logger.info(f"Vocab size changed to {value}, model will be rebuilt")

    def build_model(self) -> nn.Module:
        """Build the causal language model architecture.

        Creates a SimpleTransformer model with modern architecture features
        (RoPE, SwiGLU, optional GQA/MoE) configured via the training config.
        """
        if self.model is not None:
            return self.model

        cfg = self.config
        if isinstance(cfg, CausalLMTrainingConfig):
            train_file = cfg.train_file
        else:
            train_file = getattr(cfg, 'train_file', None)

        if train_file and os.path.exists(cfg.tokenizer_path):
            try:
                from tokenizer import ClassicalTokenizer
                self.tokenizer = ClassicalTokenizer()
                self.tokenizer.load(cfg.tokenizer_path)
                self._vocab_size = len(self.tokenizer.token_to_id)
                logger.info(f"Loaded tokenizer with vocab size: {self._vocab_size}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}, using default vocab size")

        model = SimpleTransformer(
            vocab_size=self._vocab_size,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
            mode=cfg.mode,
            num_kv_heads=cfg.num_kv_heads,
            use_moe=cfg.use_moe,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            use_checkpoint=USE_GRADIENT_CHECKPOINTING,
            use_weight_tying=cfg.use_weight_tying,
            use_sliding_window=cfg.use_sliding_window,
            window_size=cfg.window_size,
        )

        num_params = sum(p.num_el() for p in model.parameters())
        logger.info(f"Model built with {num_params:,} parameters")
        logger.info(f"Model architecture: {cfg.mode} mode")
        if cfg.num_kv_heads and cfg.num_kv_heads < cfg.nhead:
            logger.info(f"  - GQA: {cfg.nhead} Q heads, {cfg.num_kv_heads} KV heads")
        if cfg.use_moe:
            logger.info(f"  - MoE: {cfg.num_experts} experts, top-{cfg.top_k} routing")

        return model

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the causal language modeling loss.

        The loss is computed as the cross-entropy between predicted logits
        and target tokens, with optional label smoothing.

        Args:
            batch: Dictionary containing 'input_ids' and 'target_ids'.

        Returns:
            Tuple of (loss tensor, metrics dictionary).
        """
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]

        outputs, _ = self.model(input_ids)

        if self.criterion is None:
            label_smoothing = getattr(self.config, 'label_smoothing', 0.0)
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        logits_flat = outputs.view(-1, outputs.size(-1))
        targets_flat = target_ids.view(-1)

        loss = self.criterion(logits_flat, targets_flat)

        with torch.no_grad():
            preds = logits_flat.argmax(dim=-1)
            accuracy = (preds == targets_flat).float().mean().item()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy,
        }

        return loss, metrics

    def _get_extra_checkpoint_state(self) -> Dict[str, Any]:
        """Return extra state for checkpointing."""
        return {
            "vocab_size": self._vocab_size,
            "trainer_type": "causal_lm",
        }

    def create_dataloaders(
        self,
        train_file: str,
        test_file: Optional[str] = None,
        batch_size: Optional[int] = None,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create training and evaluation dataloaders.

        Args:
            train_file: Path to training data (JSONL format).
            test_file: Optional path to test/validation data.
            batch_size: Batch size (defaults to config value).
            num_workers: Number of data loading workers.

        Returns:
            Tuple of (train_loader, eval_loader).
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        train_dataset = LMDataset(
            train_file,
            context_length=self.config.context_length,
            tokenizer=self.tokenizer,
            stride=self.config.stride if hasattr(self.config, 'stride') else self.config.context_length // 2,
            max_chunks=getattr(self.config, 'max_chunks', None),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

        eval_loader = None
        if test_file and os.path.exists(test_file):
            eval_dataset = LMDataset(
                test_file,
                context_length=self.config.context_length,
                tokenizer=self.tokenizer,
                stride=self.config.context_length,
                max_chunks=getattr(self.config, 'max_chunks', None),
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(self.device.type == "cuda"),
            )
            logger.info(f"Created eval dataloader with {len(eval_dataset)} samples")

        logger.info(f"Created train dataloader with {len(train_dataset)} samples")
        return train_loader, eval_loader

    @classmethod
    def from_config_file(
        cls,
        config_path: str,
        device: Optional[torch.device] = None,
    ) -> "CausalLanguageModelTrainer":
        """Create a trainer from a YAML configuration file.

        Args:
            config_path: Path to YAML config file.
            device: Target device for training.

        Returns:
            Configured CausalLanguageModelTrainer instance.
        """
        import yaml

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        model_config = config_dict.get("model", {})
        training_config = config_dict.get("training", {})

        full_config = {**model_config, **training_config}

        config = CausalLMTrainingConfig.from_dict(full_config)

        trainer = cls(config=config, device=device)

        if training_config.get("train_file"):
            train_loader, eval_loader = trainer.create_dataloaders(
                train_file=training_config["train_file"],
                test_file=training_config.get("test_file"),
            )
            trainer.train_loader = train_loader
            trainer.eval_loader = eval_loader

        return trainer


def create_pretrain_trainer(
    train_file: str,
    test_file: Optional[str] = None,
    model_save_dir: str = MODEL_SAVE_DIR,
    tokenizer_path: str = "tokenizer.json",
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epochs: int = DEFAULT_EPOCHS,
    d_model: int = DEFAULT_D_MODEL,
    nhead: int = DEFAULT_NHEAD,
    num_layers: int = DEFAULT_NUM_LAYERS,
    dim_feedforward: int = DEFAULT_DIM_FEEDFORWARD,
    dropout: float = DEFAULT_DROPOUT,
    accumulation_steps: int = DEFAULT_ACCUMULATION_STEPS,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    use_amp: bool = True,
    device: Optional[torch.device] = None,
    mode: str = "modern",
    num_kv_heads: Optional[int] = NUM_KV_HEADS,
    use_moe: bool = False,
    num_experts: int = 8,
    top_k: int = 2,
    use_gradient_checkpointing: bool = USE_GRADIENT_CHECKPOINTING,
    use_weight_tying: bool = USE_WEIGHT_TYING,
    use_sliding_window: bool = USE_SLIDING_WINDOW,
    window_size: int = SWA_WINDOW_SIZE,
    **kwargs
) -> CausalLanguageModelTrainer:
    """Factory function to create a preconfigured pretraining trainer.

    This is a convenience function that mirrors the original train_model()
    interface for easy migration from the old training system.
    """
    config = CausalLMTrainingConfig(
        train_file=train_file,
        test_file=test_file,
        model_save_dir=model_save_dir,
        tokenizer_path=tokenizer_path,
        context_length=context_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
        checkpoint_every=checkpoint_every,
        use_amp=use_amp,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        mode=mode,
        num_kv_heads=num_kv_heads,
        use_moe=use_moe,
        num_experts=num_experts,
        top_k=top_k,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_weight_tying=use_weight_tying,
        use_sliding_window=use_sliding_window,
        window_size=window_size,
    )

    trainer = CausalLanguageModelTrainer(
        config=config,
        device=device,
    )

    train_loader, eval_loader = trainer.create_dataloaders(
        train_file=train_file,
        test_file=test_file,
    )
    trainer.train_loader = train_loader
    trainer.eval_loader = eval_loader

    return trainer
