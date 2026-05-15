"""Base trainer interface for the Lingmao Moyun training system.

2025-2026 Best Practice: This module integrates HuggingFace Accelerate for
simplified distributed training. Key features:
- Automatic multi-GPU/TPU handling via Accelerator
- Mixed precision (bf16/fp16) with automatic device management
- Simplified gradient checkpointing and accumulation
- No manual rank/sync判断 needed
- Zero code changes when switching single/multi-GPU

Usage:
    trainer = BaseTrainer(config, use_accelerate=True)  # Recommended (default)
    trainer = BaseTrainer(config, use_accelerate=False)  # Legacy mode

Accelerate Benefits:
    - 4-5 lines to enable distributed training
    - Automatic gradient scaling and device placement
    - Simplified checkpoint saving/loading across ranks
"""

from __future__ import annotations

import glob
import math
import os
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar, Union

try:
    import torch
    import torch.nn as nn
    from torch import autocast
    from torch.amp import GradScaler
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader
except Exception:
    torch = None
    nn = None
    GradScaler = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **k):  # type: ignore[misc]
        return it

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
    WARMUP_STEPS = 1000
    TOTAL_TRAINING_STEPS = 10000

try:
    from src.logger import get_logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    def get_logger(name):
        return logging.getLogger(name)

try:
    from typing import TypeVar
    if nn is not None:
        T = TypeVar("T", bound=nn.Module)
    else:
        T = TypeVar("T")
except ImportError:
    T = TypeVar("T")

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    Accelerator = None
    ACCELERATE_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Unified configuration for all trainers.

    Attributes:
        context_length: Maximum sequence length for training.
        batch_size: Training batch size per device.
        learning_rate: Peak learning rate.
        weight_decay: L2 regularization strength.
        epochs: Number of training epochs.
        accumulation_steps: Gradient accumulation steps.
        max_grad_norm: Maximum gradient norm for clipping.
        warmup_steps: Learning rate warmup steps.
        total_steps: Total training steps (for scheduler).
        min_lr_ratio: Minimum LR as ratio of peak LR.
        checkpoint_every: Save checkpoint every N epochs.
        log_interval: Log training stats every N steps.
        eval_interval: Run evaluation every N steps.
        early_stopping_patience: Stop if no improvement for N evaluations.
        early_stopping_threshold: Minimum improvement threshold.
        use_amp: Use automatic mixed precision training.
        amp_dtype: AMP dtype ('bf16' or 'fp16').
        seed: Random seed for reproducibility.
        model_save_dir: Directory to save checkpoints.
        resume_from: Checkpoint path to resume from.
        auto_resume: Automatically find and resume latest checkpoint.
    """

    context_length: int = DEFAULT_CONTEXT_LENGTH
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    epochs: int = DEFAULT_EPOCHS
    accumulation_steps: int = DEFAULT_ACCUMULATION_STEPS
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    warmup_steps: int = WARMUP_STEPS
    total_steps: int = TOTAL_TRAINING_STEPS
    min_lr_ratio: float = MIN_LR_RATIO
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    log_interval: int = 100
    eval_interval: int = 500
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    use_amp: bool = True
    amp_dtype: str = "bf16"
    seed: int = 42
    model_save_dir: str = MODEL_SAVE_DIR
    resume_from: Optional[str] = None
    auto_resume: bool = True

    d_model: int = DEFAULT_D_MODEL
    nhead: int = DEFAULT_NHEAD
    num_layers: int = DEFAULT_NUM_LAYERS
    dim_feedforward: int = DEFAULT_DIM_FEEDFORWARD
    dropout: float = DEFAULT_DROPOUT
    vocab_size: int = DEFAULT_VOCAB_SIZE
    max_len: int = 4096

    mode: str = "modern"
    num_kv_heads: Optional[int] = None
    use_moe: bool = False
    num_experts: int = 8
    top_k: int = 2
    use_gradient_checkpointing: bool = False
    gradient_checkpointing_ratio: float = 1.0
    use_weight_tying: bool = True
    use_sliding_window: bool = False
    window_size: int = 512

    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    compile_backend: str = "inductor"
    compile_dynamic: bool = True

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary, supporting nested keys."""
        flat_config: Dict[str, Any] = {}

        def flatten(d: Dict[str, Any], prefix: str = "") -> None:
            for key, value in d.items():
                new_key = f"{prefix}_{key}" if prefix else key
                if isinstance(value, dict):
                    flatten(value, new_key)
                else:
                    flat_config[new_key] = value

        flatten(config)

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        valid_config = {k: v for k, v in flat_config.items() if k in field_names}
        return cls(**valid_config)


@dataclass
class TrainingStats:
    """Container for tracking training statistics."""

    losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    eval_losses: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    best_loss: float = float("inf")
    current_epoch: int = 0
    current_step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "losses": self.losses,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
            "eval_losses": self.eval_losses,
            "steps": self.steps,
            "best_loss": self.best_loss,
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
        }


class CheckpointManager:
    """Manages checkpoint save/load operations."""

    def __init__(self, save_dir: str, version: str = "0.8.5"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.version = version

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer],
        scheduler: Optional[LambdaLR],
        scaler: Optional[GradScaler],
        stats: TrainingStats,
        epoch: int,
        step: int,
        extra_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> Path:
        """Save a training checkpoint."""
        name = f"checkpoint_epoch_{epoch + 1}_step_{step}"
        path = self.save_dir / f"{name}.pt"

        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "stats": stats.to_dict(),
            "best_loss": stats.best_loss,
            "version": self.version,
        }

        if extra_state:
            checkpoint.update(extra_state)

        torch.save(checkpoint, path, weights_only=True)

        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path, weights_only=True)

        return path

    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LambdaLR] = None,
        device: Optional[torch.device] = None,
    ) -> tuple[int, int, float, Dict[str, Any]]:
        """Load a checkpoint and return (epoch, step, best_loss, extra_state)."""
        if device is None:
            device = torch.device("cpu")

        ckpt = torch.load(path, map_location=device, weights_only=True)

        model.load_state_dict(ckpt["model_state_dict"])

        if optimizer and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Could not restore optimizer state: {e}")

        if scheduler and "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass

        extra_state = {k: v for k, v in ckpt.items() if k not in [
            "epoch", "step", "model_state_dict", "optimizer_state_dict",
            "scheduler_state_dict", "scaler_state_dict", "stats", "best_loss", "version"
        ]}

        return ckpt.get("epoch", 0), ckpt.get("step", 0), ckpt.get("best_loss", float("inf")), extra_state

    def find_latest(self) -> Optional[str]:
        """Find the most recent checkpoint in save directory."""
        pattern = str(self.save_dir / "checkpoint_epoch_*.pt")
        files = glob.glob(pattern)
        if not files:
            return None
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]


class BaseTrainer(ABC):
    """Abstract base class for all trainers.

    All training paradigms (pretraining, SFT, RL) must inherit from this class
    and implement the abstract methods. Provides common functionality including:
    - Checkpoint management
    - Gradient accumulation
    - Mixed precision training
    - Early stopping
    - Signal handling for graceful shutdown
    - Logging and monitoring
    """

    TERMINATE_TRAINING = False
    SAVE_CHECKPOINT_SIGNAL = False

    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        eval_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        use_accelerate: bool = True,
    ):
        """Initialize BaseTrainer.

        Args:
            config: Training configuration.
            model: Pre-built model (optional, can be built in fit()).
            train_loader: Training data loader.
            eval_loader: Evaluation data loader.
            device: Target device (auto-detected if None).
            use_accelerate: Use HuggingFace Accelerate for distributed training.
                Default True. Set False for legacy manual device management.

        2025-2026 Best Practice:
            use_accelerate=True enables:
            - Automatic multi-GPU/TPU handling
            - Mixed precision without manual scaler management
            - Simplified gradient accumulation
            - No manual .to(device) or rank判断 needed
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.use_accelerate = use_accelerate and ACCELERATE_AVAILABLE
        self.accelerator: Optional[Accelerator] = None

        if self.use_accelerate:
            amp_dtype = "bf16" if config.amp_dtype == "bf16" else "fp16"
            self.accelerator = Accelerator(
                mixed_precision=amp_dtype if config.use_amp else "no",
                gradient_accumulation_steps=config.accumulation_steps,
                device_placement=False,
            )
            self.device = self.accelerator.device
        else:
            self.device = device or self._get_device()

        self.stats = TrainingStats()
        self.checkpoint_manager = CheckpointManager(
            config.model_save_dir,
            version="0.8.5"
        )

        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[LambdaLR] = None
        self.scaler: Optional[GradScaler] = None
        self.criterion: Optional[nn.Module] = None

        self._setup_signal_handlers()
        self._setup_metrics()

        mode_str = "Accelerate" if self.use_accelerate else "Legacy"
        logger.info(f"Trainer initialized ({mode_str}) on device: {self.device}")

    def _get_device(self) -> torch.device:
        """Auto-detect the best available device."""
        if torch is None:
            return None

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _setup_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        def handler(sig: int, frame) -> None:
            BaseTrainer.TERMINATE_TRAINING = True
            BaseTrainer.SAVE_CHECKPOINT_SIGNAL = True
            logger.warning("Termination signal received - will save checkpoint and exit")

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _setup_metrics(self) -> None:
        """Initialize training metrics tracking."""
        self.metrics_history: List[Dict[str, float]] = []

    @abstractmethod
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute the training loss for a batch.

        Must be implemented by subclasses.

        Args:
            batch: A batch of training data.
            **kwargs: Additional arguments specific to the training paradigm.

        Returns:
            Tuple of (loss tensor, metrics dictionary).
        """
        pass

    def training_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Execute a single training step.

        2025-2026 Best Practice:
            When use_accelerate=True:
                loss, metrics = self.compute_loss(batch)
                accelerator.backward(loss)  # 自动处理混合精度和梯度缩放
                accelerator.step(self.optimizer)
                accelerator.zero_grad()

            When use_accelerate=False (Legacy):
                使用手动autocast + GradScaler管理
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        if self.use_accelerate:
            loss, metrics = self.compute_loss(batch)
            self.accelerator.backward(loss)

            if (self.stats.current_step + 1) % self.config.accumulation_steps == 0:
                if self.config.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.accelerator.step(self.optimizer)
                self.accelerator.zero_grad()
                if self.scheduler:
                    self.scheduler.step()

            return {"loss": loss.item(), **metrics}
        else:
            use_amp = self.config.use_amp and self.device.type == "cuda"
            amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16

            with autocast(enabled=use_amp, device_type=self.device.type, dtype=amp_dtype):
                loss, metrics = self.compute_loss(batch)

            loss = loss / self.config.accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            step_metrics = {"loss": loss.item() * self.config.accumulation_steps, **metrics}

            if (self.stats.current_step + 1) % self.config.accumulation_steps == 0:
                if self.config.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()

            return step_metrics

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single evaluation step."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        self.model.eval()
        use_amp = self.config.use_amp and self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16

        with torch.no_grad():
            with autocast(enabled=use_amp, device_type=self.device.type, dtype=amp_dtype):
                loss, metrics = self.compute_loss(batch)

        self.model.train()
        return {"loss": loss.item(), **metrics}

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build the model architecture.

        Must be implemented by subclasses.
        """
        pass

    def build_optimizer(self) -> Optimizer:
        """Build the optimizer with proper weight decay grouping."""
        if self.model is None:
            raise RuntimeError("Model must be built first")

        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "norm" in name or ".norm" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            fused=self.device.type == "cuda",
        )

    def build_scheduler(self) -> LambdaLR:
        """Build the learning rate scheduler with warmup and cosine decay."""
        def lr_lambda(current_step: int) -> float:
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            progress = float(current_step - self.config.warmup_steps) / float(
                max(1, self.config.total_steps - self.config.warmup_steps)
            )
            return max(self.config.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def build_scaler(self) -> Optional[GradScaler]:
        """Build the gradient scaler for mixed precision training."""
        if not self.config.use_amp or self.device.type != "cuda":
            return None

        if torch.cuda.is_bf16_supported():
            return GradScaler('cuda', bf16=True)
        return GradScaler('cuda', bf16=False)

    def fit(self) -> nn.Module:
        """Main training loop with Accelerate integration.

        2025-2026 Best Practice: When use_accelerate=True (default):
            - 4-5 lines enable distributed training
            - accelerator.prepare() handles device placement
            - accelerator.backward(loss) for gradient scaling
            - No manual rank判断 needed
            - Automatic mixed precision via Accelerator
        """
        if self.model is None:
            self.model = self.build_model()

        self.optimizer = self.build_optimizer()

        if self.use_accelerate:
            self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader
            )
            if self.eval_loader is not None:
                self.eval_loader = self.accelerator.prepare(self.eval_loader)
            self.scaler = None
        else:
            self.model.to(self.device)
            self.scaler = self.build_scaler()

        self.scheduler = self.build_scheduler()

        self._apply_compile()
        self._apply_gradient_checkpointing()

        self._before_training()

        start_epoch = 0
        if self.config.resume_from:
            if os.path.exists(self.config.resume_from):
                if self.use_accelerate:
                    self.accelerator.load_state(self.config.resume_from)
                    logger.info(f"Resumed from {self.config.resume_from}")
                else:
                    epoch, step, best_loss, _ = self.checkpoint_manager.load(
                        self.config.resume_from, self.model, self.optimizer,
                        self.scheduler, self.device
                    )
                    start_epoch = epoch + 1
                    self.stats.best_loss = best_loss
                    self.stats.current_step = step
                    logger.info(f"Resumed from epoch {epoch}, step {step}")
        elif self.config.auto_resume:
            latest = self.checkpoint_manager.find_latest()
            if latest:
                try:
                    if self.use_accelerate:
                        self.accelerator.load_state(latest)
                        logger.info(f"Auto-resumed from {latest}")
                    else:
                        epoch, step, best_loss, _ = self.checkpoint_manager.load(
                            latest, self.model, self.optimizer,
                            self.scheduler, self.device
                        )
                        start_epoch = epoch + 1
                        self.stats.best_loss = best_loss
                        self.stats.current_step = step
                        logger.info(f"Auto-resumed from {latest}")
                except Exception as e:
                    logger.warning(f"Could not load checkpoint: {e}")

        if self.train_loader is None:
            raise RuntimeError("Train loader not provided")

        best_eval_loss = float("inf")
        patience_counter = 0

        for epoch in range(start_epoch, self.config.epochs):
            if BaseTrainer.TERMINATE_TRAINING:
                logger.warning("Terminating training early")
                break

            self.stats.current_epoch = epoch
            epoch_start = time.time()

            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.epochs}",
                ncols=100,
            )

            for batch_idx, batch in enumerate(pbar):
                if BaseTrainer.TERMINATE_TRAINING:
                    break

                batch = self._prepare_batch(batch)
                metrics = self.training_step(batch)

                epoch_loss += metrics.get("loss", 0)
                num_batches += 1
                self.stats.current_step += 1

                avg_loss = epoch_loss / max(num_batches, 1)
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

                if self.stats.current_step % self.config.log_interval == 0:
                    self.stats.losses.append(avg_loss)
                    self.stats.steps.append(self.stats.current_step)
                    self.stats.learning_rates.append(self.scheduler.get_last_lr()[0])

                    self._log_metrics(metrics, step_type="train")

                if (self.eval_loader and
                    self.stats.current_step % self.config.eval_interval == 0):
                    eval_metrics = self._evaluate()
                    eval_loss = eval_metrics.get("loss", float("inf"))

                    if eval_loss < self.stats.best_loss:
                        self.stats.best_loss = eval_loss
                        is_best = True
                        patience_counter = 0
                    else:
                        is_best = False
                        patience_counter += 1

                    self._on_eval_complete(eval_metrics, is_best)

                    if (self.config.early_stopping_patience > 0 and
                        patience_counter >= self.config.early_stopping_patience):
                        improvement = self.stats.best_loss - eval_loss
                        if abs(improvement) < self.config.early_stopping_threshold:
                            logger.info(
                                f"Early stopping: no improvement for "
                                f"{patience_counter} evaluations"
                            )
                            break

            epoch_time = time.time() - epoch_start
            self.stats.epoch_times.append(epoch_time)

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"loss: {avg_epoch_loss:.4f}, time: {epoch_time:.1f}s"
            )

            if (epoch + 1) % self.config.checkpoint_every == 0 or epoch == self.config.epochs - 1:
                is_best = avg_epoch_loss < best_eval_loss
                if is_best:
                    best_eval_loss = avg_epoch_loss

                if self.use_accelerate:
                    save_dir = self.checkpoint_manager.save_dir
                    self.accelerator.save_state(str(save_dir / f"epoch_{epoch}_step_{self.stats.current_step}"))
                else:
                    self.checkpoint_manager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        stats=self.stats,
                        epoch=epoch,
                        step=self.stats.current_step,
                        is_best=is_best,
                        extra_state=self._get_extra_checkpoint_state(),
                    )

            self._on_epoch_complete(epoch, avg_epoch_loss)

        self._after_training()

        if self.model is not None:
            if self.use_accelerate:
                final_dir = self.checkpoint_manager.save_dir / "final"
                self.accelerator.save_state(str(final_dir))
                logger.info(f"Training complete. Final state saved to {final_dir}")
            else:
                final_path = self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    stats=self.stats,
                    epoch=self.config.epochs - 1,
                    step=self.stats.current_step,
                    is_best=False,
                    extra_state=self._get_extra_checkpoint_state(),
                )
                logger.info(f"Training complete. Final model saved to {final_path}")

        return self.model

    def _prepare_batch(self, batch: Union[Dict[str, torch.Tensor], tuple]) -> Dict[str, torch.Tensor]:
        """Prepare batch for training."""
        if isinstance(batch, dict):
            return {k: v.to(self.device) for k, v in batch.items()}
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            input_ids, target_ids = batch
            return {
                "input_ids": input_ids.to(self.device),
                "target_ids": target_ids.to(self.device),
            }
        else:
            raise ValueError(f"Unknown batch type: {type(batch)}")

    def _evaluate(self) -> Dict[str, float]:
        """Run evaluation on eval_loader."""
        if self.eval_loader is None:
            return {}

        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_loader:
            batch = self._prepare_batch(batch)
            metrics = self.eval_step(batch)
            total_loss += metrics.get("loss", 0)
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss}

    def _log_metrics(self, metrics: Dict[str, float], step_type: str = "train") -> None:
        """Log metrics to logger."""
        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(f"[{step_type}] Step {self.stats.current_step}: {metric_str}")

    def _get_extra_checkpoint_state(self) -> Dict[str, Any]:
        """Return extra state to save in checkpoint. Override in subclasses."""
        return {}

    def _apply_gradient_checkpointing(self) -> None:
        """Configure gradient checkpointing for memory-efficient training.
        
        Gradient checkpointing trades 20-30% more compute for 30-50% less memory.
        It recomputes activations during backward pass instead of storing them.
        """
        if not self.config.use_gradient_checkpointing:
            return
        
        if self.model is None:
            logger.warning("Model not built yet, skipping gradient checkpointing configuration")
            return
        
        if hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()
            logger.info(
                f"Gradient checkpointing enabled with ratio={self.config.gradient_checkpointing_ratio:.1f}"
            )
        else:
            logger.warning("Model does not support gradient checkpointing")

    def _apply_compile(self) -> None:
        """Apply torch.compile() to the model if configured.

        Records compilation time separately from training time.
        Compilation happens lazily on first forward pass.
        """
        if not getattr(self.config, 'compile_model', False):
            return

        if torch is None:
            logger.warning("torch.compile() requires PyTorch 2.0+, skipping")
            return

        compile_mode = getattr(self.config, 'compile_mode', 'reduce-overhead')
        compile_backend = getattr(self.config, 'compile_backend', 'inductor')
        compile_dynamic = getattr(self.config, 'compile_dynamic', True)

        logger.info(f"Applying torch.compile(mode='{compile_mode}', backend='{compile_backend}')")
        compile_start = time.time()

        if hasattr(self.model, 'compile'):
            try:
                self.model.compile(
                    mode=compile_mode,
                    backend=compile_backend,
                    dynamic=compile_dynamic,
                )
                compile_elapsed = time.time() - compile_start
                logger.info(f"Model compilation configured in {compile_elapsed:.2f}s (lazy compilation)")
            except Exception as e:
                logger.warning(f"torch.compile() failed: {e}")
                logger.warning("Training will continue without compilation")
        else:
            logger.warning("Model does not support torch.compile(), skipping")

    def _before_training(self) -> None:
        """Hook called before training starts."""
        pass

    def _after_training(self) -> None:
        """Hook called after training completes."""
        pass

    def _on_epoch_complete(self, epoch: int, epoch_loss: float) -> None:
        """Hook called after each epoch."""
        pass

    def _on_eval_complete(self, eval_metrics: Dict[str, float], is_best: bool) -> None:
        """Hook called after each evaluation."""
        self.stats.eval_losses.append(eval_metrics.get("loss", 0))
        if is_best:
            logger.info(f"New best model! Loss: {eval_metrics.get('loss'):.4f}")


logger = get_logger("LingmaoMoyun.Training")
