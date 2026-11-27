"""Training loop, evaluation, and checkpoint utilities.

All functions that live in the original monolithic ``train_model.py`` are
collected here. The top-level ``run`` function is in ``src/run.py``.
"""

import argparse
import hashlib
import math
import os
import shutil
import signal
import tempfile
from contextlib import nullcontext
import threading
import time
import psutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.amp import autocast, GradScaler
    from torch.utils.checkpoint import checkpoint as grad_checkpoint
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
except Exception:  # pragma: no cover
    torch = None
    nn = None
    GradScaler = None
    grad_checkpoint = None

    def autocast(*a, **k):  # type: ignore[misc]
        from contextlib import contextmanager

        @contextmanager
        def _null():
            yield

        return _null()

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(it, **k):  # type: ignore[misc]
        return it

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
    DEFAULT_WEIGHT_DECAY,
    FORCE_EXIT_DELAY_SECONDS,
    GRADIENT_CHECKPOINTING_CHUNKS,
    LABEL_SMOOTHING,
    LOG_DIR,
    LOG_SUBDIR,
    MIN_LR_RATIO,
    MODEL_SAVE_DIR,
    NIGHT_MODE_END_HOUR,
    NIGHT_MODE_GPU_MEMORY_FRAC,
    NIGHT_MODE_START_HOUR,
    NUM_KV_HEADS,
    SWA_WINDOW_SIZE,
    TOTAL_TRAINING_STEPS,
    USE_COMPILE,
    USE_FUSED_ADAMW,
    USE_WEIGHT_TYING,
    USE_SLIDING_WINDOW,
    USE_GRADIENT_CHECKPOINTING,
    USE_CHECKPOINT,
    VERSION,
    WARMUP_STEPS,
)
from src.dataset import LMDataset
from src.logger import get_logger
from src.model import PositionalEncoding, SimpleTransformer

logger = get_logger("LingmaoMoyun.Trainer")

# ─── Global signal state ──────────────────────────────────────────────────────
TERMINATE_TRAINING = False
SAVE_CHECKPOINT_SIGNAL = False

# ─── System monitoring functions ──────────────────────────────────────────────

def get_system_memory_info():
    """Get system memory usage information."""
    mem = psutil.virtual_memory()
    return {
        'total': mem.total,
        'used': mem.used,
        'free': mem.free,
        'percent': mem.percent,
        'available': mem.available
    }

def get_gpu_info():
    """Get GPU information if available."""
    if torch is None or not torch.cuda.is_available():
        return None
    
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        device = torch.cuda.device(i)
        props = torch.cuda.get_device_properties(i)
        mem_allocated = torch.cuda.memory_allocated(i)
        mem_reserved = torch.cuda.memory_reserved(i)
        
        gpu_info.append({
            'device_id': i,
            'name': props.name,
            'total_memory': props.total_memory,
            'memory_allocated': mem_allocated,
            'memory_reserved': mem_reserved,
            'memory_free': props.total_memory - mem_reserved,
            'utilization': torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else None
        })
    return gpu_info

def log_system_stats():
    """Log system resource statistics."""
    mem_info = get_system_memory_info()
    gpu_info = get_gpu_info()
    
    stats_log = []
    stats_log.append(f"System Memory: {mem_info['used'] / 1024**3:.2f} GB / {mem_info['total'] / 1024**3:.2f} GB ({mem_info['percent']}%)")
    
    if gpu_info:
        for gpu in gpu_info:
            stats_log.append(
                f"GPU {gpu['device_id']}: {gpu['name']} - "
                f"Allocated: {gpu['memory_allocated'] / 1024**3:.2f} GB / "
                f"Total: {gpu['total_memory'] / 1024**3:.2f} GB"
            )
            if gpu['utilization'] is not None:
                stats_log[-1] += f" | Utilization: {gpu['utilization']}%"
    
    logger.info(" | ".join(stats_log))
    
    return {
        'memory': mem_info,
        'gpu': gpu_info
    }


def _is_night_mode() -> bool:
    """Return True when the local clock is between 21:00 and 08:00."""
    h = datetime.now().hour
    return h >= NIGHT_MODE_START_HOUR or h < NIGHT_MODE_END_HOUR


def _set_cpu_limit(night_mode: bool = True) -> bool:
    """Attempt to restrict the process to a subset of CPU cores."""
    try:
        import psutil

        p = psutil.Process()
        if night_mode:
            limit = max(2, (psutil.cpu_count(logical=False) or 2) // 2)
        else:
            limit = psutil.cpu_count(logical=False) or 2
        p.cpu_affinity(list(range(limit)))
        return True
    except Exception as e:
        logger.warning(f"Cannot set CPU limit: {e}")
        return False


def _limit_gpu_memory(fraction: float) -> bool:
    """Set per-process GPU memory fraction. No-op on non-CUDA devices."""
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        torch.cuda.set_per_process_memory_fraction(fraction)
        return True
    except Exception as e:
        logger.warning(f"Cannot set GPU memory limit: {e}")
        return False


# ─── Signal handling ──────────────────────────────────────────────────────────


def _build_signal_handler() -> None:
    """Install SIGINT/SIGTERM handlers that set flags for main loop to handle."""

    def handler(sig: int, frame) -> None:  # type: ignore[no-untyped-def]
        global TERMINATE_TRAINING, SAVE_CHECKPOINT_SIGNAL
        TERMINATE_TRAINING = True
        SAVE_CHECKPOINT_SIGNAL = True
        logger.warning("Termination signal received; preserving the last completed checkpoint")

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


# ─── Checkpoint utilities ─────────────────────────────────────────────────────


def find_latest_checkpoint(model_save_dir: str) -> Optional[str]:
    """Find the highest numbered completed epoch produced by the trainer."""
    files = [p for p in Path(model_save_dir).glob("model_epoch_*.pt")
             if p.stem.removeprefix("model_epoch_").isdigit()]
    return str(max(files, key=lambda p: int(p.stem.removeprefix("model_epoch_")))) if files else None


def _atomic_torch_save(checkpoint: dict, path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            torch.save(checkpoint, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _resume_fingerprint(tokenizer_path, datasets, parameters) -> str:
    with Path(tokenizer_path).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256")
    digest.update(repr(parameters).encode("utf-8"))
    for dataset in datasets:
        digest.update(str(len(dataset) if dataset is not None else "absent").encode("ascii") + b":")
        if dataset is not None:
            for sample in dataset.samples:
                digest.update(sample["input_ids"].tobytes())
                digest.update(sample["target_ids"].tobytes())
    return digest.hexdigest()


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    scaler: Optional[GradScaler],
    stats: Dict[str, List[Any]],
    save_dir: Union[str, Path],
    *,
    best_loss: Optional[float] = None,
    name: Optional[str] = None,
    is_best: bool = False,
    fingerprint: Optional[str] = None,
) -> Path:
    """Serialize a training checkpoint.

    Args:
        epoch: Current epoch number (0-indexed).
        model, optimizer, scheduler, scaler: Training state objects.
        stats: Dictionary of training statistics (losses, learning rates, …).
        save_dir: Directory to write checkpoints to.
        best_loss: Best validation loss achieved so far.
        name: Checkpoint file stem; defaults to ``checkpoint_epoch_{epoch+1}``.
        is_best: If True, also write ``best_model.pt``.

    Returns:
        Path to the written checkpoint file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = f"checkpoint_epoch_{epoch + 1}"

    checkpoint_path = save_dir / f"{name}.pt"

    checkpoint: Dict[str, Any] = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "stats": stats,
        "resume_fingerprint": fingerprint,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None,
        "best_loss": best_loss if best_loss is not None else float("inf"),
    }

    _atomic_torch_save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved → {checkpoint_path}")

    if is_best:
        best_path = save_dir / "best_model.pt"
        _atomic_torch_save(checkpoint, best_path)
        logger.info(f"Best model saved → {best_path}")

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    device: Optional[torch.device] = None,
    *,
    scaler: Optional[GradScaler] = None,
    stats: Optional[Dict[str, List[Any]]] = None,
    fingerprint: Optional[str] = None,
) -> Tuple[int, float]:
    """Load a checkpoint and restore training state.

    Returns:
        Tuple of (starting_epoch, best_loss).
    """
    if device is None:
        device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if fingerprint is not None and ckpt.get("resume_fingerprint") != fingerprint:
        raise ValueError("Checkpoint is not a compatible completed-epoch training checkpoint")
    if scaler is not None and ckpt.get("scaler_state_dict") is None:
        raise ValueError("Checkpoint has no required AMP scaler state")
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if stats is not None:
        stats.update(ckpt["stats"])
    if fingerprint is not None:
        torch.set_rng_state(ckpt["cpu_rng_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["cuda_rng_state"]])

    return ckpt.get("epoch", 0), ckpt.get("best_loss", float("inf"))


# ─── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> float:
    """Run the model over ``test_loader`` and return average loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating", colour="blue", ncols=100, leave=False)
        for step, batch in enumerate(pbar):
            try:
                input_ids, target_ids = _unpack_batch(batch, device)
            except Exception as e:
                logger.warning(f"Skipping batch {step}: {e}")
                continue

            use_autocast = use_amp and device.type == "cuda"
            eval_dtype = amp_dtype if (use_autocast and amp_dtype) else None
            with autocast(device_type=device.type, dtype=eval_dtype) if use_autocast else nullcontext():
                outputs = model(input_ids)
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)), target_ids.view(-1)
                )

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss / (step + 1):.4f}"})

    return total_loss / max(len(test_loader), 1)


# ─── Batch unpacking helper ───────────────────────────────────────────────────


def _unpack_batch(
    batch: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalise old (dict) and new (tuple) batch formats into (input, target)."""
    if isinstance(batch, dict):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
    elif isinstance(batch, tuple) and len(batch) == 2:
        input_ids, target_ids = batch
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
    else:
        raise ValueError(f"Unknown batch type: {type(batch)}")
    return input_ids, target_ids


# ─── LR schedule ───────────────────────────────────────────────────────────────


def make_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = MIN_LR_RATIO,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Linear warmup → cosine annealing schedule with minimum LR ratio.

    Modern LLMs (LLaMA-3, Qwen3, DeepSeek-V3) use cosine annealing with a
    minimum LR ratio of 0.1-0.2 of the peak LR, plus linear warmup.

    Returns a SequentialLR that first applies linear warmup, then cosine decay.
    """
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6,  # Start near zero for stability
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=optimizer.defaults["lr"] * min_lr_ratio,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    return scheduler


# ─── Main training function ───────────────────────────────────────────────────


def train_model(
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
    use_amp: bool = True,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    accumulation_steps: int = DEFAULT_ACCUMULATION_STEPS,
    device: Optional[torch.device] = None,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    resume_from: Optional[str] = None,
    auto_resume: bool = True,
    night_mode: bool = True,
    use_gradient_checkpointing: bool = USE_GRADIENT_CHECKPOINTING,
    use_checkpoint: bool = USE_CHECKPOINT,
    use_compile: bool = USE_COMPILE,
    use_weight_tying: bool = USE_WEIGHT_TYING,
    use_sliding_window: bool = USE_SLIDING_WINDOW,
    window_size: int = SWA_WINDOW_SIZE,
    mode: str = "modern",  # Added for explicit mode selection
    num_kv_heads: Optional[int] = NUM_KV_HEADS,  # Added for configurable GQA
    use_moe: bool = False,  # Added for MoE support
    num_experts: int = 8,  # Added for MoE configuration
    top_k: int = 2,  # Added for MoE configuration
    log_stats_interval: int = 100,
    seed: Optional[int] = None,
    num_workers: int = 0,
) -> nn.Module:
    """Train a language model end-to-end.

    See ``run.py`` for the CLI entry point.

    New architecture support:
    - Modern mode: RoPE, SwiGLU, GQA, Flash Attention, optional MoE
    - Legacy mode: Original sinusoidal PE + TransformerEncoder
    """
    global TERMINATE_TRAINING, SAVE_CHECKPOINT_SIGNAL
    TERMINATE_TRAINING = SAVE_CHECKPOINT_SIGNAL = False
    if min(batch_size, context_length, epochs, checkpoint_every, accumulation_steps) < 1 or num_workers < 0:
        raise ValueError("Training sizes and checkpoint interval must be positive, workers non-negative")
    if seed is not None:
        torch.manual_seed(seed)

    # ── Device ───────────────────────────────────────────────────────────────
    if device is None:
        if (
            torch is not None
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        elif torch is not None and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    logger.info(f"Training device: {device}")

    # Log initial system stats
    logger.info("Initial system statistics:")
    log_system_stats()

    # ── Night-mode adjustments ───────────────────────────────────────────────
    if night_mode and _is_night_mode():
        logger.info("🌙 Night mode: applying low-power settings")
        _set_cpu_limit(night_mode=True)
        if device.type == "cuda":
            _limit_gpu_memory(NIGHT_MODE_GPU_MEMORY_FRAC)
            orig_bs = batch_size
            batch_size = max(2, batch_size // 2)
            accumulation_steps = max(1, accumulation_steps * orig_bs // batch_size)

    os.makedirs(model_save_dir, exist_ok=True)

    # ── Tokenizer ───────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    from tokenizer import ClassicalTokenizer

    tok = ClassicalTokenizer()
    tok.load(tokenizer_path)
    vocab_size = len(tok.token_to_id)
    logger.info(f"Tokenizer loaded, vocab size: {vocab_size}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info(f"Loading training data: {train_file}")
    train_dataset = LMDataset(
        train_file,
        context_length=context_length,
        tokenizer=tok,
        stride=context_length // 2,
    )
    if len(train_dataset) < batch_size:
        raise ValueError("Training data produces zero complete batches")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type != "cpu"),
        drop_last=True,  # Added for stable batch processing
    )

    test_loader: Optional[DataLoader] = None
    test_dataset: Optional[LMDataset] = None
    if test_file:
        logger.info(f"Loading test data: {test_file}")
        test_dataset = LMDataset(
            test_file,
            context_length=context_length,
            tokenizer=tok,
            stride=context_length,
        )
        if not len(test_dataset):
            raise ValueError("Validation data produces zero batches")
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type != "cpu"),
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(f"Creating model in {mode} mode")
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_checkpoint=use_checkpoint,
        num_kv_heads=num_kv_heads,
        use_weight_tying=use_weight_tying,
        use_sliding_window=use_sliding_window,
        window_size=window_size,
        mode=mode,
        use_moe=use_moe,
        num_experts=num_experts,
        top_k=top_k,
    ).to(device)

    # Log model architecture info
    logger.info(f"Model architecture: {mode} mode")
    if mode == "modern":
        if num_kv_heads and num_kv_heads < nhead:
            logger.info(f"  - GQA enabled: {nhead} query heads, {num_kv_heads} KV heads")
        if use_moe:
            logger.info(f"  - MoE enabled: {num_experts} experts, top-{top_k} routing")
        logger.info(f"  - Flash Attention: {'enabled' if device.type == 'cuda' else 'disabled (CPU)'}")

    # ── torch.compile (PyTorch 2.0+ Graph Compile) ─────────────────────────────
    if use_compile:
        logger.info("Compiling model with torch.compile() for ~30% speedup")
        model = torch.compile(model)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Gradient Checkpointing (memory optimization) ───────────────────────────
    # Trade compute for memory: recompute activations during backward pass
    # Standard practice in modern LLMs (LLaMA, OPT, etc.) for memory efficiency
    if use_gradient_checkpointing and model.mode == "modern":
        logger.info("Enabling gradient checkpointing for memory efficiency")
        model.gradient_checkpointing = True
    elif use_gradient_checkpointing:
        logger.warning("Gradient checkpointing only supported in modern mode")

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    # Modern weight decay: exclude bias and norm parameters from decay
    # This is standard practice in modern LLMs (LLaMA, BERT, etc.)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bias" in name or "norm" in name or ".norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Use fused AdamW when available (faster on Ampere+ GPUs with torch 2.0+)
    if USE_FUSED_ADAMW and device.type == "cuda":
        try:
            # fused=True requires CUDA and is faster
            optimizer = torch.optim.AdamW(
                optimizer_groups,
                lr=learning_rate,
                fused=True,
                foreach=False,  # disable foreach when using fused
            )
            logger.info("Using fused AdamW optimizer")
        except (TypeError, AttributeError):
            # Fallback for older PyTorch versions
            optimizer = torch.optim.AdamW(
                optimizer_groups, lr=learning_rate
            )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_groups, lr=learning_rate
        )

    scheduler = make_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=WARMUP_STEPS,
        total_steps=TOTAL_TRAINING_STEPS,
        min_lr_ratio=MIN_LR_RATIO,
    )

    # Determine AMP dtype: BF16 on supported GPUs (Ampere+), FP16 fallback
    amp_dtype = None
    scaler: Optional[GradScaler] = None
    if use_amp and device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16

        else:
            amp_dtype = torch.float16
            scaler = GradScaler('cuda')

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # ── Stats dict – Initialise with system monitoring ─────────────
    stats: Dict[str, List[Any]] = {
        "losses": [],
        "learning_rates": [],
        "epoch_times": [],
        "test_losses": [],
        "steps": [],
        "optimizer_updates": [],
        "tokens": [],
        "system_memory": [],  # Added for system memory monitoring
        "gpu_usage": [],      # Added for GPU usage monitoring
    }
    if torch is not None and torch.cuda.is_available():
        stats["gpu_memory_usage"] = []

    fingerprint = _resume_fingerprint(tokenizer_path, (train_dataset, test_dataset), (
        context_length, batch_size, num_workers, accumulation_steps, learning_rate, max_grad_norm, weight_decay,
        d_model, nhead, num_layers, dim_feedforward, dropout, device.type, amp_dtype,
        use_checkpoint, use_gradient_checkpointing, use_compile, use_weight_tying, use_sliding_window,
        window_size, mode, num_kv_heads, use_moe, num_experts, top_k,
        LABEL_SMOOTHING, WARMUP_STEPS, TOTAL_TRAINING_STEPS, MIN_LR_RATIO,
    ) + (("modern-uncached-attention-v2",) if mode == "modern" else ()))
    initial_epoch, best_loss = 0, float("inf")
    resume = resume_from or (find_latest_checkpoint(model_save_dir) if auto_resume else None)
    if resume:
        initial_epoch, best_loss = load_checkpoint(
            resume, model, optimizer, scheduler, device, scaler=scaler, stats=stats, fingerprint=fingerprint,
        )
        if initial_epoch >= epochs:
            raise ValueError("Checkpoint already reached the requested total epochs")
        logger.info(f"Restored complete epoch {initial_epoch} from {resume}")
    updates = stats["optimizer_updates"][-1] if stats["optimizer_updates"] else 0
    tokens = stats["tokens"][-1] if stats["tokens"] else 0

    # ── Install signal handler ────────────────────────────────────────────────
    _build_signal_handler()

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info(f"Starting training: {epochs} epochs from epoch {initial_epoch + 1}")

    for epoch in range(initial_epoch, epochs):
        model.train()
        total_loss = 0.0

        # FIX: initialise epoch_start_time BEFORE use (was referenced before assignment)
        epoch_start_time = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            ncols=100,
            leave=True,
            dynamic_ncols=True,
        )

        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            if TERMINATE_TRAINING:
                raise InterruptedError("Training interrupted; only completed-epoch checkpoints can be resumed")
            input_ids, target_ids = _unpack_batch(batch, device)
            tokens += target_ids.numel()
            window_start = step - step % accumulation_steps
            window_size_steps = min(accumulation_steps, len(train_loader) - window_start)
            with autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext():
                outputs, _, aux_loss = model(input_ids, return_aux_loss=True)
                ce_loss = criterion(
                    outputs.view(-1, outputs.size(-1)), target_ids.view(-1)
                )
                loss = (ce_loss + aux_loss) / window_size_steps

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(
                train_loader
            ):
                # Gradient clipping: unscale first when using AMP, then clip
                if max_grad_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm, error_if_nonfinite=scaler is None
                    )

                performed_update = True
                if scaler is not None:
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    performed_update = scaler.get_scale() >= old_scale
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if performed_update:
                    scheduler.step()
                    updates += 1
            total_loss += loss.item() * window_size_steps
            avg_loss = total_loss / max(step + 1, 1)
            cur_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{cur_lr:.8f}"})

            current_step = epoch * len(train_loader) + step
            stats["steps"].append(current_step)
            stats["losses"].append(avg_loss)
            stats["learning_rates"].append(cur_lr)
            stats["optimizer_updates"].append(updates)
            stats["tokens"].append(tokens)

            # Log system stats periodically
            if step % log_stats_interval == 0:
                system_stats = log_system_stats()
                stats["system_memory"].append(system_stats["memory"]["percent"])
                if system_stats["gpu"]:
                    stats["gpu_usage"].append(system_stats["gpu"][0].get("utilization", 0))

            if (
                torch is not None
                and torch.cuda.is_available()
                and "gpu_memory_usage" in stats
            ):
                stats["gpu_memory_usage"].append(torch.cuda.memory_allocated(0))

        # ── Epoch summary ──────────────────────────────────────────────────
        epoch_loss = total_loss / max(len(train_loader), 1)
        epoch_elapsed = time.time() - epoch_start_time
        stats["epoch_times"].append(epoch_elapsed)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} done – loss={epoch_loss:.4f}, "
            f"time={epoch_elapsed:.1f}s, optimizer_updates={updates}, tokens={tokens}"
        )
        
        # Log epoch end system stats
        logger.info("End of epoch system statistics:")
        log_system_stats()

        if TERMINATE_TRAINING:
            raise InterruptedError("Training interrupted; only completed-epoch checkpoints can be resumed")

        is_best = False
        if test_loader is not None:
            test_loss = evaluate_model(
                model, test_loader, criterion, device,
                use_amp=(use_amp and device.type == "cuda"),
                amp_dtype=amp_dtype,
            )
            logger.info(f"Test loss: {test_loss:.4f}")
            stats["test_losses"].append(test_loss)
            if test_loss < best_loss:
                best_loss = test_loss
                is_best = True

        # ── Checkpoint ──────────────────────────────────────────────────────
        if (epoch + 1) % checkpoint_every == 0 or is_best or epoch + 1 == epochs:
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                stats=stats,
                save_dir=model_save_dir,
                best_loss=best_loss,
                name=f"model_epoch_{epoch + 1}",
                is_best=is_best,
                fingerprint=fingerprint,
            )

    # ── Final save ────────────────────────────────────────────────────────────
    save_checkpoint(
        epoch=epochs - 1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        stats=stats,
        save_dir=model_save_dir,
        best_loss=best_loss,
        name=f"final_model_v{VERSION}",
        fingerprint=fingerprint,
    )
    logger.info(f"✅ Training complete. Final model saved.")

    return model
