"""Configuration schemas for Lingmao Moyun training system.

This module defines Pydantic models for validating and structuring configuration
data across different training modes: pretraining, SFT, and RL.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Model architecture configuration."""
    d_model: int = Field(default=768, ge=128, le=8192, description="Model hidden dimension")
    nhead: int = Field(default=12, ge=1, le=64, description="Number of attention heads")
    num_layers: int = Field(default=12, ge=1, le=128, description="Number of transformer layers")
    dim_feedforward: int = Field(default=3072, ge=512, le=32768, description="FFN hidden dimension")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="Dropout rate")
    max_len: int = Field(default=1024, ge=64, le=8192, description="Maximum sequence length")
    vocab_size: int = Field(default=50000, ge=1000, le=200000, description="Vocabulary size")
    num_kv_heads: Optional[int] = Field(default=None, description="Number of KV heads for GQA")
    head_dim: int = Field(default=64, ge=32, le=128, description="Attention head dimension")
    use_weight_tying: bool = Field(default=True, description="Tie embedding and output weights")
    use_sliding_window: bool = Field(default=False, description="Enable sliding window attention")
    swa_window_size: int = Field(default=512, ge=64, le=4096, description="SWA window size")
    use_mla: bool = Field(default=False, description="Enable Multi-Head Latent Attention")
    mla_latent_dim: int = Field(default=128, ge=32, le=512, description="MLA latent dimension")
    mla_num_latent_heads: int = Field(default=4, ge=1, le=16, description="MLA latent heads")
    use_moe: bool = Field(default=False, description="Enable Mixture of Experts")
    num_experts: int = Field(default=8, ge=2, le=64, description="Number of MoE experts")
    top_k: int = Field(default=2, ge=1, le=8, description="Top-k routing for MoE")
    mode: str = Field(default="modern", description="Model mode: 'modern' or 'legacy'")

    @field_validator('num_kv_heads')
    @classmethod
    def validate_kv_heads(cls, v, info):
        if v is not None and v.nhead > info.data['nhead']:
            raise ValueError(f"num_kv_heads ({v}) cannot exceed nhead ({info.data['nhead']})")
        return v


class TrainingConfig(BaseModel):
    """Training hyperparameters configuration."""
    context_length: int = Field(default=512, ge=64, le=8192, description="Context/window length")
    batch_size: int = Field(default=8, ge=1, le=512, description="Training batch size")
    learning_rate: float = Field(default=5e-5, ge=1e-7, le=0.1, description="Initial learning rate")
    epochs: int = Field(default=10, ge=1, le=1000, description="Number of training epochs")
    accumulation_steps: int = Field(default=4, ge=1, le=128, description="Gradient accumulation steps")
    max_grad_norm: float = Field(default=1.0, ge=0.1, le=10.0, description="Gradient clipping norm")
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0, description="L2 regularization")
    checkpoint_every: int = Field(default=1, ge=1, le=100, description="Checkpoint frequency (epochs)")
    warmup_steps: int = Field(default=1000, ge=0, le=100000, description="LR warmup steps")
    total_training_steps: int = Field(default=10000, ge=100, le=10000000, description="Total training steps")
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0, description="Min LR as ratio of peak")
    label_smoothing: float = Field(default=0.1, ge=0.0, le=1.0, description="Label smoothing factor")
    use_amp: bool = Field(default=True, description="Enable automatic mixed precision")
    use_compile: bool = Field(default=False, description="Enable torch.compile")
    use_gradient_checkpointing: bool = Field(default=False, description="Enable gradient checkpointing")
    gradient_checkpointing_ratio: float = Field(default=1.0, ge=0.0, le=1.0, description="Ratio of layers to apply checkpointing")
    log_stats_interval: int = Field(default=100, ge=1, le=10000, description="Stats logging interval")


class DatasetConfig(BaseModel):
    """Dataset and data processing configuration."""
    stride: int = Field(default=256, ge=1, le=2048, description="Sliding window stride")
    max_chunks: Optional[int] = Field(default=None, description="Maximum chunks to process")
    memmap_threshold: int = Field(default=10000000, ge=0, description="Memmap threshold (tokens)")
    batch_load_size: int = Field(default=100, ge=1, le=10000, description="Batch load size for tokenization")
    sample_array_init_size: int = Field(default=1000, ge=100, le=100000, description="Sample array init size")

    train_file: Optional[str] = Field(default=None, description="Training data file path")
    test_file: Optional[str] = Field(default=None, description="Test data file path")
    val_file: Optional[str] = Field(default=None, description="Validation data file path")
    data_format: str = Field(default="text", description="Data format: 'text', 'json', 'jsonl'")


class MemoryConfig(BaseModel):
    """Memory and performance configuration."""
    max_cache_size: int = Field(default=10000, ge=100, description="Tokenizer cache size")
    memory_map_suffix: str = Field(default=".dat", description="Memmap file suffix")
    gpu_memory_frac: float = Field(default=0.90, ge=0.1, le=1.0, description="GPU memory fraction")
    cpu_offload: bool = Field(default=False, description="Offload model to CPU")
    use_fused_adamw: bool = Field(default=True, description="Use fused AdamW optimizer")


class NightModeConfig(BaseModel):
    """Night mode power-saving configuration."""
    enabled: bool = Field(default=False, description="Enable night mode")
    start_hour: int = Field(default=21, ge=0, le=23, description="Night mode start hour")
    end_hour: int = Field(default=8, ge=0, le=23, description="Night mode end hour")
    batch_divisor: int = Field(default=2, ge=1, le=8, description="Batch size divisor")
    gpu_memory_frac: float = Field(default=0.50, ge=0.1, le=1.0, description="Night GPU memory fraction")
    cpu_cores_divisor: int = Field(default=2, ge=1, le=16, description="CPU cores divisor")


class TokenizerConfig(BaseModel):
    """Tokenizer configuration."""
    path: str = Field(default="tokenizer.json", description="Tokenizer file path")
    cache_maxsize: int = Field(default=16384, ge=1000, description="Tokenizer cache max size")
    vocab_size: int = Field(default=16000, ge=1000, le=100000, description="Tokenizer vocabulary size")
    special_tokens: List[str] = Field(
        default=["<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<cls>"],
        description="Special tokens list"
    )
    use_bpe: bool = Field(default=False, description="Use BPE tokenizer")
    bpe_vocab_size: int = Field(default=32768, ge=1000, le=100000, description="BPE vocabulary size")
    bpe_model_type: str = Field(default="unigram", description="BPE model type")
    char_coverage: float = Field(default=1.0, ge=0.5, le=1.0, description="Character coverage")
    byte_fallback: bool = Field(default=False, description="Enable byte-level fallback")


class PathsConfig(BaseModel):
    """Paths configuration."""
    log_dir: str = Field(default="logs", description="Log directory")
    log_subdir: str = Field(default="train_model", description="Log subdirectory")
    model_save_dir: str = Field(default="model_weights", description="Model save directory")
    checkpoint_name_prefix: str = Field(default="checkpoint_epoch", description="Checkpoint name prefix")
    best_model_name: str = Field(default="best_model", description="Best model filename")
    final_model_name_prefix: str = Field(default="final_model", description="Final model prefix")
    tokenizer_path: str = Field(default="tokenizer.json", description="Tokenizer path")


class RLConfig(BaseModel):
    """Reinforcement Learning specific configuration."""
    reward_model_path: Optional[str] = Field(default=None, description="Reward model path")
    reference_model_path: Optional[str] = Field(default=None, description="Reference model for KL divergence")
    reward_scale: float = Field(default=1.0, ge=0.1, le=10.0, description="Reward scaling factor")
    kl_coef: float = Field(default=0.1, ge=0.0, le=1.0, description="KL divergence coefficient")
    gamma: float = Field(default=0.99, ge=0.9, le=1.0, description="Discount factor")
    lam: float = Field(default=0.95, ge=0.8, le=1.0, description="GAE lambda")
    clip_range: float = Field(default=0.2, ge=0.01, le=0.5, description="PPO clip range")
    entropy_coef: float = Field(default=0.01, ge=0.0, le=0.1, description="Entropy bonus coefficient")
    value_loss_coef: float = Field(default=0.5, ge=0.0, le=1.0, description="Value loss coefficient")
    ppo_epochs: int = Field(default=4, ge=1, le=100, description="PPO update epochs")
    mini_batch_size: int = Field(default=4, ge=1, le=128, description="Mini batch size for PPO")


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    version: str = Field(default="0.8.5", description="Configuration version")
    experiment_name: Optional[str] = Field(default=None, description="Experiment name for logging")
    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    tags: List[str] = Field(default_factory=list, description="Experiment tags")

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    night_mode: NightModeConfig = Field(default_factory=NightModeConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    rl: Optional[RLConfig] = Field(default=None, description="RL-specific config (for RL training)")

    def to_trainer_kwargs(self) -> Dict[str, Any]:
        """Convert experiment config to keyword arguments for trainer."""
        kwargs = {
            "context_length": self.training.context_length,
            "batch_size": self.training.batch_size,
            "learning_rate": self.training.learning_rate,
            "epochs": self.training.epochs,
            "accumulation_steps": self.training.accumulation_steps,
            "max_grad_norm": self.training.max_grad_norm,
            "weight_decay": self.training.weight_decay,
            "checkpoint_every": self.training.checkpoint_every,
            "use_amp": self.training.use_amp,
            "use_compile": self.training.use_compile,
            "use_gradient_checkpointing": self.training.use_gradient_checkpointing,
            "gradient_checkpointing_ratio": self.training.gradient_checkpointing_ratio,
            "log_stats_interval": self.training.log_stats_interval,
            "train_file": self.dataset.train_file,
            "test_file": self.dataset.test_file,
            "tokenizer_path": self.tokenizer.path,
            "model_save_dir": self.paths.model_save_dir,
            "night_mode": self.night_mode.enabled,
        }

        if self.model.num_kv_heads is not None:
            kwargs["num_kv_heads"] = self.model.num_kv_heads
        if self.model.use_sliding_window:
            kwargs["use_sliding_window"] = True
            kwargs["window_size"] = self.model.swa_window_size
        if self.model.use_moe:
            kwargs["use_moe"] = True
            kwargs["num_experts"] = self.model.num_experts
            kwargs["top_k"] = self.model.top_k

        return kwargs

    def to_model_kwargs(self) -> Dict[str, Any]:
        """Convert model config to keyword arguments for SimpleTransformer."""
        return {
            "vocab_size": self.model.vocab_size,
            "d_model": self.model.d_model,
            "nhead": self.model.nhead,
            "num_layers": self.model.num_layers,
            "dim_feedforward": self.model.dim_feedforward,
            "dropout": self.model.dropout,
            "num_kv_heads": self.model.num_kv_heads,
            "use_weight_tying": self.model.use_weight_tying,
            "use_sliding_window": self.model.use_sliding_window,
            "window_size": self.model.swa_window_size,
            "mode": self.model.mode,
            "use_moe": self.model.use_moe,
            "num_experts": self.model.num_experts,
            "top_k": self.model.top_k,
        }


class PretrainConfig(ExperimentConfig):
    """Pretraining specific configuration with defaults optimized for pretraining."""
    model: ModelConfig = Field(default_factory=lambda: ModelConfig(
        d_model=768,
        nhead=12,
        num_layers=12,
        dropout=0.1,
    ))
    training: TrainingConfig = Field(default_factory=lambda: TrainingConfig(
        context_length=512,
        batch_size=8,
        learning_rate=5e-5,
        epochs=10,
        accumulation_steps=4,
        warmup_steps=1000,
        total_training_steps=10000,
    ))


class SFTConfig(ExperimentConfig):
    """Supervised Fine-Tuning specific configuration with defaults optimized for SFT."""
    model: ModelConfig = Field(default_factory=lambda: ModelConfig(
        d_model=512,
        nhead=8,
        num_layers=6,
        dropout=0.05,
    ))
    training: TrainingConfig = Field(default_factory=lambda: TrainingConfig(
        context_length=256,
        batch_size=16,
        learning_rate=2e-5,
        epochs=3,
        accumulation_steps=2,
        warmup_steps=100,
        total_training_steps=1000,
        label_smoothing=0.05,
    ))


class RLExperimentConfig(ExperimentConfig):
    """Reinforcement Learning experiment configuration."""
    rl: RLConfig = Field(default_factory=RLConfig)
