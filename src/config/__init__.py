"""Configuration system for Lingmao Moyun training.

This module provides configuration schemas and validation utilities
for different training modes: pretraining, SFT, and reinforcement learning.
"""

import os
import sys
from pathlib import Path

LOG_DIR = os.environ.get("LOG_DIR", "logs")
LOG_SUBDIR = os.environ.get("LOG_SUBDIR", "train_model")
LOG_FORMAT_CONSOLE = "%(message)s"
LOG_FORMAT_FILE = "%(asctime)s - %(levelname)s - %(message)s"

try:
    from src.config.schema import (
        ExperimentConfig,
        PretrainConfig,
        SFTConfig,
        RLExperimentConfig,
        ModelConfig,
        TrainingConfig,
        DatasetConfig,
        MemoryConfig,
        NightModeConfig,
        TokenizerConfig,
        PathsConfig,
        RLConfig,
    )
except ImportError:
    ExperimentConfig = PretrainConfig = SFTConfig = RLExperimentConfig = None
    ModelConfig = TrainingConfig = DatasetConfig = MemoryConfig = None
    NightModeConfig = TokenizerConfig = PathsConfig = RLConfig = None

try:
    from src.config.validator import (
        ConfigValidator,
        ConfigValidationError,
        load_and_validate_config,
        create_config_template,
    )
except ImportError:
    ConfigValidator = ConfigValidationError = None
    load_and_validate_config = create_config_template = None

try:
    from src.config import (
        VERSION,
        DEFAULT_D_MODEL,
        DEFAULT_NHEAD,
        DEFAULT_NUM_LAYERS,
        DEFAULT_DIM_FEEDFORWARD,
        DEFAULT_DROPOUT,
        DEFAULT_MAX_LEN,
        DEFAULT_VOCAB_SIZE,
        DEFAULT_CONTEXT_LENGTH,
        DEFAULT_BATCH_SIZE,
        DEFAULT_LEARNING_RATE,
        DEFAULT_EPOCHS,
        DEFAULT_ACCUMULATION_STEPS,
        DEFAULT_MAX_GRAD_NORM,
        DEFAULT_WEIGHT_DECAY,
        DEFAULT_CHECKPOINT_EVERY,
        DEFAULT_STRIDE,
        DEFAULT_MAX_CHUNKS,
        MEMMAP_THRESHOLD,
        BATCH_LOAD_SIZE,
        SAMPLE_ARRAY_INIT_SIZE,
        MAX_CACHE_SIZE,
        MEMORY_MAP_SUFFIX,
        NIGHT_MODE_START_HOUR,
        NIGHT_MODE_END_HOUR,
        NIGHT_MODE_BATCH_DIVISOR,
        NIGHT_MODE_GPU_MEMORY_FRAC,
        DEFAULT_GPU_MEMORY_FRAC,
        WARMUP_STEPS,
        TOTAL_TRAINING_STEPS,
        MIN_LR_RATIO,
        WARMUP_RATIO,
        USE_FUSED_ADAMW,
        USE_GRADIENT_CHECKPOINTING,
        GRADIENT_CHECKPOINTING_CHUNKS,
        CPU_OFFLOAD,
        LABEL_SMOOTHING,
        USE_FLASH_ATTENTION,
        NUM_KV_HEADS,
        HEAD_DIM,
        USE_CHECKPOINT,
        USE_COMPILE,
        USE_WEIGHT_TYING,
        USE_SLIDING_WINDOW,
        SWA_WINDOW_SIZE,
        USE_MLA,
        MLA_LATENT_DIM,
        MLA_NUM_LATENT_HEADS,
        USE_BPE_TOKENIZER,
        BPE_VOCAB_SIZE,
        BPE_MODEL_TYPE,
        CHAR_COVERAGE,
        BYTE_FALLBACK,
        EXCLUDED_DIRS,
        EXCLUDED_EXTENSIONS,
        PROTECTED_PATTERNS,
        FORCE_EXIT_DELAY_SECONDS,
        TOKENIZER_CACHE_MAXSIZE,
        SPECIAL_TOKENS,
        DEFAULT_NIGHT_MODE_CORES_DIVISOR,
        CHECKPOINT_NAME_PREFIX,
        BEST_MODEL_NAME,
        FINAL_MODEL_NAME_PREFIX,
        DEFAULT_TOKENIZER_PATH,
        MODEL_SAVE_DIR,
        SECONDS_PER_MINUTE,
        SECONDS_PER_HOUR,
        KB,
        MB,
        GB,
    )
except ImportError:
    VERSION = "0.8.5"
    DEFAULT_D_MODEL = 768
    DEFAULT_NHEAD = 12
    DEFAULT_NUM_LAYERS = 12
    DEFAULT_DIM_FEEDFORWARD = 3072
    DEFAULT_DROPOUT = 0.1
    DEFAULT_MAX_LEN = 1024
    DEFAULT_VOCAB_SIZE = 30000
    DEFAULT_CONTEXT_LENGTH = 512
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_LEARNING_RATE = 5e-5
    DEFAULT_EPOCHS = 10
    DEFAULT_ACCUMULATION_STEPS = 4
    DEFAULT_MAX_GRAD_NORM = 1.0
    DEFAULT_WEIGHT_DECAY = 0.01
    DEFAULT_CHECKPOINT_EVERY = 1
    DEFAULT_STRIDE = 256
    DEFAULT_MAX_CHUNKS = None
    MEMMAP_THRESHOLD = 10_000_000
    BATCH_LOAD_SIZE = 100
    SAMPLE_ARRAY_INIT_SIZE = 1000
    MAX_CACHE_SIZE = 10000
    MEMORY_MAP_SUFFIX = ".dat"
    NIGHT_MODE_START_HOUR = 21
    NIGHT_MODE_END_HOUR = 8
    NIGHT_MODE_BATCH_DIVISOR = 2
    NIGHT_MODE_GPU_MEMORY_FRAC = 0.50
    DEFAULT_GPU_MEMORY_FRAC = 0.90
    WARMUP_STEPS = 1000
    TOTAL_TRAINING_STEPS = 10000
    MIN_LR_RATIO = 0.1
    WARMUP_RATIO = 0.1
    USE_FUSED_ADAMW = True
    USE_GRADIENT_CHECKPOINTING = False
    GRADIENT_CHECKPOINTING_CHUNKS = 1
    GRADIENT_CHECKPOINTING_RATIO = 1.0
    CPU_OFFLOAD = False
    LABEL_SMOOTHING = 0.1
    USE_FLASH_ATTENTION = True
    NUM_KV_HEADS = None
    HEAD_DIM = 64
    USE_CHECKPOINT = False
    USE_COMPILE = False
    USE_WEIGHT_TYING = True
    USE_SLIDING_WINDOW = False
    SWA_WINDOW_SIZE = 512
    USE_MLA = False
    MLA_LATENT_DIM = 128
    MLA_NUM_LATENT_HEADS = 4
    USE_BPE_TOKENIZER = False
    BPE_VOCAB_SIZE = 32768
    BPE_MODEL_TYPE = "unigram"
    CHAR_COVERAGE = 1.0
    BYTE_FALLBACK = False
    EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".tox", "venv", ".venv", "node_modules"}
    EXCLUDED_EXTENSIONS = {".pyc", ".pyo", ".so", ".dylib", ".woff", ".woff2", ".ttf", ".otf"}
    PROTECTED_PATTERNS = {".git", "__pycache__"}
    FORCE_EXIT_DELAY_SECONDS = 10.0
    TOKENIZER_CACHE_MAXSIZE = 16384
    SPECIAL_TOKENS = {"<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<cls>", "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[CLS]", "[MASK]"}
    DEFAULT_NIGHT_MODE_CORES_DIVISOR = 2
    CHECKPOINT_NAME_PREFIX = "checkpoint_epoch"
    BEST_MODEL_NAME = "best_model"
    FINAL_MODEL_NAME_PREFIX = "final_model"
    DEFAULT_TOKENIZER_PATH = "tokenizer.json"
    MODEL_SAVE_DIR = "model_weights"
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 3600
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB

__all__ = [
    "ExperimentConfig",
    "PretrainConfig",
    "SFTConfig",
    "RLExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "DatasetConfig",
    "MemoryConfig",
    "NightModeConfig",
    "TokenizerConfig",
    "PathsConfig",
    "RLConfig",
    "ConfigValidator",
    "ConfigValidationError",
    "load_and_validate_config",
    "create_config_template",
    "LOG_DIR",
    "LOG_SUBDIR",
    "LOG_FORMAT_CONSOLE",
    "LOG_FORMAT_FILE",
    "VERSION",
    "DEFAULT_D_MODEL",
    "DEFAULT_NHEAD",
    "DEFAULT_NUM_LAYERS",
    "DEFAULT_DIM_FEEDFORWARD",
    "DEFAULT_DROPOUT",
    "DEFAULT_MAX_LEN",
    "DEFAULT_VOCAB_SIZE",
    "DEFAULT_CONTEXT_LENGTH",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_EPOCHS",
    "DEFAULT_ACCUMULATION_STEPS",
    "DEFAULT_MAX_GRAD_NORM",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_CHECKPOINT_EVERY",
    "DEFAULT_STRIDE",
    "DEFAULT_MAX_CHUNKS",
    "MEMMAP_THRESHOLD",
    "BATCH_LOAD_SIZE",
    "SAMPLE_ARRAY_INIT_SIZE",
    "MAX_CACHE_SIZE",
    "MEMORY_MAP_SUFFIX",
    "NIGHT_MODE_START_HOUR",
    "NIGHT_MODE_END_HOUR",
    "NIGHT_MODE_BATCH_DIVISOR",
    "NIGHT_MODE_GPU_MEMORY_FRAC",
    "DEFAULT_GPU_MEMORY_FRAC",
    "WARMUP_STEPS",
    "TOTAL_TRAINING_STEPS",
    "MIN_LR_RATIO",
    "WARMUP_RATIO",
    "USE_FUSED_ADAMW",
    "USE_GRADIENT_CHECKPOINTING",
    "GRADIENT_CHECKPOINTING_CHUNKS",
    "GRADIENT_CHECKPOINTING_RATIO",
    "CPU_OFFLOAD",
    "LABEL_SMOOTHING",
    "USE_FLASH_ATTENTION",
    "NUM_KV_HEADS",
    "HEAD_DIM",
    "USE_CHECKPOINT",
    "USE_COMPILE",
    "USE_WEIGHT_TYING",
    "USE_SLIDING_WINDOW",
    "SWA_WINDOW_SIZE",
    "USE_MLA",
    "MLA_LATENT_DIM",
    "MLA_NUM_LATENT_HEADS",
    "USE_BPE_TOKENIZER",
    "BPE_VOCAB_SIZE",
    "BPE_MODEL_TYPE",
    "CHAR_COVERAGE",
    "BYTE_FALLBACK",
    "EXCLUDED_DIRS",
    "EXCLUDED_EXTENSIONS",
    "PROTECTED_PATTERNS",
    "FORCE_EXIT_DELAY_SECONDS",
    "TOKENIZER_CACHE_MAXSIZE",
    "SPECIAL_TOKENS",
    "DEFAULT_NIGHT_MODE_CORES_DIVISOR",
    "CHECKPOINT_NAME_PREFIX",
    "BEST_MODEL_NAME",
    "FINAL_MODEL_NAME_PREFIX",
    "DEFAULT_TOKENIZER_PATH",
    "MODEL_SAVE_DIR",
    "SECONDS_PER_MINUTE",
    "SECONDS_PER_HOUR",
    "KB",
    "MB",
    "GB",
]
