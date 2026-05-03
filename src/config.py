"""Configuration constants for the Lingmao Moyun training system.

All magic numbers are defined here as named constants for maintainability.
"""

# ─── Version ─────────────────────────────────────────────────────────────────
VERSION = "0.8.5"

# ─── Model Architecture ──────────────────────────────────────────────────────
DEFAULT_D_MODEL = 768
DEFAULT_NHEAD = 12
DEFAULT_NUM_LAYERS = 12
DEFAULT_DIM_FEEDFORWARD = 3072
DEFAULT_DROPOUT = 0.1
DEFAULT_MAX_LEN = 1024
DEFAULT_VOCAB_SIZE = 30000

# ─── Training ─────────────────────────────────────────────────────────────────
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_EPOCHS = 10
DEFAULT_ACCUMULATION_STEPS = 4
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_CHECKPOINT_EVERY = 1

# ─── Dataset ──────────────────────────────────────────────────────────────────
DEFAULT_STRIDE = 256
DEFAULT_MAX_CHUNKS = None
MEMMAP_THRESHOLD = 10_000_000  # tokens; use memmap above this
BATCH_LOAD_SIZE = 100  # chunk batch size for tokenization
SAMPLE_ARRAY_INIT_SIZE = 1000  # checkpoint for memory tracking

# ─── Memory & Performance ─────────────────────────────────────────────────────
MAX_CACHE_SIZE = 10000  # tokenizer cache eviction threshold
MEMORY_MAP_SUFFIX = ".dat"

# ─── Night Mode ───────────────────────────────────────────────────────────────
NIGHT_MODE_START_HOUR = 21  # 9 PM
NIGHT_MODE_END_HOUR = 8     # 8 AM
NIGHT_MODE_BATCH_DIVISOR = 2
NIGHT_MODE_GPU_MEMORY_FRAC = 0.50
DEFAULT_GPU_MEMORY_FRAC = 0.90

# ─── LR Schedule ─────────────────────────────────────────────────────────────
WARMUP_STEPS = 1000
TOTAL_TRAINING_STEPS = 10000
MIN_LR_RATIO = 0.1  # Minimum LR as ratio of peak (for cosine annealing)
WARMUP_RATIO = 0.1  # Warmup as ratio of total steps (0.1 = 10% warmup)

# ─── Modern Training Optimizations ───────────────────────────────────────────
USE_FUSED_ADAMW = True  # Use fused AdamW kernel (faster on Ampere+ GPUs)
USE_GRADIENT_CHECKPOINTING = True  # Now enabled by default - improves effective depth
GRADIENT_CHECKPOINTING_CHUNKS = 1  # Number of checkpoint chunks (1 = uniform)
CPU_OFFLOAD = False  # Offload model to CPU when not in use (for large models)
LABEL_SMOOTHING = 0.1  # Standard in modern LLMs (LLaMA-3, Qwen3) for better generation

# ─── Attention & Architecture ────────────────────────────────────────────────
USE_FLASH_ATTENTION = True  # Use Flash Attention via PyTorch SDPA (auto-detected)
NUM_KV_HEADS = None  # None = MHA, < nhead = GQA (modern LLMs use GQA)
HEAD_DIM = 64  # Standard head dimension for modern LLMs
USE_CHECKPOINT = False  # Use activation checkpointing in legacy mode
USE_COMPILE = False  # Use torch.compile() for ~30% speedup on PyTorch 2.0+

# ─── Weight Tying ───────────────────────────────────────────────────────────────
USE_WEIGHT_TYING = True  # Tie embedding and output layer weights (reduces params ~20%)

# ─── Sliding Window Attention ─────────────────────────────────────────────────
USE_SLIDING_WINDOW = False  # Enable sliding window attention (SWA)
SWA_WINDOW_SIZE = 512  # Window size for SWA (must be power of 2 for efficiency)

# ─── Modern Tokenization ──────────────────────────────────────────────────────
USE_BPE_TOKENIZER = False  # Use sentencepiece BPE instead of max-match
BPE_VOCAB_SIZE = 32768  # Standard BPE vocab for Chinese
BPE_MODEL_TYPE = "unigram"  # BPE algorithm: "unigram", "bpe", or "char"
CHAR_COVERAGE = 1.0  # Character coverage for Chinese tokenization
BYTE_FALLBACK = False  # Use byte-level tokenization as fallback

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = "logs"
LOG_SUBDIR = "train_model"
LOG_FORMAT_CONSOLE = "%(message)s"
LOG_FORMAT_FILE = "%(asctime)s - %(levelname)s - %(message)s"

# ─── Cleanup / Scanning ───────────────────────────────────────────────────────
EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".tox", "venv", ".venv", "node_modules"}
EXCLUDED_EXTENSIONS = {".pyc", ".pyo", ".so", ".dylib", ".woff", ".woff2", ".ttf", ".otf"}
PROTECTED_PATTERNS = {".git", "__pycache__"}

# ─── Signal Handling ─────────────────────────────────────────────────────────
FORCE_EXIT_DELAY_SECONDS = 10.0

# ─── Tokenizer ────────────────────────────────────────────────────────────────
TOKENIZER_CACHE_MAXSIZE = 16384
SPECIAL_TOKENS = {"<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<cls>", "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[CLS]", "[MASK]"}

# ─── CPU Limit ────────────────────────────────────────────────────────────────
DEFAULT_NIGHT_MODE_CORES_DIVISOR = 2

# ─── Checkpoint ───────────────────────────────────────────────────────────────
CHECKPOINT_NAME_PREFIX = "checkpoint_epoch"
BEST_MODEL_NAME = "best_model"
FINAL_MODEL_NAME_PREFIX = "final_model"

# ─── Paths ────────────────────────────────────────────────────────────────────
DEFAULT_TOKENIZER_PATH = "tokenizer.json"
MODEL_SAVE_DIR = "model_weights"


# ─── Time Formatting ──────────────────────────────────────────────────────────
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
KB = 1024
MB = 1024 * KB
GB = 1024 * MB
