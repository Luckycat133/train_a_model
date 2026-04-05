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
DEFAULT_VOCAB_SIZE = 50000

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
