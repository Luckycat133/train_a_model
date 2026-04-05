"""Dataset module for the Lingmao Moyun language model training system."""

import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np

# Use lazy import for torch – the module-level fallback in train_model.py handles environments
# where torch is unavailable.
try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - fallback for test environments
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass

from src.config import (
    BATCH_LOAD_SIZE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_STRIDE,
    MAX_CACHE_SIZE,
    MEMMAP_THRESHOLD,
    MEMORY_MAP_SUFFIX,
    SAMPLE_ARRAY_INIT_SIZE,
    SPECIAL_TOKENS,
)
from src.logger import get_logger

logger = get_logger("LingmaoMoyun.Dataset")

# ─── Memory tracking ──────────────────────────────────────────────────────────


def _track_memory() -> float:
    """Return current process RSS in MB, or 0.0 if unavailable."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


# ─── LMDataset ────────────────────────────────────────────────────────────────


class LMDataset(Dataset):
    """Language model dataset with sliding-window sampling.

    Args:
        data_path: Path to a JSONL file or a directory containing JSONL files.
        context_length: Context window size in tokens.
        tokenizer: Tokenizer instance (must implement ``tokenize`` and/or
            ``batch_tokenize``). If ``None``, character-level encoding is used.
        stride: Sliding window step size.
        max_chunks: Maximum number of text chunks to load (memory guard).
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        tokenizer: Optional[Any] = None,
        stride: int = DEFAULT_STRIDE,
        max_chunks: Optional[int] = DEFAULT_MAX_CHUNKS,
    ) -> None:
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.stride = stride
        self.max_chunks = max_chunks

        self.np = np
        self.use_memmap = False
        self.memmap_path: Optional[str] = None

        self.token_chunks: List[str] = []
        self.samples: List[Dict[str, Any]] = []

        self.load_time = 0.0
        self.process_time = 0.0
        self.peak_memory_mb = 0.0
        _track_memory()

        start = time.time()
        self._load_data(data_path)
        self.load_time = time.time() - start

        start = time.time()
        self._create_samples()
        self.process_time = time.time() - start

        self.peak_memory_mb = max(self.peak_memory_mb, _track_memory())
        logger.info(
            f"Dataset {data_path} loaded: {len(self.samples)} samples, "
            f"peak memory {self.peak_memory_mb:.2f} MB"
        )

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_data(self, data_path: Union[str, Path]) -> None:
        """Load JSONL files, respecting ``max_chunks`` as a memory ceiling."""
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")

        if path.is_dir():
            files: Iterable[Path] = path.glob("**/*.jsonl")
            logger.info(f"Loading JSONL files from directory {path}")
        else:
            files = [path]
            logger.info(f"Loading single file: {path}")

        chunks = 0
        for fp in files:
            n = self._process_jsonl_file(fp)
            chunks += n
            logger.info(f"Processed {fp}: extracted {n} chunks")

            # Guard: stop loading if we hit the memory ceiling
            if self.max_chunks is not None and chunks >= self.max_chunks:
                logger.info(f"max_chunks limit ({self.max_chunks}) reached, stopping load")
                break

            # Periodic memory check – abort if we're consuming too much
            current_mb = _track_memory()
            if current_mb > 2048:  # hard cap: 2 GB per process
                logger.warning(
                    f"Memory usage ({current_mb:.0f} MB) exceeded 2 GB limit, "
                    "stopping data load early"
                )
                break

        logger.info(f"Total chunks loaded: {len(self.token_chunks)}")

    def _process_jsonl_file(self, file_path: Path) -> int:
        """Parse one JSONL file and append content strings to ``token_chunks``."""
        new_chunks = 0
        with open(file_path, "r", encoding="utf-8", errors="strict") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON at line {line_idx + 1}")
                    continue

                # Extract text from common field names
                content: Optional[str] = None
                for field in ("content", "text", "body", "paragraphs"):
                    if field in item and item[field]:
                        content = item[field]
                        break

                if content is None and "title" in item:
                    content = item.get("title", "")

                # Flatten lists
                if isinstance(content, list):
                    content = "\n".join(str(p) for p in content if p)

                if not isinstance(content, str) or not content.strip():
                    continue

                self.token_chunks.append(content)
                new_chunks += 1

                if (
                    self.max_chunks is not None
                    and len(self.token_chunks) >= self.max_chunks
                ):
                    break

        return new_chunks

    # ── sample creation ─────────────────────────────────────────────────────

    def _create_samples(self) -> None:
        """Build sliding-window (input, target) samples from token chunks."""
        logger.info("Creating training samples ...")
        t0 = time.time()

        all_tokens: List[int] = []

        # Batch tokenization
        for i in range(0, len(self.token_chunks), BATCH_LOAD_SIZE):
            batch = self.token_chunks[i : i + BATCH_LOAD_SIZE]

            if self.tokenizer is not None:
                if hasattr(self.tokenizer, "batch_tokenize"):
                    token_batches = self.tokenizer.batch_tokenize(batch)
                    for tokens in token_batches:
                        if tokens and len(tokens) > 1:
                            all_tokens.extend(tokens)
                else:
                    for chunk in batch:
                        try:
                            tokens = self.tokenizer.tokenize(chunk)
                            if tokens and len(tokens) > 1:
                                all_tokens.extend(tokens)
                        except Exception as e:
                            logger.warning(f"Tokenization error: {e}")
            else:
                # Character-level fallback
                for chunk in batch:
                    char_ids = [ord(c) % 30000 for c in chunk]
                    if char_ids:
                        all_tokens.extend(char_ids)

        if not all_tokens:
            logger.warning("No tokens generated – skipping sample creation")
            self.samples = []
            return

        if len(all_tokens) < self.context_length:
            logger.warning(
                f"Total tokens ({len(all_tokens)}) < context_length "
                f"({self.context_length}), repeating data"
            )
            repeats = self.context_length // len(all_tokens) + 1
            all_tokens = all_tokens * repeats

        arr: Union[np.ndarray, np.memmap]
        if len(all_tokens) > MEMMAP_THRESHOLD:
            self.use_memmap = True
            fd, self.memmap_path = tempfile.mkstemp(suffix=MEMORY_MAP_SUFFIX)
            os.close(fd)
            arr = np.memmap(
                self.memmap_path,
                dtype=np.int32,
                mode="w+",
                shape=(len(all_tokens),),
            )
            arr[:] = all_tokens
            arr.flush()
            arr = np.memmap(
                self.memmap_path,
                dtype=np.int32,
                mode="r",
                shape=(len(all_tokens),),
            )
            logger.info(f"Using memmap for {len(all_tokens)} tokens")
        else:
            arr = np.array(all_tokens, dtype=np.int32)

        total = max(0, (len(arr) - self.context_length) // self.stride + 1)
        logger.info(f"Will create {total} training samples")

        if total <= 0:
            self.samples = []
            return

        input_arrays: List[np.ndarray] = []
        target_arrays: List[np.ndarray] = []

        for i in range(0, len(arr) - self.context_length, self.stride):
            inp = arr[i : i + self.context_length]
            tgt = arr[i + 1 : i + self.context_length + 1]
            if len(inp) == self.context_length and len(tgt) == self.context_length:
                input_arrays.append(inp)
                target_arrays.append(tgt)

            if len(input_arrays) % SAMPLE_ARRAY_INIT_SIZE == 0:
                self.peak_memory_mb = max(self.peak_memory_mb, _track_memory())

        for inp, tgt in zip(input_arrays, target_arrays):
            self.samples.append({"input_ids": inp, "target_ids": tgt})

        logger.info(
            f"Sample creation done in {time.time() - t0:.2f}s: "
            f"{len(self.samples)} samples"
        )

    # ── torch Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a dict with ``input_ids`` and ``target_ids`` tensors.

        The dict format is the canonical one used throughout the codebase.
        ``__getitem__`` is also compatible with code that unpacks a 2-tuple.
        """
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "target_ids": torch.tensor(sample["target_ids"], dtype=torch.long),
        }

    def __del__(self) -> None:
        """Release memmap temporary file."""
        if self.use_memmap and self.memmap_path and os.path.exists(self.memmap_path):
            try:
                os.unlink(self.memmap_path)
            except Exception:
                pass
