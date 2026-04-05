#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backward-compatibility shim for the old monolithic ``train_model.py``.

All logic lives in ``src/`` now.  This file re-exports the public API so that
existing importers (``test_*.py``, ``generate.py``, ``processor.py``,
``cleanup.py`` …) keep working without changes.
"""

# ── Core training components ───────────────────────────────────────────────────
from src.trainer import train_model

# ── Dataset ────────────────────────────────────────────────────────────────────
from src.dataset import LMDataset

# ── Model architecture ─────────────────────────────────────────────────────────
from src.model import PositionalEncoding, SimpleTransformer

# ── Utilities ───────────────────────────────────────────────────────────────────
from src.logger import get_logger
from src.utils import format_memory_size, format_time, plot_training_stats

# ── Version constant ───────────────────────────────────────────────────────────
from src.config import VERSION

__all__ = [
    "LMDataset",
    "PositionalEncoding",
    "SimpleTransformer",
    "format_memory_size",
    "format_time",
    "get_logger",
    "plot_training_stats",
    "train_model",
    "VERSION",
]
