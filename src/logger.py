"""Logging utilities for the Lingmao Moyun training system."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import LOG_DIR, LOG_SUBDIR, LOG_FORMAT_CONSOLE, LOG_FORMAT_FILE

# ─── Module-level logger (lazily initialized) ──────────────────────────────────
_loggers: dict[str, logging.Logger] = {}


def setup_logger(
    name: str = "LingmaoMoyun",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Configure and return a logger with both console and file handlers.

    Args:
        name: Logger name.
        log_file: Custom log file path. If None, auto-generates one in logs/train_model/.
        level: Logging level.
        console: Whether to add a console handler.

    Returns:
        Configured logger instance.
    """
    # Return cached logger if already set up (prevents duplicate handlers)
    if name in _loggers and _loggers[name].handlers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Determine log file path
    if log_file is None:
        log_dir = Path(LOG_DIR) / LOG_SUBDIR
    else:
        log_dir = Path(log_file).parent

    log_dir.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        log_file = str(
            log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

    # File handler – full format with timestamps
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT_FILE))

    # Console handler – simplified format
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(LOG_FORMAT_CONSOLE))

    logger.addHandler(fh)
    if console:
        logger.addHandler(ch)

    _loggers[name] = logger
    return logger


def log_info(message: str, logger_name: str = "LingmaoMoyun") -> None:
    setup_logger(logger_name).info(message)


def log_warning(message: str, logger_name: str = "LingmaoMoyun") -> None:
    setup_logger(logger_name).warning(message)


def log_error(message: str, logger_name: str = "LingmaoMoyun") -> None:
    setup_logger(logger_name).error(message)


def log_success(message: str, logger_name: str = "LingmaoMoyun") -> None:
    setup_logger(logger_name).info(message)


def get_logger(name: str = "LingmaoMoyun") -> logging.Logger:
    """Get or create a logger by name."""
    return setup_logger(name)
