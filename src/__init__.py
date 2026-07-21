"""Core package for the Lingmao Moyun training experiment.

Heavy NumPy/PyTorch modules are loaded lazily so metadata and CLI help remain
available in lightweight environments.
"""

from src.logger import log_error, log_info, log_success, log_warning, setup_logger

__all__ = [
    "LMDataset",
    "SimpleTransformer",
    "PositionalEncoding",
    "setup_logger",
    "log_info",
    "log_warning",
    "log_error",
    "log_success",
]


def __getattr__(name):
    if name == "LMDataset":
        from src.dataset import LMDataset

        return LMDataset
    if name in {"SimpleTransformer", "PositionalEncoding"}:
        from src.model import PositionalEncoding, SimpleTransformer

        return {
            "SimpleTransformer": SimpleTransformer,
            "PositionalEncoding": PositionalEncoding,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
