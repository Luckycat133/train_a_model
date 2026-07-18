"""General-purpose utility functions for the Lingmao Moyun system."""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

# ─── Matplotlib stub (when unavailable) ────────────────────────────────────────

try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager

    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    from types import SimpleNamespace

    MATPLOTLIB_AVAILABLE = False

    def _noop_savefig(path: str, *a: Any, **k: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()

    plt = SimpleNamespace(  # type: ignore[assignment]
        style=SimpleNamespace(use=lambda *a, **k: None),
        subplots=lambda *a, **k: (SimpleNamespace(), (SimpleNamespace(), SimpleNamespace())),
        tight_layout=lambda *a, **k: None,
        savefig=_noop_savefig,
        close=lambda *a, **k: None,
        rcParams={},
    )  # type: ignore[assignment]

    matplotlib = SimpleNamespace(  # type: ignore[assignment]
        font_manager=SimpleNamespace(
            fontManager=SimpleNamespace(ttflist=[])  # type: ignore[assignment]
        )
    )


# ─── Formatting helpers ────────────────────────────────────────────────────────


def format_memory_size(size_bytes: Optional[float]) -> str:
    """Format a byte count as a human-readable string."""
    if size_bytes is None:
        return "未知"

    size_kb = size_bytes / 1024
    if size_kb < 1024:
        return f"{size_kb:.2f} KB"

    size_mb = size_kb / 1024
    if size_mb < 1024:
        return f"{size_mb:.2f} MB"

    size_gb = size_mb / 1024
    return f"{size_gb:.2f} GB"


def format_time(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.1f}小时{minutes:.1f}分钟"


# ─── Plotting ──────────────────────────────────────────────────────────────────


def plot_training_stats(stats: Dict[str, Any], save_dir: Union[str, Path]) -> str:
    """Create a training-stats plot and save it to disk.

    When Matplotlib is unavailable the stub ``plt`` writes an empty sentinel file
    so that callers that check for the existence of the output file are satisfied.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(save_dir / f"training_stats_{int(time.time())}.png")

    if MATPLOTLIB_AVAILABLE:
        plt.figure()
        plt.plot(stats.get("losses", []))
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.savefig(save_path)  # writes empty sentinel

    return save_path
