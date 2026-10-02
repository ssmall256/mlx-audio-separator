"""Where this package keeps its Neural Engine assets, and how to name it to users."""
from __future__ import annotations

from pathlib import Path

#: The pip distribution whose extras install the Core ML runtime and converter.
DISTRIBUTION = "mlx-audio-separator"
#: The module that converts the Core ML assets (``python -m <module> convert``).
ANE_MODULE = "mlx_audio_separator.demucs_mlx.ane"


def ane_cache_dir() -> Path:
    """Converted Core ML assets and the compiled native bridge.

    Kept apart from demucs-mlx's cache, like the converted weights: each
    package validates its assets against its own weight cache.
    """
    return Path.home() / ".cache" / "mlx-audio-separator" / "ane"
