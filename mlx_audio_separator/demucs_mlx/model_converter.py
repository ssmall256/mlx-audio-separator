"""Convert PyTorch models to MLX models."""

import logging
import os
import typing as tp
from pathlib import Path

logger = logging.getLogger(__name__)

#: Where converted Demucs weights are written.
#:
#: This used to be `~/.cache/demucs-mlx`, which is the demucs-mlx package's own
#: directory. Both packages wrote `<model>_config.json` and
#: `<model>.safetensors` there under schemas that reject each other, and both
#: rebuild a cache they cannot read -- so with the two installed side by side
#: (mlx-weights absent, which is its default, since demucs-mlx does not depend
#: on it) every alternating run reconverted, each time needing torch and the
#: upstream checkpoint. Writing somewhere named after this package ends that.
_CACHE_DIR_ENV = "MLX_AUDIO_SEPARATOR_DEMUCS_CACHE_DIR"
_LEGACY_CACHE_DIR = Path.home() / ".cache" / "demucs-mlx"


def get_mlx_cache_dir() -> Path:
    """Get or create the MLX model cache directory."""
    override = os.getenv(_CACHE_DIR_ENV, "").strip()
    cache_dir = (
        Path(override).expanduser()
        if override
        else Path.home() / ".cache" / "mlx-audio-separator" / "demucs"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _legacy_cache_dir() -> tp.Optional[Path]:
    """The pre-0.1.12 location, read from but never written to.

    Existing users keep their converted weights instead of paying for a
    conversion they have already done. A cache demucs-mlx left there is
    rejected as before and simply reconverted into the new directory once.
    """
    if os.getenv(_CACHE_DIR_ENV, "").strip():
        return None
    return _LEGACY_CACHE_DIR if _LEGACY_CACHE_DIR.is_dir() else None


def get_mlx_model(name: str, repo: tp.Optional[Path] = None):
    """
    Get an MLX model, loading from cache or converting from PyTorch if needed.
    """
    from .mlx_convert import SafeCacheError, convert_htdemucs_weights, load_mlx_model

    cache_dir = get_mlx_cache_dir()

    # NOTE: load_mlx_model should handle caching, but we wrap it
    # to ensure we don't accidentally trigger conversion logic on every run.
    try:
        # auto_convert=False ensures we fail fast if not found,
        # allowing us to handle the conversion step explicitly below.
        model = load_mlx_model(name, cache_dir=str(cache_dir), auto_convert=False, verbose=False)
        return model
    except FileNotFoundError:
        legacy = _legacy_cache_dir()
        if legacy is not None:
            try:
                model = load_mlx_model(
                    name, cache_dir=str(legacy), auto_convert=False, verbose=False
                )
                logger.info("Using the cache for '%s' in %s.", name, legacy)
                return model
            except (FileNotFoundError, SafeCacheError) as exc:
                logger.debug("No usable cache for '%s' in %s (%s).", name, legacy, exc)
        # If we are here, the model is missing.
        logger.info("Cache miss for '%s'. Converting from PyTorch...", name)
    except SafeCacheError as exc:
        # A cache that exists but cannot be trusted. Caches written before
        # 0.1.8, and caches written into the same directory by demucs-mlx,
        # lack fields this loader requires -- so without this branch an
        # unreadable cache is a hard failure rather than the self-healing
        # cache miss it should be. Regenerating is also the right response to a
        # digest mismatch: discard the suspect file and rebuild it.
        logger.info(
            "Unusable cache for '%s' (%s). Regenerating...", name, exc
        )

    # This step might take a few seconds but only happens once.
    convert_htdemucs_weights(
        name,
        output_dir=str(cache_dir),
        verify=False,
        verbose=True
    )

    # Load the newly converted model
    logger.info("Loading converted model...")
    return load_mlx_model(name, cache_dir=str(cache_dir), auto_convert=False, verbose=True)
