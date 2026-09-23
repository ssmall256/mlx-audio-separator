"""Convert PyTorch models to MLX models."""

import logging
import typing as tp
from pathlib import Path

logger = logging.getLogger(__name__)


def get_mlx_cache_dir() -> Path:
    """Get or create the MLX model cache directory."""
    cache_dir = Path.home() / '.cache' / 'demucs-mlx'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


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
