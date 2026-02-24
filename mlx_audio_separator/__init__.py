try:
    from importlib.metadata import version

    __version__ = version("mlx-audio-separator")
except Exception:
    __version__ = "0.1.1-dev"

from .core import Separator  # noqa: F401
