"""Restricted loading for the official Demucs model repository."""

from __future__ import annotations

import math
import typing as tp
from fractions import Fraction

from packaging.version import Version

MIN_SAFE_TORCH_VERSION = Version("2.6")


def _require_safe_torch(torch: tp.Any) -> None:
    installed = Version(str(torch.__version__).split("+", 1)[0])
    if installed < MIN_SAFE_TORCH_VERSION:
        raise RuntimeError(
            "PyTorch 2.6 or newer is required to safely load Demucs checkpoints."
        )


def _safe_globals() -> list[tp.Any]:
    """Return the narrow compatibility allowlist used by official Demucs files."""
    import numpy as np
    import numpy._core.multiarray as np_core_multiarray
    from demucs.demucs import Demucs
    from demucs.hdemucs import HDemucs
    from demucs.htdemucs import HTDemucs

    allowed: list[tp.Any] = [
        Demucs,
        HDemucs,
        HTDemucs,
        Fraction,
        np.dtype,
        (np_core_multiarray.scalar, "numpy.core.multiarray.scalar"),
        (np_core_multiarray.scalar, "numpy._core.multiarray.scalar"),
    ]

    # NumPy dtype classes are constructed dynamically and therefore are not
    # reported by get_unsafe_globals_in_checkpoint(). PyTorch documents that
    # they must be allowlisted explicitly for weights-only loading.
    dtype_specs = (
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    )
    allowed.extend({type(np.dtype(spec)) for spec in dtype_specs})

    # Quantized official MDX packages reference this class. It is optional in
    # Demucs, so leave it unavailable unless the user installed diffq.
    try:
        from diffq.diffq import DiffQuantizer
    except ImportError:
        pass
    else:
        allowed.append(DiffQuantizer)

    return allowed


def _validate_constructor_value(value: tp.Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int, Fraction)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite constructor value at {path}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_constructor_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"Constructor mapping at {path} must have string keys")
        for key, item in value.items():
            _validate_constructor_value(item, f"{path}.{key}")
        return
    raise ValueError(f"Unsupported constructor value at {path}: {type(value).__name__}")


def _validate_package(package: tp.Any, torch: tp.Any) -> dict[str, tp.Any]:
    from demucs.demucs import Demucs
    from demucs.hdemucs import HDemucs
    from demucs.htdemucs import HTDemucs

    if not isinstance(package, dict):
        raise ValueError(f"Unexpected Demucs checkpoint type: {type(package).__name__}")

    required = {"klass", "args", "kwargs", "state"}
    missing = required.difference(package)
    if missing:
        raise ValueError(f"Demucs checkpoint is missing required keys: {sorted(missing)}")

    if not any(package["klass"] is model_class for model_class in (Demucs, HDemucs, HTDemucs)):
        raise ValueError(f"Unsupported Demucs model class: {package['klass']!r}")
    if not isinstance(package["args"], (list, tuple)):
        raise ValueError("Demucs checkpoint args must be a list or tuple")
    if not isinstance(package["kwargs"], dict):
        raise ValueError("Demucs checkpoint kwargs must be a dictionary")

    _validate_constructor_value(package["args"], "args")
    _validate_constructor_value(package["kwargs"], "kwargs")

    state = package["state"]
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("Demucs checkpoint state must be a string-keyed dictionary")
    if state.get("__quantized") is not True:
        invalid = [key for key, value in state.items() if not isinstance(value, torch.Tensor)]
        if invalid:
            raise ValueError(
                "Demucs checkpoint state contains non-tensor values: "
                f"{invalid[:5]}"
            )

    return package


def _load_package_from_url(url: str, torch: tp.Any) -> dict[str, tp.Any]:
    _require_safe_torch(torch)
    try:
        with torch.serialization.safe_globals(_safe_globals()):
            package = torch.hub.load_state_dict_from_url(
                url,
                map_location="cpu",
                check_hash=True,
                weights_only=True,
            )
    except Exception as exc:
        raise RuntimeError(
            f"Refusing to load Demucs checkpoint {url!r} with unrestricted deserialization"
        ) from exc
    return _validate_package(package, torch)


def get_restricted_demucs_model(model_name: str) -> tp.Any:
    """Load an official Demucs model without unrestricted pickle execution."""
    try:
        import torch
        from demucs.pretrained import REMOTE_ROOT, _parse_remote_files
        from demucs.repo import AnyModelRepo, BagOnlyRepo, ModelLoadingError, ModelOnlyRepo
        from demucs.states import load_model
    except ImportError:
        raise ImportError(
            "Model conversion requires the [convert] extras. "
            "Install with: pip install 'mlx-audio-separator[convert]'"
        ) from None

    _require_safe_torch(torch)

    class RestrictedRemoteRepo(ModelOnlyRepo):
        def __init__(self, models: dict[str, str]):
            self._models = models

        def has_model(self, signature: str) -> bool:
            return signature in self._models

        def get_model(self, signature: str) -> tp.Any:
            try:
                url = self._models[signature]
            except KeyError:
                raise ModelLoadingError(
                    f"Could not find a pre-trained model with signature {signature}."
                ) from None
            package = _load_package_from_url(url, torch)
            model = load_model(package)
            model.eval()
            return model

    models = _parse_remote_files(REMOTE_ROOT / "files.txt")
    model_repo = RestrictedRemoteRepo(models)
    bag_repo = BagOnlyRepo(REMOTE_ROOT, model_repo)
    model = AnyModelRepo(model_repo, bag_repo).get_model(model_name)
    model.eval()
    return model
