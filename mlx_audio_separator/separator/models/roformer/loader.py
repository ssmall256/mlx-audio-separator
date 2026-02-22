"""MLX-only Roformer model loader with PyTorch weight conversion."""

import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .bs_roformer import BSRoformerMLX, create_compiled_model
from .mel_band_roformer import MelBandRoformerMLX

logger = logging.getLogger(__name__)


def detect_model_type(model_path: str, config: Dict[str, Any]) -> str:
    """Detect whether a model is bs_roformer or mel_band_roformer."""
    # Check config first
    if "num_bands" in config.get("model", config):
        return "mel_band_roformer"
    if "freqs_per_bands" in config.get("model", config):
        return "bs_roformer"

    # Check filename
    path_lower = model_path.lower()
    if any(s in path_lower for s in ("mel_band_roformer", "mel-band-roformer", "melband")):
        return "mel_band_roformer"
    if any(s in path_lower for s in ("bs_roformer", "bs-roformer", "bsroformer")):
        return "bs_roformer"
    if "roformer" in path_lower:
        return "bs_roformer"

    raise ValueError(f"Cannot determine Roformer model type from path: {model_path}")


def create_bs_roformer_mlx(config: Dict[str, Any]) -> BSRoformerMLX:
    """Create BS-Roformer MLX model from config."""
    os.environ.setdefault("MLX_USE_FAST_SDP", "1")
    os.environ.setdefault("MLX_ENABLE_AMP", "1")
    os.environ.setdefault("MLX_ENABLE_COMPILE", "1")

    model_cfg = config.get("model", config)

    model_args = {
        "dim": model_cfg["dim"],
        "depth": model_cfg["depth"],
        "stereo": model_cfg.get("stereo", False),
        "num_stems": model_cfg.get("num_stems", 2),
        "time_transformer_depth": model_cfg.get("time_transformer_depth", 2),
        "freq_transformer_depth": model_cfg.get("freq_transformer_depth", 2),
        "linear_transformer_depth": model_cfg.get("linear_transformer_depth", 0),
        "freqs_per_bands": tuple(model_cfg["freqs_per_bands"]),
        "dim_head": model_cfg.get("dim_head", 64),
        "heads": model_cfg.get("heads", 8),
        "attn_dropout": 0.0,
        "ff_dropout": 0.0,
        "mlp_expansion_factor": model_cfg.get("mlp_expansion_factor", 4),
        "mask_estimator_depth": model_cfg.get("mask_estimator_depth", 2),
    }

    for key in ("stft_n_fft", "stft_hop_length", "stft_win_length"):
        if key in model_cfg:
            model_args[key] = model_cfg[key]

    logger.debug(f"Creating BSRoformerMLX with args: {list(model_args.keys())}")
    return create_compiled_model(**model_args)


def create_mel_band_roformer_mlx(config: Dict[str, Any]) -> MelBandRoformerMLX:
    """Create MelBand-Roformer MLX model from config."""
    os.environ.setdefault("MLX_USE_FAST_SDP", "1")
    os.environ.setdefault("MLX_ENABLE_AMP", "1")
    os.environ.setdefault("MLX_ENABLE_COMPILE", "1")

    model_cfg = config.get("model", config)

    model_args = {
        "dim": model_cfg["dim"],
        "depth": model_cfg["depth"],
        "stereo": model_cfg.get("stereo", False),
        "num_stems": model_cfg.get("num_stems", 2),
        "time_transformer_depth": model_cfg.get("time_transformer_depth", 2),
        "freq_transformer_depth": model_cfg.get("freq_transformer_depth", 2),
        "linear_transformer_depth": model_cfg.get("linear_transformer_depth", 0),
        "num_bands": model_cfg.get("num_bands", 60),
        "dim_head": model_cfg.get("dim_head", 64),
        "heads": model_cfg.get("heads", 8),
        "attn_dropout": 0.0,
        "ff_dropout": 0.0,
        "mlp_expansion_factor": model_cfg.get("mlp_expansion_factor", 4),
        "mask_estimator_depth": model_cfg.get("mask_estimator_depth", 2),
        "sample_rate": config.get("audio", {}).get("sample_rate", model_cfg.get("sample_rate", 44100)),
    }

    for key in ("stft_n_fft", "stft_hop_length", "stft_win_length"):
        if key in model_cfg:
            model_args[key] = model_cfg[key]

    logger.debug(f"Creating MelBandRoformerMLX with args: {list(model_args.keys())}")
    return MelBandRoformerMLX(**model_args)


def convert_torch_to_mlx_weights(state_dict: Dict[str, Any]) -> Dict[str, mx.array]:
    """Convert PyTorch state dict to MLX weights format.

    Handles:
    1. Parameter name mapping (gamma -> weight for norms)
    2. Module path restructuring for MLX module tree
    3. Sequential layer indexing (module.N -> module.layers.N)
    """
    # Detect if model has linear transformers
    has_linear_transformers = any(
        re.match(r"^layers\.\d+\.2\.", key) for key in state_dict.keys()
    )
    logger.info(f"Checkpoint has linear transformers: {has_linear_transformers}")

    mlx_weights = {}

    for key, value in state_dict.items():
        # Skip rotary embedding buffers
        if "rotary_embed.freqs" in key:
            continue

        # Convert to numpy
        numpy_weight = _to_numpy(value)

        # Map parameter names
        mlx_key = key.replace(".gamma", ".weight")

        # Restructure band_split paths
        if "band_split.to_features." in mlx_key:
            parts = mlx_key.split(".")
            if len(parts) >= 5 and parts[2].isdigit():
                band_idx = parts[2]
                submodule_idx = parts[3]
                param_name = ".".join(parts[4:])
                if submodule_idx == "0":
                    mlx_key = f"band_split.to_features_{band_idx}.norm.{param_name}"
                elif submodule_idx == "1":
                    mlx_key = f"band_split.to_features_{band_idx}.linear.{param_name}"

        # Restructure main block transformer paths
        main_block_match = re.match(r"^layers\.(\d+)\.([012])\.(.+)$", mlx_key)
        if main_block_match:
            block_idx = main_block_match.group(1)
            transformer_idx = main_block_match.group(2)
            rest = main_block_match.group(3)

            if has_linear_transformers:
                names = ["linear_transformer", "time_transformer", "freq_transformer"]
                mlx_key = f"layers_{block_idx}.{names[int(transformer_idx)]}.{rest}"
            else:
                if transformer_idx == "0":
                    mlx_key = f"layers_{block_idx}.time_transformer.{rest}"
                elif transformer_idx == "1":
                    mlx_key = f"layers_{block_idx}.freq_transformer.{rest}"

        # Restructure individual transformer layer paths
        transformer_match = re.search(r"(\.layers)\.(\d+)\.([01])\.", mlx_key)
        if transformer_match:
            prefix = mlx_key[: transformer_match.start()]
            layer_idx = transformer_match.group(2)
            submodule = transformer_match.group(3)
            suffix = mlx_key[transformer_match.end() :]

            if submodule == "0":
                mlx_key = f"{prefix}.layers_{layer_idx}.attn.{suffix}"
            elif submodule == "1":
                mlx_key = f"{prefix}.layers_{layer_idx}.ff.{suffix}"

        # Restructure mask_estimators
        if "mask_estimators." in mlx_key:
            mlx_key = re.sub(r"mask_estimators\.(\d+)", r"mask_estimators_\1", mlx_key)
            mlx_key = re.sub(r"to_freqs\.(\d+)\.0\.", r"to_freqs_\1.", mlx_key)

        # MLX Sequential: insert "layers." before numeric indices
        mlx_key = re.sub(r"\.net\.(\d+)\.", r".net.layers.\1.", mlx_key)
        mlx_key = re.sub(r"\.to_out\.(\d+)\.", r".to_out.layers.\1.", mlx_key)
        mlx_key = re.sub(r"(to_freqs_\d+)\.(\d+)\.", r"\1.layers.\2.", mlx_key)

        mlx_weights[mlx_key] = mx.array(numpy_weight)

    logger.debug(f"Converted {len(mlx_weights)} tensors from PyTorch to MLX format")
    return mlx_weights


def _to_numpy(value) -> np.ndarray:
    """Convert a weight value to numpy array, handling torch tensors."""
    try:
        # Try torch tensor first
        return value.cpu().numpy()
    except AttributeError:
        return np.array(value)


def load_roformer_model(
    model_path: str,
    config: Dict[str, Any],
) -> Tuple[nn.Module, str]:
    """Load a Roformer model with MLX weights.

    Tries safetensors first, falls back to PyTorch checkpoint conversion.

    Args:
        model_path: Path to model checkpoint (.ckpt, .pth, or .safetensors)
        config: Model configuration dict

    Returns:
        Tuple of (model, model_type)
    """
    model_type = detect_model_type(model_path, config)
    logger.info(f"Loading {model_type} model with MLX from {model_path}")

    if not os.path.exists(model_path):
        # No checkpoint to inspect — create model from config as-is
        if model_type == "bs_roformer":
            model = create_bs_roformer_mlx(config)
        elif model_type == "mel_band_roformer":
            model = create_mel_band_roformer_mlx(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        logger.warning(f"Model path {model_path} does not exist, using random initialization")
        return model, model_type

    # Try safetensors first
    safetensors_path = _find_safetensors(model_path)
    if safetensors_path:
        logger.info(f"Loading pre-converted safetensors from {safetensors_path}")
        # Detect actual depth from safetensors keys
        weights = mx.load(safetensors_path)
        _override_depth_from_weights(config, set(weights.keys()))
        if model_type == "bs_roformer":
            model = create_bs_roformer_mlx(config)
        else:
            model = create_mel_band_roformer_mlx(config)
        model.load_weights(list(weights.items()), strict=False)
        return model, model_type

    # Fall back to PyTorch checkpoint conversion
    logger.info("No safetensors found, converting from PyTorch checkpoint...")
    state_dict = _load_state_dict(model_path)
    mlx_weights = convert_torch_to_mlx_weights(state_dict)
    logger.info(f"Converted {len(mlx_weights)} weight tensors from PyTorch to MLX")

    # Detect actual mask_estimator_depth from converted keys and override config
    _override_depth_from_weights(config, set(mlx_weights.keys()))

    if model_type == "bs_roformer":
        model = create_bs_roformer_mlx(config)
    elif model_type == "mel_band_roformer":
        model = create_mel_band_roformer_mlx(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_weights(list(mlx_weights.items()), strict=False)
    logger.info("Successfully loaded converted weights into MLX model")

    # Optionally save safetensors for future use
    if os.environ.get("MLX_SAVE_SAFETENSORS") == "1":
        base = os.path.splitext(model_path)[0]
        st_path = base + ".safetensors"
        try:
            mx.save_safetensors(st_path, dict(mlx_weights))
            logger.info(f"Saved converted weights to {st_path}")
        except Exception as e:
            logger.warning(f"Could not save safetensors: {e}")

    return model, model_type


def _find_safetensors(model_path: str) -> Optional[str]:
    """Check if a safetensors version of the model exists."""
    base = os.path.splitext(model_path)[0]
    safetensors_path = base + ".safetensors"
    if os.path.exists(safetensors_path):
        return safetensors_path
    return None


def _load_state_dict(model_path: str) -> Dict[str, Any]:
    """Load a PyTorch checkpoint and extract the state dict."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            f"PyTorch is required to convert checkpoint {model_path} to MLX format. "
            "Either install torch (`pip install torch`) or pre-convert the model to "
            "safetensors format."
        )

    logger.debug(f"Loading PyTorch checkpoint from {model_path}")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

    return state_dict


def _override_depth_from_weights(config: Dict[str, Any], weight_keys: set):
    """Detect actual mask_estimator_depth from weight keys and override config.

    Some checkpoints are trained with a different mask_estimator_depth than
    what their YAML config specifies. We detect the actual depth by counting
    the number of linear layers in the mask estimator MLP.
    """
    model_cfg = config.get("model", config)

    # Find the highest layer index in mask_estimators_0.to_freqs_0.layers.N
    max_idx = -1
    for k in weight_keys:
        m = re.match(r"mask_estimators_0\.to_freqs_0\.layers\.(\d+)\.", k)
        if m:
            max_idx = max(max_idx, int(m.group(1)))

    if max_idx < 0:
        return

    # The MLP has alternating Linear/Tanh layers: indices 0,2,4,... are Linear
    # num_linears = (max_idx // 2) + 1, and MLP depth = num_linears
    actual_depth = (max_idx // 2) + 1
    config_depth = model_cfg.get("mask_estimator_depth", 2)

    if actual_depth != config_depth:
        logger.info(
            f"Overriding mask_estimator_depth: config={config_depth}, "
            f"checkpoint={actual_depth}"
        )
        model_cfg["mask_estimator_depth"] = actual_depth
