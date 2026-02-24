"""Unified MDXC model loader (Roformer + MDX23C/TFC_TDF_v3)."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio_separator.separator.models.roformer.loader import load_roformer_model

from .tfc_tdf_v3_mlx import TfcTdfV3MLX

logger = logging.getLogger(__name__)

SCHEMA_ROFORMER = "roformer"
SCHEMA_MDX23C_TFC_TDF_V3 = "mdx23c_tfc_tdf_v3"

_KNOWN_MDX23C_FILENAMES = {
    "mdx23c-8kfft-instvoc_hq.ckpt",
    "mdx23c-de-reverb-aufr33-jarredou.ckpt",
    "mdx23c-drumsep-aufr33-jarredou.ckpt",
}

_MDX23C_REQUIRED_MODEL_KEYS = {
    "num_subbands",
    "num_scales",
    "num_blocks_per_scale",
    "num_channels",
    "growth",
    "bottleneck_factor",
    "scale",
}
_MDX23C_REQUIRED_AUDIO_KEYS = {"dim_f", "n_fft", "hop_length", "num_channels"}


def classify_mdxc_schema(model_path: str, config: Dict[str, Any]) -> str:
    """Classify MDXC model schema."""
    model_cfg = config.get("model", config)
    audio_cfg = config.get("audio", {})
    model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
    audio_cfg = audio_cfg if isinstance(audio_cfg, dict) else {}

    base_name = os.path.basename(model_path).lower()
    if base_name in _KNOWN_MDX23C_FILENAMES:
        return SCHEMA_MDX23C_TFC_TDF_V3

    if config.get("is_roformer"):
        return SCHEMA_ROFORMER
    if "num_bands" in model_cfg or "freqs_per_bands" in model_cfg:
        return SCHEMA_ROFORMER

    has_mdx23c_model = _MDX23C_REQUIRED_MODEL_KEYS.issubset(model_cfg.keys())
    has_mdx23c_audio = _MDX23C_REQUIRED_AUDIO_KEYS.issubset(audio_cfg.keys())
    if has_mdx23c_model and has_mdx23c_audio:
        return SCHEMA_MDX23C_TFC_TDF_V3

    missing_model = sorted(_MDX23C_REQUIRED_MODEL_KEYS.difference(model_cfg.keys()))
    missing_audio = sorted(_MDX23C_REQUIRED_AUDIO_KEYS.difference(audio_cfg.keys()))
    roformer_markers = sorted(k for k in ("num_bands", "freqs_per_bands") if k in model_cfg)
    raise ValueError(
        "Unsupported MDXC schema for "
        f"{os.path.basename(model_path)}. "
        f"Missing MDX23C model keys: {missing_model}; "
        f"missing MDX23C audio keys: {missing_audio}; "
        f"Roformer markers present: {roformer_markers}."
    )


def load_mdxc_model(
    model_path: str,
    config: Dict[str, Any],
) -> Tuple[nn.Module, str]:
    """Load MDXC model via schema-specific MLX path."""
    schema = classify_mdxc_schema(model_path, config)
    if schema == SCHEMA_ROFORMER:
        model, model_type = load_roformer_model(model_path=model_path, config=config)
        return model, model_type
    if schema == SCHEMA_MDX23C_TFC_TDF_V3:
        model = load_mdx23c_model(model_path=model_path, config=config)
        return model, SCHEMA_MDX23C_TFC_TDF_V3
    raise ValueError(f"Unsupported MDXC schema: {schema}")


def create_mdx23c_model(config: Dict[str, Any]) -> TfcTdfV3MLX:
    return TfcTdfV3MLX(config=config)


def load_mdx23c_model(model_path: str, config: Dict[str, Any]) -> TfcTdfV3MLX:
    logger.info("Loading MDX23C TFC_TDF_v3 model with MLX from %s", model_path)
    model = create_mdx23c_model(config)

    if not os.path.exists(model_path):
        logger.warning("Model path %s does not exist, using random initialization", model_path)
        return model

    safetensors_path = _find_safetensors(model_path)
    if safetensors_path:
        logger.info("Loading pre-converted safetensors from %s", safetensors_path)
        weights = mx.load(safetensors_path)
        model.load_weights(list(weights.items()), strict=False)
        return model

    logger.info("No safetensors found, converting MDX23C checkpoint from PyTorch...")
    state_dict = _load_state_dict(model_path)
    mlx_weights = convert_mdx23c_torch_to_mlx_weights(state_dict)
    logger.info("Converted %s MDX23C tensors to MLX format", len(mlx_weights))
    model.load_weights(list(mlx_weights.items()), strict=False)

    if os.environ.get("MLX_SAVE_SAFETENSORS") == "1":
        try:
            mx.save_safetensors(os.path.splitext(model_path)[0] + ".safetensors", dict(mlx_weights))
        except Exception as exc:  # pragma: no cover - non-critical
            logger.warning("Could not save converted MDX23C safetensors: %s", exc)

    return model


def convert_mdx23c_torch_to_mlx_weights(state_dict: Dict[str, Any]) -> Dict[str, mx.array]:
    """Convert PyTorch MDX23C checkpoint tensors to MLX tensor names/layouts."""
    mlx_weights: Dict[str, mx.array] = {}
    for key, value in state_dict.items():
        mlx_key = _translate_torch_key(key)
        if mlx_key is None:
            continue
        weight = _to_numpy(value)
        if _is_conv_weight(mlx_key, weight):
            # Conv2d: OIHW -> OHWI
            weight = np.transpose(weight, (0, 2, 3, 1))
        elif _is_conv_transpose_weight(mlx_key, weight):
            # ConvTranspose2d: IOHW -> OHWI
            weight = np.transpose(weight, (1, 2, 3, 0))
        mlx_weights[mlx_key] = mx.array(weight)
    return mlx_weights


def _find_safetensors(model_path: str) -> Optional[str]:
    candidate = os.path.splitext(model_path)[0] + ".safetensors"
    return candidate if os.path.exists(candidate) else None


def _load_state_dict(model_path: str) -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        raise ImportError(
            f"PyTorch is required to convert checkpoint {model_path} to MLX format. "
            "Install torch or provide a pre-converted .safetensors file."
        )

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unexpected checkpoint structure for {model_path}: {type(state_dict)}")
    return state_dict


def _to_numpy(value: Any) -> np.ndarray:
    try:
        return value.detach().cpu().numpy()
    except AttributeError:
        return np.array(value)


def _strip_prefix(key: str) -> str:
    for prefix in ("state_dict.", "model.", "module."):
        if key.startswith(prefix):
            return _strip_prefix(key[len(prefix):])
    return key


def _translate_torch_key(key: str) -> Optional[str]:
    key = _strip_prefix(key)
    if key.startswith("stft."):
        return None

    if key == "first_conv.weight":
        return "first_conv.weight"

    # Encoder TFC/TDF blocks
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tfc1\.0\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tfc1_norm.{m.group(3)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tfc1\.2\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tfc1_conv.{m.group(3)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tdf\.0\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tdf_norm1.{m.group(3)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tdf\.2\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tdf_linear1.{m.group(3)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tdf\.3\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tdf_norm2.{m.group(3)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tdf\.5\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tdf_linear2.{m.group(3)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tfc2\.0\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tfc2_norm.{m.group(3)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tfc2\.2\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tfc2_conv.{m.group(3)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.shortcut\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.shortcut.{m.group(3)}"

    # Encoder downscale
    m = re.match(r"^encoder_blocks\.(\d+)\.downscale\.conv\.0\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.downscale.norm.{m.group(2)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.downscale\.conv\.2\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.downscale.conv.{m.group(2)}"

    # Bottleneck
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.tfc1\.0\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.tfc1_norm.{m.group(2)}"
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.tfc1\.2\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.tfc1_conv.{m.group(2)}"
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.tdf\.0\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.tdf_norm1.{m.group(2)}"
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.tdf\.2\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.tdf_linear1.{m.group(2)}"
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.tdf\.3\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.tdf_norm2.{m.group(2)}"
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.tdf\.5\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.tdf_linear2.{m.group(2)}"
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.tfc2\.0\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.tfc2_norm.{m.group(2)}"
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.tfc2\.2\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.tfc2_conv.{m.group(2)}"
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.shortcut\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.shortcut.{m.group(2)}"

    # Decoder upscale
    m = re.match(r"^decoder_blocks\.(\d+)\.upscale\.conv\.0\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.upscale.norm.{m.group(2)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.upscale\.conv\.2\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.upscale.conv.{m.group(2)}"

    # Decoder TFC/TDF blocks
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tfc1\.0\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tfc1_norm.{m.group(3)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tfc1\.2\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tfc1_conv.{m.group(3)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tdf\.0\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tdf_norm1.{m.group(3)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tdf\.2\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tdf_linear1.{m.group(3)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tdf\.3\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tdf_norm2.{m.group(3)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tdf\.5\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tdf_linear2.{m.group(3)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tfc2\.0\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tfc2_norm.{m.group(3)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tfc2\.2\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tfc2_conv.{m.group(3)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.shortcut\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.shortcut.{m.group(3)}"

    if key == "final_conv.0.weight":
        return "final_conv1.weight"
    if key == "final_conv.2.weight":
        return "final_conv2.weight"

    # Ignore non-parameter artifacts and unsupported paths explicitly.
    if key.endswith(".num_batches_tracked"):
        return None
    logger.debug("Unhandled MDX23C checkpoint key: %s", key)
    return None


def _is_conv_weight(mlx_key: str, value: np.ndarray) -> bool:
    if value.ndim != 4:
        return False
    if mlx_key.endswith("upscale.conv.weight"):
        return False
    return mlx_key.endswith(".weight") and any(
        token in mlx_key
        for token in (
            "first_conv.weight",
            "final_conv1.weight",
            "final_conv2.weight",
            "tfc1_conv.weight",
            "tfc2_conv.weight",
            "shortcut.weight",
            "downscale.conv.weight",
        )
    )


def _is_conv_transpose_weight(mlx_key: str, value: np.ndarray) -> bool:
    return value.ndim == 4 and mlx_key.endswith("upscale.conv.weight")
