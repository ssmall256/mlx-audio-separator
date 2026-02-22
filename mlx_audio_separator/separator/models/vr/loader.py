"""VR model weight loader for MLX.

Handles conversion of PyTorch .pth weights to MLX format:
- Conv2d weight transposition: OIHW → OHWI (MLX NHWC layout)
- BatchNorm parameter renaming: .weight/.bias → MLX conventions
- Caching converted weights as .safetensors

At runtime, loads safetensors directly with zero PyTorch dependency.
"""

import json
import logging
import math
import os

import mlx.core as mx
import numpy as np

from .nets import determine_model_capacity
from .nets_new import CascadedNet

logger = logging.getLogger(__name__)


class ModelParameters:
    """Load VR model parameters from JSON config file."""

    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.param = json.loads(f.read(), object_pairs_hook=_int_keys)

        for k in ["mid_side", "mid_side_b", "mid_side_b2", "stereo_w", "stereo_n", "reverse"]:
            if k not in self.param:
                self.param[k] = False

        if "n_bins" in self.param:
            self.param["bins"] = self.param["n_bins"]


def _int_keys(d):
    """Convert string digit keys to int (JSON object_pairs_hook)."""
    result = {}
    for key, value in d:
        if key.isdigit():
            key = int(key)
        result[key] = value
    return result


def load_vr_model(model_path, model_data):
    """Load a VR model with MLX weights.

    Tries safetensors cache first, falls back to PyTorch conversion.

    Args:
        model_path: Path to the .pth model file
        model_data: Model metadata dict (from hash lookup)

    Returns:
        (model, model_params, is_vr_51_model) tuple
    """
    # Load model parameters
    package_root = os.path.dirname(os.path.abspath(__file__))
    params_dir = os.path.join(package_root, "modelparams")
    params_file = os.path.join(params_dir, f"{model_data['vr_model_param']}.json")
    model_params = ModelParameters(params_file)

    # Determine model type
    is_vr_51_model = "nout" in model_data and "nout_lstm" in model_data
    model_capacity = (model_data.get("nout", 32), model_data.get("nout_lstm", 128))

    nn_arch_sizes = [31191, 33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
    vr_5_1_models = [56817, 218409]
    model_size = math.ceil(os.stat(model_path).st_size / 1024)
    nn_arch_size = min(nn_arch_sizes, key=lambda x: abs(x - model_size))

    if nn_arch_size in vr_5_1_models or is_vr_51_model:
        model = CascadedNet(model_params.param["bins"] * 2, nn_arch_size, nout=model_capacity[0], nout_lstm=model_capacity[1])
        is_vr_51_model = True
    else:
        model = determine_model_capacity(model_params.param["bins"] * 2, nn_arch_size)

    # Try loading safetensors cache
    safetensors_path = model_path.rsplit(".", 1)[0] + ".safetensors"
    if os.path.exists(safetensors_path):
        logger.info(f"Loading cached MLX weights from {safetensors_path}")
        weights = mx.load(safetensors_path)
        model.load_weights(list(weights.items()))
        model.eval()
        return model, model_params, is_vr_51_model

    # Convert from PyTorch
    logger.info("Converting PyTorch weights to MLX format...")
    weights = convert_torch_to_mlx_weights(model_path)
    model.load_weights(list(weights.items()))
    model.eval()

    # Cache as safetensors
    try:
        mx.save_safetensors(safetensors_path, weights)
        logger.info(f"Cached MLX weights to {safetensors_path}")
    except Exception as e:
        logger.warning(f"Could not cache safetensors: {e}")

    return model, model_params, is_vr_51_model


def convert_torch_to_mlx_weights(model_path):
    """Convert PyTorch .pth state dict to MLX weight dict.

    Handles:
    - Conv2d weight transposition from OIHW → OHWI
    - BatchNorm renaming: running_mean → running_mean, running_var → running_var
    - Sequential layer indexing (conv.0 → first child, conv.1 → second child, etc.)
    - nn.Sequential wrapping in CascadedNet stages
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for initial weight conversion. "
            "Install with: pip install 'mlx-audio-separator[convert]'"
        )

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    mlx_weights = {}

    # First pass: collect all weights with converted keys
    pending_lstm_biases = {}  # key_prefix → {ih: array, hh: array}

    for key, value in state_dict.items():
        arr = value.numpy()
        mlx_key = _convert_key(key)

        # Skip num_batches_tracked (not used in MLX BatchNorm)
        if mlx_key is None:
            continue

        # Handle LSTM parameter name differences
        # PyTorch: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
        # MLX: Wx, Wh, bias (bias = bias_ih + bias_hh)
        if ".weight_ih_l0" in mlx_key:
            mlx_key = mlx_key.replace(".weight_ih_l0", ".Wx")
        elif ".weight_hh_l0" in mlx_key:
            mlx_key = mlx_key.replace(".weight_hh_l0", ".Wh")
        elif ".bias_ih_l0" in mlx_key:
            prefix = mlx_key.replace(".bias_ih_l0", "")
            pending_lstm_biases.setdefault(prefix, {})["ih"] = arr
            continue
        elif ".bias_hh_l0" in mlx_key:
            prefix = mlx_key.replace(".bias_hh_l0", "")
            pending_lstm_biases.setdefault(prefix, {})["hh"] = arr
            continue

        # Transpose Conv2d weights: OIHW → OHWI
        if _is_conv_weight(key, arr):
            arr = np.transpose(arr, (0, 2, 3, 1))

        mlx_weights[mlx_key] = mx.array(arr)

    # Combine LSTM biases: MLX uses single bias = bias_ih + bias_hh
    for prefix, biases in pending_lstm_biases.items():
        if "ih" in biases and "hh" in biases:
            combined = biases["ih"] + biases["hh"]
            mlx_weights[f"{prefix}.bias"] = mx.array(combined)

    return mlx_weights


def _is_conv_weight(key, arr):
    """Check if a parameter is a Conv2d weight (4D and named 'weight' under conv)."""
    return arr.ndim == 4 and "weight" in key


def _convert_key(key):
    """Convert PyTorch state dict key to MLX module path.

    PyTorch Sequential containers use numeric indices (conv.0.weight, conv.1.weight).
    In our MLX implementation, Conv2DBNActiv has .conv and .bn attributes directly,
    so we need to map the Sequential indices.

    Also handles:
    - CascadedNet stg*_low_band_net Sequential: .0. → _base., .1. → _proj.
    - ASPP bottleneck Sequential: bottleneck.0.* → bottleneck_conv.*
    - ASPP conv1 Sequential: conv1.1.conv.N → conv1_proj.conv/bn (skip AdaptiveAvgPool)
    - ASPP separable conv blocks: conv3/4/5.conv.[0|1|2] → dw_conv/pw_conv/bn
    - Decoder naming drift: dec*.conv.conv.N → dec*.conv1.conv/bn
    - LSTM bidirectional weight mappings

    Returns None for parameters that should be skipped (num_batches_tracked).
    """
    # Skip num_batches_tracked
    if key.endswith("num_batches_tracked"):
        return None

    parts = key.split(".")
    new_parts = []
    i = 0

    while i < len(parts):
        part = parts[i]

        # Handle nn.Sequential wrapping in CascadedNet stages
        # stg1_low_band_net.0.xxx → stg1_low_band_net_base.xxx
        # stg1_low_band_net.1.xxx → stg1_low_band_net_proj.xxx
        if part in ("stg1_low_band_net", "stg2_low_band_net") and i + 1 < len(parts) and parts[i + 1].isdigit():
            idx = int(parts[i + 1])
            if idx == 0:
                new_parts.append(f"{part}_base")
            else:
                new_parts.append(f"{part}_proj")
            i += 2
            continue

        # Handle ASPP conv1 Sequential wrapper
        # PyTorch: conv1 = Sequential(AdaptiveAvgPool2d, Conv2DBNActiv)
        # conv1.0 is AdaptiveAvgPool2d (no params), conv1.1 is Conv2DBNActiv
        # conv1.1.conv.N.param → conv1_proj.conv/bn.param
        if part == "conv1" and i + 1 < len(parts) and parts[i + 1].isdigit():
            idx = int(parts[i + 1])
            if idx == 1:
                new_parts.append("conv1_proj")
                i += 2  # skip "conv1" and "1"
                continue
            # idx == 0 would be AdaptiveAvgPool2d (no learnable params)
            i += 2
            continue

        # Handle ASPP bottleneck Sequential wrapper
        # PyTorch: aspp.bottleneck.0.conv.N.param
        # MLX: aspp.bottleneck_conv.conv/bn.param
        if part == "bottleneck" and i + 1 < len(parts) and parts[i + 1].isdigit():
            idx = int(parts[i + 1])
            if idx == 0:
                new_parts.append("bottleneck_conv")
            else:
                new_parts.append(f"bottleneck_{idx}")
            i += 2
            continue

        # Decoder naming in source checkpoints is dec*.conv.* while our model uses dec*.conv1.*
        # Example: dec1.conv.conv.0.weight -> dec1.conv1.conv.weight
        if (
            part == "conv"
            and i > 0
            and parts[i - 1].startswith("dec")
            and i + 1 < len(parts)
            and parts[i + 1] == "conv"
        ):
            new_parts.append("conv1")
            i += 1
            continue

        # Handle Conv2DBNActiv/Separable Sequential: conv.N → concrete submodule
        # Conv2DBNActiv: conv.0 = Conv2d, conv.1 = BatchNorm
        # ASPP separable blocks: conv3/conv4/conv5 use conv.0=dw, conv.1=pw, conv.2=bn
        if part == "conv" and i + 1 < len(parts) and parts[i + 1].isdigit():
            idx = int(parts[i + 1])
            in_aspp_sep = (
                len(new_parts) >= 2
                and new_parts[-2] == "aspp"
                and new_parts[-1] in {"conv3", "conv4", "conv5"}
            )
            if in_aspp_sep:
                if idx == 0:
                    new_parts.append("dw_conv")
                elif idx == 1:
                    new_parts.append("pw_conv")
                elif idx == 2:
                    new_parts.append("bn")
                else:
                    i += 2
                    continue
                i += 2
                continue

            if idx == 0:
                new_parts.append("conv")
            elif idx in (1, 2):
                new_parts.append("bn")
            else:
                i += 2
                continue
            i += 2
            continue

        new_parts.append(part)
        i += 1

    result = ".".join(new_parts)

    # LSTM mappings
    result = result.replace(".dense.0.", ".dense_linear.")
    result = result.replace(".dense.1.", ".dense_bn.")

    # Squeeze conv mapping
    result = result.replace("lstm_dec2.conv.", "lstm_dec2.squeeze_conv.")

    # LSTM weight mappings for bidirectional
    # PyTorch bidirectional LSTM stores weights as:
    #   lstm.weight_ih_l0, lstm.weight_hh_l0 (forward)
    #   lstm.weight_ih_l0_reverse, lstm.weight_hh_l0_reverse (backward)
    if "lstm.weight_ih_l0_reverse" in result or "lstm.weight_hh_l0_reverse" in result:
        result = result.replace("lstm.", "lstm_bwd.")
        result = result.replace("_reverse", "")
    elif "lstm.bias_ih_l0_reverse" in result or "lstm.bias_hh_l0_reverse" in result:
        result = result.replace("lstm.", "lstm_bwd.")
        result = result.replace("_reverse", "")
    elif "lstm.weight_ih_l0" in result or "lstm.weight_hh_l0" in result:
        result = result.replace("lstm.", "lstm_fwd.")
    elif "lstm.bias_ih_l0" in result or "lstm.bias_hh_l0" in result:
        result = result.replace("lstm.", "lstm_fwd.")

    return result
