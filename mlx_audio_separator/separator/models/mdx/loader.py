"""ONNX-to-MLX weight converter and loader for MDX ConvTDFNet models."""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .convtdfnet import ConvTDFNet

logger = logging.getLogger(__name__)

# Default model parameters for known MDX models
# These are inferred from the ONNX model structure and UVR model data
MDX_DEFAULT_PARAMS = {
    "dim_c": 4,
    "num_blocks": 9,
    "l": 3,
    "g": 32,
    "k": 3,
    "bn": 16,
    "bias": True,
    "optimizer": "rmsprop",
}

_NUMERIC_MAPPING_DISABLED_BASENAMES = {"uvr-mdx-net-voc_ft.onnx"}


def create_mdx_model(model_data: Dict[str, Any]) -> ConvTDFNet:
    """Create a ConvTDFNet model from model data parameters.

    Args:
        model_data: Model metadata dict (from JSON hash lookup) containing:
            - mdx_dim_f_set: frequency dimension
            - mdx_dim_t_set: log2 of time dimension
            - mdx_n_fft_scale_set: FFT size

    Returns:
        ConvTDFNet model instance
    """
    dim_f = model_data["mdx_dim_f_set"]
    dim_t = 2 ** model_data["mdx_dim_t_set"]
    n_fft = model_data["mdx_n_fft_scale_set"]

    # Extract optional override params from model_data
    params = dict(MDX_DEFAULT_PARAMS)
    for key in (
        "num_blocks", "l", "g", "k", "bn", "bias", "optimizer", "dim_c",
    ):
        if key in model_data:
            params[key] = model_data[key]

    hop_length = model_data.get("mdx_hop_length", 1024)

    model = ConvTDFNet(
        dim_c=params["dim_c"],
        dim_f=dim_f,
        dim_t=dim_t,
        n_fft=n_fft,
        hop_length=hop_length,
        num_blocks=params["num_blocks"],
        num_tdf_layers=params["l"],
        g=params["g"],
        k=params["k"],
        bn=params["bn"],
        bias=params["bias"],
        optimizer=params["optimizer"],
    )

    logger.debug(
        f"Created ConvTDFNet: dim_f={dim_f}, dim_t={dim_t}, n_fft={n_fft}, "
        f"num_blocks={params['num_blocks']}, g={params['g']}, bn={params['bn']}"
    )
    return model


def _load_onnx_weights(onnx_path: str) -> Dict[str, np.ndarray]:
    """Load all initializer weights from an ONNX model file."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        raise ImportError(
            "The 'onnx' package is required to convert MDX models. "
            "Install with: pip install onnx"
        )

    logger.info(f"Loading ONNX model from {onnx_path}")
    model = onnx.load(onnx_path)

    onnx_weights = {}
    for initializer in model.graph.initializer:
        numpy_weight = numpy_helper.to_array(initializer)
        onnx_weights[initializer.name] = numpy_weight

    logger.info(f"Extracted {len(onnx_weights)} weight tensors from ONNX model")
    return onnx_weights


def _infer_params_from_onnx(
    onnx_weights: Dict[str, np.ndarray],
    dim_f: int,
) -> Dict[str, int]:
    """Infer architecture parameters (g, num_blocks, bn, l) from ONNX weights.

    ONNX models may have numeric IDs for some weights (fused BN into Conv)
    and named weights for others. We infer the architecture from weight shapes.
    """
    # Separate named vs numeric weights
    named = {}
    numeric = []
    for name, weight in onnx_weights.items():
        if name[0].isdigit() and "." not in name:
            numeric.append((int(name), weight))
        else:
            named[name] = weight

    params = {}

    # Infer g/dim_c from first plausible first_conv-like 1x1 kernel.
    for _, weight in onnx_weights.items():
        if (
            weight.ndim == 4
            and weight.shape[2] == 1
            and weight.shape[3] == 1
            and int(weight.shape[1]) <= 8
            and int(weight.shape[0]) > int(weight.shape[1])
        ):
            params["g"] = int(weight.shape[0])
            params["dim_c"] = int(weight.shape[1])
            logger.info(
                "Inferred g=%s, dim_c=%s from 1x1 conv shape %s",
                params["g"],
                params["dim_c"],
                tuple(weight.shape),
            )
            break

    if "g" not in params and numeric:
        numeric.sort(key=lambda x: x[0])
        for _, weight in numeric:
            if (
                weight.ndim == 4
                and weight.shape[2] == 1
                and weight.shape[3] == 1
                and int(weight.shape[1]) <= 8
            ):
                params["g"] = int(weight.shape[0])
                params["dim_c"] = int(weight.shape[1])
                logger.info(
                    "Inferred g=%s, dim_c=%s from numeric 1x1 conv shape %s",
                    params["g"],
                    params["dim_c"],
                    tuple(weight.shape),
                )
                break

    # Infer num_blocks from encoding_blocks indices.
    max_enc_idx = -1
    for name in named:
        m = re.match(r"encoding_blocks\.(\d+)\.", name)
        if m:
            max_enc_idx = max(max_enc_idx, int(m.group(1)))
    if max_enc_idx >= 0:
        n = max_enc_idx + 1
        params["num_blocks"] = 2 * n + 1
        logger.info("Inferred num_blocks=%s from %s encoding blocks", params["num_blocks"], n)

    # Alternate naming profile (kuielab family): ds_dense / is_dense / mid_dense.
    if "num_blocks" not in params:
        max_ds_dense_idx = -1
        for name in named:
            m = re.match(r"ds_dense\.(\d+)\.", name)
            if m:
                max_ds_dense_idx = max(max_ds_dense_idx, int(m.group(1)))
        if max_ds_dense_idx >= 0:
            n = max_ds_dense_idx + 1
            params["num_blocks"] = 2 * n + 1
            logger.info("Inferred num_blocks=%s from ds_dense depth n=%s", params["num_blocks"], n)

    # Last-resort depth estimate from max conv width and inferred g.
    if "num_blocks" not in params and "g" in params:
        g = int(params["g"])
        max_channel = 0
        for weight in onnx_weights.values():
            if weight.ndim == 4:
                max_channel = max(max_channel, int(weight.shape[0]), int(weight.shape[1]))
        if max_channel > 0 and max_channel % g == 0:
            n_est = (max_channel // g) - 1
            if 1 <= n_est <= 12:
                params["num_blocks"] = 2 * n_est + 1
                logger.info(
                    "Inferred num_blocks=%s from max conv channels=%s and g=%s",
                    params["num_blocks"],
                    max_channel,
                    g,
                )

    # Infer bn from TDF linear weight shapes.
    for name, weight in named.items():
        if (
            re.match(r"encoding_blocks\.0\.tdf\.0\.", name)
            or re.match(r"ds_dense\.0\.tdf\.0\.", name)
        ) and weight.ndim == 2:
            if weight.shape[0] == dim_f:
                params["bn"] = dim_f // weight.shape[1]
            elif weight.shape[1] == dim_f:
                params["bn"] = dim_f // weight.shape[0]
            logger.info(f"Inferred bn={params['bn']} from TDF linear shape")
            break

    if "bn" not in params and numeric:
        for _, weight in numeric:
            if weight.ndim == 2 and dim_f in weight.shape:
                other_dim = weight.shape[1] if weight.shape[0] == dim_f else weight.shape[0]
                if dim_f % other_dim == 0 and other_dim < dim_f:
                    params["bn"] = dim_f // other_dim
                    logger.info(f"Inferred bn={params['bn']} from numeric TDF linear")
                    break

    # Infer num_tdf_layers (l) from numeric path if present.
    if numeric and "g" in params:
        g = int(params["g"])
        tfc_count = 0
        for _, weight in numeric:
            if weight.ndim != 4:
                continue
            if weight.shape[1] != g or weight.shape[0] != g:
                if tfc_count > 0:
                    break
                continue
            tfc_count += 1
        if tfc_count > 0:
            params["l"] = tfc_count
            logger.info(f"Inferred l={params['l']} TFC layers from weight shapes")

    return params


def convert_onnx_to_mlx_weights(
    onnx_weights: Dict[str, np.ndarray],
    g: int,
    n: int,
    num_tdf_layers: int,
    dim_f: int,
    bn: int,
    include_numeric: bool = True,
) -> Dict[str, mx.array]:
    """Convert ONNX weights to MLX format.

    Handles both named ONNX weights (proper module paths) and numeric
    ONNX weights (IDs from fused BN export). For numeric weights, we map
    positionally based on the known architecture structure.

    Conv2d weights are transposed from (O, I, H, W) to (O, H, W, I).
    Fused BN layers get identity parameters.
    """
    # Separate named vs numeric weights
    named = {}
    numeric = []
    for name, weight in onnx_weights.items():
        if name[0].isdigit() and "." not in name:
            numeric.append((int(name), weight))
        else:
            named[name] = weight

    mlx_weights = {}

    # Process named weights using existing translation
    for name, weight in named.items():
        mlx_name = _translate_weight_name(name)
        if mlx_name is None:
            continue

        if _is_conv_weight(mlx_name, weight):
            if _is_conv_transpose_weight(mlx_name):
                # ConvTranspose2d: PyTorch (I, O, H, W) → MLX (O, H, W, I)
                weight = np.transpose(weight, (1, 2, 3, 0))
            else:
                # Conv2d: PyTorch (O, I, H, W) → MLX (O, H, W, I)
                weight = np.transpose(weight, (0, 2, 3, 1))

        mlx_weights[mlx_name] = mx.array(weight)

    used_conv_profile = False
    if not _has_structured_named_weights(named):
        conv_sequence = _extract_conv_profile_sequence(named)
        if conv_sequence:
            # Conv_* exports keep convolution kernels in unnamed buckets and
            # often keep TDF linears in numeric IDs.
            linear_numeric = sorted(
                [(idx, weight) for idx, weight in numeric if weight.ndim == 2],
                key=lambda x: x[0],
            )
            positional = conv_sequence + linear_numeric
            _map_numeric_weights(positional, mlx_weights, g, n, num_tdf_layers, dim_f, bn)
            used_conv_profile = True

    # Process numeric weights positionally
    if include_numeric and numeric and not used_conv_profile:
        numeric.sort(key=lambda x: x[0])
        _map_numeric_weights(
            numeric, mlx_weights, g, n, num_tdf_layers, dim_f, bn,
        )

    logger.info(f"Converted {len(mlx_weights)} weight tensors to MLX format")
    return mlx_weights


def _map_numeric_weights(
    numeric: List[Tuple[int, np.ndarray]],
    mlx_weights: Dict[str, mx.array],
    g: int,
    n: int,
    num_tdf_layers: int,
    dim_f: int,
    bn: int,
):
    """Map numeric ONNX weights positionally to MLX model paths.

    ONNX models with fused BatchNorm have numeric weight IDs instead of
    named parameters. The structure is predictable:
    - first_conv: weight + bias
    - Per encoder block: l TFC conv (weight+bias) + ds conv (weight+bias)
    - Bottleneck: l TFC conv (weight+bias)
    - Per decoder block: l TFC conv (weight+bias)
    - TDF linear weights (2D): 2 per block (linear1, linear2)

    For fused BN layers, we set identity parameters so they act as no-ops.
    """
    idx = 0

    def take():
        nonlocal idx
        if idx >= len(numeric):
            return None
        _, w = numeric[idx]
        idx += 1
        return w

    def add_conv(prefix, weight):
        """Add a conv weight (transposed) and its bias."""
        if weight.ndim != 4:
            raise ValueError(
                f"MDX numeric mapping expected 4D conv tensor for '{prefix}', "
                f"got shape {weight.shape}."
            )
        try:
            transposed = np.transpose(weight, (0, 2, 3, 1))
        except ValueError as exc:
            raise ValueError(
                f"MDX numeric mapping transpose failed for '{prefix}' "
                f"with shape {weight.shape}; this ONNX layout is unsupported "
                "by positional mapping."
            ) from exc
        mlx_weights[f"{prefix}.weight"] = mx.array(transposed)
        bias = take()
        if bias is not None:
            mlx_weights[f"{prefix}.bias"] = mx.array(bias)

    def add_identity_bn(prefix, num_features):
        """Add identity BN parameters (no-op) for fused conv layers."""
        mlx_weights[f"{prefix}.bn.weight"] = mx.ones((num_features,))
        mlx_weights[f"{prefix}.bn.bias"] = mx.zeros((num_features,))
        mlx_weights[f"{prefix}.bn.running_mean"] = mx.zeros((num_features,))
        mlx_weights[f"{prefix}.bn.running_var"] = mx.ones((num_features,))

    def add_tfc_block(block_prefix, c):
        """Add TFC conv weights and identity BN for a block."""
        for j in range(num_tdf_layers):
            w = take()
            if w is not None and w.ndim == 4:
                add_conv(f"{block_prefix}.tfc.conv_{j}", w)
                add_identity_bn(f"{block_prefix}.tfc.norm_{j}", c)

    # first_conv
    w = take()
    if w is not None:
        add_conv("first_conv", w)
        add_identity_bn("first_norm", g)

    c = g
    f = dim_f

    # Encoder blocks
    for i in range(n):
        add_tfc_block(f"enc_{i}", c)
        # ds conv
        w = take()
        if w is not None and w.ndim == 4:
            add_conv(f"ds_{i}_conv", w)
            add_identity_bn(f"ds_{i}_norm", c + g)
        f = f // 2
        c += g

    # Bottleneck
    add_tfc_block("bottleneck", c)

    # Decoder blocks
    for i in range(n):
        c -= g
        add_tfc_block(f"dec_{i}", c)

    # TDF linear weights (2D tensors)
    # Order: enc_0..enc_{n-1}, bottleneck, dec_0..dec_{n-1}
    f = dim_f
    c = g
    for i in range(n):
        _add_tdf_linears(take, mlx_weights, f"enc_{i}", f, bn)
        f = f // 2
        c += g

    _add_tdf_linears(take, mlx_weights, "bottleneck", f, bn)

    for i in range(n):
        f = f * 2
        c -= g
        _add_tdf_linears(take, mlx_weights, f"dec_{i}", f, bn)

    if idx < len(numeric):
        logger.debug(
            f"Warning: {len(numeric) - idx} unmapped numeric weights remaining"
        )


def _add_tdf_linears(take, mlx_weights, prefix, f, bn):
    """Add TDF linear weights for a block.

    ONNX stores linear weights as (in_features, out_features) but
    MLX nn.Linear expects (out_features, in_features), so we transpose.
    """
    w1 = take()
    if w1 is not None and w1.ndim == 2:
        mlx_weights[f"{prefix}.tdf_linear1.weight"] = mx.array(w1.T)
    w2 = take()
    if w2 is not None and w2.ndim == 2:
        mlx_weights[f"{prefix}.tdf_linear2.weight"] = mx.array(w2.T)


def _is_conv_transpose_weight(name: str) -> bool:
    """Check if a weight is a ConvTranspose2d kernel (upsampling layers)."""
    return "us_" in name and "conv." in name


def _is_conv_weight(name: str, weight: np.ndarray) -> bool:
    """Check if a weight tensor is a Conv2d/ConvTranspose2d kernel."""
    if weight.ndim != 4:
        return False
    # Conv weights end with .weight and are in Conv2d modules
    if name.endswith(".weight") and any(
        k in name for k in ("conv.", "first_conv.", "final_conv.", "ds_", "us_")
    ):
        return True
    # Also check TFC conv weights
    if "tfc.conv_" in name and name.endswith(".weight") and weight.ndim == 4:
        return True
    return False


def _translate_weight_name(onnx_name: str) -> Optional[str]:
    """Translate an ONNX weight name to MLX ConvTDFNet path.

    This handles the mapping from PyTorch's module naming to our flattened naming.
    Only processes named (non-numeric) weights.
    """
    name = onnx_name

    # Skip numeric-only names (handled by positional mapping)
    if name[0].isdigit() and "." not in name:
        return None

    # Handle PyTorch Lightning wrapper prefixes
    for prefix in ("model.", "module."):
        if name.startswith(prefix):
            name = name[len(prefix):]

    # first_conv Sequential: 0=Conv2d, 1=BatchNorm
    if name.startswith("first_conv.0."):
        return name.replace("first_conv.0.", "first_conv.")
    if name.startswith("first_conv.1."):
        param = name.split(".")[-1]
        if param == "running_mean":
            return "first_norm.bn.running_mean"
        if param == "running_var":
            return "first_norm.bn.running_var"
        return f"first_norm.bn.{param}"

    # final_conv Sequential: 0=Conv2d
    if name.startswith("final_conv.0."):
        return name.replace("final_conv.0.", "final_conv.")

    # Encoding blocks
    enc_match = re.match(r"encoding_blocks\.(\d+)\.(.*)", name)
    if enc_match:
        idx = enc_match.group(1)
        rest = enc_match.group(2)
        translated = _translate_tfc_tdf_path(rest)
        return f"enc_{idx}.{translated}" if translated else None

    # Downsampling: ds.N.0.param → ds_N_conv.param, ds.N.1.param → ds_N_norm.bn.param
    ds_match = re.match(r"ds\.(\d+)\.(\d+)\.(.*)", name)
    if ds_match:
        idx = ds_match.group(1)
        sub_idx = ds_match.group(2)
        param = ds_match.group(3)
        if sub_idx == "0":
            return f"ds_{idx}_conv.{param}"
        elif sub_idx == "1":
            return f"ds_{idx}_norm.bn.{param}"

    # Bottleneck
    if name.startswith("bottleneck_block."):
        rest = name[len("bottleneck_block."):]
        translated = _translate_tfc_tdf_path(rest)
        return f"bottleneck.{translated}" if translated else None

    # Upsampling: us.N.0.param → us_N_conv.param, us.N.1.param → us_N_norm.bn.param
    us_match = re.match(r"us\.(\d+)\.(\d+)\.(.*)", name)
    if us_match:
        idx = us_match.group(1)
        sub_idx = us_match.group(2)
        param = us_match.group(3)
        if sub_idx == "0":
            return f"us_{idx}_conv.{param}"
        elif sub_idx == "1":
            return f"us_{idx}_norm.bn.{param}"

    # Decoding blocks
    dec_match = re.match(r"decoding_blocks\.(\d+)\.(.*)", name)
    if dec_match:
        idx = dec_match.group(1)
        rest = dec_match.group(2)
        translated = _translate_tfc_tdf_path(rest)
        return f"dec_{idx}.{translated}" if translated else None

    # Window and freq_pad parameters (skip, not needed in MLX)
    if name in ("window", "freq_pad"):
        return None

    logger.debug(f"Unhandled ONNX weight name: {name}")
    return None


def _translate_tfc_tdf_path(path: str) -> Optional[str]:
    """Translate TFC_TDF internal paths.

    PyTorch:
        tfc.H.M.0.weight → tfc.conv_M.weight (Conv2d)
        tfc.H.M.1.weight → tfc.norm_M.bn.weight (BatchNorm)
        tdf.0.weight → tdf_linear1.weight (Linear)
        tdf.1.weight → tdf_norm1.bn.weight (BatchNorm)
        tdf.3.weight → tdf_linear2.weight (for bottleneck)
        tdf.4.weight → tdf_norm2.bn.weight
    """
    # TFC convolutions: tfc.H.M.0.param → tfc.conv_M.param
    tfc_conv_match = re.match(r"tfc\.H\.(\d+)\.0\.(.*)", path)
    if tfc_conv_match:
        layer_idx = tfc_conv_match.group(1)
        param = tfc_conv_match.group(2)
        return f"tfc.conv_{layer_idx}.{param}"

    # TFC norms: tfc.H.M.1.param → tfc.norm_M.bn.param
    tfc_norm_match = re.match(r"tfc\.H\.(\d+)\.1\.(.*)", path)
    if tfc_norm_match:
        layer_idx = tfc_norm_match.group(1)
        param = tfc_norm_match.group(2)
        return f"tfc.norm_{layer_idx}.bn.{param}"

    # TDF: 0→linear1, 1→norm1, 2→ReLU(skip), 3→linear2, 4→norm2
    tdf_match = re.match(r"tdf\.(\d+)\.(.*)", path)
    if tdf_match:
        idx = int(tdf_match.group(1))
        param = tdf_match.group(2)
        if idx == 0:
            return f"tdf_linear1.{param}"
        elif idx == 1:
            return f"tdf_norm1.bn.{param}"
        elif idx == 3:
            return f"tdf_linear2.{param}"
        elif idx == 4:
            return f"tdf_norm2.bn.{param}"
        return None

    # Direct pass-through for already-mapped paths
    return path


def load_mdx_model(
    model_path: str,
    model_data: Dict[str, Any],
) -> Tuple[ConvTDFNet, Dict[str, Any]]:
    """Load an MDX model with MLX weights.

    Tries safetensors first, falls back to ONNX conversion.
    Infers architecture parameters from ONNX weight shapes when needed.

    Args:
        model_path: Path to model file (.onnx or .safetensors)
        model_data: Model metadata dict from JSON hash lookup

    Returns:
        Tuple of (model, model_data)
    """
    if not os.path.exists(model_path):
        model = create_mdx_model(model_data)
        logger.warning(f"Model path {model_path} does not exist")
        model.eval()
        return model, model_data

    # Try safetensors first
    base = os.path.splitext(model_path)[0]
    safetensors_path = base + ".safetensors"
    if os.path.exists(safetensors_path):
        logger.info(f"Loading pre-converted safetensors from {safetensors_path}")
        weights = mx.load(safetensors_path)
        # Infer params from safetensors keys
        _override_mdx_params_from_weights(model_data, weights)
        model = create_mdx_model(model_data)
        model.load_weights(list(weights.items()), strict=False)
        model.eval()
        return model, model_data

    # Convert from ONNX
    if model_path.lower().endswith(".onnx"):
        logger.info("Converting ONNX model to MLX weights...")
        onnx_weights = _load_onnx_weights(model_path)

        # Infer architecture params from ONNX weight shapes
        dim_f = model_data["mdx_dim_f_set"]
        inferred = _infer_params_from_onnx(onnx_weights, dim_f)

        # Override model_data with inferred params
        for key, value in inferred.items():
            if key not in model_data:
                model_data[key] = value
                logger.info(f"Using inferred {key}={value}")

        # Create model with correct params
        model = create_mdx_model(model_data)

        # Get architecture params for weight mapping
        params = dict(MDX_DEFAULT_PARAMS)
        for key in (
            "num_blocks", "l", "g", "k", "bn", "bias", "optimizer", "dim_c",
        ):
            if key in model_data:
                params[key] = model_data[key]

        n = params["num_blocks"] // 2

        include_numeric = _should_include_numeric_mapping(model_path)
        if not include_numeric:
            logger.info(
                "Numeric positional mapping disabled for model %s; "
                "using named/Conv-profile conversion only.",
                os.path.basename(model_path),
            )

        mlx_weights = convert_onnx_to_mlx_weights(
            onnx_weights,
            g=params["g"],
            n=n,
            num_tdf_layers=params["l"],
            dim_f=dim_f,
            bn=params["bn"],
            include_numeric=include_numeric,
        )
        if not mlx_weights:
            raise ValueError(
                "Unsupported MDX ONNX weight schema: no compatible weights were mapped "
                f"for {os.path.basename(model_path)}."
            )
        model.load_weights(list(mlx_weights.items()), strict=False)

        # Optionally save safetensors for future use
        if os.environ.get("MLX_SAVE_SAFETENSORS") == "1":
            try:
                mx.save_safetensors(safetensors_path, dict(mlx_weights))
                logger.info(f"Saved converted weights to {safetensors_path}")
            except Exception as e:
                logger.warning(f"Could not save safetensors: {e}")

        model.eval()
        return model, model_data

    raise ValueError(f"Unsupported model format: {model_path}")


def _override_mdx_params_from_weights(
    model_data: Dict[str, Any],
    weights: Dict[str, mx.array],
):
    """Infer g from safetensors weight shapes and override model_data."""
    for key, value in weights.items():
        if key == "first_conv.weight":
            # Shape: (g, H, W, dim_c) in MLX OHWI format
            g = value.shape[0]
            if "g" not in model_data:
                model_data["g"] = g
                logger.info(f"Inferred g={g} from safetensors")
            break


def _should_include_numeric_mapping(model_path: str) -> bool:
    base = os.path.basename(model_path).lower()
    return base not in _NUMERIC_MAPPING_DISABLED_BASENAMES


def _has_structured_named_weights(named_weights: Dict[str, np.ndarray]) -> bool:
    prefixes = (
        "first_conv.",
        "encoding_blocks.",
        "decoding_blocks.",
        "bottleneck_block.",
        "ds.",
        "us.",
    )
    return any(name.startswith(prefixes) for name in named_weights)


def _extract_conv_profile_sequence(named_weights: Dict[str, np.ndarray]) -> List[Tuple[int, np.ndarray]]:
    """Build positional Conv_* sequence for ONNX exports with unnamed kernels."""
    grouped: Dict[int, Dict[str, np.ndarray]] = {}
    for name, weight in named_weights.items():
        m = re.match(r"Conv_(\d+)\.(weight|bias)$", name)
        if not m:
            continue
        idx = int(m.group(1))
        grouped.setdefault(idx, {})[m.group(2)] = weight

    if not grouped:
        return []

    ordered: List[Tuple[int, np.ndarray]] = []
    flat_idx = 0
    for conv_idx in sorted(grouped):
        pair = grouped[conv_idx]
        if "weight" in pair:
            ordered.append((flat_idx, pair["weight"]))
            flat_idx += 1
        if "bias" in pair:
            ordered.append((flat_idx, pair["bias"]))
            flat_idx += 1
    return ordered
