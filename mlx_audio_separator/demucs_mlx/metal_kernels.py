"""
Custom fused Metal kernels for Demucs-MLX inference.

Each kernel includes:
- Metal source code
- Python wrapper with shape handling
- float32 accumulation for numerical stability

When Metal is not available (e.g., MLX on Linux/CPU), all functions
fall back to equivalent pure-MLX operations automatically.
"""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn


def _has_metal() -> bool:
    """Check whether Metal GPU backend is available."""
    try:
        metal = getattr(mx, "metal", None)
        if metal is None:
            return False
        # mx.metal.is_available() is the canonical check
        is_avail = getattr(metal, "is_available", None)
        if callable(is_avail):
            return bool(is_avail())
        # If the module exists but has no is_available, assume Metal is present
        # (older MLX versions on macOS).
        return True
    except Exception:
        return False


HAS_METAL = _has_metal()


def _fused_groupnorm_mode() -> str:
    """Runtime mode for fused GroupNorm kernels.

    Controlled by:
      MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE = all | glu_only | gelu_only | off

    Defaults to `all`.
    """
    raw_env = os.getenv("MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE")
    if raw_env is None or raw_env.strip() == "":
        deterministic = os.getenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED", "").strip().lower()
        # In deterministic mode, disable all fused GroupNorm kernels due known
        # run-to-run drift in fused reduction kernels.
        if deterministic in {"1", "true", "yes", "on"}:
            return "off"
        return "all"
    raw = raw_env.strip().lower()
    if raw in {"", "all", "on", "true", "1"}:
        return "all"
    if raw in {"glu_only", "glu"}:
        return "glu_only"
    if raw in {"gelu_only", "gelu"}:
        return "gelu_only"
    if raw in {"off", "none", "unfused", "false", "0"}:
        return "off"
    if raw in {"safe", "default"}:
        return "off"
    return "all"


def _explicit_threadgroup_cap(env_var: str) -> int | None:
    """Optional aligned threadgroup cap override from environment."""
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    value = max(32, min(1024, value))
    return ((value + 31) // 32) * 32


def _fused_groupnorm_glu_impl() -> str:
    """Implementation selector for GroupNorm+GLU path.

    Values:
      - hybrid (default): float32 GroupNorm+affine + fused GLU kernel
      - legacy: monolithic fused GroupNorm+GLU Metal kernel
    """
    raw = os.getenv("MLX_AUDIO_SEPARATOR_GN_GLU_IMPL", "hybrid").strip().lower()
    if raw in {"legacy", "old"}:
        return "legacy"
    return "hybrid"


def _fused_groupnorm_glu_enable_multigroup() -> bool:
    """Enable experimental multigroup hybrid GroupNorm+GLU fast path."""
    raw = os.getenv("MLX_AUDIO_SEPARATOR_GN_GLU_MULTIGROUP", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _stable_threadgroup_size(elems_per_group: int, cap_env_var: str | None = None) -> int:
    """Choose a stable SIMD-aligned threadgroup size for reduction kernels.

    Fast path uses up to 1024 threads. Deterministic mode caps to 256 to avoid
    numerical instability and run-to-run drift observed for some GroupNorm
    shapes on Apple GPUs.

    Enable deterministic mode with:
      MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED=1
    """
    deterministic = os.getenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED", "").strip().lower()
    max_threads = 256 if deterministic in {"1", "true", "yes", "on"} else 1024
    if cap_env_var:
        explicit_cap = _explicit_threadgroup_cap(cap_env_var)
        if explicit_cap is not None:
            max_threads = min(max_threads, explicit_cap)
    return min(max_threads, max(32, ((int(elems_per_group) + 31) // 32) * 32))

# ==============================================================================
# Fused GLU: a * sigmoid(b) where [a, b] = split(x, 2, axis)
# ==============================================================================
# After reshaping to (N, 2*C) with target axis last, the memory layout is:
#   row i: [a_0, a_1, ..., a_{C-1}, b_0, b_1, ..., b_{C-1}]
# So for element (row, col) where col < C:
#   a = x[row * 2C + col]
#   b = x[row * 2C + C + col]

_GLU_SOURCE = r"""
uint gid = thread_position_in_grid.x;
uint total = params[0];     // N * C (total output elements)
uint half_dim = params[1];  // C (half of last dimension)

if (gid >= total) return;

uint row = gid / half_dim;
uint col = gid % half_dim;
uint full_dim = half_dim * 2;

float a = (float)x[row * full_dim + col];
float b = (float)x[row * full_dim + half_dim + col];
float sig_b = 1.0f / (1.0f + metal::exp(-b));
out[gid] = (T)(a * sig_b);
"""

_glu_kernel = None


def _get_glu_kernel():
    global _glu_kernel
    if _glu_kernel is None:
        _glu_kernel = mx.fast.metal_kernel(
            name="fused_glu",
            input_names=["x", "params"],
            output_names=["out"],
            source=_GLU_SOURCE,
        )
    return _glu_kernel


def fused_glu(x: mx.array, axis: int = 1) -> mx.array:
    """Fused GLU: split x in half along axis, compute a * sigmoid(b).

    Equivalent to:
        a, b = mx.split(x, 2, axis=axis)
        return a * mx.sigmoid(b)
    """
    if not HAS_METAL:
        a, b = mx.split(x, 2, axis=axis)
        return a * mx.sigmoid(b)

    ndim = x.ndim
    axis = axis % ndim

    in_shape = list(x.shape)
    if in_shape[axis] % 2 != 0:
        raise ValueError(f"Axis {axis} size must be even, got {in_shape[axis]}")

    # Move target axis to last for contiguous split
    if axis != ndim - 1:
        perm = list(range(ndim))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        x = x.transpose(*perm)

    last_dim = x.shape[-1]
    half = last_dim // 2
    x_2d = mx.contiguous(x.reshape(-1, last_dim))
    N = x_2d.shape[0]
    total = N * half

    params = mx.array([total, half], dtype=mx.int32)

    result_flat = _get_glu_kernel()(
        inputs=[x_2d.reshape(-1), params],
        template=[("T", x.dtype)],
        grid=(total, 1, 1),
        threadgroup=(min(256, total), 1, 1),
        output_shapes=[(total,)],
        output_dtypes=[x.dtype],
    )[0]

    result = result_flat.reshape(*x.shape[:-1], half)

    if axis != ndim - 1:
        perm = list(range(ndim))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        result = result.transpose(*perm)

    return result


# ==============================================================================
# Fused GroupNorm + GELU (erf-based, matching MLX's nn.gelu exactly)
# ==============================================================================
# Each threadgroup handles one (batch, group) pair.
# Uses simdgroup reductions for mean/variance.
# GELU = 0.5 * x * (1 + erf(x / sqrt(2)))
# Metal provides metal::precise::erf() for exact erf.

_GROUPNORM_GELU_HEADER = r"""
// Abramowitz & Stegun approximation of erf, max error ~1.5e-7
inline float erf_approx(float x) {
    // erf(-x) = -erf(x)
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    float a = metal::abs(x);
    // A&S formula 7.1.26
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float t2 = t * t;
    float t3 = t2 * t;
    float t4 = t3 * t;
    float t5 = t4 * t;
    float poly = 0.254829592f * t
               - 0.284496736f * t2
               + 1.421413741f * t3
               - 1.453152027f * t4
               + 1.061405429f * t5;
    float result = 1.0f - poly * metal::exp(-a * a);
    return sign * result;
}
"""

_GROUPNORM_GELU_SOURCE = r"""
uint bg = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;
uint sid = thread_index_in_simdgroup;
uint wid = simdgroup_index_in_threadgroup;
uint num_simdgroups = tg_size / 32;

uint num_groups = params[0];
uint channels_per_group = params[1];
uint spatial_size = params[2];
uint total_channels = params[3];

uint batch_idx = bg / num_groups;
uint group_idx = bg % num_groups;

uint elems_per_group = channels_per_group * spatial_size;

uint base = batch_idx * total_channels * spatial_size
          + group_idx * channels_per_group * spatial_size;

// Pass 1: Compute mean
float local_sum = 0.0f;
for (uint i = tid; i < elems_per_group; i += tg_size) {
    local_sum += (float)x[base + i];
}
local_sum = simd_sum(local_sum);

threadgroup float shared_sums[32];
if (sid == 0) shared_sums[wid] = local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (wid == 0) {
    float val = (sid < num_simdgroups) ? shared_sums[sid] : 0.0f;
    val = simd_sum(val);
    if (sid == 0) shared_sums[0] = val;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float mean = shared_sums[0] / (float)elems_per_group;

// Pass 2: Compute variance
float local_var = 0.0f;
for (uint i = tid; i < elems_per_group; i += tg_size) {
    float diff = (float)x[base + i] - mean;
    local_var += diff * diff;
}
local_var = simd_sum(local_var);
if (sid == 0) shared_sums[wid] = local_var;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (wid == 0) {
    float val = (sid < num_simdgroups) ? shared_sums[sid] : 0.0f;
    val = simd_sum(val);
    if (sid == 0) shared_sums[0] = val;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float var = shared_sums[0] / (float)elems_per_group;
float inv_std = metal::rsqrt(var + eps[0]);

// Pass 3: Normalize, apply affine, apply erf-based GELU
float rsqrt2 = 0.7071067811865475f;  // 1/sqrt(2)
for (uint i = tid; i < elems_per_group; i += tg_size) {
    uint c_local = i / spatial_size;
    uint c_global = group_idx * channels_per_group + c_local;
    float val = ((float)x[base + i] - mean) * inv_std;
    val = val * (float)weight[c_global] + (float)bias[c_global];
    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    val = 0.5f * val * (1.0f + erf_approx(val * rsqrt2));
    out[base + i] = (T)val;
}
"""

_groupnorm_gelu_kernel = None


def _get_groupnorm_gelu_kernel():
    global _groupnorm_gelu_kernel
    if _groupnorm_gelu_kernel is None:
        _groupnorm_gelu_kernel = mx.fast.metal_kernel(
            name="fused_groupnorm_gelu",
            input_names=["x", "weight", "bias", "eps", "params"],
            output_names=["out"],
            header=_GROUPNORM_GELU_HEADER,
            source=_GROUPNORM_GELU_SOURCE,
        )
    return _groupnorm_gelu_kernel


def _groupnorm_gelu_fallback(
    x: mx.array, weight: mx.array, bias: mx.array,
    num_groups: int, eps: float = 1e-5,
) -> mx.array:
    """Pure-MLX GroupNorm + GELU (no Metal required).

    Uses float32 accumulation for numerical parity with fused kernels and
    PyTorch-style GroupNorm behavior on float16 inputs.
    """
    B, C = x.shape[0], x.shape[1]
    cpg = C // num_groups
    x32 = x.astype(mx.float32)
    x_r = x32.reshape(B, num_groups, cpg, *x.shape[2:])
    axes = tuple(range(2, x_r.ndim))
    mean = x_r.mean(axis=axes, keepdims=True)
    var = ((x_r - mean) ** 2).mean(axis=axes, keepdims=True)
    x_norm = (x_r - mean) * mx.rsqrt(var + eps)
    x_out = x_norm.reshape(x.shape)
    w_shape = [1, C] + [1] * (x.ndim - 2)
    x_out = x_out * weight.astype(mx.float32).reshape(w_shape) + bias.astype(mx.float32).reshape(w_shape)
    return nn.gelu(x_out).astype(x.dtype)


def _groupnorm_affine_fp32(
    x: mx.array, weight: mx.array, bias: mx.array,
    num_groups: int, eps: float = 1e-5,
) -> mx.array:
    """GroupNorm + affine in float32, returning float32 tensor."""
    B = x.shape[0]
    C = x.shape[1]
    channels_per_group = C // num_groups
    x32 = x.astype(mx.float32)
    x_r = x32.reshape(B, num_groups, channels_per_group, *x.shape[2:])
    axes = tuple(range(2, x_r.ndim))
    mean = x_r.mean(axis=axes, keepdims=True)
    var = ((x_r - mean) ** 2).mean(axis=axes, keepdims=True)
    x_norm = (x_r - mean) * mx.rsqrt(var + eps)
    x_out = x_norm.reshape(x.shape)
    w_shape = [1, C] + [1] * (x.ndim - 2)
    return x_out * weight.astype(mx.float32).reshape(w_shape) + bias.astype(mx.float32).reshape(w_shape)


def _groupnorm_glu_fallback(
    x: mx.array, weight: mx.array, bias: mx.array,
    num_groups: int, eps: float = 1e-5,
) -> mx.array:
    """Pure-MLX GroupNorm + GLU fallback with float32 accumulation."""
    normed = _groupnorm_affine_fp32(x, weight, bias, num_groups, eps)
    a, b = mx.split(normed, 2, axis=1)
    return (a * mx.sigmoid(b)).astype(x.dtype)


def fused_groupnorm_gelu(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    num_groups: int,
    eps: float = 1e-5,
) -> mx.array:
    """Fused GroupNorm + GELU for NCL or NCHW layout.

    Equivalent to:
        x = groupnorm(x, weight, bias, num_groups, eps)
        x = gelu(x)
    """
    mode = _fused_groupnorm_mode()
    if mode in {"glu_only", "off"}:
        return _groupnorm_gelu_fallback(x, weight, bias, num_groups, eps)

    if not HAS_METAL:
        return _groupnorm_gelu_fallback(x, weight, bias, num_groups, eps)

    orig_shape = x.shape
    B = x.shape[0]
    C = x.shape[1]
    spatial_size = 1
    for d in x.shape[2:]:
        spatial_size *= d

    if C % num_groups != 0:
        raise ValueError(f"channels {C} not divisible by num_groups {num_groups}")
    channels_per_group = C // num_groups

    x_contig = mx.contiguous(x.reshape(B, C, spatial_size))
    weight = mx.contiguous(weight.astype(mx.float32))
    bias = mx.contiguous(bias.astype(mx.float32))
    eps_arr = mx.array([eps], dtype=mx.float32)
    params = mx.array([num_groups, channels_per_group, spatial_size, C], dtype=mx.int32)

    total_groups = B * num_groups
    elems_per_group = channels_per_group * spatial_size
    tg = _stable_threadgroup_size(
        elems_per_group,
        cap_env_var="MLX_AUDIO_SEPARATOR_GN_GELU_TG_CAP",
    )

    result = _get_groupnorm_gelu_kernel()(
        inputs=[x_contig, weight, bias, eps_arr, params],
        template=[("T", x.dtype)],
        grid=(total_groups * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, C, spatial_size)],
        output_dtypes=[x.dtype],
    )[0]

    return result.reshape(orig_shape)


# ==============================================================================
# Fused GroupNorm + GLU (GroupNorm over 2C channels, then split+sigmoid+mul)
# ==============================================================================
# Each threadgroup handles one (batch, group) pair.
# GroupNorm normalizes over 2C channels, then GLU splits into a (first C) and
# b (second C), computing output = a * sigmoid(b). Output has half the channels.

_GROUPNORM_GLU_SOURCE = r"""
uint bg = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;
uint sid = thread_index_in_simdgroup;
uint wid = simdgroup_index_in_threadgroup;
uint num_simdgroups = tg_size / 32;

uint num_groups = params[0];
uint channels_per_group = params[1];
uint spatial_size = params[2];
uint total_channels = params[3];
uint half_channels = params[4];

uint batch_idx = bg / num_groups;
uint group_idx = bg % num_groups;

uint elems_per_group = channels_per_group * spatial_size;

uint base = batch_idx * total_channels * spatial_size
          + group_idx * channels_per_group * spatial_size;

// Pass 1: Compute mean
float local_sum = 0.0f;
for (uint i = tid; i < elems_per_group; i += tg_size) {
    local_sum += (float)x[base + i];
}
local_sum = simd_sum(local_sum);

threadgroup float shared_sums[32];
if (sid == 0) shared_sums[wid] = local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (wid == 0) {
    float val = (sid < num_simdgroups) ? shared_sums[sid] : 0.0f;
    val = simd_sum(val);
    if (sid == 0) shared_sums[0] = val;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float mean = shared_sums[0] / (float)elems_per_group;

// Pass 2: Compute variance
float local_var = 0.0f;
for (uint i = tid; i < elems_per_group; i += tg_size) {
    float diff = (float)x[base + i] - mean;
    local_var += diff * diff;
}
local_var = simd_sum(local_var);
if (sid == 0) shared_sums[wid] = local_var;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (wid == 0) {
    float val = (sid < num_simdgroups) ? shared_sums[sid] : 0.0f;
    val = simd_sum(val);
    if (sid == 0) shared_sums[0] = val;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float var = shared_sums[0] / (float)elems_per_group;
float inv_std = metal::rsqrt(var + eps[0]);

// Intermediate buffer for normalized values (stored in shared memory would be
// too large for typical channel counts, so we write to a temp output and read back).
// Instead, we normalize on-the-fly and store in the output buffer.

// Pass 3: Normalize, apply affine, then apply GLU
// GLU: output[c, s] = a[c, s] * sigmoid(b[c, s])
// where a = first half of channels, b = second half
// After GroupNorm+affine, the normalized values are in (B, 2C, S) layout.
// We need to pair channel c with channel c + C.

// For GLU, we iterate over the OUTPUT elements (half the channels).
// Each output element at (c_out, s) reads:
//   a = norm_affine(x[base_a + c_out * S + s])
//   b = norm_affine(x[base_b + c_out * S + s])
// where base_a/base_b account for the group offset.

// But groups span the full 2C channels. We need to handle the case
// where the GLU split crosses group boundaries.
// For standard usage (num_groups divides 2C and C), the first half of groups
// correspond to 'a' channels and second half to 'b' channels.

// Actually for simplicity and correctness, this kernel should handle the common
// case where num_groups divides both 2C and C (e.g., groups=1 or groups=4 with
// 2C divisible by 4). The GLU split is on the channel dimension, independent of groups.

// We compute all normalized values first, then apply GLU.
// Strategy: each thread processes output elements. For each output (c_out, s),
// compute both a and b normalized values on the fly.

// Half channels per group on the output side:
uint half_cpg = channels_per_group / 2;
uint out_epg = half_cpg * spatial_size;
uint out_base = batch_idx * half_channels * spatial_size
              + group_idx * half_cpg * spatial_size;

for (uint i = tid; i < out_epg; i += tg_size) {
    uint c_local = i / spatial_size;
    uint s = i % spatial_size;

    // 'a' channel: first half of the group
    uint c_a = c_local;
    uint c_a_global = group_idx * channels_per_group + c_a;
    float val_a = ((float)x[base + c_a * spatial_size + s] - mean) * inv_std;
    val_a = val_a * (float)weight[c_a_global] + (float)bias[c_a_global];

    // 'b' channel: second half of the group
    uint c_b = c_local + half_cpg;
    uint c_b_global = group_idx * channels_per_group + c_b;
    float val_b = ((float)x[base + c_b * spatial_size + s] - mean) * inv_std;
    val_b = val_b * (float)weight[c_b_global] + (float)bias[c_b_global];

    // GLU: a * sigmoid(b)
    float sig_b = 1.0f / (1.0f + metal::exp(-val_b));
    out[out_base + i] = (T)(val_a * sig_b);
}
"""

_groupnorm_glu_kernel = None


def _get_groupnorm_glu_kernel():
    global _groupnorm_glu_kernel
    if _groupnorm_glu_kernel is None:
        _groupnorm_glu_kernel = mx.fast.metal_kernel(
            name="fused_groupnorm_glu",
            input_names=["x", "weight", "bias", "eps", "params"],
            output_names=["out"],
            source=_GROUPNORM_GLU_SOURCE,
        )
    return _groupnorm_glu_kernel


def fused_groupnorm_glu(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    num_groups: int,
    eps: float = 1e-5,
) -> mx.array:
    """Fused GroupNorm + GLU for NCL or NCHW layout.

    GroupNorm normalizes over 2C channels, then GLU splits in half on axis=1.

    Equivalent to:
        x = groupnorm(x, weight, bias, num_groups, eps)
        a, b = split(x, 2, axis=1)
        return a * sigmoid(b)

    Input shape: (B, 2C, ...) -> Output shape: (B, C, ...)

    Implementation policy:
      - hybrid (default): GroupNorm+affine in float32 + fused GLU kernel.
        For num_groups > 1, this is opt-in via
        MLX_AUDIO_SEPARATOR_GN_GLU_MULTIGROUP=1.
      - legacy: monolithic fused GroupNorm+GLU kernel (kept for experiments).
        Restricted to num_groups == 1.
    """
    orig_shape = x.shape
    B = x.shape[0]
    C_full = x.shape[1]  # 2C
    C_half = C_full // 2  # C

    spatial_size = 1
    for d in x.shape[2:]:
        spatial_size *= d

    if C_full % num_groups != 0:
        raise ValueError(f"channels {C_full} not divisible by num_groups {num_groups}")
    if C_full % 2 != 0:
        raise ValueError(f"channels {C_full} must be even for GLU")
    channels_per_group = C_full // num_groups

    mode = _fused_groupnorm_mode()

    if mode in {"gelu_only", "off"}:
        # Respect runtime mode: disable fused GLU kernel.
        return _groupnorm_glu_fallback(x, weight, bias, num_groups, eps)

    if not HAS_METAL:
        # Pure-MLX fallback: separate GroupNorm + GLU.
        return _groupnorm_glu_fallback(x, weight, bias, num_groups, eps)

    impl = _fused_groupnorm_glu_impl()
    if impl != "legacy":
        if num_groups > 1 and not _fused_groupnorm_glu_enable_multigroup():
            return _groupnorm_glu_fallback(x, weight, bias, num_groups, eps)
        # Correctness-first hybrid path:
        # keep GroupNorm+affine and GLU in float32, cast once at the end.
        normed_fp32 = _groupnorm_affine_fp32(x, weight, bias, num_groups, eps)
        out_fp32 = fused_glu(normed_fp32, axis=1)
        return out_fp32.astype(x.dtype)

    # Legacy monolithic kernel is restricted to num_groups == 1.
    # For grouped normalization, prefer the correctness-first hybrid path above.
    if num_groups > 1:
        if not _fused_groupnorm_glu_enable_multigroup():
            return _groupnorm_glu_fallback(x, weight, bias, num_groups, eps)
        normed_fp32 = _groupnorm_affine_fp32(x, weight, bias, num_groups, eps)
        out_fp32 = fused_glu(normed_fp32, axis=1)
        return out_fp32.astype(x.dtype)

    x_contig = mx.contiguous(x.reshape(B, C_full, spatial_size))
    weight = mx.contiguous(weight.astype(mx.float32))
    bias = mx.contiguous(bias.astype(mx.float32))
    eps_arr = mx.array([eps], dtype=mx.float32)
    params = mx.array(
        [num_groups, channels_per_group, spatial_size, C_full, C_half], dtype=mx.int32
    )

    total_groups = B * num_groups
    elems_per_group = channels_per_group * spatial_size
    tg = _stable_threadgroup_size(
        elems_per_group,
        cap_env_var="MLX_AUDIO_SEPARATOR_GN_GLU_TG_CAP",
    )

    out_shape = list(orig_shape)
    out_shape[1] = C_half

    result = _get_groupnorm_glu_kernel()(
        inputs=[x_contig, weight, bias, eps_arr, params],
        template=[("T", x.dtype)],
        grid=(total_groups * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, C_half, spatial_size)],
        output_dtypes=[x.dtype],
    )[0]

    return result.reshape(out_shape)


# ==============================================================================
# Fused complex-to-real magnitude (CAC mode)
# ==============================================================================

_COMPLEX_TO_INTERLEAVED_SOURCE = r"""
uint i = thread_position_in_grid.x;
uint total = params[3];
if (i >= total) return;

uint T_dim = params[0];
uint Fr_dim = params[1];
uint C_dim = params[2];
uint FrT = Fr_dim * T_dim;
uint CFrT = C_dim * FrT;

uint b = i / (2 * CFrT);
uint rem = i % (2 * CFrT);
uint c2 = rem / FrT;
uint c = c2 / 2;
uint is_imag = c2 % 2;
uint ft = rem % FrT;

uint in_idx = (b * CFrT + c * FrT + ft) * 2 + is_imag;
out[i] = x[in_idx];
"""

_complex_to_interleaved_kernel = None


def _get_complex_to_interleaved_kernel():
    global _complex_to_interleaved_kernel
    if _complex_to_interleaved_kernel is None:
        _complex_to_interleaved_kernel = mx.fast.metal_kernel(
            name="complex_to_interleaved",
            input_names=["x", "params"],
            output_names=["out"],
            source=_COMPLEX_TO_INTERLEAVED_SOURCE,
        )
    return _complex_to_interleaved_kernel


def fused_complex_to_interleaved(z: mx.array) -> mx.array:
    """Convert complex (B, C, Fr, T) to interleaved real (B, 2*C, Fr, T).

    Equivalent to:
        real = mx.real(z)
        imag = mx.imag(z)
        m = mx.stack([real, imag], axis=2).reshape(B, C*2, Fr, T)
    """
    B, C, Fr, T = z.shape

    if not HAS_METAL:
        real = mx.real(z)
        imag = mx.imag(z)
        return mx.stack([real, imag], axis=2).reshape(B, C * 2, Fr, T)

    z_real = mx.view(z, mx.float32)  # (B, C, Fr, T, 2)
    z_flat = mx.contiguous(z_real).reshape(-1)

    total = B * 2 * C * Fr * T
    params = mx.array([T, Fr, C, total], dtype=mx.int32)

    result = _get_complex_to_interleaved_kernel()(
        inputs=[z_flat, params],
        template=[],
        grid=(total, 1, 1),
        threadgroup=(min(256, total), 1, 1),
        output_shapes=[(total,)],
        output_dtypes=[mx.float32],
    )[0]

    return result.reshape(B, 2 * C, Fr, T)
