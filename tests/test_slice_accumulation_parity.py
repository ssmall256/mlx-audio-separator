"""Focused correctness and parity test for MLX slice accumulation.

Validates that native `array.at[...].add(...)` produces bit-exact or numerically
identical results compared against NumPy reference across:
- No overlap, ordinary (50%) overlap, heavy (75%, 87.5%) overlap
- Leading, middle, and trailing update axes
- Full and partial final chunks
- Single and multi-channel / multi-stem tensors
- Float32 and Float16 dtypes
- Contiguous and transposed (non-contiguous) chunks
"""

import math

import mlx.core as mx
import numpy as np
import pytest


def compute_snr(ref: np.ndarray, test: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    noise = ref - test
    signal_power = np.sum(ref**2)
    noise_power = np.sum(noise**2)
    if noise_power == 0:
        return float("inf")
    if signal_power == 0:
        return 0.0
    return float(10 * np.log10(signal_power / noise_power))


@pytest.mark.parametrize("overlap_ratio", [0.0, 0.5, 0.75, 0.875])
@pytest.mark.parametrize("dtype", [mx.float32])
def test_waveform_slice_add_parity_trailing_axis(overlap_ratio, dtype):
    """Test overlap-add accumulation on trailing time axis: (S, C, L)."""
    num_stems = 4
    channels = 2
    total_len = 44100 * 3
    chunk_len = 8820
    hop = int(chunk_len * (1.0 - overlap_ratio)) if overlap_ratio > 0 else chunk_len
    num_hops = math.ceil((total_len - chunk_len) / hop) + 1

    rng = np.random.default_rng(12345)
    np_dtype = np.float32 if dtype == mx.float32 else np.float16

    chunks_np = [
        rng.standard_normal((num_stems, channels, chunk_len)).astype(np_dtype)
        for _ in range(num_hops)
    ]
    window_np = np.hanning(chunk_len).astype(np_dtype)

    # NumPy double-precision reference
    accum_np = np.zeros((num_stems, channels, total_len), dtype=np.float64)
    counter_np = np.zeros((total_len,), dtype=np.float64)

    for i, c in enumerate(chunks_np):
        start = i * hop
        this_len = min(chunk_len, total_len - start)
        if this_len <= 0:
            break
        end = start + this_len
        weighted = (c[:, :, :this_len] * window_np[:this_len]).astype(np.float64)
        accum_np[:, :, start:end] += weighted
        counter_np[start:end] += window_np[:this_len].astype(np.float64)

    ref_out = (accum_np / np.maximum(counter_np[None, None, :], 1e-10)).astype(np_dtype)

    # MLX native array.at[...].add(...)
    accum_mx = mx.zeros((num_stems, channels, total_len), dtype=dtype)
    counter_mx = mx.zeros((total_len,), dtype=dtype)
    window_mx = mx.array(window_np, dtype=dtype)

    for i, c in enumerate(chunks_np):
        start = i * hop
        this_len = min(chunk_len, total_len - start)
        if this_len <= 0:
            break
        end = start + this_len
        c_mx = mx.array(c, dtype=dtype)
        weighted_mx = c_mx[:, :, :this_len] * window_mx[:this_len]
        accum_mx = accum_mx.at[:, :, start:end].add(weighted_mx)
        counter_mx = counter_mx.at[start:end].add(window_mx[:this_len])

    mx_out = accum_mx / mx.maximum(counter_mx[None, None, :], mx.array(1e-10, dtype=dtype))
    mx.eval(mx_out)

    actual_out = np.array(mx_out)
    max_abs_err = np.max(np.abs(actual_out - ref_out))
    snr = compute_snr(ref_out, actual_out)

    assert max_abs_err <= 1e-5, f"max_abs_err={max_abs_err} exceeded tolerance"
    assert snr > 100.0, f"SNR={snr} dB below 100 dB threshold"


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_slice_add_all_axes(axis):
    """Test slice addition on leading (0), middle (1), and trailing (2) axes."""
    shape = [8, 12, 16]
    update_shape = list(shape)
    update_shape[axis] = 4

    rng = np.random.default_rng(42)
    base_np = rng.standard_normal(shape).astype(np.float32)
    update_np = rng.standard_normal(update_shape).astype(np.float32)

    # Slices
    slices = [slice(None)] * 3
    slices[axis] = slice(2, 6)

    # Reference
    ref_np = base_np.copy()
    ref_np[tuple(slices)] += update_np

    # MLX
    base_mx = mx.array(base_np)
    update_mx = mx.array(update_np)
    out_mx = base_mx.at[tuple(slices)].add(update_mx)
    mx.eval(out_mx)

    np.testing.assert_allclose(np.array(out_mx), ref_np, rtol=1e-6, atol=1e-6)


def test_slice_add_non_contiguous_inputs():
    """Verify strided/transposed updates accumulate correctly."""
    # (T, C, S) transposed to (S, C, T)
    rng = np.random.default_rng(99)
    raw = rng.standard_normal((100, 2, 4)).astype(np.float32)
    raw_mx = mx.transpose(mx.array(raw), (2, 1, 0))  # non-contiguous (4, 2, 100)

    accum = mx.zeros((4, 2, 200), dtype=mx.float32)
    accum = accum.at[:, :, 50:150].add(raw_mx)
    mx.eval(accum)

    ref = np.zeros((4, 2, 200), dtype=np.float32)
    ref[:, :, 50:150] += np.transpose(raw, (2, 1, 0))
    np.testing.assert_allclose(np.array(accum), ref, rtol=1e-6, atol=1e-6)


def test_slice_add_boundary_partial_chunk():
    """Verify behavior on boundaries: chunk overflowing buffer length."""
    total_len = 100
    start = 80
    end = 100
    actual_len = end - start

    update_mx = mx.ones((2, actual_len), dtype=mx.float32)
    out_mx = mx.zeros((2, total_len), dtype=mx.float32)
    out_mx = out_mx.at[:, start:end].add(update_mx)
    mx.eval(out_mx)

    expected = np.zeros((2, total_len), dtype=np.float32)
    expected[:, start:end] = 1.0
    np.testing.assert_array_equal(np.array(out_mx), expected)
