"""Tests for fused Metal kernel runtime controls and fallback math."""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

from mlx_audio_separator.demucs_mlx import metal_kernels as mk


def test_stable_threadgroup_size_default_allows_large_groups():
    key = "MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED"
    previous = os.environ.pop(key, None)
    try:
        size = mk._stable_threadgroup_size(33072)
        assert size == 1024
    finally:
        if previous is not None:
            os.environ[key] = previous


def test_stable_threadgroup_size_deterministic_caps_to_256():
    key = "MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED"
    previous = os.environ.get(key)
    os.environ[key] = "1"
    try:
        size = mk._stable_threadgroup_size(33072)
        assert size == 256
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def test_fused_groupnorm_mode_defaults_to_all():
    key_mode = "MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE"
    key_det = "MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED"
    previous_mode = os.environ.pop(key_mode, None)
    previous_det = os.environ.pop(key_det, None)
    try:
        assert mk._fused_groupnorm_mode() == "all"
    finally:
        if previous_mode is not None:
            os.environ[key_mode] = previous_mode
        if previous_det is not None:
            os.environ[key_det] = previous_det


def test_fused_groupnorm_mode_defaults_to_gelu_only_in_deterministic_mode():
    key_mode = "MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE"
    key_det = "MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED"
    previous_mode = os.environ.pop(key_mode, None)
    previous_det = os.environ.get(key_det)
    os.environ[key_det] = "1"
    try:
        assert mk._fused_groupnorm_mode() == "gelu_only"
    finally:
        if previous_mode is not None:
            os.environ[key_mode] = previous_mode
        if previous_det is None:
            os.environ.pop(key_det, None)
        else:
            os.environ[key_det] = previous_det


def test_fused_groupnorm_mode_aliases():
    key = "MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE"
    previous = os.environ.get(key)
    try:
        os.environ[key] = "glu"
        assert mk._fused_groupnorm_mode() == "glu_only"
        os.environ[key] = "gelu_only"
        assert mk._fused_groupnorm_mode() == "gelu_only"
        os.environ[key] = "off"
        assert mk._fused_groupnorm_mode() == "off"
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def test_explicit_threadgroup_cap_parsing():
    key = "MLX_AUDIO_SEPARATOR_GN_GELU_TG_CAP"
    previous = os.environ.get(key)
    try:
        os.environ[key] = "190"
        assert mk._explicit_threadgroup_cap(key) == 192
        os.environ[key] = "abc"
        assert mk._explicit_threadgroup_cap(key) is None
        os.environ[key] = "4096"
        assert mk._explicit_threadgroup_cap(key) == 1024
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def test_fused_groupnorm_glu_impl_defaults_to_hybrid():
    key = "MLX_AUDIO_SEPARATOR_GN_GLU_IMPL"
    previous = os.environ.pop(key, None)
    try:
        assert mk._fused_groupnorm_glu_impl() == "hybrid"
        os.environ[key] = "legacy"
        assert mk._fused_groupnorm_glu_impl() == "legacy"
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def test_groupnorm_gelu_fallback_float16_matches_fp32_reference():
    x = mx.random.normal((1, 64, 128), dtype=mx.float16)
    w = mx.random.normal((64,), dtype=mx.float32)
    b = mx.random.normal((64,), dtype=mx.float32)
    out = mk._groupnorm_gelu_fallback(x, w, b, num_groups=4, eps=1e-5)
    mx.eval(out)

    x32 = x.astype(mx.float32)
    xr = x32.reshape(1, 4, 16, 128)
    mean = xr.mean(axis=(2, 3), keepdims=True)
    var = ((xr - mean) ** 2).mean(axis=(2, 3), keepdims=True)
    ref = ((xr - mean) * mx.rsqrt(var + 1e-5)).reshape(x.shape)
    ref = ref * w.reshape(1, 64, 1) + b.reshape(1, 64, 1)
    ref = nn.gelu(ref).astype(mx.float16)
    mx.eval(ref)

    diff = (out.astype(mx.float32) - ref.astype(mx.float32)).abs().max()
    assert float(diff.item()) <= 1e-3


def test_groupnorm_glu_fallback_float16_matches_fp32_reference():
    x = mx.random.normal((1, 128, 128), dtype=mx.float16)
    w = mx.random.normal((128,), dtype=mx.float32)
    b = mx.random.normal((128,), dtype=mx.float32)
    out = mk._groupnorm_glu_fallback(x, w, b, num_groups=1, eps=1e-5)
    mx.eval(out)

    x32 = x.astype(mx.float32)
    xr = x32.reshape(1, 1, 128, 128)
    mean = xr.mean(axis=(2, 3), keepdims=True)
    var = ((xr - mean) ** 2).mean(axis=(2, 3), keepdims=True)
    ref = ((xr - mean) * mx.rsqrt(var + 1e-5)).reshape(x.shape)
    ref = ref * w.reshape(1, 128, 1) + b.reshape(1, 128, 1)
    a, bb = mx.split(ref, 2, axis=1)
    ref = (a * mx.sigmoid(bb)).astype(mx.float16)
    mx.eval(ref)

    diff = (out.astype(mx.float32) - ref.astype(mx.float32)).abs().max()
    assert float(diff.item()) <= 1e-3
