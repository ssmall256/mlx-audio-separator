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


def test_fused_groupnorm_mode_defaults_to_off_in_deterministic_mode():
    key_mode = "MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE"
    key_det = "MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED"
    previous_mode = os.environ.pop(key_mode, None)
    previous_det = os.environ.get(key_det)
    os.environ[key_det] = "1"
    try:
        assert mk._fused_groupnorm_mode() == "off"
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


def test_fused_groupnorm_glu_enable_multigroup_flag_parsing():
    key = "MLX_AUDIO_SEPARATOR_GN_GLU_MULTIGROUP"
    previous = os.environ.get(key)
    try:
        os.environ.pop(key, None)
        assert mk._fused_groupnorm_glu_enable_multigroup() is False
        os.environ[key] = "1"
        assert mk._fused_groupnorm_glu_enable_multigroup() is True
        os.environ[key] = "true"
        assert mk._fused_groupnorm_glu_enable_multigroup() is True
        os.environ[key] = "0"
        assert mk._fused_groupnorm_glu_enable_multigroup() is False
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


def test_groupnorm_glu_hybrid_path_used_for_multigroup(monkeypatch):
    monkeypatch.setattr(mk, "HAS_METAL", True)
    monkeypatch.setattr(mk, "_fused_groupnorm_mode", lambda: "all")
    monkeypatch.setattr(mk, "_fused_groupnorm_glu_impl", lambda: "hybrid")
    monkeypatch.setattr(mk, "_fused_groupnorm_glu_enable_multigroup", lambda: True)

    calls = {"fallback": 0, "affine": 0, "glu": 0}

    def fake_fallback(x, weight, bias, num_groups, eps):
        calls["fallback"] += 1
        return x[..., : x.shape[-1]]

    def fake_affine(x, weight, bias, num_groups, eps):
        calls["affine"] += 1
        return x.astype(mx.float32)

    def fake_glu(x, axis=1):
        calls["glu"] += 1
        return mx.zeros((x.shape[0], x.shape[1] // 2, *x.shape[2:]), dtype=mx.float32)

    monkeypatch.setattr(mk, "_groupnorm_glu_fallback", fake_fallback)
    monkeypatch.setattr(mk, "_groupnorm_affine_fp32", fake_affine)
    monkeypatch.setattr(mk, "fused_glu", fake_glu)

    x = mx.random.normal((1, 8, 16), dtype=mx.float16)
    w = mx.ones((8,), dtype=mx.float32)
    b = mx.zeros((8,), dtype=mx.float32)
    out = mk.fused_groupnorm_glu(x, w, b, num_groups=2, eps=1e-5)
    mx.eval(out)

    assert calls["fallback"] == 0
    assert calls["affine"] == 1
    assert calls["glu"] == 1
    assert out.shape == (1, 4, 16)
    assert out.dtype == mx.float16


def test_groupnorm_glu_hybrid_runs_glu_in_fp32_then_cast(monkeypatch):
    monkeypatch.setattr(mk, "HAS_METAL", True)
    monkeypatch.setattr(mk, "_fused_groupnorm_mode", lambda: "all")
    monkeypatch.setattr(mk, "_fused_groupnorm_glu_impl", lambda: "hybrid")
    monkeypatch.setattr(mk, "_fused_groupnorm_glu_enable_multigroup", lambda: True)

    dtypes = {"affine_out": None, "glu_in": None}

    def fake_affine(x, weight, bias, num_groups, eps):
        out = x.astype(mx.float32)
        dtypes["affine_out"] = out.dtype
        return out

    def fake_glu(x, axis=1):
        dtypes["glu_in"] = x.dtype
        return mx.ones((x.shape[0], x.shape[1] // 2, *x.shape[2:]), dtype=mx.float32)

    monkeypatch.setattr(mk, "_groupnorm_affine_fp32", fake_affine)
    monkeypatch.setattr(mk, "fused_glu", fake_glu)

    x = mx.random.normal((1, 8, 16), dtype=mx.float16)
    w = mx.ones((8,), dtype=mx.float32)
    b = mx.zeros((8,), dtype=mx.float32)
    out = mk.fused_groupnorm_glu(x, w, b, num_groups=2, eps=1e-5)
    mx.eval(out)

    assert dtypes["affine_out"] == mx.float32
    assert dtypes["glu_in"] == mx.float32
    assert out.dtype == mx.float16


def test_groupnorm_glu_legacy_multigroup_falls_back_to_hybrid(monkeypatch):
    monkeypatch.setattr(mk, "HAS_METAL", True)
    monkeypatch.setattr(mk, "_fused_groupnorm_mode", lambda: "all")
    monkeypatch.setattr(mk, "_fused_groupnorm_glu_impl", lambda: "legacy")
    monkeypatch.setattr(mk, "_fused_groupnorm_glu_enable_multigroup", lambda: True)

    calls = {"affine": 0, "glu": 0, "legacy_kernel": 0}

    def fake_affine(x, weight, bias, num_groups, eps):
        calls["affine"] += 1
        return x.astype(mx.float32)

    def fake_glu(x, axis=1):
        calls["glu"] += 1
        return mx.zeros((x.shape[0], x.shape[1] // 2, *x.shape[2:]), dtype=mx.float32)

    def fake_get_kernel():
        def kernel(*args, **kwargs):
            calls["legacy_kernel"] += 1
            raise AssertionError("legacy kernel should not run for num_groups > 1")
        return kernel

    monkeypatch.setattr(mk, "_groupnorm_affine_fp32", fake_affine)
    monkeypatch.setattr(mk, "fused_glu", fake_glu)
    monkeypatch.setattr(mk, "_get_groupnorm_glu_kernel", fake_get_kernel)

    x = mx.random.normal((1, 8, 16), dtype=mx.float16)
    w = mx.ones((8,), dtype=mx.float32)
    b = mx.zeros((8,), dtype=mx.float32)
    out = mk.fused_groupnorm_glu(x, w, b, num_groups=2, eps=1e-5)
    mx.eval(out)

    assert calls["affine"] == 1
    assert calls["glu"] == 1
    assert calls["legacy_kernel"] == 0
    assert out.shape == (1, 4, 16)


def test_groupnorm_glu_hybrid_numerically_matches_fallback_multigroup(monkeypatch):
    if not mk.HAS_METAL:
        return

    monkeypatch.setattr(mk, "_fused_groupnorm_mode", lambda: "all")
    monkeypatch.setattr(mk, "_fused_groupnorm_glu_impl", lambda: "hybrid")
    monkeypatch.setattr(mk, "_fused_groupnorm_glu_enable_multigroup", lambda: True)

    x = mx.random.normal((1, 64, 128), dtype=mx.float16)
    w = mx.random.normal((64,), dtype=mx.float32)
    b = mx.random.normal((64,), dtype=mx.float32)

    out = mk.fused_groupnorm_glu(x, w, b, num_groups=4, eps=1e-5)
    ref = mk._groupnorm_glu_fallback(x, w, b, num_groups=4, eps=1e-5)
    mx.eval(out, ref)

    out32 = out.astype(mx.float32)
    ref32 = ref.astype(mx.float32)
    denom = float(mx.sqrt(mx.sum(ref32 * ref32)).item())
    err = float(mx.sqrt(mx.sum((out32 - ref32) * (out32 - ref32))).item())
    rel_l2 = 0.0 if denom == 0.0 else err / denom
    assert rel_l2 <= 5e-4


def test_groupnorm_glu_hybrid_multigroup_disabled_defaults_to_fallback(monkeypatch):
    monkeypatch.setattr(mk, "HAS_METAL", True)
    monkeypatch.setattr(mk, "_fused_groupnorm_mode", lambda: "all")
    monkeypatch.setattr(mk, "_fused_groupnorm_glu_impl", lambda: "hybrid")
    monkeypatch.setattr(mk, "_fused_groupnorm_glu_enable_multigroup", lambda: False)

    calls = {"fallback": 0, "affine": 0, "glu": 0}

    def fake_fallback(x, weight, bias, num_groups, eps):
        calls["fallback"] += 1
        return mx.zeros((x.shape[0], x.shape[1] // 2, *x.shape[2:]), dtype=x.dtype)

    def fake_affine(x, weight, bias, num_groups, eps):
        calls["affine"] += 1
        return x.astype(mx.float32)

    def fake_glu(x, axis=1):
        calls["glu"] += 1
        return x[:, : x.shape[1] // 2, ...]

    monkeypatch.setattr(mk, "_groupnorm_glu_fallback", fake_fallback)
    monkeypatch.setattr(mk, "_groupnorm_affine_fp32", fake_affine)
    monkeypatch.setattr(mk, "fused_glu", fake_glu)

    x = mx.random.normal((1, 8, 16), dtype=mx.float16)
    w = mx.ones((8,), dtype=mx.float32)
    b = mx.zeros((8,), dtype=mx.float32)
    out = mk.fused_groupnorm_glu(x, w, b, num_groups=2, eps=1e-5)
    mx.eval(out)

    assert calls["fallback"] == 1
    assert calls["affine"] == 0
    assert calls["glu"] == 0
    assert out.shape == (1, 4, 16)
