"""Tests for Demucs Wiener runtime controls."""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

from mlx_audio_separator.demucs_mlx import wiener_mlx
from mlx_audio_separator.demucs_mlx.mlx_hdemucs import _demucs_strict_eval_enabled as hdemucs_strict_eval
from mlx_audio_separator.demucs_mlx.mlx_hdemucs import _demucs_wiener_use_vmap as hdemucs_use_vmap
from mlx_audio_separator.demucs_mlx.mlx_htdemucs import _demucs_strict_eval_enabled as htdemucs_strict_eval
from mlx_audio_separator.demucs_mlx.mlx_htdemucs import _demucs_wiener_use_vmap as htdemucs_use_vmap


def test_demucs_wiener_use_vmap_defaults_true():
    key = "MLX_AUDIO_SEPARATOR_DEMUCS_WIENER_USE_VMAP"
    previous = os.environ.pop(key, None)
    try:
        assert hdemucs_use_vmap() is True
        assert htdemucs_use_vmap() is True
    finally:
        if previous is not None:
            os.environ[key] = previous


def test_demucs_wiener_use_vmap_honors_false_values():
    key = "MLX_AUDIO_SEPARATOR_DEMUCS_WIENER_USE_VMAP"
    previous = os.environ.get(key)
    try:
        for value in ("0", "false", "no", "off"):
            os.environ[key] = value
            assert hdemucs_use_vmap() is False
            assert htdemucs_use_vmap() is False
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def test_demucs_strict_eval_defaults_false():
    key = "MLX_AUDIO_SEPARATOR_DEMUCS_STRICT_EVAL"
    previous = os.environ.pop(key, None)
    try:
        assert hdemucs_strict_eval() is False
        assert htdemucs_strict_eval() is False
    finally:
        if previous is not None:
            os.environ[key] = previous


def test_demucs_strict_eval_honors_true_values():
    key = "MLX_AUDIO_SEPARATOR_DEMUCS_STRICT_EVAL"
    previous = os.environ.get(key)
    try:
        for value in ("1", "true", "yes", "on"):
            os.environ[key] = value
            assert hdemucs_strict_eval() is True
            assert htdemucs_strict_eval() is True
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def test_wiener_complex_phase_fallback_when_angle_missing(monkeypatch):
    monkeypatch.delattr(wiener_mlx.mx, "angle", raising=False)

    x = mx.array([1.0 + 1.0j, -2.0 + 0.5j], dtype=mx.complex64)
    got = wiener_mlx._complex_phase(x)
    expected = mx.arctan2(mx.imag(x), mx.real(x))
    np.testing.assert_allclose(np.array(got), np.array(expected), rtol=1e-6, atol=1e-6)


def test_wiener_phase_copy_path_uses_phase_helper(monkeypatch):
    # Force deterministic phase to validate helper wiring in softmask=False path.
    monkeypatch.setattr(
        wiener_mlx,
        "_complex_phase",
        lambda x: mx.zeros(x.shape, dtype=mx.float32) + 0.5,
    )

    targets = mx.array(np.abs(np.random.default_rng(0).normal(size=(3, 4, 2, 2))).astype(np.float32))
    mix_stft = mx.array(np.random.default_rng(1).normal(size=(3, 4, 2, 2)).astype(np.float32))

    out = wiener_mlx.wiener(targets, mix_stft, iterations=0, softmask=False, residual=False)
    expected_complex = targets.astype(mx.complex64) * mx.exp(1j * (mx.zeros((3, 4, 2, 1), dtype=mx.float32) + 0.5))
    expected = wiener_mlx._from_complex(expected_complex)
    np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-6, atol=1e-6)


def test_wiener_prealloc_output_matches_list_concatenate(monkeypatch):
    rng = np.random.default_rng(42)
    y = mx.array(
        rng.normal(size=(5, 3, 2, 2)).astype(np.float32)
        + 1j * rng.normal(size=(5, 3, 2, 2)).astype(np.float32),
        dtype=mx.complex64,
    )
    x = mx.array(
        rng.normal(size=(5, 3, 2)).astype(np.float32)
        + 1j * rng.normal(size=(5, 3, 2)).astype(np.float32),
        dtype=mx.complex64,
    )
    # Keep this test focused on branch parity between list+concat vs preallocated writes.
    monkeypatch.setattr(
        wiener_mlx,
        "_compute_covariance_batch",
        lambda y_slice: mx.zeros(
            (y_slice.shape[-1], y_slice.shape[1], y_slice.shape[2], y_slice.shape[2]),
            dtype=mx.complex64,
        ),
    )
    monkeypatch.setattr(
        wiener_mlx,
        "_apply_wiener_batch",
        lambda x_slice, v_slice, R, eps: mx.broadcast_to(
            x_slice[..., None],
            (x_slice.shape[0], x_slice.shape[1], x_slice.shape[2], v_slice.shape[-1]),
        ),
    )

    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_DEMUCS_WIENER_PREALLOC_OUTPUT", "0")
    out_list, v_list, r_list = wiener_mlx.expectation_maximization(y, x, iterations=1, batch_size=2)

    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_DEMUCS_WIENER_PREALLOC_OUTPUT", "1")
    out_pre, v_pre, r_pre = wiener_mlx.expectation_maximization(y, x, iterations=1, batch_size=2)

    np.testing.assert_allclose(np.array(out_pre), np.array(out_list), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(v_pre), np.array(v_list), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(r_pre), np.array(r_list), rtol=1e-6, atol=1e-6)
