"""Tests for experimental Roformer fast norm path."""

import mlx.core as mx
import numpy as np

from mlx_audio_separator.separator.models.roformer.bs_roformer import L2Norm


def test_l2norm_default_path_matches_reference(monkeypatch):
    monkeypatch.delenv("MLX_AUDIO_SEPARATOR_ROFORMER_FAST_NORM", raising=False)

    layer = L2Norm(dim=16, eps=1e-12)
    assert layer.use_fast_norm is False

    x = mx.random.normal((2, 32, 16), dtype=mx.float32)
    y = layer(x)

    norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
    denom = mx.maximum(norm, layer.eps)
    ref = (x / denom) * layer.scale * layer.weight

    np.testing.assert_allclose(np.array(y), np.array(ref), rtol=1e-6, atol=1e-6)


def test_l2norm_fast_path_matches_rms_norm_formula(monkeypatch):
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_ROFORMER_FAST_NORM", "1")

    layer = L2Norm(dim=16, eps=1e-12)
    assert layer.use_fast_norm is True

    x = mx.random.normal((2, 32, 16), dtype=mx.float32)
    y = layer(x)
    ref = mx.fast.rms_norm(x, layer.weight, layer.eps) * layer.scale

    np.testing.assert_allclose(np.array(y), np.array(ref), rtol=1e-6, atol=1e-6)

