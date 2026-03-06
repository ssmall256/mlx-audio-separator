"""Tests for grouped BS-Roformer band/mask experimental paths."""

import mlx.core as mx
import numpy as np

from mlx_audio_separator.separator.models.roformer.bs_roformer import (
    BSRoformerMLX,
    BandSplit,
    MaskEstimator,
)


def test_band_split_grouped_matches_legacy():
    dim_inputs = (4, 4, 8, 8, 8)
    band_split = BandSplit(dim=16, dim_inputs=dim_inputs, use_grouped=False)
    x = mx.random.normal((2, 5, sum(dim_inputs)), dtype=mx.float32)

    legacy = band_split(x)
    band_split.use_grouped = True
    grouped = band_split(x)

    np.testing.assert_allclose(np.array(grouped), np.array(legacy), rtol=1e-5, atol=1e-5)


def test_mask_estimator_grouped_matches_legacy():
    dim_inputs = (4, 4, 6, 6)
    estimator = MaskEstimator(dim=12, dim_inputs=dim_inputs, depth=2, mlp_expansion_factor=2, use_grouped=False)
    x = mx.random.normal((2, 7, len(dim_inputs), 12), dtype=mx.float32)

    legacy = estimator(x)
    estimator.use_grouped = True
    grouped = estimator(x)

    np.testing.assert_allclose(np.array(grouped), np.array(legacy), rtol=1e-5, atol=1e-5)


def test_band_split_grouped_weight_cache_reuses_packed_tensors(monkeypatch):
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_ROFORMER_GROUPED_WEIGHT_CACHE", "1")
    dim_inputs = (4, 4, 8, 8)
    band_split = BandSplit(dim=16, dim_inputs=dim_inputs, use_grouped=True)
    x = mx.random.normal((2, 5, sum(dim_inputs)), dtype=mx.float32)

    _ = band_split(x)
    packed_first = band_split._grouped_pack_cache[(0, 1)]
    norm_id = id(packed_first["norm_weights"])
    linear_id = id(packed_first["linear_weights"])

    _ = band_split(x)
    packed_second = band_split._grouped_pack_cache[(0, 1)]
    assert id(packed_second["norm_weights"]) == norm_id
    assert id(packed_second["linear_weights"]) == linear_id


def test_mask_estimator_grouped_weight_cache_reuses_packed_tensors(monkeypatch):
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_ROFORMER_GROUPED_WEIGHT_CACHE", "1")
    dim_inputs = (4, 4, 6, 6)
    estimator = MaskEstimator(dim=12, dim_inputs=dim_inputs, depth=2, mlp_expansion_factor=2, use_grouped=True)
    x = mx.random.normal((2, 7, len(dim_inputs), 12), dtype=mx.float32)

    _ = estimator(x)
    pack_first = estimator._grouped_mlp_pack_cache[(0, 1)]
    assert pack_first is not None
    first_linear = next(op for op in pack_first["ops"] if op["kind"] == "linear")
    first_weight_id = id(first_linear["weights"])

    _ = estimator(x)
    pack_second = estimator._grouped_mlp_pack_cache[(0, 1)]
    assert pack_second is not None
    second_linear = next(op for op in pack_second["ops"] if op["kind"] == "linear")
    assert id(second_linear["weights"]) == first_weight_id


def test_forward_model_compile_fullgraph_shape_cache(monkeypatch):
    compile_calls = {"count": 0}

    def fake_compile(fn, **kwargs):
        compile_calls["count"] += 1
        return fn

    monkeypatch.setattr(mx, "compile", fake_compile)

    model = BSRoformerMLX.__new__(BSRoformerMLX)
    model.experimental_compile_fullgraph = True
    model._forward_model_compile_cache = {}
    model._forward_model_compile_disabled = set()
    model._forward_model_impl = lambda stft_repr: stft_repr + 1.0

    x = mx.zeros((1, 4, 5, 2), dtype=mx.float32)
    y1 = model._forward_model(x)
    y2 = model._forward_model(x)

    np.testing.assert_allclose(np.array(y1), np.ones((1, 4, 5, 2), dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(y2), np.ones((1, 4, 5, 2), dtype=np.float32), rtol=1e-6, atol=1e-6)
    assert compile_calls["count"] == 1
