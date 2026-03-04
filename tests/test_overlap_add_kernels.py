"""Tests for overlap-add experimental fusion gating behavior."""

import mlx.core as mx

from mlx_audio_separator.separator.models.roformer import overlap_add_kernels as oak


def test_accumulate_span_does_not_use_metal_when_not_opted_in(monkeypatch):
    def _fail(*_args, **_kwargs):
        raise AssertionError("Metal kernel path should not run when use_compiled=False")

    monkeypatch.setattr(oak._MetalKernelCache, "get_weighted", classmethod(lambda cls: _fail()))
    monkeypatch.setattr(oak._MetalKernelCache, "get_counter", classmethod(lambda cls: _fail()))

    cache = oak.OverlapAddFusionCache()
    weighted = mx.zeros((2, 1, 2, 8), dtype=mx.float32)
    starts_batch = [0, 4]
    window = mx.ones((8,), dtype=mx.float32)

    span_result, span_counter = cache.accumulate_span(
        weighted=weighted,
        starts_batch=starts_batch,
        span_start=0,
        safe_len=8,
        window_safe=window,
        num_stems=1,
        channels=2,
        use_compiled=False,
    )

    assert span_result.shape == (1, 2, 12)
    assert span_counter.shape == (12,)
