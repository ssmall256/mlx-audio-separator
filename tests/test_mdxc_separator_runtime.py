"""Runtime behavior tests for MDXC separator experimental paths."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_audio_separator.separator.architectures.mdxc_separator import MDXCSeparator


def test_run_model_callable_honors_defer_batch_eval(monkeypatch):
    calls = {"eval": 0}
    orig_eval = mx.eval

    def wrapped_eval(*args, **kwargs):
        calls["eval"] += 1
        return orig_eval(*args, **kwargs)

    monkeypatch.setattr(mx, "eval", wrapped_eval)

    sep = MDXCSeparator.__new__(MDXCSeparator)
    sep.experimental_mlx_stream_pipeline = False
    sep._pipeline_stream = None
    sep.experimental_mdxc_defer_batch_eval = True

    batch = mx.ones((1, 2, 8), dtype=mx.float32)
    out = sep._run_model_callable(lambda x: x, batch)
    mx.eval(out)
    assert calls["eval"] == 1  # only the explicit eval above

    calls["eval"] = 0
    sep.experimental_mdxc_defer_batch_eval = False
    out = sep._run_model_callable(lambda x: x, batch)
    mx.eval(out)
    assert calls["eval"] == 2  # one internal + one explicit


def test_compute_gather_idx_precomputed_matches_legacy():
    starts = [0, 7, 19, 35, 44]
    chunk_size = 8
    arange_chunk = mx.arange(chunk_size, dtype=mx.int32)
    precomputed = mx.array(starts, dtype=mx.int32)[:, None] + arange_chunk[None, :]
    start_to_row = {int(start): idx for idx, start in enumerate(starts)}

    starts_batch = [7, 35, 35, 0]
    got = MDXCSeparator._compute_gather_idx(
        starts_batch,
        arange_chunk,
        precomputed_gather_idx=precomputed,
        start_to_row=start_to_row,
    )
    legacy = mx.array(starts_batch, dtype=mx.int32)[:, None] + arange_chunk[None, :]

    np.testing.assert_array_equal(np.array(got), np.array(legacy))


def test_compute_gather_idx_falls_back_when_start_missing():
    starts = [0, 10]
    arange_chunk = mx.arange(4, dtype=mx.int32)
    precomputed = mx.array(starts, dtype=mx.int32)[:, None] + arange_chunk[None, :]
    start_to_row = {0: 0, 10: 1}

    starts_batch = [0, 99]
    got = MDXCSeparator._compute_gather_idx(
        starts_batch,
        arange_chunk,
        precomputed_gather_idx=precomputed,
        start_to_row=start_to_row,
    )
    legacy = mx.array(starts_batch, dtype=mx.int32)[:, None] + arange_chunk[None, :]
    np.testing.assert_array_equal(np.array(got), np.array(legacy))
