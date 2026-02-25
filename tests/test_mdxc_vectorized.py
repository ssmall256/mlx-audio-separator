"""Tests for experimental vectorized MDXC chunking path."""

import mlx.core as mx
import numpy as np

from mlx_audio_separator.separator.architectures import mdxc_separator as mdxc_mod


class _NoopLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


def _make_separator(model_run, batch_size=3):
    sep = mdxc_mod.MDXCSeparator.__new__(mdxc_mod.MDXCSeparator)
    sep.batch_size = batch_size
    sep.model_run = model_run
    return sep


def test_chunk_starts_covers_tail():
    starts = mdxc_mod.MDXCSeparator._chunk_starts(total_samples=301, chunk_size=80, step=45)
    assert starts[0] == 0
    assert starts[-1] == 221
    assert starts == sorted(starts)


def test_vectorized_overlap_add_identity_single_stem(monkeypatch):
    monkeypatch.setattr(mdxc_mod, "tqdm", lambda it, **kwargs: it)

    rng = np.random.default_rng(0)
    mix = rng.standard_normal((2, 301), dtype=np.float32)
    chunk_size = 80
    step = 45
    starts = mdxc_mod.MDXCSeparator._chunk_starts(mix.shape[1], chunk_size, step)
    window = np.hamming(chunk_size).astype(np.float32)

    def model_run(batch):
        # Identity model output: (B, 1, C, T)
        return mx.expand_dims(batch, axis=1)

    sep = _make_separator(model_run=model_run, batch_size=3)
    out = sep._run_chunked_model_vectorized(
        mix_mx=mx.array(mix),
        starts=starts,
        chunk_size=chunk_size,
        window_mx=mx.array(window),
        num_stems=1,
    )

    assert out.shape == (1, 2, mix.shape[1])
    np.testing.assert_allclose(out[0], mix, rtol=1e-5, atol=1e-5)


def test_vectorized_overlap_add_multi_stem_sign_split(monkeypatch):
    monkeypatch.setattr(mdxc_mod, "tqdm", lambda it, **kwargs: it)

    rng = np.random.default_rng(1)
    mix = rng.standard_normal((2, 257), dtype=np.float32)
    chunk_size = 64
    step = 32
    starts = mdxc_mod.MDXCSeparator._chunk_starts(mix.shape[1], chunk_size, step)
    window = np.hamming(chunk_size).astype(np.float32)

    def model_run(batch):
        # Two deterministic stems for parity checks.
        return mx.stack([batch, -batch], axis=1)

    sep = _make_separator(model_run=model_run, batch_size=4)
    out = sep._run_chunked_model_vectorized(
        mix_mx=mx.array(mix),
        starts=starts,
        chunk_size=chunk_size,
        window_mx=mx.array(window),
        num_stems=2,
    )

    assert out.shape == (2, 2, mix.shape[1])
    np.testing.assert_allclose(out[0], mix, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[1], -mix, rtol=1e-5, atol=1e-5)


def test_fixed_compiled_batch_tail_padding_masks_extra_slots():
    rng = np.random.default_rng(2)
    mix = rng.standard_normal((2, 120), dtype=np.float32)

    sep = _make_separator(model_run=lambda batch: batch, batch_size=4)
    sep._compiled_model_run = lambda batch: mx.expand_dims(batch, axis=1)

    starts = [0, 17, 34]
    chunk_size = 16
    arange_chunk = mx.arange(chunk_size, dtype=mx.int32)

    x, starts_batch = sep._run_fixed_compiled_batch(
        mix_mx=mx.array(mix),
        starts=starts,
        start_idx=0,
        current_batch_size=3,
        chunk_size=chunk_size,
        arange_chunk=arange_chunk,
    )

    assert starts_batch == starts
    assert x.shape == (4, 1, 2, chunk_size)

    x_np = np.array(x, dtype=np.float32, copy=False)
    for idx, st in enumerate(starts_batch):
        np.testing.assert_allclose(x_np[idx, 0], mix[:, st : st + chunk_size], rtol=1e-5, atol=1e-5)

    # Tail slot is padded and masked to zeros.
    np.testing.assert_allclose(x_np[3, 0], np.zeros((2, chunk_size), dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_roformer_static_compiled_demix_uses_shapeless_flag(monkeypatch):
    monkeypatch.setattr(mdxc_mod, "tqdm", lambda it, **kwargs: it)

    compile_calls = []

    def fake_compile(fun, **kwargs):
        compile_calls.append(kwargs)
        return fun

    monkeypatch.setattr(mdxc_mod.mx, "compile", fake_compile)

    rng = np.random.default_rng(3)
    mix = rng.standard_normal((2, 257), dtype=np.float32)
    chunk_size = 64
    step = 32
    starts = mdxc_mod.MDXCSeparator._chunk_starts(mix.shape[1], chunk_size, step)
    window = np.hamming(chunk_size).astype(np.float32)

    sep = mdxc_mod.MDXCSeparator.__new__(mdxc_mod.MDXCSeparator)
    sep.model_run = lambda batch: mx.expand_dims(batch, axis=1)
    sep._compiled_demix_fn_cache = {}
    sep._compiled_demix_shapeless_disabled = set()
    sep.experimental_compile_shapeless = True

    out = sep._run_roformer_static_compiled_demix(
        mix_mx=mx.array(mix),
        starts=starts,
        chunk_size=chunk_size,
        window_mx=mx.array(window),
        num_stems=1,
    )

    assert out.shape == (1, 2, mix.shape[1])
    np.testing.assert_allclose(out[0], mix, rtol=1e-5, atol=1e-5)
    assert compile_calls and compile_calls[0].get("shapeless") is True


def test_roformer_static_compiled_demix_disables_shapeless_after_failure(monkeypatch):
    monkeypatch.setattr(mdxc_mod, "tqdm", lambda it, **kwargs: it)

    compile_calls = []

    def fake_compile(fun, **kwargs):
        compile_calls.append(dict(kwargs))
        if kwargs.get("shapeless"):
            def _bad(*args, **kw):
                raise RuntimeError("CustomKernel cannot infer output shapes")
            return _bad
        return fun

    monkeypatch.setattr(mdxc_mod.mx, "compile", fake_compile)

    rng = np.random.default_rng(4)
    mix = rng.standard_normal((2, 257), dtype=np.float32)
    chunk_size = 64
    step = 32
    starts = mdxc_mod.MDXCSeparator._chunk_starts(mix.shape[1], chunk_size, step)
    window = np.hamming(chunk_size).astype(np.float32)

    sep = mdxc_mod.MDXCSeparator.__new__(mdxc_mod.MDXCSeparator)
    sep.model_run = lambda batch: mx.expand_dims(batch, axis=1)
    sep._compiled_demix_fn_cache = {}
    sep._compiled_demix_shapeless_disabled = set()
    sep.experimental_compile_shapeless = True
    sep.logger = _NoopLogger()

    out1 = sep._run_roformer_static_compiled_demix(
        mix_mx=mx.array(mix),
        starts=starts,
        chunk_size=chunk_size,
        window_mx=mx.array(window),
        num_stems=1,
    )
    out2 = sep._run_roformer_static_compiled_demix(
        mix_mx=mx.array(mix),
        starts=starts,
        chunk_size=chunk_size,
        window_mx=mx.array(window),
        num_stems=1,
    )

    assert out1.shape == (1, 2, mix.shape[1])
    assert out2.shape == (1, 2, mix.shape[1])
    np.testing.assert_allclose(out1[0], mix, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out2[0], mix, rtol=1e-5, atol=1e-5)

    shapeless_calls = [c for c in compile_calls if c.get("shapeless") is True]
    shaped_calls = [c for c in compile_calls if c.get("shapeless") is False]
    assert len(shapeless_calls) == 1
    assert len(shaped_calls) == 1


def test_demix_static_path_forces_shaped_compile(monkeypatch):
    mix = np.zeros((2, 33000), dtype=np.float32)
    captured: dict[str, object] = {}

    sep = mdxc_mod.MDXCSeparator.__new__(mdxc_mod.MDXCSeparator)
    sep.logger = _NoopLogger()
    sep.model_type = "bs_roformer"
    sep.model_run = lambda batch: mx.expand_dims(batch, axis=1)
    sep.model_data = {
        "training": {"instruments": ["Instrumental"]},
        "inference": {"dim_t": 64},
        "audio": {"sample_rate": 44100, "hop_length": 512},
        "model": {"stft_hop_length": 512},
    }
    sep.override_model_segment_size = False
    sep.segment_size = 256
    sep.overlap = 8
    sep.batch_size = 1
    sep.sample_rate = 44100
    sep.experimental_compile_model_forward = True
    sep.experimental_compile_shapeless = True
    sep.experimental_roformer_static_compiled_demix = True
    sep.experimental_vectorized_chunking = False
    sep._np_window_cache = {}
    sep._mlx_window_cache = {}
    sep._logged_static_shapeless_disable = False

    def fake_static(**kwargs):
        captured.update(kwargs)
        in_mix = kwargs["mix_mx"]
        return np.zeros((1, int(in_mix.shape[0]), int(in_mix.shape[1])), dtype=np.float32)

    monkeypatch.setattr(sep, "_run_roformer_static_compiled_demix", fake_static)
    out = sep._demix_mlx(mix)

    assert isinstance(out, np.ndarray)
    assert out.shape == (2, mix.shape[1])
    assert captured.get("shapeless_override") is False


def test_separate_writes_complementary_secondary_for_single_target_model():
    sep = mdxc_mod.MDXCSeparator.__new__(mdxc_mod.MDXCSeparator)
    sep.logger = _NoopLogger()
    sep.sample_rate = 44100
    sep.normalization_threshold = 0.9
    sep.amplification_threshold = 0.0
    sep.output_single_stem = None
    sep.output_dir = "/tmp"
    sep.override_model_segment_size = False
    sep.model_data = {"training": {"target_instrument": "Vocals", "instruments": []}}
    sep.primary_stem_name = "Vocals"
    sep.secondary_stem_name = "Instrumental"
    sep.reset_perf_metrics = lambda: None
    sep.add_perf_time = lambda *args, **kwargs: None
    sep.prepare_mix = lambda _: np.zeros((2, 32), dtype=np.float32)
    sep._demix_mlx = lambda _: {
        "Vocals": np.zeros((2, 32), dtype=np.float32),
        "Instrumental": np.zeros((2, 32), dtype=np.float32),
    }
    sep.get_stem_output_path = lambda stem, _: f"track_({stem}).wav"

    written = []
    sep.write_audio = lambda path, data: written.append(path)

    out = sep.separate("/tmp/track.wav")

    assert len(out) == 2
    assert len(written) == 2
    assert any("Vocals" in p for p in out)
    assert any("Instrumental" in p for p in out)
