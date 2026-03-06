"""Runtime toggle tests for Demucs apply_mlx helpers."""

import mlx.core as mx
import numpy as np

from mlx_audio_separator.demucs_mlx import apply_mlx


def test_deterministic_accumulation_default_off(monkeypatch):
    monkeypatch.delenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_ACCUMULATION", raising=False)
    monkeypatch.delenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED", raising=False)
    assert apply_mlx._deterministic_accumulation_enabled() is False


def test_deterministic_accumulation_inherits_deterministic_fused(monkeypatch):
    monkeypatch.delenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_ACCUMULATION", raising=False)
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED", "1")
    assert apply_mlx._deterministic_accumulation_enabled() is True


def test_deterministic_accumulation_explicit_override_off(monkeypatch):
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_ACCUMULATION", "0")
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED", "1")
    assert apply_mlx._deterministic_accumulation_enabled() is False


def test_deterministic_accumulation_explicit_override_on(monkeypatch):
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_ACCUMULATION", "true")
    monkeypatch.delenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED", raising=False)
    assert apply_mlx._deterministic_accumulation_enabled() is True


def test_demucs_apply_concat_batching_flag_parsing(monkeypatch):
    monkeypatch.delenv("MLX_AUDIO_SEPARATOR_DEMUCS_APPLY_CONCAT_BATCHING", raising=False)
    assert apply_mlx._demucs_apply_concat_batching_enabled() is False
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_DEMUCS_APPLY_CONCAT_BATCHING", "1")
    assert apply_mlx._demucs_apply_concat_batching_enabled() is True


class _DummyBatchModel:
    samplerate = 8
    segment = 1.0
    sources = ["vocals", "other"]

    def valid_length(self, length):
        return int(length)

    def __call__(self, x):
        return mx.stack([x, x * mx.array(0.5, dtype=x.dtype)], axis=1)


def test_apply_model_concat_batching_matches_legacy(monkeypatch):
    model = _DummyBatchModel()
    mix = mx.arange(32, dtype=mx.float32).reshape(1, 2, 16)

    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_DEMUCS_APPLY_CONCAT_BATCHING", "0")
    legacy = apply_mlx.apply_model(
        model,
        mix,
        shifts=0,
        split=True,
        overlap=0.5,
        transition_power=1.0,
        batch_size=2,
        segment=1.0,
    )

    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_DEMUCS_APPLY_CONCAT_BATCHING", "1")
    candidate = apply_mlx.apply_model(
        model,
        mix,
        shifts=0,
        split=True,
        overlap=0.5,
        transition_power=1.0,
        batch_size=2,
        segment=1.0,
    )

    mx.eval(legacy, candidate)
    np.testing.assert_array_equal(np.array(legacy), np.array(candidate))
