"""Runtime toggle tests for Demucs apply_mlx helpers."""

import math

import mlx.core as mx
import pytest

from mlx_audio_separator.demucs_mlx import apply_mlx


class _IdentityDemucsModel:
    samplerate = 44100
    segment = 7.8
    sources = ["mix"]
    audio_channels = 2

    def valid_length(self, length):
        return int(length)

    def __call__(self, x):
        return x[:, None, :, :]


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_apply_model_split_overlap_add_reconstructs_identity(overlap, batch_size):
    """Overlap-add must reconstruct an identity model exactly.

    MLX < 0.32.0 corrupts strided slice scatter-add, which silently skews this
    accumulator. Sweeping overlap and batch size exercises the 1-D, 2-D and
    3-D Metal dispatch grids where the bad index arithmetic shows up.
    """
    sr = 44100
    length = sr * 20
    t = mx.arange(length, dtype=mx.float32) / sr
    left = 0.6 * mx.sin(2.0 * math.pi * 220.0 * t)
    right = 0.3 * mx.sin(2.0 * math.pi * 440.0 * t)
    mix = mx.stack([left, right], axis=0).reshape(1, 2, length).astype(mx.float32)

    out = apply_mlx.apply_model(
        _IdentityDemucsModel(),
        mix,
        shifts=0,
        split=True,
        overlap=overlap,
        transition_power=1.0,
        batch_size=batch_size,
        segment=7.8,
    )

    target = mix[:, None, :, :]
    assert mx.max(mx.abs(out - target)).item() <= 1e-4
    assert mx.max(mx.abs(out)).item() <= 0.91


def test_apply_model_long_track_overlap_add_reconstructs_identity():
    """The long-track per-update boundary preserves the full signal."""
    sr = 44_100
    length = sr * 60
    t = mx.arange(length, dtype=mx.float32) / sr
    mix = mx.stack([
        0.6 * mx.sin(2.0 * math.pi * 220.0 * t),
        0.3 * mx.sin(2.0 * math.pi * 440.0 * t),
    ], axis=0).reshape(1, 2, length)
    out = apply_mlx.apply_model(
        _IdentityDemucsModel(), mix, shifts=0, split=True,
        overlap=0.25, batch_size=2, segment=7.8,
    )
    assert mx.max(mx.abs(out[:, 0] - mix)).item() <= 1e-4
