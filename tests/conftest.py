"""Shared fixtures for mlx-audio-separator tests."""

import mlx.core as mx
import numpy as np
import pytest


@pytest.fixture
def synthetic_mono():
    """Return a factory for mono sine-wave audio as mx.array."""
    def _make(sr=44100, duration=1.0):
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        wave = np.sin(2 * np.pi * 440 * t)
        return mx.array(wave.reshape(1, 1, -1))
    return _make


@pytest.fixture
def synthetic_stereo():
    """Return a factory for stereo sine-wave audio as mx.array."""
    def _make(sr=44100, duration=1.0):
        n = int(sr * duration)
        t = np.linspace(0, duration, n, dtype=np.float32)
        left = np.sin(2 * np.pi * 440 * t)
        right = np.sin(2 * np.pi * 880 * t)
        wave = np.stack([left, right])
        return mx.array(wave.reshape(1, 2, -1))
    return _make


# --- Small model configs for testing ---

@pytest.fixture
def bs_roformer_config():
    return {
        "model": {
            "dim": 64,
            "depth": 1,
            "stereo": True,
            "num_stems": 1,
            "time_transformer_depth": 1,
            "freq_transformer_depth": 1,
            "linear_transformer_depth": 0,
            "freqs_per_bands": tuple([25] * 41),  # must sum to n_fft//2+1 = 1025
            "dim_head": 64,
            "heads": 8,
            "attn_dropout": 0.0,
            "ff_dropout": 0.0,
            "mlp_expansion_factor": 4,
            "mask_estimator_depth": 1,
        }
    }


@pytest.fixture
def mel_band_roformer_config():
    return {
        "model": {
            "dim": 64,
            "depth": 1,
            "stereo": True,
            "num_stems": 1,
            "time_transformer_depth": 1,
            "freq_transformer_depth": 1,
            "linear_transformer_depth": 0,
            "num_bands": 60,
            "dim_head": 64,
            "heads": 8,
            "attn_dropout": 0.0,
            "ff_dropout": 0.0,
            "mlp_expansion_factor": 4,
            "mask_estimator_depth": 1,
        },
        "audio": {
            "sample_rate": 44100,
        },
    }


@pytest.fixture
def convtdfnet_config():
    return {
        "dim_c": 4,
        "dim_f": 2048,
        "dim_t": 256,
        "n_fft": 6144,
        "hop_length": 1024,
        "num_blocks": 4,
        "num_tdf_layers": 3,
        "g": 16,
        "k": 3,
        "bn": 8,
        "bias": True,
        "optimizer": "rmsprop",
    }
