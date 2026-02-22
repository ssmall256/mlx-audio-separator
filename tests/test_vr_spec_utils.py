"""Tests for VR spectral utilities."""

import numpy as np
import pytest

from mlx_audio_separator.separator.models.vr.spec_utils import (
    crop_center,
    make_padding,
    spectrogram_to_wave,
    wave_to_spectrogram,
)


class MockMP:
    """Minimal mock for ModelParameters."""
    def __init__(self):
        self.param = {
            "reverse": False,
            "mid_side": False,
            "mid_side_b": False,
            "mid_side_b2": False,
            "stereo_w": False,
            "stereo_n": False,
        }


class TestWaveSpectrogramRoundTrip:
    def test_round_trip(self):
        """wave -> spectrogram -> wave should preserve most energy."""
        sr = 44100
        duration = 0.5
        n = int(sr * duration)
        t = np.linspace(0, duration, n, dtype=np.float64)
        wave = np.stack([
            np.sin(2 * np.pi * 440 * t),
            np.sin(2 * np.pi * 880 * t),
        ])

        hop_length = 1024
        n_fft = 2048
        mp = MockMP()

        spec = wave_to_spectrogram(wave, hop_length, n_fft, mp, band=1)
        recovered = spectrogram_to_wave(spec, hop_length=hop_length, mp=mp, is_v51_model=False)

        # Trim to common length, skip edges (STFT windowing artifacts)
        min_len = min(wave.shape[1], recovered.shape[1])
        edge = n_fft  # skip edge artifacts
        if min_len > 2 * edge:
            orig = wave[:, edge:min_len - edge]
            rec = recovered[:, edge:min_len - edge]
            # Check correlation is high (signal preserved)
            for ch in range(2):
                corr = np.corrcoef(orig[ch], rec[ch])[0, 1]
                assert corr > 0.9, f"Channel {ch} correlation {corr:.3f} too low"


class TestCropCenter:
    def test_same_size(self):
        h1 = np.random.randn(1, 1, 1, 100)
        h2 = np.random.randn(1, 1, 1, 100)
        result = crop_center(h1, h2)
        assert result.shape == h1.shape

    def test_crop(self):
        h1 = np.random.randn(1, 1, 1, 200)
        h2 = np.random.randn(1, 1, 1, 100)
        result = crop_center(h1, h2)
        assert result.shape[3] == 100

    def test_raises_if_h1_smaller(self):
        h1 = np.random.randn(1, 1, 1, 50)
        h2 = np.random.randn(1, 1, 1, 100)
        with pytest.raises(ValueError):
            crop_center(h1, h2)


class TestMakePadding:
    def test_basic(self):
        left, right, roi_size = make_padding(100, 64, 8)
        assert left == 8
        assert roi_size == 48  # cropsize - offset*2 = 64 - 16 = 48
        # right = roi_size - (width % roi_size) + left = 48 - (100 % 48) + 8 = 48 - 4 + 8 = 52
        assert right == 52

    def test_zero_offset(self):
        left, right, roi_size = make_padding(100, 64, 0)
        assert left == 0
        assert roi_size == 64
        # right = 64 - (100 % 64) + 0 = 64 - 36 + 0 = 28
        assert right == 28

    def test_offset_equals_half_cropsize(self):
        left, right, roi_size = make_padding(100, 64, 32)
        assert left == 32
        assert roi_size == 64  # cropsize - offset*2 = 0, so falls back to cropsize
