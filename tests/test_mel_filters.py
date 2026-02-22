"""Tests for mel filter bank and Hz/mel conversion functions."""

import numpy as np
import pytest

from mlx_audio_separator.separator.models.roformer.mel_band_roformer import (
    _hz_to_mel,
    _mel_to_hz,
    create_mel_filter_bank,
)


def _has_librosa():
    try:
        import librosa  # noqa: F401
        return True
    except ImportError:
        return False


class TestHzMelConversion:
    def test_round_trip_slaney(self):
        freqs = np.array([0.0, 200.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
        mels = _hz_to_mel(freqs.copy())
        recovered = _mel_to_hz(mels.copy())
        np.testing.assert_allclose(recovered, freqs, atol=1e-3)

    def test_round_trip_htk(self):
        freqs = np.array([0.0, 200.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
        mels = _hz_to_mel(freqs.copy(), htk=True)
        recovered = _mel_to_hz(mels.copy(), htk=True)
        np.testing.assert_allclose(recovered, freqs, atol=1e-3)

    def test_zero_hz_is_zero_mel(self):
        assert _hz_to_mel(np.array([0.0]))[0] == pytest.approx(0.0)
        assert _hz_to_mel(np.array([0.0]), htk=True)[0] == pytest.approx(0.0)

    def test_monotonic(self):
        freqs = np.linspace(0, 22050, 100)
        mels = _hz_to_mel(freqs.copy())
        assert np.all(np.diff(mels) >= 0)


class TestMelFilterBank:
    def test_shape(self):
        fb = create_mel_filter_bank(44100, 2048, 60)
        assert fb.shape == (60, 1025)

    def test_shape_different_bands(self):
        for n_mels in (30, 60, 128):
            fb = create_mel_filter_bank(44100, 2048, n_mels)
            assert fb.shape == (n_mels, 1025)

    def test_non_negative(self):
        fb = create_mel_filter_bank(44100, 2048, 60)
        assert np.all(fb >= 0)

    def test_all_freqs_covered_after_fixup(self):
        """Each frequency bin should be covered by at least one mel band."""
        fb = create_mel_filter_bank(44100, 2048, 128)
        covered = fb.sum(axis=0)
        # DC and Nyquist may not be covered, skip them
        assert np.all(covered[1:-1] > 0)

    def test_slaney_norm(self):
        fb = create_mel_filter_bank(44100, 2048, 60, norm="slaney")
        assert fb.shape == (60, 1025)
        assert np.all(fb >= 0)

    @pytest.mark.skipif(
        not _has_librosa(), reason="librosa not installed"
    )
    def test_matches_librosa_shape_and_sparsity(self):
        """Our filter bank should have same shape and similar sparsity pattern as librosa."""
        import librosa
        our_fb = create_mel_filter_bank(44100, 2048, 60)
        lib_fb = librosa.filters.mel(sr=44100, n_fft=2048, n_mels=60)
        assert our_fb.shape == lib_fb.shape
        # Both should have same nonzero pattern (which bands cover which bins)
        our_nonzero = our_fb > 0
        lib_nonzero = lib_fb > 0
        # Allow small differences in border bins
        overlap = np.sum(our_nonzero & lib_nonzero) / max(np.sum(lib_nonzero), 1)
        assert overlap > 0.9, f"Filter bank overlap with librosa is only {overlap:.2%}"
