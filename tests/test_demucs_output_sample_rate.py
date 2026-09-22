"""Demucs must resample its stems to the requested output sample rate.

Demucs models run at their own trained rate (44100 Hz). Before this fix the
separator loaded and inferred at 44100 but handed the stems to ``write_audio``,
which stamps the header with ``self.sample_rate``. With ``--sample_rate 48000``
that produced a file whose header claimed 48 kHz over a 44.1 kHz payload, so it
played back 8.8% short (a 262.4 s input became 241.1 s).
"""

from __future__ import annotations

import logging

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio_separator.separator.architectures.demucs_separator import DemucsSeparator

MODEL_SR = 44100
DURATION_S = 2


class _StubDemucsSeparator:
    samplerate = MODEL_SR

    def separate_tensor(self, wav_mx, return_mx=True):
        frames = wav_mx.shape[-1]
        stems = {
            name: mx.zeros((2, frames)) + value
            for value, name in enumerate(["drums", "vocals"], start=1)
        }
        return None, stems


def _build(sample_rate):
    """Assemble only the attributes separate() touches, skipping model load."""
    sep = DemucsSeparator.__new__(DemucsSeparator)
    sep.logger = logging.getLogger("test")
    sep.sample_rate = sample_rate
    sep.output_single_stem = None
    sep.output_dir = None
    sep._demucs_model_name = "htdemucs"
    sep._demucs_separator = _StubDemucsSeparator()
    sep.written = {}

    sep.reset_perf_metrics = lambda: None
    sep.add_perf_time = lambda *a, **k: None
    sep.get_stem_output_path = lambda name, custom=None: f"{name}.wav"
    sep.write_audio = lambda path, data, **kw: sep.written.__setitem__(path, np.asarray(data))
    return sep


@pytest.fixture
def clip(tmp_path):
    mac = pytest.importorskip("mlx_audio_io")
    path = tmp_path / "clip.wav"
    t = np.arange(MODEL_SR * DURATION_S, dtype=np.float32) / MODEL_SR
    tone = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    mac.save(str(path), mx.array(np.stack([tone, tone], axis=-1)), MODEL_SR)
    return str(path)


@pytest.mark.parametrize("target_sr", [44100, 48000, 96000, 22050])
def test_stem_frame_count_matches_requested_sample_rate(clip, target_sr):
    sep = _build(target_sr)
    sep.separate(clip)

    assert sep.written, "no stems were written"
    expected = round(MODEL_SR * DURATION_S * target_sr / MODEL_SR)
    for path, data in sep.written.items():
        frames = data.shape[0] if data.ndim == 2 and data.shape[0] > data.shape[1] else data.shape[-1]
        # The written duration must equal the input duration once the header
        # rate is applied -- that is exactly what regressed.
        assert frames == pytest.approx(expected, abs=2), (
            f"{path}: {frames} frames at {target_sr} Hz is "
            f"{frames / target_sr:.4f}s, expected {DURATION_S}s"
        )


def test_no_resample_when_rates_match(clip, monkeypatch):
    """The default path must stay a no-op, not a needless resample."""
    import mlx_audio_io as mac

    calls = []
    real = mac.resample
    monkeypatch.setattr(mac, "resample", lambda *a, **k: calls.append(a[1:3]) or real(*a, **k))

    _build(MODEL_SR).separate(clip)
    assert calls == [], f"resample should not run at the model rate, got {calls}"
