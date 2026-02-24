"""Tests for CommonSeparator write behavior."""

import logging
from pathlib import Path

import numpy as np
import pytest

from mlx_audio_separator.separator.common_separator import CommonSeparator


class _DummySeparator(CommonSeparator):
    def separate(self, audio_file_path):
        raise NotImplementedError


def _make_separator(tmp_path: Path, model_name: str) -> _DummySeparator:
    return _DummySeparator(
        {
            "logger": logging.getLogger("test_common_separator"),
            "log_level": logging.DEBUG,
            "model_name": model_name,
            "model_path": "/tmp/model",
            "model_data": {},
            "output_dir": str(tmp_path),
            "output_format": "WAV",
            "output_bitrate": None,
            "normalization_threshold": 0.9,
            "amplification_threshold": 0.0,
            "enable_denoise": False,
            "output_single_stem": None,
            "invert_using_spec": False,
            "sample_rate": 44100,
            "performance_params": {"write_workers": 1},
        }
    )


def test_write_audio_writes_near_silent_stem(tmp_path, monkeypatch):
    def fake_save(path, stem_source, sample_rate, encoding="pcm16", bitrate="auto"):
        Path(path).write_bytes(b"RIFF")

    monkeypatch.setattr("mlx_audio_separator.separator.common_separator.mac.save", fake_save)

    separator = _make_separator(tmp_path, "mel_band_roformer_karaoke_gabox_v2")
    silent = np.zeros((1024, 2), dtype=np.float32)
    output_name = "f8_(Vocals)_mel_band_roformer_karaoke_gabox_v2.wav"

    separator.write_audio(output_name, silent)

    out_path = tmp_path / output_name
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


@pytest.mark.parametrize(
    ("model_name", "stems"),
    [
        ("mel_band_roformer_karaoke_gabox_v2", ["Vocals"]),
        ("mel_band_roformer_karaoke_becruily", ["Vocals", "Instrumental"]),
    ],
)
def test_karaoke_models_still_materialize_silent_vocals_files(tmp_path, monkeypatch, model_name, stems):
    def fake_save(path, stem_source, sample_rate, encoding="pcm16", bitrate="auto"):
        Path(path).write_bytes(b"RIFF")

    monkeypatch.setattr("mlx_audio_separator.separator.common_separator.mac.save", fake_save)

    separator = _make_separator(tmp_path, model_name)
    separator.audio_file_base = "f8"

    written = []
    for stem_name in stems:
        stem_path = separator.get_stem_output_path(stem_name, custom_output_names=None)
        # Reproduces prior failure mode: vocals can be effectively silent.
        if stem_name == "Vocals":
            stem_source = np.zeros((1024, 2), dtype=np.float32)
        else:
            stem_source = np.full((1024, 2), 0.01, dtype=np.float32)
        separator.write_audio(stem_path, stem_source)
        written.append(tmp_path / stem_path)

    for output_path in written:
        assert output_path.is_file(), f"missing output file: {output_path}"
        assert output_path.stat().st_size > 0
