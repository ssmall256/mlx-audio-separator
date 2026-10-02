"""Tests for CommonSeparator write behavior."""

import logging
from pathlib import Path

import mlx.core as mx
import mlx_audio_io as mac
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
    def fake_save(path, stem_source, sample_rate, **kwargs):
        Path(path).write_bytes(b"RIFF")

    monkeypatch.setattr("mlx_audio_separator.separator.common_separator.mac.save", fake_save)

    separator = _make_separator(tmp_path, "mel_band_roformer_karaoke_gabox_v2")
    silent = mx.zeros((1024, 2), dtype=mx.float32)
    output_name = "f8_(Vocals)_mel_band_roformer_karaoke_gabox_v2.wav"

    separator.write_audio(output_name, silent)

    out_path = tmp_path / output_name
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_write_audio_flac_fast_write_requests_backend_mode(tmp_path, monkeypatch):
    captured = {}

    def fake_save(path, stem_source, sample_rate, flac_compression="default", **kwargs):
        captured["flac_compression"] = flac_compression
        Path(path).write_bytes(b"fLaC")

    monkeypatch.setattr("mlx_audio_separator.separator.common_separator.mac.save", fake_save)

    separator = _make_separator(tmp_path, "BS-Roformer-SW")
    separator.output_format = "FLAC"
    separator.experimental_flac_fast_write = True
    stem_source = mx.zeros((512, 2), dtype=mx.float32)

    separator.write_audio("track_(Vocals)_BS-Roformer-SW.flac", stem_source)

    assert captured["flac_compression"] == "fast"


def test_write_audio_flac_fast_write_falls_back_when_backend_missing_kw(tmp_path, monkeypatch):
    calls = {"count": 0}

    def fake_save(path, stem_source, sample_rate, layout="auto", encoding="pcm16", bitrate="auto"):
        calls["count"] += 1
        Path(path).write_bytes(b"fLaC")

    monkeypatch.setattr("mlx_audio_separator.separator.common_separator.mac.save", fake_save)

    separator = _make_separator(tmp_path, "BS-Roformer-SW")
    separator.output_format = "FLAC"
    separator.experimental_flac_fast_write = True
    stem_source = mx.zeros((512, 2), dtype=mx.float32)

    separator.write_audio("track_(Vocals)_BS-Roformer-SW.flac", stem_source)

    # First attempt with flac_compression raises TypeError; fallback retry succeeds.
    assert calls["count"] == 1


@pytest.mark.parametrize(
    ("model_name", "stems"),
    [
        ("mel_band_roformer_karaoke_gabox_v2", ["Vocals"]),
        ("mel_band_roformer_karaoke_becruily", ["Vocals", "Instrumental"]),
    ],
)
def test_karaoke_models_still_materialize_silent_vocals_files(tmp_path, monkeypatch, model_name, stems):
    def fake_save(path, stem_source, sample_rate, **kwargs):
        Path(path).write_bytes(b"RIFF")

    monkeypatch.setattr("mlx_audio_separator.separator.common_separator.mac.save", fake_save)

    separator = _make_separator(tmp_path, model_name)
    separator.audio_file_base = "f8"

    written = []
    for stem_name in stems:
        stem_path = separator.get_stem_output_path(stem_name, custom_output_names=None)
        # Reproduces prior failure mode: vocals can be effectively silent.
        if stem_name == "Vocals":
            stem_source = mx.zeros((1024, 2), dtype=mx.float32)
        else:
            stem_source = mx.full((1024, 2), 0.01, dtype=mx.float32)
        separator.write_audio(stem_path, stem_source)
        written.append(tmp_path / stem_path)

    for output_path in written:
        assert output_path.is_file(), f"missing output file: {output_path}"
        assert output_path.stat().st_size > 0


def test_write_audio_mlx_array_zero_copy(tmp_path, monkeypatch):
    saved_types = []

    def fake_save(path, stem_source, sample_rate, **kwargs):
        saved_types.append(type(stem_source))
        Path(path).write_bytes(b"RIFF")

    monkeypatch.setattr("mlx_audio_separator.separator.common_separator.mac.save", fake_save)

    separator = _make_separator(tmp_path, "htdemucs")
    arr_mx = mx.zeros((2, 1024), dtype=mx.float32)
    output_name = "test_stem.wav"

    separator.write_audio(output_name, arr_mx)

    out_path = tmp_path / output_name
    assert out_path.is_file()
    assert len(saved_types) == 1
    assert issubclass(saved_types[0], mx.array)


def test_async_stem_writer_mlx_array_zero_copy(tmp_path, monkeypatch):
    from mlx_audio_separator.utils.performance import AsyncStemWriter

    saved_types = []

    def fake_save(path, stem_source, sample_rate, **kwargs):
        saved_types.append(type(stem_source))
        Path(path).write_bytes(b"RIFF")

    monkeypatch.setattr("mlx_audio_io.save", fake_save)

    writer = AsyncStemWriter(workers=2)
    arr_mx = mx.zeros((2, 1024), dtype=mx.float32)
    dest = tmp_path / "async_stem.wav"

    writer.submit(
        stem_path=str(dest),
        stem_source=arr_mx,
        sample_rate=44100,
        layout="channels_first",
        encoding="pcm16",
        bitrate="auto",
    )
    writer.flush()
    writer.close()

    assert dest.is_file()
    assert len(saved_types) == 1
    assert issubclass(saved_types[0], mx.array)



def _stereo_stem(frames=4410):
    """A (channels, frames) stem whose channels differ, so a layout mix-up shows."""
    stem = mx.random.uniform(-0.5, 0.5, (2, frames), key=mx.random.key(11))
    mx.eval(stem)
    return stem


@pytest.mark.parametrize("write_workers", [1, 2])
@pytest.mark.parametrize("suffix", ["wav", "flac"])
def test_write_audio_preserves_channels_first_layout(tmp_path, write_workers, suffix):
    """Regression: 0.1.19 wrote (channels, frames) MLX stems with scrambled channels."""
    separator = _make_separator(tmp_path, "htdemucs")
    separator.write_workers = write_workers
    separator.input_encoding = "float32"
    # AudioToolbox writes no FLAC packet for under 4608 frames; use one second.
    stem = _stereo_stem(frames=44100)

    separator.write_audio(f"stem.{suffix}", stem)
    separator.flush_pending_writes()

    loaded, _ = mac.load(str(tmp_path / f"stem.{suffix}"), layout="channels_first")
    tolerance = 1e-6 if suffix == "wav" else 1e-4
    assert loaded.shape == stem.shape
    assert mx.max(mx.abs(loaded - stem)).item() < tolerance


@pytest.mark.parametrize("write_workers", [1, 2])
def test_write_audio_preserves_transposed_view(tmp_path, write_workers):
    """A (frames, channels) transposed view must be written in logical order."""
    separator = _make_separator(tmp_path, "htdemucs")
    separator.write_workers = write_workers
    separator.input_encoding = "float32"
    stem = _stereo_stem()

    separator.write_audio("stem.wav", stem.T)
    separator.flush_pending_writes()

    loaded, _ = mac.load(str(tmp_path / "stem.wav"), layout="channels_first")
    assert mx.max(mx.abs(loaded - stem)).item() < 1e-6


def test_write_audio_scales_silent_stem_without_dividing_by_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mlx_audio_separator.separator.common_separator.mac.save",
        lambda path, *args, **kwargs: Path(path).write_bytes(b"RIFF"),
    )
    separator = _make_separator(tmp_path, "htdemucs")
    separator.amplification_threshold = 0.5

    separator.write_audio("silent.wav", mx.zeros((2, 1024), dtype=mx.float32))

    assert (tmp_path / "silent.wav").is_file()


def _run_with_timeout(fn, seconds=10.0):
    """Run fn in a thread; fail instead of hanging the suite."""
    import threading

    outcome = {}

    def target():
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"{fn} did not return within {seconds}s"
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def test_async_stem_writer_reports_failure_in_fallback_save(tmp_path, monkeypatch):
    """A failure in the no-flac_compression retry must reach flush(), and close() must not hang."""
    from mlx_audio_separator.utils.performance import AsyncStemWriter

    def fake_save(path, stem_source, sample_rate, layout="auto", encoding="pcm16", bitrate="auto"):
        raise OSError("disk full")

    monkeypatch.setattr("mlx_audio_io.save", fake_save)

    writer = AsyncStemWriter(workers=2)
    writer.submit(
        stem_path=str(tmp_path / "stem.flac"),
        stem_source=_stereo_stem(),
        sample_rate=44100,
        layout="channels_first",
        encoding="pcm16",
        bitrate="auto",
        flac_compression="default",
    )

    with pytest.raises(OSError, match="disk full"):
        _run_with_timeout(writer.flush)
    # The error is reported once; the writer is reusable and closes cleanly.
    _run_with_timeout(writer.flush)
    _run_with_timeout(writer.close)


def test_write_audio_failed_save_fails_the_file(tmp_path):
    """A real save failure in a writer thread must surface, not hang or vanish."""
    separator = _make_separator(tmp_path, "htdemucs")
    separator.write_workers = 2

    separator.write_audio("missing_directory/stem.wav", _stereo_stem())

    with pytest.raises(Exception):
        _run_with_timeout(separator.flush_pending_writes)
    assert not (tmp_path / "missing_directory" / "stem.wav").exists()
    _run_with_timeout(separator.clear_file_specific_paths)
