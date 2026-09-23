"""The Demucs cache must not share a directory with demucs-mlx.

Both packages write `<model>_config.json` and `<model>.safetensors`, under
config schemas that reject each other, and both now rebuild a cache they cannot
read. Sharing `~/.cache/demucs-mlx` therefore meant that with the two installed
side by side every alternating run reconverted -- each time needing torch and
the upstream checkpoint, or failing outright without them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_audio_separator.demucs_mlx import mlx_convert, model_converter
from mlx_audio_separator.demucs_mlx.mlx_convert import SafeCacheError


def test_the_write_location_is_named_after_this_package(monkeypatch, tmp_path):
    monkeypatch.delenv(model_converter._CACHE_DIR_ENV, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    cache_dir = model_converter.get_mlx_cache_dir()
    assert cache_dir == tmp_path / ".cache" / "mlx-audio-separator" / "demucs"
    assert cache_dir.is_dir()
    assert "demucs-mlx" not in str(cache_dir), "must not be demucs-mlx's directory"


def test_the_env_override_is_honoured(monkeypatch, tmp_path):
    monkeypatch.setenv(model_converter._CACHE_DIR_ENV, str(tmp_path / "elsewhere"))
    assert model_converter.get_mlx_cache_dir() == tmp_path / "elsewhere"
    assert model_converter._legacy_cache_dir() is None, "an override replaces both"


def test_the_legacy_directory_is_still_read(monkeypatch, tmp_path):
    """Existing users must not pay for a conversion they already did."""
    monkeypatch.delenv(model_converter._CACHE_DIR_ENV, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        model_converter, "_LEGACY_CACHE_DIR", tmp_path / ".cache" / "demucs-mlx"
    )
    legacy = tmp_path / ".cache" / "demucs-mlx"
    legacy.mkdir(parents=True)

    seen = []

    def fake_load(name, *, cache_dir, **kwargs):
        seen.append(Path(cache_dir))
        if Path(cache_dir) == legacy:
            return f"model:{name}"
        raise FileNotFoundError("no cache")

    monkeypatch.setattr(mlx_convert, "load_mlx_model", fake_load)
    monkeypatch.setattr(
        mlx_convert, "convert_htdemucs_weights",
        lambda *a, **k: pytest.fail("must not reconvert when the legacy cache works"),
    )
    assert model_converter.get_mlx_model("htdemucs") == "model:htdemucs"
    assert seen[0] != legacy and seen[1] == legacy, "new location first, then legacy"


def test_a_foreign_legacy_cache_converts_into_the_new_directory(monkeypatch, tmp_path):
    """A demucs-mlx cache in the old directory is rejected once, then replaced
    here -- rather than being rewritten there for demucs-mlx to reject next."""
    monkeypatch.delenv(model_converter._CACHE_DIR_ENV, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        model_converter, "_LEGACY_CACHE_DIR", tmp_path / ".cache" / "demucs-mlx"
    )
    (tmp_path / ".cache" / "demucs-mlx").mkdir(parents=True)
    new_dir = tmp_path / ".cache" / "mlx-audio-separator" / "demucs"

    converted_into = []
    loads = {"n": 0}

    def fake_load(name, *, cache_dir, **kwargs):
        loads["n"] += 1
        if loads["n"] == 1:
            raise FileNotFoundError("no cache")
        if loads["n"] == 2:
            raise SafeCacheError("missing fields: ['torch_signatures']")
        return f"model:{name}"

    monkeypatch.setattr(mlx_convert, "load_mlx_model", fake_load)
    monkeypatch.setattr(
        mlx_convert, "convert_htdemucs_weights",
        lambda *a, output_dir, **k: converted_into.append(Path(output_dir)),
    )
    assert model_converter.get_mlx_model("htdemucs") == "model:htdemucs"
    assert converted_into == [new_dir], "must never write to demucs-mlx's directory"
