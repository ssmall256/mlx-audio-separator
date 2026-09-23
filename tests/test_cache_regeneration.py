"""An unusable Demucs cache must regenerate, not raise.

0.1.8 hardened the safetensors cache, and its loader requires fields that no
earlier cache contains. demucs-mlx writes its own, differently-shaped config
into the same `~/.cache/demucs-mlx` directory, which this loader also rejects.
`get_mlx_model` caught only `FileNotFoundError`, so either of those raised
`SafeCacheError` straight out to the caller -- turning a self-healing cache
miss into a hard failure, with nothing saying that regenerating was the fix.
"""

from __future__ import annotations

import pytest

from mlx_audio_separator.demucs_mlx import mlx_convert, model_converter
from mlx_audio_separator.demucs_mlx.mlx_convert import SafeCacheError


@pytest.fixture
def converting(monkeypatch, tmp_path):
    """Record whether conversion ran, and how often the model was loaded."""
    calls = {"converted": 0, "loads": 0}
    monkeypatch.setattr(model_converter, "get_mlx_cache_dir", lambda: tmp_path)
    # Keep the real ~/.cache/demucs-mlx read-through out of these cases; it is
    # covered by tests/test_cache_dir_namespacing.py.
    monkeypatch.setattr(model_converter, "_legacy_cache_dir", lambda: None)
    monkeypatch.setattr(
        mlx_convert, "convert_htdemucs_weights",
        lambda *a, **k: calls.__setitem__("converted", calls["converted"] + 1),
    )

    def install(first_error):
        def fake_load(name, **kwargs):
            calls["loads"] += 1
            if calls["loads"] == 1 and first_error is not None:
                raise first_error
            return f"model:{name}"

        monkeypatch.setattr(mlx_convert, "load_mlx_model", fake_load)
        return calls

    return install


@pytest.mark.parametrize(
    "error",
    [
        SafeCacheError("Demucs cache config is missing fields: ['torch_signatures']"),
        SafeCacheError("Demucs cache config has unknown fields: ['source_artifacts']"),
        SafeCacheError("Unsupported Demucs cache format version: None"),
        SafeCacheError("safetensors digest mismatch"),
    ],
    ids=["missing-fields", "foreign-fields", "old-format", "digest-mismatch"],
)
def test_an_unusable_cache_regenerates(converting, error):
    calls = converting(error)
    assert model_converter.get_mlx_model("htdemucs") == "model:htdemucs"
    assert calls["converted"] == 1, "an unreadable cache must be rebuilt"
    assert calls["loads"] == 2, "and reloaded afterwards"


def test_a_missing_cache_still_converts(converting):
    calls = converting(FileNotFoundError("no cache"))
    assert model_converter.get_mlx_model("htdemucs") == "model:htdemucs"
    assert calls["converted"] == 1
    assert calls["loads"] == 2


def test_a_good_cache_is_not_reconverted(converting):
    calls = converting(None)
    assert model_converter.get_mlx_model("htdemucs") == "model:htdemucs"
    assert calls["converted"] == 0, "a readable cache must not trigger conversion"
    assert calls["loads"] == 1


def test_an_unrelated_error_is_not_swallowed(converting):
    """Only an unusable cache regenerates. A bug elsewhere must still surface."""
    calls = converting(RuntimeError("metal device lost"))
    with pytest.raises(RuntimeError, match="metal device lost"):
        model_converter.get_mlx_model("htdemucs")
    assert calls["converted"] == 0
