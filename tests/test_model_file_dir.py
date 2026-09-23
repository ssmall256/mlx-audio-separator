"""Downloaded model files must not default into /tmp.

macOS clears `/tmp` on boot, so the old `/tmp/audio-separator-models/` default
cost a multi-gigabyte re-download after every reboot, and `/tmp` is
world-writable -- whichever user creates that directory first owns it, and
`download_file_if_not_exists` returns early for any file already sitting at the
target path, without checking it. Every checkpoint load is `weights_only=True`
or protobuf parsing, so a planted file means wrong output rather than code
execution; a cache in a directory another user controls is still wrong.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from mlx_audio_separator import core
from mlx_audio_separator.demucs_mlx.defaults import (
    DEFAULT_MODEL_FILE_DIR,
    LEGACY_MODEL_FILE_DIR,
)


def test_the_default_is_not_in_tmp():
    assert not DEFAULT_MODEL_FILE_DIR.startswith("/tmp")
    assert not DEFAULT_MODEL_FILE_DIR.startswith("/private/tmp")
    assert DEFAULT_MODEL_FILE_DIR.endswith(
        os.path.join(".cache", "mlx-audio-separator", "models")
    )
    assert LEGACY_MODEL_FILE_DIR == "/tmp/audio-separator-models/"


class _Sep:
    """The real download/adoption methods, without constructing a Separator."""

    _link_from_legacy_model_dir = core.Separator._link_from_legacy_model_dir
    _adopt_legacy_converted_weights = core.Separator._adopt_legacy_converted_weights
    _adopt_legacy_model_file = core.Separator._adopt_legacy_model_file
    download_file_if_not_exists = core.Separator.download_file_if_not_exists

    def __init__(self, legacy_dir):
        self._legacy_model_file_dir = legacy_dir
        self.logger = logging.getLogger("test-separator")


def test_a_file_in_the_old_location_is_reused(tmp_path, caplog):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "model.ckpt").write_bytes(b"weights")
    target = tmp_path / "new" / "model.ckpt"

    sep = _Sep(str(legacy))
    assert sep._adopt_legacy_model_file(str(target)) is True
    assert target.read_bytes() == b"weights", "must not re-download several GB"


def test_nothing_is_reused_when_the_caller_chose_a_directory(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "model.ckpt").write_bytes(b"weights")
    target = tmp_path / "new" / "model.ckpt"

    sep = _Sep(None)
    assert sep._adopt_legacy_model_file(str(target)) is False
    assert not target.exists()


def test_a_missing_legacy_file_falls_through_to_download(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    sep = _Sep(str(legacy))
    assert sep._adopt_legacy_model_file(str(tmp_path / "new" / "absent.ckpt")) is False


@pytest.mark.parametrize("chosen", ["param", "env"])
def test_an_explicit_directory_disables_the_legacy_fallback(monkeypatch, tmp_path, chosen):
    """The caller's choice wins; adopting files into it would be a surprise."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    kwargs = {"output_dir": str(tmp_path / "out")}
    if chosen == "param":
        monkeypatch.delenv("AUDIO_SEPARATOR_MODEL_DIR", raising=False)
        kwargs["model_file_dir"] = str(tmp_path / "chosen")
    else:
        (tmp_path / "chosen").mkdir()
        monkeypatch.setenv("AUDIO_SEPARATOR_MODEL_DIR", str(tmp_path / "chosen"))
    sep = core.Separator(**kwargs)
    assert sep.model_file_dir == str(tmp_path / "chosen")
    assert sep._legacy_model_file_dir is None


def test_the_converted_safetensors_beside_a_model_is_adopted(tmp_path):
    """The loaders write their MLX conversion next to the checkpoint and never
    download it, so the download path alone would leave it behind -- and a
    torch-free install would then fail to convert a model that already worked.
    """
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "m.pth").write_bytes(b"torch")
    (legacy / "m.safetensors").write_bytes(b"mlx")
    target = tmp_path / "new" / "m.pth"

    sep = _Sep(str(legacy))
    assert sep._adopt_legacy_model_file(str(target)) is True
    assert (tmp_path / "new" / "m.safetensors").read_bytes() == b"mlx"


def test_the_conversion_is_adopted_even_when_the_model_is_already_local(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "m.safetensors").write_bytes(b"mlx")
    new = tmp_path / "new"
    new.mkdir()
    (new / "m.pth").write_bytes(b"torch")       # model present, conversion not

    sep = _Sep(str(legacy))
    sep.download_file_if_not_exists("http://unused", str(new / "m.pth"))
    assert (new / "m.safetensors").read_bytes() == b"mlx"


def test_a_json_or_yaml_does_not_grow_a_safetensors_sibling(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "vr_model_data.json").write_bytes(b"{}")
    (legacy / "vr_model_data.safetensors").write_bytes(b"nonsense")
    target = tmp_path / "new" / "vr_model_data.json"

    sep = _Sep(str(legacy))
    assert sep._adopt_legacy_model_file(str(target)) is True
    assert not (tmp_path / "new" / "vr_model_data.safetensors").exists()
