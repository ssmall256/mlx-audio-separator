"""The legacy /tmp model cache must be skippable.

The fallback reaches outside `$HOME`, so a release smoke test that only points
`HOME` at an empty directory still starts with a warm model cache and cannot
prove first-run behaviour. `MLX_AUDIO_SEPARATOR_NO_LEGACY_CACHE=1` turns it off.
"""

import pytest

from mlx_audio_separator import core
from mlx_audio_separator.core import Separator


@pytest.fixture
def legacy_dir(monkeypatch, tmp_path):
    """Point the legacy default at a directory that exists."""
    legacy = tmp_path / "legacy-models"
    legacy.mkdir()
    monkeypatch.setattr(core, "LEGACY_MODEL_FILE_DIR", str(legacy))
    monkeypatch.setattr(core, "DEFAULT_MODEL_FILE_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("AUDIO_SEPARATOR_MODEL_DIR", raising=False)
    return legacy


def test_the_legacy_directory_is_adopted_by_default(legacy_dir, monkeypatch, tmp_path):
    monkeypatch.delenv("MLX_AUDIO_SEPARATOR_NO_LEGACY_CACHE", raising=False)
    sep = Separator(info_only=True, output_dir=str(tmp_path / "out"))
    assert sep._legacy_model_file_dir == str(legacy_dir)


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_the_opt_out_disables_adoption(legacy_dir, monkeypatch, tmp_path, value):
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_NO_LEGACY_CACHE", value)
    sep = Separator(info_only=True, output_dir=str(tmp_path / "out"))
    assert sep._legacy_model_file_dir is None, f"{value!r} should disable adoption"


@pytest.mark.parametrize("value", ["0", "false", "", "no"])
def test_other_values_leave_adoption_alone(legacy_dir, monkeypatch, tmp_path, value):
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_NO_LEGACY_CACHE", value)
    sep = Separator(info_only=True, output_dir=str(tmp_path / "out"))
    assert sep._legacy_model_file_dir == str(legacy_dir)


def test_an_explicit_model_dir_still_wins(legacy_dir, monkeypatch, tmp_path):
    monkeypatch.delenv("MLX_AUDIO_SEPARATOR_NO_LEGACY_CACHE", raising=False)
    sep = Separator(
        info_only=True,
        model_file_dir=str(tmp_path / "chosen"),
        output_dir=str(tmp_path / "out"),
    )
    assert sep._legacy_model_file_dir is None
