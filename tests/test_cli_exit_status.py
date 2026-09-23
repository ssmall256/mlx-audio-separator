"""A file that fails to separate must make the CLI exit non-zero.

`Separator.separate` logs a failed file and moves on, so the batch survives one
bad input. The CLI then printed "Separation complete!" and exited 0 regardless
-- so a run that produced no stems at all looked like a success to any script
wrapping it. That is how a total failure went unnoticed.
"""

from __future__ import annotations

import logging

import pytest

from mlx_audio_separator import core
from mlx_audio_separator.utils import cli


class _Separator:
    """Stands in for Separator: records failures the way the real one does."""

    def __init__(self, outputs, failures):
        self._outputs = outputs
        self.failed_files = failures

    def load_model(self, **kwargs):
        return None

    def separate(self, audio_files, custom_output_names=None):
        return list(self._outputs)


@pytest.fixture
def run_cli(monkeypatch, tmp_path):
    def run(outputs, failures):
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"RIFF")
        # main() imports Separator from core at call time.
        monkeypatch.setattr(
            core, "Separator", lambda **kwargs: _Separator(outputs, failures)
        )
        monkeypatch.setattr(
            "sys.argv", ["mlx-audio-separator", str(clip), "--output_dir", str(tmp_path)]
        )
        try:
            cli.main()
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0

    return run


def test_success_exits_zero(run_cli):
    assert run_cli(["a_(vocals).wav"], []) == 0


def test_a_total_failure_exits_non_zero(run_cli):
    assert run_cli([], [("clip.wav", "no usable threadgroup size")]) == 1


def test_a_partial_failure_exits_non_zero(run_cli):
    """Some stems written is still not a success -- a batch job must know."""
    assert run_cli(["a_(vocals).wav"], [("b.wav", "boom")]) == 1


def test_the_failure_is_reported_at_error_level(run_cli, caplog):
    with caplog.at_level(logging.ERROR):
        run_cli([], [("clip.wav", "no usable threadgroup size")])
    text = caplog.text
    assert "clip.wav" in text
    assert "no usable threadgroup size" in text
    assert "no output written" in text


def test_success_does_not_claim_failure(run_cli, caplog):
    with caplog.at_level(logging.INFO):
        run_cli(["a_(vocals).wav"], [])
    assert "Separation complete!" in caplog.text
    assert "Failed" not in caplog.text
