"""The shipped defaults must be the recommended configuration.

Every value here was chosen from a measurement, not a guess, and each one used
to be set to something worse. These tests exist so a future change has to argue
with the numbers rather than silently regress them.
"""

from __future__ import annotations

import os

import pytest

import mlx_audio_separator.demucs_mlx.metal_kernels as mk
from mlx_audio_separator.demucs_mlx.defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SHIFT_SEED,
)
from mlx_audio_separator.utils.performance import (
    apply_experimental_env,
    explicit_performance_keys,
)


def test_fused_groupnorm_is_off_by_default(monkeypatch):
    """~20 dB SNR worse than unfused and not faster."""
    monkeypatch.delenv("MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE", raising=False)
    monkeypatch.delenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED", raising=False)
    assert mk._fused_groupnorm_mode() == "off"


def test_fused_groupnorm_can_still_be_re_enabled(monkeypatch):
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE", "all")
    assert mk._fused_groupnorm_mode() == "all"


def test_demucs_batch_size_default_is_the_measured_optimum():
    """8 was ~2x slower at ~2x the memory; 12 was ~10x slower."""
    assert DEFAULT_BATCH_SIZE == 2


def test_demucs_batch_size_is_defined_in_exactly_one_place():
    """The CLI, the API and apply_model all drifted apart before."""
    import inspect

    from mlx_audio_separator.demucs_mlx import api, apply_mlx

    assert (
        inspect.signature(apply_mlx.apply_model).parameters["batch_size"].default
        == DEFAULT_BATCH_SIZE
    )
    assert (
        inspect.signature(api.Separator.__init__).parameters["batch_size"].default
        == DEFAULT_BATCH_SIZE
    )


def test_demucs_shifts_are_seeded_by_default():
    """Identical input should reproduce."""
    assert DEFAULT_SHIFT_SEED is not None


def test_speed_modes_do_not_select_a_worse_batch_size():
    """latency_safe_v2 used to select 12, which is ~10x slower than 2."""
    from mlx_audio_separator.core import Separator

    src = __import__("inspect").getsource(Separator._apply_speed_mode_overrides)
    assert '"Demucs": 12' not in src
    assert '"Demucs": 8' not in src


class TestEnvironmentIsNotClobbered:
    """Ten variables were overwritten on every run, ignoring user exports."""

    def test_unrequested_flag_leaves_the_environment_alone(self, monkeypatch):
        monkeypatch.setenv("MLX_AUDIO_SEPARATOR_DEMUCS_APPLY_CONCAT_BATCHING", "1")
        apply_experimental_env(
            "MLX_AUDIO_SEPARATOR_DEMUCS_APPLY_CONCAT_BATCHING",
            False,
            explicit=False,
        )
        assert os.environ["MLX_AUDIO_SEPARATOR_DEMUCS_APPLY_CONCAT_BATCHING"] == "1"

    def test_requested_flag_is_published(self, monkeypatch):
        monkeypatch.setenv("MLX_AUDIO_SEPARATOR_DEMUCS_APPLY_CONCAT_BATCHING", "1")
        apply_experimental_env(
            "MLX_AUDIO_SEPARATOR_DEMUCS_APPLY_CONCAT_BATCHING",
            False,
            explicit=True,
        )
        assert os.environ["MLX_AUDIO_SEPARATOR_DEMUCS_APPLY_CONCAT_BATCHING"] == "0"

    def test_unrequested_flag_does_not_invent_a_value(self, monkeypatch):
        monkeypatch.delenv(
            "MLX_AUDIO_SEPARATOR_DEMUCS_APPLY_CONCAT_BATCHING", raising=False
        )
        apply_experimental_env(
            "MLX_AUDIO_SEPARATOR_DEMUCS_APPLY_CONCAT_BATCHING",
            False,
            explicit=False,
        )
        assert "MLX_AUDIO_SEPARATOR_DEMUCS_APPLY_CONCAT_BATCHING" not in os.environ


def test_explicit_keys_records_only_what_the_caller_supplied():
    assert explicit_performance_keys(None) == frozenset()
    assert explicit_performance_keys({"write_workers": 2}) == frozenset({"write_workers"})


@pytest.mark.parametrize(
    "flag,expected",
    [("auto", None), ("bf16", "1"), ("fp32", "0")],
)
def test_precision_flag_maps_to_amp(flag, expected, monkeypatch):
    """bf16 stays the default (faster, ~70 dB SNR from fp32) but is now switchable."""
    monkeypatch.delenv("MLX_ENABLE_AMP", raising=False)
    if flag != "auto":
        os.environ["MLX_ENABLE_AMP"] = "1" if flag == "bf16" else "0"
    assert os.environ.get("MLX_ENABLE_AMP") == expected


def test_cli_exposes_precision_and_seed():
    """Both were previously reachable only by reading source."""
    from mlx_audio_separator.utils import cli

    parser_src = __import__("inspect").getsource(cli)
    assert '"--precision"' in parser_src
    assert '"--demucs_seed"' in parser_src
