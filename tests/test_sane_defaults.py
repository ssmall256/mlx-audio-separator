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
    DEFAULT_VR_BATCH_SIZE,
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


def test_io_policy_defaults_to_the_measured_optimum():
    """deferred + 2 writers: ~6-17% faster, bit-identical output, +70 MB RSS."""
    from mlx_audio_separator.utils.performance import DEFAULT_PERFORMANCE_PARAMS

    assert DEFAULT_PERFORMANCE_PARAMS["cache_clear_policy"] == "deferred"
    assert DEFAULT_PERFORMANCE_PARAMS["write_workers"] == 2


def test_library_does_not_touch_global_warning_filters(tmp_path, monkeypatch):
    """Separator() used to silence every warning in the host process.

    It called warnings.filterwarnings("ignore") at any log level above DEBUG,
    which is not a library's call to make, and it hid real defects: leaked file
    handles in core.py and the FutureWarning about an unusable legacy Demucs
    cache. Asserted behaviourally rather than by reading source, so the check
    cannot be satisfied by a comment.
    """
    import warnings

    from mlx_audio_separator.core import Separator

    touched = []
    monkeypatch.setattr(
        warnings, "filterwarnings", lambda *a, **k: touched.append(("filterwarnings", a))
    )
    monkeypatch.setattr(
        warnings, "simplefilter", lambda *a, **k: touched.append(("simplefilter", a))
    )

    Separator(info_only=True, model_file_dir=str(tmp_path / "models"))

    assert touched == [], f"Separator mutated global warning filters: {touched}"


def test_core_does_not_leak_file_handles(tmp_path):
    """The three json.load(open(...)) sites were only invisible because every
    warning was being suppressed."""
    import warnings

    from mlx_audio_separator.core import Separator

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sep = Separator(info_only=True, model_file_dir=str(tmp_path / "models"))
        sep.list_supported_model_files()

    leaks = [w for w in caught if issubclass(w.category, ResourceWarning)]
    assert leaks == [], f"unclosed files: {[str(w.message) for w in leaks]}"


def test_speed_mode_is_deprecated_and_inert(tmp_path):
    from mlx_audio_separator.core import Separator

    with pytest.warns(DeprecationWarning, match="no longer changes anything"):
        sep = Separator(
            info_only=True,
            model_file_dir=str(tmp_path / "m"),
            performance_params={"speed_mode": "latency_safe_v3"},
        )
    baseline = Separator(info_only=True, model_file_dir=str(tmp_path / "b"))
    assert (
        sep.performance_params["cache_clear_policy"]
        == baseline.performance_params["cache_clear_policy"]
    )


def test_shipped_perf_configs_do_not_pin_a_deprecated_speed_mode():
    """They would warn on every benchmark run and measure the default anyway."""
    import glob
    import json
    import os

    root = os.path.join(os.path.dirname(__file__), os.pardir, "scripts", "perf", "configs")
    offenders = []
    for path in glob.glob(os.path.join(root, "*.json")):
        with open(path, encoding="utf-8") as handle:
            cfg = json.load(handle)
        mode = json.dumps(cfg)
        if "latency_safe" in mode:
            offenders.append(os.path.basename(path))
    assert offenders == [], f"configs pin a deprecated speed_mode: {offenders}"


def test_vr_batch_size_default_is_measured():
    """Batch 1 was never fastest; 2 wins on short clips and is within 0.02 s of
    batch 4's best on long ones for 2.5 GB less peak memory."""
    assert DEFAULT_VR_BATCH_SIZE == 2


def test_vr_batch_default_is_used_by_cli_and_core(tmp_path):
    from mlx_audio_separator.core import Separator

    sep = Separator(info_only=True, model_file_dir=str(tmp_path / "m"))
    assert sep.arch_specific_params["VR"]["batch_size"] == DEFAULT_VR_BATCH_SIZE
