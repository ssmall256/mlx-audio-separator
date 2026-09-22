"""Tests for MDXC experimental Roformer fast-norm flag plumbing."""

import mlx_audio_separator.separator.architectures.mdxc_separator as mdxc_mod

_ALL_KEYS = frozenset({
    "experimental_roformer_fast_norm",
    "experimental_roformer_grouped_band_split",
    "experimental_roformer_grouped_mask_estimator",
    "experimental_roformer_grouped_weight_cache",
    "experimental_roformer_chunk_gather_batching",
    "experimental_roformer_ola_simd_tuning",
    "experimental_roformer_compile_fullgraph",
})


class _NoopLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


def _run_load_model_with_flag(
    monkeypatch,
    enabled: bool,
    grouped: bool = False,
    compile_fullgraph: bool = False,
    explicit: "frozenset | None" = None,
) -> dict:
    seen_env = {"value": None, "grouped": None, "mask": None, "ola_tuning": None, "fullgraph": None}

    def fake_loader(*, model_path, config):
        seen_env["value"] = mdxc_mod.os.environ.get("MLX_AUDIO_SEPARATOR_ROFORMER_FAST_NORM")
        seen_env["grouped"] = mdxc_mod.os.environ.get("MLX_AUDIO_SEPARATOR_ROFORMER_GROUPED_BAND_SPLIT")
        seen_env["mask"] = mdxc_mod.os.environ.get("MLX_AUDIO_SEPARATOR_ROFORMER_GROUPED_MASK_ESTIMATOR")
        seen_env["ola_tuning"] = mdxc_mod.os.environ.get("MLX_AUDIO_SEPARATOR_ROFORMER_OLA_SIMD_TUNING")
        seen_env["fullgraph"] = mdxc_mod.os.environ.get("MLX_AUDIO_SEPARATOR_ROFORMER_COMPILE_FULLGRAPH")
        return (lambda x: x), "bs_roformer"

    monkeypatch.setattr("mlx_audio_separator.separator.models.mdxc.loader.load_mdxc_model", fake_loader)

    sep = mdxc_mod.MDXCSeparator.__new__(mdxc_mod.MDXCSeparator)
    sep.logger = _NoopLogger()
    sep.experimental_roformer_fast_norm = bool(enabled)
    sep.experimental_compile_model_forward = False
    sep.experimental_compile_shapeless = False
    sep.experimental_roformer_static_compiled_demix = False
    sep.experimental_roformer_grouped_band_split = bool(grouped)
    sep.experimental_roformer_grouped_mask_estimator = bool(grouped)
    sep.experimental_roformer_ola_simd_tuning = bool(grouped)
    sep.experimental_roformer_compile_fullgraph = bool(compile_fullgraph)
    sep.model_path = "/tmp/fake.ckpt"
    sep.model_data = {}
    sep._compiled_model_run = None
    sep._fixed_batch_compiled_forward = False
    # Only keys the caller actually supplied are published to the environment.
    sep.performance_params_explicit_keys = (
        _ALL_KEYS if explicit is None else frozenset(explicit)
    )

    sep._load_model()
    return seen_env


def test_mdxc_load_model_sets_roformer_fast_norm_env_true(monkeypatch):
    assert _run_load_model_with_flag(monkeypatch, enabled=True)["value"] == "1"


def test_mdxc_load_model_sets_roformer_fast_norm_env_false(monkeypatch):
    assert _run_load_model_with_flag(monkeypatch, enabled=False)["value"] == "0"


def test_mdxc_load_model_sets_roformer_grouped_flags(monkeypatch):
    seen = _run_load_model_with_flag(monkeypatch, enabled=False, grouped=True)
    assert seen["grouped"] == "1"
    assert seen["mask"] == "1"
    assert seen["ola_tuning"] == "1"


def test_mdxc_load_model_sets_roformer_compile_fullgraph_flag(monkeypatch):
    seen = _run_load_model_with_flag(monkeypatch, enabled=False, compile_fullgraph=True)
    assert seen["fullgraph"] == "1"


def test_unrequested_flags_do_not_clobber_the_environment(monkeypatch):
    """An exported value must survive a run that never mentions that flag.

    These variables are read lazily during model construction, and the loader
    used to publish all seven unconditionally, so anything a user exported was
    overwritten before it could take effect. Six of the seven have no CLI flag,
    making the environment the only way to set them at all.
    """
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_ROFORMER_FAST_NORM", "1")
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_ROFORMER_GROUPED_BAND_SPLIT", "1")

    seen = _run_load_model_with_flag(
        monkeypatch, enabled=False, explicit=frozenset()
    )

    assert seen["value"] == "1", "exported fast-norm value was clobbered"
    assert seen["grouped"] == "1", "exported grouped-band-split value was clobbered"


def test_explicitly_requested_flag_still_overrides_the_environment(monkeypatch):
    """Asking for a value in performance_params wins over the environment."""
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_ROFORMER_FAST_NORM", "1")

    seen = _run_load_model_with_flag(
        monkeypatch,
        enabled=False,
        explicit=frozenset({"experimental_roformer_fast_norm"}),
    )

    assert seen["value"] == "0"
