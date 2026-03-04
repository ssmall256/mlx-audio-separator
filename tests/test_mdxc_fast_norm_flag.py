"""Tests for MDXC experimental Roformer fast-norm flag plumbing."""

import mlx_audio_separator.separator.architectures.mdxc_separator as mdxc_mod


class _NoopLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


def _run_load_model_with_flag(monkeypatch, enabled: bool, grouped: bool = False, compile_fullgraph: bool = False) -> dict:
    seen_env = {"value": None, "grouped": None, "mask": None, "fullgraph": None}

    def fake_loader(*, model_path, config):
        seen_env["value"] = mdxc_mod.os.environ.get("MLX_AUDIO_SEPARATOR_ROFORMER_FAST_NORM")
        seen_env["grouped"] = mdxc_mod.os.environ.get("MLX_AUDIO_SEPARATOR_ROFORMER_GROUPED_BAND_SPLIT")
        seen_env["mask"] = mdxc_mod.os.environ.get("MLX_AUDIO_SEPARATOR_ROFORMER_GROUPED_MASK_ESTIMATOR")
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
    sep.experimental_roformer_compile_fullgraph = bool(compile_fullgraph)
    sep.model_path = "/tmp/fake.ckpt"
    sep.model_data = {}
    sep._compiled_model_run = None
    sep._fixed_batch_compiled_forward = False

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


def test_mdxc_load_model_sets_roformer_compile_fullgraph_flag(monkeypatch):
    seen = _run_load_model_with_flag(monkeypatch, enabled=False, compile_fullgraph=True)
    assert seen["fullgraph"] == "1"

