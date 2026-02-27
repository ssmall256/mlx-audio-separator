"""Tests for MDXC experimental Roformer fast-norm flag plumbing."""

import mlx_audio_separator.separator.architectures.mdxc_separator as mdxc_mod


class _NoopLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


def _run_load_model_with_flag(monkeypatch, enabled: bool) -> str:
    seen_env = {"value": None}

    def fake_loader(*, model_path, config):
        seen_env["value"] = mdxc_mod.os.environ.get("MLX_AUDIO_SEPARATOR_ROFORMER_FAST_NORM")
        return (lambda x: x), "bs_roformer"

    monkeypatch.setattr("mlx_audio_separator.separator.models.mdxc.loader.load_mdxc_model", fake_loader)

    sep = mdxc_mod.MDXCSeparator.__new__(mdxc_mod.MDXCSeparator)
    sep.logger = _NoopLogger()
    sep.experimental_roformer_fast_norm = bool(enabled)
    sep.experimental_compile_model_forward = False
    sep.experimental_compile_shapeless = False
    sep.experimental_roformer_static_compiled_demix = False
    sep.model_path = "/tmp/fake.ckpt"
    sep.model_data = {}
    sep._compiled_model_run = None
    sep._fixed_batch_compiled_forward = False

    sep._load_model()
    return str(seen_env["value"])


def test_mdxc_load_model_sets_roformer_fast_norm_env_true(monkeypatch):
    assert _run_load_model_with_flag(monkeypatch, enabled=True) == "1"


def test_mdxc_load_model_sets_roformer_fast_norm_env_false(monkeypatch):
    assert _run_load_model_with_flag(monkeypatch, enabled=False) == "0"

