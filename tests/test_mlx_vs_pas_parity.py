import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "perf" / "mlx_vs_pas_parity.py"
    spec = importlib.util.spec_from_file_location("mlx_vs_pas_parity", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_normalize_output_paths_prefers_output_dir(tmp_path):
    mod = _load_module()
    out_dir = tmp_path / "outs"
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "mix_(Vocals).wav"
    f.write_bytes(b"abc")

    resolved = mod._normalize_output_paths(["mix_(Vocals).wav"], output_dir=out_dir)
    assert len(resolved) == 1
    assert Path(resolved[0]) == f


def test_validate_outputs_reports_empty_file(tmp_path):
    mod = _load_module()
    out_dir = tmp_path / "outs"
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "a.wav"
    f.write_bytes(b"")

    ok, reason, _ = mod._validate_outputs(["a.wav"], output_dir=out_dir, wait_seconds=0.0)
    assert ok is False
    assert reason and "empty output files" in reason


def test_build_summary_includes_termination_flags():
    mod = _load_module()
    rows = [
        {"status": "ok", "pass": True},
        {"status": "error: boom", "pass": False},
    ]
    summary = mod._build_summary(rows, fail_fast=True, terminated_early=True, stop_reason="model: error")

    assert summary["total_models"] == 2
    assert summary["ok_models"] == 1
    assert summary["failed_models"] == 1
    assert summary["pass_models"] == 1
    assert summary["terminated_early"] is True
    assert summary["stop_reason"] == "model: error"


def test_demucs_strict_mlx_env_sets_and_restores(monkeypatch):
    mod = _load_module()

    for key in (
        "MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED",
        "MLX_AUDIO_SEPARATOR_DEMUCS_ISTFT_ALLOW_FUSED",
        "MLX_AUDIO_SEPARATOR_DEMUCS_WIENER_USE_VMAP",
        "MLX_AUDIO_SEPARATOR_DEMUCS_STRICT_EVAL",
    ):
        monkeypatch.delenv(key, raising=False)

    with mod._demucs_strict_mlx_env(enabled=True):
        assert mod.os.environ["MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED"] == "1"
        assert mod.os.environ["MLX_AUDIO_SEPARATOR_DEMUCS_ISTFT_ALLOW_FUSED"] == "0"
        assert mod.os.environ["MLX_AUDIO_SEPARATOR_DEMUCS_WIENER_USE_VMAP"] == "0"
        assert mod.os.environ["MLX_AUDIO_SEPARATOR_DEMUCS_STRICT_EVAL"] == "1"

    for key in (
        "MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED",
        "MLX_AUDIO_SEPARATOR_DEMUCS_ISTFT_ALLOW_FUSED",
        "MLX_AUDIO_SEPARATOR_DEMUCS_WIENER_USE_VMAP",
        "MLX_AUDIO_SEPARATOR_DEMUCS_STRICT_EVAL",
    ):
        assert key not in mod.os.environ
