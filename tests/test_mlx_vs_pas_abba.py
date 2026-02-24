import importlib.util
from pathlib import Path


def _load_abba_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "perf" / "mlx_vs_pas_abba.py"
    spec = importlib.util.spec_from_file_location("mlx_vs_pas_abba", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_normalize_output_paths_prefers_output_dir(tmp_path):
    mod = _load_abba_module()
    out_dir = tmp_path / "outs"
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "mix_(Vocals).wav"
    f.write_bytes(b"abc")

    resolved = mod._normalize_output_paths(["mix_(Vocals).wav"], output_dir=out_dir)
    assert len(resolved) == 1
    assert Path(resolved[0]) == f


def test_validate_outputs_accepts_relative_paths_in_output_dir(tmp_path):
    mod = _load_abba_module()
    out_dir = tmp_path / "outs"
    out_dir.mkdir(parents=True, exist_ok=True)
    f1 = out_dir / "a.wav"
    f2 = out_dir / "b.wav"
    f1.write_bytes(b"1")
    f2.write_bytes(b"2")

    ok, reason, resolved = mod._validate_outputs(["a.wav", "b.wav"], output_dir=out_dir, wait_seconds=0.0)
    assert ok is True
    assert reason is None
    assert [Path(x).name for x in resolved] == ["a.wav", "b.wav"]


def test_validate_outputs_missing_file_returns_reason(tmp_path):
    mod = _load_abba_module()
    out_dir = tmp_path / "outs"
    out_dir.mkdir(parents=True, exist_ok=True)
    f1 = out_dir / "a.wav"
    f1.write_bytes(b"1")

    ok, reason, resolved = mod._validate_outputs(["a.wav", "missing.wav"], output_dir=out_dir, wait_seconds=0.0)
    assert ok is False
    assert reason and "missing output files" in reason
    assert len(resolved) == 2
