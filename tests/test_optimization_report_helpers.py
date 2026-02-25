"""Unit tests for optimization report helper functions."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_report_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "perf" / "run_optimization_report.py"
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("run_optimization_report", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_optimization_report module.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_si_sdr_db_identical_is_large():
    rpt = _load_report_module()
    x = np.random.default_rng(0).standard_normal((2048, 2)).astype(np.float32)
    value = rpt._si_sdr_db(x, x)
    assert value > 100.0


def test_rel_l2_identical_is_zero():
    rpt = _load_report_module()
    x = np.random.default_rng(0).standard_normal((512, 2)).astype(np.float32)
    assert rpt._rel_l2(x, x) == 0.0


def test_align_audio_pair_trims_to_shorter_shape():
    rpt = _load_report_module()
    a = np.zeros((100, 2), dtype=np.float32)
    b = np.zeros((80, 1), dtype=np.float32)
    a2, b2, aligned = rpt._align_audio_pair(a, b)
    assert aligned is True
    assert a2.shape == (80, 1)
    assert b2.shape == (80, 1)


def test_load_reference_manifest_normalizes_paths_and_stems(tmp_path: Path):
    rpt = _load_report_module()
    stem = tmp_path / "vocals.wav"
    stem.write_bytes(b"dummy")
    payload = {str(tmp_path / "song.wav"): {"Vocals": str(stem)}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = rpt._load_reference_manifest(manifest_path)
    key = str((tmp_path / "song.wav").resolve())
    assert key in manifest
    assert manifest[key]["vocals"] == str(stem.resolve())


def test_sha256_file_matches_known_value(tmp_path: Path):
    rpt = _load_report_module()
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert rpt._sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_run_cmd_success():
    rpt = _load_report_module()
    ok, out = rpt._run_cmd([sys.executable, "-c", "print('ok')"])
    assert ok is True
    assert out == "ok"


def test_collect_repro_metadata_includes_manifests(tmp_path: Path):
    rpt = _load_report_module()
    corpus_file = tmp_path / "clip.wav"
    corpus_file.write_bytes(b"not-a-real-wav")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_file = model_dir / "foo.onnx"
    model_file.write_bytes(b"model-bytes")

    meta = rpt.collect_repro_metadata(
        corpus=[str(corpus_file)],
        models=["foo.onnx", "missing.ckpt"],
        model_file_dir=str(model_dir),
        capture_corpus_hashes=False,
        capture_model_hashes=False,
    )
    assert "git" in meta
    assert meta["corpus_manifest"][0]["path"] == str(corpus_file)
    assert meta["model_manifest"][0]["exists"] is True
    assert meta["model_manifest"][1]["exists"] is False


def test_run_python_mps_parity_fail_fast_summary(monkeypatch, tmp_path: Path):
    rpt = _load_report_module()

    def fake_run_model(**kwargs):
        model = kwargs["model"]
        if model == "good.onnx":
            return {
                "model": model,
                "status": "ok",
                "pass": True,
                "arch_mlx": "MDX",
                "files_checked": 1,
                "files_passed": 1,
                "max_rel_l2": 0.0,
            }
        return {
            "model": model,
            "status": "error: boom",
            "pass": False,
            "arch_mlx": "MDX",
            "files_checked": 0,
            "files_passed": 0,
            "max_rel_l2": 1.0,
        }

    monkeypatch.setattr(rpt, "run_python_mps_parity_model", fake_run_model)
    payload = rpt.run_python_mps_parity(
        corpus=["/tmp/f.wav"],
        models=["good.onnx", "bad.onnx", "skipped.onnx"],
        output_root=tmp_path,
        model_file_dir="/tmp/models",
        mlx_separator_kwargs={"output_format": "WAV"},
        pas_separator_kwargs={"output_format": "WAV"},
        threshold_rel_l2=1e-5,
        seed=12345,
        demucs_shifts_zero=True,
        demucs_mlx_strict_kernels=True,
        max_files=1,
        fail_fast=True,
    )

    assert payload["status"] == "ok"
    assert payload["all_pass"] is False
    assert payload["summary"]["terminated_early"] is True
    assert payload["summary"]["total_models"] == 2
    assert payload["summary"]["ok_models"] == 1
    assert payload["summary"]["failed_models"] == 1
