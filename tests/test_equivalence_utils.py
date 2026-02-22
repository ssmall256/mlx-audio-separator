"""Tests for deterministic equivalence utility helpers."""

import os

import numpy as np

from mlx_audio_separator.utils import equivalence as eq
from mlx_audio_separator.utils.equivalence import compare_stem_maps


def test_compare_stem_maps_pass_on_identical_audio():
    x = np.zeros((100, 2), dtype=np.float32)
    baseline = {"vocals": (x.copy(), 44100)}
    candidate = {"vocals": (x.copy(), 44100)}

    result = compare_stem_maps(baseline, candidate, threshold_rel_l2=1e-5)

    assert result["pass"] is True
    assert result["counts_match"] is True
    assert result["stems_match"] is True
    assert result["max_rel_l2"] == 0.0
    assert result["stems"][0]["status"] == "ok"


def test_compare_stem_maps_fails_on_drift():
    baseline = {"vocals": (np.ones((64, 2), dtype=np.float32), 44100)}
    candidate = {"vocals": (np.zeros((64, 2), dtype=np.float32), 44100)}

    result = compare_stem_maps(baseline, candidate, threshold_rel_l2=1e-5)

    assert result["pass"] is False
    assert result["stems"][0]["status"] == "drift"
    assert result["stems"][0]["pass_rel_l2"] is False
    assert result["max_rel_l2"] > 0.1


def test_compare_stem_maps_fails_on_missing_stem():
    baseline = {"vocals": (np.zeros((32, 2), dtype=np.float32), 44100)}
    candidate = {"instrumental": (np.zeros((32, 2), dtype=np.float32), 44100)}

    result = compare_stem_maps(baseline, candidate, threshold_rel_l2=1e-5)

    assert result["pass"] is False
    assert result["counts_match"] is True
    assert result["stems_match"] is False
    assert any(row["status"] == "missing" for row in result["stems"])


def test_equivalence_suite_default_gating_skips_demucs(monkeypatch):
    def fake_run_model_equivalence(**kwargs):
        return {
            "model": kwargs["model_filename"],
            "status": "ok",
            "arch": "Demucs",
            "error": None,
            "per_file": {},
            "max_rel_l2": 0.2,
            "pass": False,
        }

    monkeypatch.setattr(eq, "run_model_equivalence", fake_run_model_equivalence)
    payload = eq.run_equivalence_suite(
        corpus=["/tmp/fake.wav"],
        models=["htdemucs.yaml"],
        baseline_separator_kwargs={},
        candidate_separator_kwargs={},
    )

    row = payload["summary"][0]
    assert row["arch"] == "Demucs"
    assert row["strict_pass"] is False
    assert row["gated"] is False
    assert row["pass"] is True
    assert payload["all_pass"] is True


def test_equivalence_suite_can_gate_demucs(monkeypatch):
    def fake_run_model_equivalence(**kwargs):
        return {
            "model": kwargs["model_filename"],
            "status": "ok",
            "arch": "Demucs",
            "error": None,
            "per_file": {},
            "max_rel_l2": 0.2,
            "pass": False,
        }

    monkeypatch.setattr(eq, "run_model_equivalence", fake_run_model_equivalence)
    payload = eq.run_equivalence_suite(
        corpus=["/tmp/fake.wav"],
        models=["htdemucs.yaml"],
        baseline_separator_kwargs={},
        candidate_separator_kwargs={},
        gated_arches={"Demucs"},
    )

    row = payload["summary"][0]
    assert row["gated"] is True
    assert row["pass"] is False
    assert payload["all_pass"] is False


def test_temporary_env_restores_previous_value():
    key = "MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED"
    previous = os.environ.get(key)
    os.environ[key] = "0"
    try:
        with eq._temporary_env(key, "1"):
            assert os.environ.get(key) == "1"
        assert os.environ.get(key) == "0"
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def test_temporary_env_restores_missing_value():
    key = "MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED"
    previous = os.environ.pop(key, None)
    try:
        with eq._temporary_env(key, "1"):
            assert os.environ.get(key) == "1"
        assert key not in os.environ
    finally:
        if previous is not None:
            os.environ[key] = previous


def test_run_model_equivalence_sets_istft_unfused_for_determinism(monkeypatch):
    observed: dict[str, str | None] = {}

    def fake_separator_from_kwargs(*args, **kwargs):
        observed["det_fused"] = os.environ.get("MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED")
        observed["istft_allow_fused"] = os.environ.get("MLX_AUDIO_SEPARATOR_DEMUCS_ISTFT_ALLOW_FUSED")
        observed["wiener_use_vmap"] = os.environ.get("MLX_AUDIO_SEPARATOR_DEMUCS_WIENER_USE_VMAP")
        observed["strict_eval"] = os.environ.get("MLX_AUDIO_SEPARATOR_DEMUCS_STRICT_EVAL")
        raise RuntimeError("stop after env capture")

    monkeypatch.setattr(eq, "_separator_from_kwargs", fake_separator_from_kwargs)

    result = eq.run_model_equivalence(
        model_filename="htdemucs.yaml",
        corpus=["/tmp/fake.wav"],
        baseline_separator_kwargs={},
        candidate_separator_kwargs={},
    )

    assert observed["det_fused"] == "1"
    assert observed["istft_allow_fused"] == "0"
    assert observed["wiener_use_vmap"] == "0"
    assert observed["strict_eval"] == "1"
    assert result["status"] == "error"
