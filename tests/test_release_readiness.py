"""Tests for scripts/release/check_release_readiness.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_release_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "release"
        / "check_release_readiness.py"
    )
    spec = importlib.util.spec_from_file_location("check_release_readiness", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load check_release_readiness module.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_release_gate_passes_on_all_ok():
    mod = _load_release_module()
    results = [
        {"filename": "a", "arch": "MDX", "status": "ok"},
        {"filename": "b", "arch": "MDXC", "status": "ok"},
    ]
    summary = mod.evaluate_release_readiness(
        results,
        require_total_models=2,
        max_failures=0,
        min_success_rate=1.0,
        require_models_ok=["a"],
    )
    assert summary["pass"] is True
    assert summary["failures"] == 0
    assert summary["success_rate"] == 1.0


def test_release_gate_fails_on_required_model_not_ok():
    mod = _load_release_module()
    results = [
        {"filename": "a", "arch": "MDX", "status": "error: boom"},
        {"filename": "b", "arch": "MDXC", "status": "ok"},
    ]
    summary = mod.evaluate_release_readiness(
        results,
        require_total_models=2,
        max_failures=0,
        min_success_rate=1.0,
        require_models_ok=["a", "b"],
    )
    assert summary["pass"] is False
    assert summary["failures"] == 1
    assert any("required models not ok" in reason for reason in summary["reasons"])


def test_release_gate_can_allow_skipped_rows():
    mod = _load_release_module()
    results = [
        {"filename": "a", "arch": "Demucs", "status": "skipped: missing dependency"},
        {"filename": "b", "arch": "MDX", "status": "ok"},
    ]

    strict_summary = mod.evaluate_release_readiness(
        results,
        require_total_models=2,
        max_failures=0,
        min_success_rate=1.0,
        allow_skipped=False,
    )
    assert strict_summary["pass"] is False

    allow_summary = mod.evaluate_release_readiness(
        results,
        require_total_models=2,
        max_failures=0,
        min_success_rate=1.0,
        allow_skipped=True,
    )
    assert allow_summary["pass"] is True
