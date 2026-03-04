"""Unit tests for latency comparison helpers."""

import importlib.util
from pathlib import Path
import sys

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "perf" / "compare_latency.py"
_SPEC = importlib.util.spec_from_file_location("compare_latency", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["compare_latency"] = _MOD
_SPEC.loader.exec_module(_MOD)

STAGE_KEYS = _MOD.STAGE_KEYS
_timed_separate = _MOD._timed_separate
compare_results = _MOD.compare_results
parse_args = _MOD.parse_args
_extract_speed_mode = _MOD._extract_speed_mode
_validate_speed_mode_alignment = _MOD._validate_speed_mode_alignment


class _FakeSep:
    def __init__(self):
        self.calls = 0
        self.last_perf_metrics = {}

    def separate(self, _audio_path):
        self.calls += 1
        base = float(self.calls)
        self.last_perf_metrics = {
            "decode_s": base * 0.1,
            "preprocess_s": base * 0.01,
            "inference_s": base * 0.2,
            "postprocess_s": base * 0.02,
            "write_s": base * 0.3,
            "cleanup_s": base * 0.04,
            "total_s": base * 0.67,
        }


def test_timed_separate_emits_stage_payloads():
    sep = _FakeSep()
    runs = _timed_separate(sep, "/tmp/in.wav", repeats=3)

    assert len(runs) == 3
    for run in runs:
        assert "total_s" in run
        assert "stages" in run
        for stage in STAGE_KEYS:
            assert stage in run["stages"]


def test_compare_results_includes_stage_and_p95():
    baseline = {
        "BS-Roformer-SW.ckpt": {
            "status": "ok",
            "arch": "MDXC",
            "median_s": 10.0,
            "p95_s": 12.0,
            "stage_medians_s": {k: 1.0 for k in STAGE_KEYS},
        }
    }
    candidate = {
        "BS-Roformer-SW.ckpt": {
            "status": "ok",
            "arch": "MDXC",
            "median_s": 9.0,
            "p95_s": 10.0,
            "stage_medians_s": {k: 0.9 for k in STAGE_KEYS},
        }
    }

    rows = compare_results(
        models=["BS-Roformer-SW.ckpt"],
        baseline=baseline,
        candidate=candidate,
        target_improvement_demucs_mdxc=5.0,
        max_regression_mdx_vr=5.0,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["delta_pct"] < 0
    assert row["p95_delta_pct"] < 0
    assert isinstance(row["baseline_stage_medians_s"], dict)
    assert isinstance(row["candidate_stage_medians_s"], dict)
    assert row["pass"] is True


def test_parse_args_supports_cooldown_controls():
    args = parse_args(
        [
            "--corpus-file",
            "corpus.txt",
            "--baseline-config",
            "baseline.json",
            "--candidate-config",
            "candidate.json",
            "--cooldown-seconds-after-file",
            "15",
            "--cooldown-seconds-after-model",
            "20",
            "--cooldown-seconds-between-configs",
            "120",
            "--cooldown-seconds-before-equivalence",
            "60",
            "--warmup-override",
            "0",
            "--repeats-override",
            "1",
            "--allow-speed-mode-mismatch",
        ]
    )

    assert args.cooldown_seconds_after_file == 15.0
    assert args.cooldown_seconds_after_model == 20.0
    assert args.cooldown_seconds_between_configs == 120.0
    assert args.cooldown_seconds_before_equivalence == 60.0
    assert args.warmup_override == 0
    assert args.repeats_override == 1
    assert args.allow_speed_mode_mismatch is True


def test_extract_speed_mode_defaults_and_override():
    assert _extract_speed_mode({}) == "default"
    assert _extract_speed_mode({"separator": {"performance_params": {}}}) == "default"
    assert _extract_speed_mode({"separator": {"performance_params": {"speed_mode": "latency_safe_v3"}}}) == "latency_safe_v3"


def test_speed_mode_mismatch_guard_requires_opt_in():
    with pytest.raises(ValueError):
        _validate_speed_mode_alignment("default", "latency_safe_v3", allow_mismatch=False)
    _validate_speed_mode_alignment("default", "latency_safe_v3", allow_mismatch=True)
