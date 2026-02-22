"""Tests for perf trace emission and cache policy behavior."""

import json

from mlx_audio_separator.core import Separator
from mlx_audio_separator.utils.performance import PerfTraceWriter


class _FakeModel:
    def __init__(self):
        self.clear_calls = 0

    def clear_gpu_cache(self):
        self.clear_calls += 1


def test_perf_trace_writer_jsonl(tmp_path):
    path = tmp_path / "perf_trace.jsonl"
    writer = PerfTraceWriter(str(path))
    writer.write({"model": "x", "metrics": {"total_s": 1.23}})

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["model"] == "x"
    assert payload["metrics"]["total_s"] == 1.23


def test_core_emits_perf_trace(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    sep = Separator(
        info_only=True,
        model_file_dir=str(tmp_path / "models"),
        output_dir=str(tmp_path / "out"),
        performance_params={"perf_trace": True, "perf_trace_path": str(trace_path)},
    )
    sep.model_name = "demo"
    sep.model_type = "MDX"
    sep._emit_perf_trace(
        audio_file_path=str(tmp_path / "song.wav"),
        metrics={
            "decode_s": 0.1,
            "preprocess_s": 0.2,
            "inference_s": 0.3,
            "postprocess_s": 0.4,
            "write_s": 0.5,
            "cleanup_s": 0.6,
            "total_s": 2.1,
        },
    )

    payload = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert payload["model"] == "demo"
    assert payload["arch"] == "MDX"
    assert payload["metrics"]["inference_s"] == 0.3
    assert "params" in payload


def test_deferred_cache_policy_periodic_clear(tmp_path):
    sep = Separator(
        info_only=True,
        model_file_dir=str(tmp_path / "models"),
        performance_params={"cache_clear_policy": "deferred"},
    )
    sep.model_instance = _FakeModel()

    for _ in range(9):
        sep._apply_cache_policy_after_file()
    assert sep.model_instance.clear_calls == 0

    sep._apply_cache_policy_after_file()
    assert sep.model_instance.clear_calls == 1
