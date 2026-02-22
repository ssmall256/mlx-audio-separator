"""Tests for performance params and tuning helpers."""

import json

import pytest

from mlx_audio_separator.core import Separator
from mlx_audio_separator.utils.performance import normalize_performance_params, select_best_candidate


class TestPerformanceParams:
    def test_defaults_present(self, tmp_path):
        sep = Separator(info_only=True, model_file_dir=str(tmp_path / "models"))
        perf = sep.performance_params
        assert perf["speed_mode"] == "default"
        assert perf["auto_tune_batch"] is False
        assert perf["tune_probe_seconds"] == 8.0
        assert perf["cache_clear_policy"] == "aggressive"
        assert perf["write_workers"] == 1
        assert perf["perf_trace"] is False
        assert perf["perf_trace_path"] is None

    def test_latency_safe_batch_overrides(self, tmp_path):
        sep = Separator(
            info_only=True,
            model_file_dir=str(tmp_path / "models"),
            performance_params={"speed_mode": "latency_safe"},
        )
        assert sep.arch_specific_params["Demucs"]["batch_size"] == 8
        assert sep.arch_specific_params["MDXC"]["batch_size"] == 1
        assert sep.arch_specific_params["MDX"]["batch_size"] == 1
        assert sep.arch_specific_params["VR"]["batch_size"] == 1

    def test_invalid_speed_mode(self):
        with pytest.raises(ValueError, match="speed_mode"):
            normalize_performance_params({"speed_mode": "fastest"})

    def test_invalid_cache_policy(self):
        with pytest.raises(ValueError, match="cache_clear_policy"):
            normalize_performance_params({"cache_clear_policy": "never"})

    def test_invalid_write_workers(self):
        with pytest.raises(ValueError, match="write_workers"):
            normalize_performance_params({"write_workers": 0})

    def test_batch_selection_prefers_smallest_on_tie(self):
        best = select_best_candidate(
            {
                1: [1.10, 1.12],
                2: [0.98, 1.01],  # best median
                4: [1.00, 1.00],  # within tie threshold with 2
            },
            tie_ratio=0.03,
        )
        assert best == 2


def test_cli_performance_params_propagation(monkeypatch, tmp_path):
    import mlx_audio_separator.utils.cli as cli

    input_wav = tmp_path / "in.wav"
    input_wav.write_bytes(b"fake")
    captured = {}

    class FakeSeparator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def load_model(self, model_filename):
            captured["model_filename"] = model_filename

        def separate(self, audio_files, custom_output_names=None):
            captured["audio_files"] = list(audio_files)
            captured["custom_output_names"] = custom_output_names
            return []

    monkeypatch.setattr("mlx_audio_separator.core.Separator", FakeSeparator)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mlx-audio-separator",
            str(input_wav),
            "--speed_mode",
            "latency_safe",
            "--auto_tune_batch",
            "--tune_probe_seconds",
            "6.5",
            "--cache_clear_policy",
            "deferred",
            "--write_workers",
            "2",
            "--perf_trace",
            "--perf_trace_path",
            str(tmp_path / "perf.jsonl"),
            "--custom_output_names",
            json.dumps({"vocals": "vox"}),
        ],
    )
    cli.main()

    perf = captured["performance_params"]
    assert perf["speed_mode"] == "latency_safe"
    assert perf["auto_tune_batch"] is True
    assert perf["tune_probe_seconds"] == 6.5
    assert perf["cache_clear_policy"] == "deferred"
    assert perf["write_workers"] == 2
    assert perf["perf_trace"] is True
    assert str(perf["perf_trace_path"]).endswith("perf.jsonl")
