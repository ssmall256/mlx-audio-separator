"""Tests for benchmark reliability and recovery behavior."""

import json
from collections import defaultdict
from pathlib import Path

from mlx_audio_separator.utils import benchmark


class StubSeparator:
    """Configurable benchmark stub for Separator."""

    model_list = {}
    load_plan = {}
    separate_plan = {}
    load_attempts = defaultdict(int)
    separate_attempts = defaultdict(int)
    load_invocations = []

    def __init__(self, info_only=False, **kwargs):
        self.info_only = info_only
        self.kwargs = kwargs
        self.current_model = None
        self.last_perf_metrics = {}
        self.strict_errors = False

    @classmethod
    def reset(cls):
        cls.model_list = {}
        cls.load_plan = {}
        cls.separate_plan = {}
        cls.load_attempts = defaultdict(int)
        cls.separate_attempts = defaultdict(int)
        cls.load_invocations = []

    def get_simplified_model_list(self, filter_sort_by=None):
        return dict(self.model_list)

    def load_model(self, model_filename):
        self.current_model = model_filename
        self.load_invocations.append(model_filename)
        idx = self.load_attempts[model_filename]
        self.load_attempts[model_filename] += 1
        actions = self.load_plan.get(model_filename, ["ok"])
        action = actions[min(idx, len(actions) - 1)]
        if isinstance(action, Exception):
            raise action

    def _set_strict_separation_errors(self, enabled):
        self.strict_errors = bool(enabled)

    def separate(self, audio_file):
        idx = self.separate_attempts[self.current_model]
        self.separate_attempts[self.current_model] += 1
        plan = self.separate_plan.get(self.current_model, [])
        if callable(plan):
            return plan(self, idx)
        if isinstance(plan, list):
            if not plan:
                return []
            action = plan[min(idx, len(plan) - 1)]
            if isinstance(action, Exception):
                raise action
            return action
        return []


def _write_audio(path: Path):
    path.write_bytes(b"RIFF")


def _read_results(path: Path):
    with path.open() as f:
        return json.load(f)


def _create_nonempty_output(instance: StubSeparator, repeat_idx: int):
    out = Path(instance.kwargs["output_dir"]) / f"{instance.current_model}-{repeat_idx}.wav"
    out.write_bytes(b"stem")
    return [str(out)]


def _run_benchmark(tmp_path, monkeypatch, model_file_dir=None):
    monkeypatch.setattr("mlx_audio_separator.core.Separator", StubSeparator)
    monkeypatch.setattr("mlx_audio_separator.utils.benchmark.clear_mlx_cache", lambda: None)
    audio = tmp_path / "input.wav"
    _write_audio(audio)
    benchmark.run_benchmark(
        audio_file=str(audio),
        output_dir=str(tmp_path),
        model_file_dir=str(model_file_dir or tmp_path),
        cooldown=0,
        wait_nominal=False,
        skip_download=False,
        resume=False,
        repeats=1,
        warmup=0,
        profile=False,
    )
    return _read_results(tmp_path / "benchmark_results.json")


def test_stems_zero_marks_failure(tmp_path, monkeypatch):
    StubSeparator.reset()
    StubSeparator.model_list = {"m.onnx": {"Name": "MDX model", "Type": "MDX"}}
    StubSeparator.separate_plan = {"m.onnx": [[]]}

    data = _run_benchmark(tmp_path, monkeypatch)
    result = data["results"][0]
    assert result["status"].startswith("error:")
    assert result["validation"]["stems_nonzero"] is False


def test_missing_output_file_marks_failure(tmp_path, monkeypatch):
    StubSeparator.reset()
    StubSeparator.model_list = {"m.onnx": {"Name": "MDX model", "Type": "MDX"}}

    def missing_file_output(instance, repeat_idx):
        out = Path(instance.kwargs["output_dir"]) / "missing.wav"
        return [str(out)]

    StubSeparator.separate_plan = {"m.onnx": missing_file_output}

    data = _run_benchmark(tmp_path, monkeypatch)
    result = data["results"][0]
    assert result["status"].startswith("error:")
    assert result["validation"]["outputs_exist"] is False
    assert len(result["validation"]["missing_files"]) == 1


def test_empty_output_file_marks_failure(tmp_path, monkeypatch):
    StubSeparator.reset()
    StubSeparator.model_list = {"m.onnx": {"Name": "MDX model", "Type": "MDX"}}

    def empty_file_output(instance, repeat_idx):
        out = Path(instance.kwargs["output_dir"]) / f"{repeat_idx}.wav"
        out.write_bytes(b"")
        return [str(out)]

    StubSeparator.separate_plan = {"m.onnx": empty_file_output}

    data = _run_benchmark(tmp_path, monkeypatch)
    result = data["results"][0]
    assert result["status"].startswith("error:")
    assert result["validation"]["outputs_nonempty"] is False
    assert len(result["validation"]["empty_files"]) == 1


def test_status_ok_requires_validation_success(tmp_path, monkeypatch):
    StubSeparator.reset()
    StubSeparator.model_list = {"m.onnx": {"Name": "MDX model", "Type": "MDX"}}
    StubSeparator.separate_plan = {"m.onnx": _create_nonempty_output}

    data = _run_benchmark(tmp_path, monkeypatch)
    result = data["results"][0]
    assert result["status"] == "ok"
    assert result["validation"]["stems_nonzero"] is True
    assert result["validation"]["outputs_exist"] is True
    assert result["validation"]["outputs_nonempty"] is True


def test_zip_load_error_retries_once(tmp_path, monkeypatch):
    StubSeparator.reset()
    model_filename = "mel_band_roformer_instrumental_instv7n_gabox.ckpt"
    StubSeparator.model_list = {model_filename: {"Name": "Roformer", "Type": "MDXC"}}
    StubSeparator.load_plan = {
        model_filename: [
            RuntimeError("PytorchStreamReader failed reading zip archive: failed finding central directory"),
            "ok",
        ]
    }
    StubSeparator.separate_plan = {model_filename: _create_nonempty_output}

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / model_filename).write_bytes(b"corrupt")
    (model_dir / "mel_band_roformer_instrumental_instv7n_gabox.safetensors").write_bytes(b"cache")

    data = _run_benchmark(tmp_path, monkeypatch, model_file_dir=model_dir)
    result = data["results"][0]
    assert StubSeparator.load_attempts[model_filename] == 2
    assert result["status"] == "ok"
    assert result["retry"]["attempted"] is True
    assert result["retry"]["deleted_model_file"] is True
    assert not (model_dir / model_filename).exists()
    assert not (model_dir / "mel_band_roformer_instrumental_instv7n_gabox.safetensors").exists()


def test_demucs_preflight_skips_demucs_and_continues(tmp_path, monkeypatch):
    StubSeparator.reset()
    StubSeparator.model_list = {
        "htdemucs.yaml": {"Name": "Demucs", "Type": "Demucs"},
        "m.onnx": {"Name": "MDX model", "Type": "MDX"},
    }
    StubSeparator.separate_plan = {"m.onnx": _create_nonempty_output}
    monkeypatch.setattr(
        "mlx_audio_separator.utils.benchmark._demucs_conversion_dependency_available",
        lambda: False,
    )

    data = _run_benchmark(tmp_path, monkeypatch)
    results = {row["filename"]: row for row in data["results"]}
    assert results["htdemucs.yaml"]["status"].startswith("skipped:")
    assert results["m.onnx"]["status"] == "ok"
    assert "htdemucs.yaml" not in StubSeparator.load_invocations


def test_strict_benchmark_records_inner_exception_details(tmp_path, monkeypatch):
    StubSeparator.reset()
    StubSeparator.model_list = {"m.onnx": {"Name": "MDX model", "Type": "MDX"}}
    StubSeparator.separate_plan = {"m.onnx": [RuntimeError("boom from inner separation")]}

    data = _run_benchmark(tmp_path, monkeypatch)
    result = data["results"][0]
    assert result["status"].startswith("error:")
    assert result["diagnostic"]["exception_type"] == "RuntimeError"
    assert "boom from inner separation" in result["diagnostic"]["message"]


def test_karaoke_models_missing_vocals_regression(tmp_path, monkeypatch):
    StubSeparator.reset()
    karaoke_models = {
        "mel_band_roformer_karaoke_gabox_v2.ckpt": {
            "Name": "Roformer Model: MelBand Roformer | Karaoke V2 by Gabox",
            "Type": "MDXC",
        },
        "mel_band_roformer_karaoke_becruily.ckpt": {
            "Name": "Roformer Model: MelBand Roformer | Karaoke by becruily",
            "Type": "MDXC",
        },
    }
    StubSeparator.model_list = karaoke_models

    def write_karaoke_outputs(instance, repeat_idx):
        model_stem = Path(instance.current_model).stem
        base = Path(instance.kwargs["output_dir"])
        if "becruily" in model_stem:
            outputs = [
                base / f"f8_(Vocals)_{model_stem}.wav",
                base / f"f8_(Instrumental)_{model_stem}.wav",
            ]
        else:
            outputs = [base / f"f8_(Vocals)_{model_stem}.wav"]
        for out in outputs:
            out.write_bytes(b"stem")
        return [str(out) for out in outputs]

    StubSeparator.separate_plan = {
        "mel_band_roformer_karaoke_gabox_v2.ckpt": write_karaoke_outputs,
        "mel_band_roformer_karaoke_becruily.ckpt": write_karaoke_outputs,
    }

    data = _run_benchmark(tmp_path, monkeypatch)
    results = {row["filename"]: row for row in data["results"]}
    assert results["mel_band_roformer_karaoke_gabox_v2.ckpt"]["status"] == "ok"
    assert results["mel_band_roformer_karaoke_becruily.ckpt"]["status"] == "ok"
