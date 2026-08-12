"""Security regression tests for Demucs ingestion and MLX caches."""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_audio_separator.demucs_mlx import mlx_convert, secure_demucs


class _ExecutablePayload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return os.system, (f"touch {self.marker}",)


def _cache_config(
    model_name: str = "htdemucs",
    *,
    model_class: str = "HTDemucsMLX",
    sub_model_class=None,
    num_models: int = 1,
    weights=None,
):
    return {
        "format_version": mlx_convert.SAFE_CACHE_FORMAT_VERSION,
        "model_name": model_name,
        "model_class": model_class,
        "sub_model_class": sub_model_class,
        "args": [],
        "kwargs": {"segment": Fraction(39, 5)},
        "mlx_version": "0.31.0",
        "num_models": num_models,
        "weights": weights,
        "conversion_date": datetime.now().isoformat(),
        "torch_signatures": mlx_convert.MLX_MODEL_REGISTRY[model_name]["signatures"],
        "safetensors_sha256": "",
        "verification_passed": False,
    }


def _mock_url_load(monkeypatch, torch, checkpoint):
    captured = {}

    def load_from_url(url, **kwargs):
        captured.update(url=url, **kwargs)
        load_kwargs = {
            "map_location": kwargs["map_location"],
            "weights_only": kwargs["weights_only"],
        }
        return torch.load(checkpoint, **load_kwargs)

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", load_from_url)
    return captured


def test_restricted_demucs_loader_rejects_vulnerable_torch_before_download():
    download_called = False

    def download(*args, **kwargs):
        nonlocal download_called
        download_called = True

    torch = SimpleNamespace(
        __version__="2.5.1+cpu",
        hub=SimpleNamespace(load_state_dict_from_url=download),
    )

    with pytest.raises(RuntimeError, match="PyTorch 2.6 or newer"):
        secure_demucs._load_package_from_url("https://example.invalid/model.th", torch)

    assert not download_called


def test_restricted_demucs_loader_does_not_execute_pickle_payload(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "malicious.th"
    marker = tmp_path / "executed"
    torch.save({"payload": _ExecutablePayload(marker)}, checkpoint)
    captured = _mock_url_load(monkeypatch, torch, checkpoint)

    with pytest.raises(RuntimeError, match="Refusing to load Demucs checkpoint"):
        secure_demucs._load_package_from_url("https://example.invalid/model.th", torch)

    assert not marker.exists()
    assert captured == {
        "url": "https://example.invalid/model.th",
        "map_location": "cpu",
        "check_hash": True,
        "weights_only": True,
    }


def test_restricted_demucs_loader_accepts_known_package(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    demucs_module = pytest.importorskip("demucs.demucs")
    model = demucs_module.Demucs(
        sources=["left", "right"],
        audio_channels=2,
        channels=4,
        depth=2,
        dconv_mode=0,
    )
    args, kwargs = model._init_args_kwargs
    checkpoint = tmp_path / "demucs.th"
    torch.save(
        {
            "klass": demucs_module.Demucs,
            "args": args,
            "kwargs": kwargs,
            "state": model.state_dict(),
        },
        checkpoint,
    )
    _mock_url_load(monkeypatch, torch, checkpoint)

    package = secure_demucs._load_package_from_url(
        "https://example.invalid/model.th", torch
    )

    assert package["klass"] is demucs_module.Demucs
    assert package["state"].keys() == model.state_dict().keys()


def test_restricted_demucs_loader_rejects_malformed_package(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "malformed.th"
    torch.save({"klass": "not-a-class", "args": [], "kwargs": {}, "state": {}}, checkpoint)
    _mock_url_load(monkeypatch, torch, checkpoint)

    with pytest.raises(ValueError, match="Unsupported Demucs model class"):
        secure_demucs._load_package_from_url("https://example.invalid/model.th", torch)


def test_legacy_pickle_is_never_deserialized(tmp_path):
    marker = tmp_path / "executed"
    legacy = tmp_path / "htdemucs_mlx.pkl"
    with legacy.open("wb") as handle:
        pickle.dump(_ExecutablePayload(marker), handle)

    with pytest.warns(FutureWarning, match="Ignoring unsafe legacy Demucs cache"):
        with pytest.raises(
            FileNotFoundError,
            match="legacy cache.*ignored.*python -m.*mlx_convert htdemucs",
        ):
            mlx_convert.load_mlx_model(
                "htdemucs", cache_dir=str(tmp_path), auto_convert=False
            )

    assert not marker.exists()
    assert legacy.exists()


def test_legacy_pickle_regenerates_without_reading_or_deleting(monkeypatch, tmp_path):
    legacy = tmp_path / "htdemucs_mlx.pkl"
    legacy.write_bytes(b"not a pickle")
    expected = object()
    conversions = []

    def convert(model_name, output_dir, **kwargs):
        conversions.append((model_name, output_dir, kwargs))
        (tmp_path / "htdemucs.safetensors").write_bytes(b"safe")
        (tmp_path / "htdemucs_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(mlx_convert, "convert_htdemucs_weights", convert)
    monkeypatch.setattr(
        mlx_convert,
        "load_mlx_model_from_safetensors",
        lambda *args, **kwargs: expected,
    )

    with pytest.warns(FutureWarning, match="Ignoring unsafe legacy Demucs cache"):
        loaded = mlx_convert.load_mlx_model("htdemucs", cache_dir=str(tmp_path))

    assert loaded is expected
    assert legacy.read_bytes() == b"not a pickle"
    assert conversions == [
        (
            "htdemucs",
            str(tmp_path),
            {"verify": False, "verbose": False},
        )
    ]


def test_invalid_safe_cache_never_downgrades_to_pickle(tmp_path):
    marker = tmp_path / "executed"
    with (tmp_path / "htdemucs_mlx.pkl").open("wb") as handle:
        pickle.dump(_ExecutablePayload(marker), handle)
    (tmp_path / "htdemucs.safetensors").write_bytes(b"invalid")
    (tmp_path / "htdemucs_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(mlx_convert.SafeCacheError, match="missing fields"):
        mlx_convert.load_mlx_model("htdemucs", cache_dir=str(tmp_path))

    assert not marker.exists()


def test_incomplete_safe_cache_fails_closed(tmp_path):
    (tmp_path / "htdemucs.safetensors").write_bytes(b"incomplete")

    with pytest.raises(mlx_convert.SafeCacheError, match="Incomplete Demucs safe cache"):
        mlx_convert.load_mlx_model("htdemucs", cache_dir=str(tmp_path))


def test_safe_cache_round_trips_fraction_and_digest(tmp_path):
    config = _cache_config()
    weights = {"encoder.weight": mx.array([[1.0, 2.0]])}

    output = mlx_convert._save_safe_cache("htdemucs", str(tmp_path), weights, config)
    loaded = mlx_convert._load_safe_cache_config(
        tmp_path / "htdemucs_config.json", "htdemucs"
    )

    assert output == str(tmp_path / "htdemucs.safetensors")
    assert loaded["kwargs"]["segment"] == Fraction(39, 5)
    assert loaded["mlx_version"] == mx.__version__
    assert loaded["safetensors_sha256"] == mlx_convert._sha256_file(Path(output))
    assert set(mx.load(output)) == {"encoder.weight"}
    assert not (tmp_path / "htdemucs_mlx.pkl").exists()


def test_single_model_safe_cache_loads_with_null_sub_model_class(monkeypatch, tmp_path):
    from mlx_audio_separator.demucs_mlx import mlx_htdemucs

    class FakeHTDemucsMLX:
        def __init__(self, *args, **kwargs):
            self.evaluated = False

        def eval(self):
            self.evaluated = True

    config = _cache_config()
    mlx_convert._save_safe_cache(
        "htdemucs",
        str(tmp_path),
        {"weight": mx.array([1.0])},
        config,
    )
    monkeypatch.setattr(mlx_htdemucs, "HTDemucsMLX", FakeHTDemucsMLX)
    monkeypatch.setattr(mlx_convert, "_load_weights_into_model", lambda *args: None)

    model = mlx_convert.load_mlx_model_from_safetensors(
        "htdemucs", cache_dir=str(tmp_path)
    )

    assert isinstance(model, FakeHTDemucsMLX)
    assert model.evaluated


def test_safe_cache_validates_ensemble_metadata(tmp_path):
    config = _cache_config(
        model_name="htdemucs_ft",
        model_class="BagOfModelsMLX",
        sub_model_class="HTDemucsMLX",
        num_models=4,
        weights=[[1.0, 1.0, 1.0, 1.0]] * 4,
    )
    config["per_model_args"] = [[], [], [], []]
    config["per_model_kwargs"] = [{}, {}, {}, {}]

    mlx_convert._save_safe_cache(
        "htdemucs_ft",
        str(tmp_path),
        {"model_0.weight": mx.array([1.0])},
        config,
    )
    loaded = mlx_convert._load_safe_cache_config(
        tmp_path / "htdemucs_ft_config.json", "htdemucs_ft"
    )

    assert loaded["num_models"] == 4
    assert loaded["sub_model_class"] == "HTDemucsMLX"


def test_safe_cache_rejects_unknown_model_class():
    config = _cache_config()
    config["model_class"] = "ArbitraryPythonModel"
    config["safetensors_sha256"] = "0" * 64

    with pytest.raises(mlx_convert.SafeCacheError, match="Unknown MLX model class"):
        mlx_convert._validate_safe_cache_config(
            mlx_convert._encode_json_value(config), "htdemucs"
        )


def test_safe_cache_rejects_digest_mismatch(tmp_path):
    config = _cache_config()
    output = Path(
        mlx_convert._save_safe_cache(
            "htdemucs",
            str(tmp_path),
            {"weight": mx.array([1.0])},
            config,
        )
    )
    output.write_bytes(output.read_bytes() + b"tampered")

    with pytest.raises(mlx_convert.SafeCacheError, match="digest mismatch"):
        mlx_convert.load_mlx_model_from_safetensors(
            "htdemucs", cache_dir=str(tmp_path)
        )


def test_converter_writes_only_safe_cache(monkeypatch, tmp_path):
    pytest.importorskip("demucs.apply")

    class HTDemucs:
        sources = ["drums", "bass", "other", "vocals"]
        segment = Fraction(39, 5)
        _init_args_kwargs = ((), {"segment": segment})

    HTDemucsMLX = type(
        "HTDemucsMLX",
        (),
        {"state_dict": lambda self: {"weight": mx.array([1.0])}},
    )

    monkeypatch.setattr(
        secure_demucs, "get_restricted_demucs_model", lambda model_name: HTDemucs()
    )
    monkeypatch.setattr(
        mlx_convert, "convert_single_model", lambda model, verbose=False: HTDemucsMLX()
    )

    output = mlx_convert.convert_htdemucs_weights(
        "htdemucs", output_dir=str(tmp_path), verbose=False
    )

    assert output == str(tmp_path / "htdemucs.safetensors")
    assert (tmp_path / "htdemucs_config.json").exists()
    assert not (tmp_path / "htdemucs_mlx.pkl").exists()
