"""Tests for Roformer loader safetensors conversion and cache paths."""

import numpy as np


class _DummyModel:
    def __init__(self):
        self.loaded_weights = None

    def load_weights(self, weights, strict=False):
        self.loaded_weights = list(weights)


def test_load_roformer_model_saves_safetensors_when_enabled(monkeypatch, tmp_path):
    from mlx_audio_separator.separator.models.roformer import loader as rof_loader

    model_path = tmp_path / "BS-Roformer-SW.ckpt"
    model_path.write_bytes(b"ckpt")
    config = {"model": {"dim": 16, "depth": 1, "freqs_per_bands": [1025]}}

    saved = {}
    model = _DummyModel()

    monkeypatch.setenv("MLX_SAVE_SAFETENSORS", "1")
    monkeypatch.setattr(rof_loader, "detect_model_type", lambda *_: "bs_roformer")
    monkeypatch.setattr(rof_loader, "_find_safetensors", lambda *_: None)
    monkeypatch.setattr(rof_loader, "_load_state_dict", lambda *_: {"layer.weight": np.zeros((2, 2), dtype=np.float32)})
    monkeypatch.setattr(
        rof_loader,
        "convert_torch_to_mlx_weights",
        lambda *_: {"layer.weight": np.zeros((2, 2), dtype=np.float32)},
    )
    monkeypatch.setattr(rof_loader, "create_bs_roformer_mlx", lambda *_: model)
    monkeypatch.setattr(
        rof_loader.mx,
        "save_safetensors",
        lambda path, data: saved.update({"path": path, "keys": sorted(data.keys())}),
    )

    loaded_model, model_type = rof_loader.load_roformer_model(str(model_path), config)

    assert loaded_model is model
    assert model_type == "bs_roformer"
    assert saved["path"] == str(model_path.with_suffix(".safetensors"))
    assert saved["keys"] == ["layer.weight"]


def test_load_roformer_model_prefers_existing_safetensors(monkeypatch, tmp_path):
    from mlx_audio_separator.separator.models.roformer import loader as rof_loader

    model_path = tmp_path / "BS-Roformer-SW.ckpt"
    model_path.write_bytes(b"ckpt")
    st_path = tmp_path / "BS-Roformer-SW.safetensors"
    st_path.write_bytes(b"st")
    config = {"model": {"dim": 16, "depth": 1, "freqs_per_bands": [1025]}}

    model = _DummyModel()

    monkeypatch.setattr(rof_loader, "detect_model_type", lambda *_: "bs_roformer")
    monkeypatch.setattr(rof_loader, "_find_safetensors", lambda *_: str(st_path))
    monkeypatch.setattr(rof_loader, "_load_state_dict", lambda *_: (_ for _ in ()).throw(AssertionError("unexpected ckpt load")))
    monkeypatch.setattr(rof_loader.mx, "load", lambda *_: {"layer.weight": np.zeros((2, 2), dtype=np.float32)})
    monkeypatch.setattr(rof_loader, "create_bs_roformer_mlx", lambda *_: model)

    loaded_model, model_type = rof_loader.load_roformer_model(str(model_path), config)

    assert loaded_model is model
    assert model_type == "bs_roformer"
    assert model.loaded_weights is not None
    assert len(model.loaded_weights) == 1
