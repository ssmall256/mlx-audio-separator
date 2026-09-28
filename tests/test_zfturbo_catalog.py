"""Catalog and download coverage for ZFTurbo vocals v1."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_audio_separator.core import Separator
from mlx_audio_separator.hf_mel_roformer import download_model, load_model_data

MODEL_ID = "mel-roformer-zfturbo-vocals-v1-mlx"


def _hub_config():
    return {
        "checkpoint_family": "zfturbo_vocals_v1",
        "_source_input": "model_vocals_mel_band_roformer_sdr_8.42.ckpt",
        "model_type": "mel_band_roformer",
        "mask_estimator_depth": 1,
        "num_stems": 1,
        "sample_rate": 44100,
        "chunk_size": 352800,
        "num_overlap": 2,
        "dim": 192,
        "depth": 8,
        "heads": 8,
        "dim_head": 64,
        "num_bands": 60,
        "mlp_expansion_factor": 4,
        "n_fft": 2048,
        "hop_length": 512,
        "win_length": 2048,
    }


def _catalog_entry():
    catalog_path = Path(__file__).resolve().parents[1] / "mlx_audio_separator" / "models.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    return catalog["mlx_hf_roformer_download_list"]["ZFTurbo Mel-Band-RoFormer Vocals v1"]


def _separator(tmp_path, monkeypatch):
    separator = Separator(info_only=True, model_file_dir=str(tmp_path / "models"))

    def provide_uvr_list(_url, path):
        Path(path).write_text(
            json.dumps({
                "demucs_download_list": {}, "vr_download_list": {},
                "mdx_download_list": {}, "mdx23c_download_list": {},
                "roformer_download_list": {},
            }),
            encoding="utf-8",
        )

    monkeypatch.setattr(separator, "download_file_if_not_exists", provide_uvr_list)
    return separator


def test_catalog_lists_zfturbo_as_mdxc_vocals_without_invented_score(tmp_path, monkeypatch):
    separator = _separator(tmp_path, monkeypatch)
    listed = separator.list_supported_model_files()["MDXC"]["ZFTurbo Mel-Band-RoFormer Vocals v1"]

    assert listed["filename"] == MODEL_ID
    assert listed["source_checkpoint"] == "model_vocals_mel_band_roformer_sdr_8.42.ckpt"
    assert listed["target_stem"] == "Vocals"
    assert listed["stems"] == ["Vocals", "Instrumental"]
    assert listed["scores"] == {}
    assert separator.get_simplified_model_list(filter_sort_by="vocals")[MODEL_ID]["Stems"] == [
        "Vocals*", "Instrumental",
    ]


def test_download_reuses_complete_files_and_retries_truncated_weights(tmp_path, monkeypatch):
    entry = _catalog_entry()
    entry["weight_size_bytes"] = 12
    calls = []

    class FakeResponse:
        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            filename = self.url.rsplit("/", 1)[-1]
            yield b"weights-data" if filename == "model.safetensors" else json.dumps(_hub_config()).encode()

    def fake_get(url, *, stream, timeout):
        calls.append(url)
        return FakeResponse(url)

    monkeypatch.setattr("mlx_audio_separator.hf_mel_roformer.requests.get", fake_get)

    model_dir = download_model(entry, tmp_path)
    assert model_dir == tmp_path / MODEL_ID
    assert [call.rsplit("/", 1)[-1] for call in calls] == ["model.safetensors", "config.json"]
    assert all(f"/{entry['revision']}/" in call for call in calls)

    download_model(entry, tmp_path)
    assert len(calls) == 2

    (model_dir / "model.safetensors").write_bytes(b"partial")
    download_model(entry, tmp_path)
    assert calls[-1].endswith("/model.safetensors")


def test_config_adapter_preserves_v1_architecture_and_stems(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_hub_config()), encoding="utf-8")
    metadata = load_model_data(tmp_path)
    assert metadata["training"]["target_instrument"] == "Vocals"
    assert metadata["inference"] == {"chunk_size": 352800, "num_overlap": 2}
    assert metadata["model"]["mask_estimator_depth"] == 2

    wrong = _hub_config()
    wrong["mask_estimator_depth"] = 2
    (tmp_path / "config.json").write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="mask_estimator_depth"):
        load_model_data(tmp_path)


def test_download_model_and_data_and_loader_use_local_hub_directory(tmp_path, monkeypatch):
    from mlx_audio_separator.separator.architectures.mdxc_separator import MDXCSeparator

    model_dir = tmp_path / "models" / MODEL_ID
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"weights-data")
    (model_dir / "config.json").write_text(json.dumps(_hub_config()), encoding="utf-8")
    separator = _separator(tmp_path, monkeypatch)
    monkeypatch.setattr("mlx_audio_separator.core.hf_mel_roformer.download_model", lambda *_: model_dir)

    separator.download_model_and_data(MODEL_ID)

    loaded = SimpleNamespace()
    seen = []
    monkeypatch.setattr(
        "mlx_audio_separator.hf_mel_roformer.load_local_model",
        lambda path, data: seen.append((path, data)) or loaded,
    )
    stub = SimpleNamespace(model_data=load_model_data(model_dir), model_path=str(model_dir), logger=logging.getLogger(__name__))
    MDXCSeparator._load_model(stub)
    assert stub.model_run is loaded
    assert stub.model_type == "mel_band_roformer"
    assert seen == [(str(model_dir), stub.model_data)]


def test_model_sample_rate_is_restored_when_switching_away(tmp_path, monkeypatch):
    model_dir = tmp_path / MODEL_ID
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(_hub_config()), encoding="utf-8")

    class FakeArchModule:
        class MDXCSeparator:
            def __init__(self, common_config, arch_config):
                self.sample_rate = common_config["sample_rate"]

    separator = Separator(info_only=True, model_file_dir=str(tmp_path), sample_rate=48000)
    monkeypatch.setattr(
        separator,
        "download_model_files",
        lambda name: (name, "MDXC", name, str(model_dir if name == MODEL_ID else tmp_path / "other.yaml"), None),
    )
    monkeypatch.setattr(separator, "load_model_data_from_yaml", lambda *_: {"training": {"instruments": ["Vocals"]}})
    monkeypatch.setattr("mlx_audio_separator.core.importlib.import_module", lambda *_: FakeArchModule)

    separator.load_model(MODEL_ID)
    assert separator.sample_rate == separator.model_instance.sample_rate == 44100

    separator.load_model("other.yaml")
    assert separator.sample_rate == separator.model_instance.sample_rate == 48000
