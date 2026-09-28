"""Catalog download and metadata for the published ZFTurbo vocals v1 MLX model."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests

MODEL_SOURCE = "huggingface_mel_roformer"
CHECKPOINT_FAMILY = "zfturbo_vocals_v1"
SOURCE_CHECKPOINT = "model_vocals_mel_band_roformer_sdr_8.42.ckpt"


def load_model_data(model_dir: str | Path) -> dict:
    """Adapt the release config to the separator's stem and inference metadata."""
    config_path = Path(model_dir) / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    expected = {
        "checkpoint_family": CHECKPOINT_FAMILY,
        "_source_input": SOURCE_CHECKPOINT,
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
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected ZFTurbo vocals v1 config {key}: {config.get(key)!r}")

    return {
        "backend": MODEL_SOURCE,
        "is_roformer": True,
        "audio": {"sample_rate": config["sample_rate"]},
        "model": {
            "dim": config["dim"],
            "depth": config["depth"],
            "stereo": True,
            "num_stems": config["num_stems"],
            "time_transformer_depth": 1,
            "freq_transformer_depth": 1,
            "linear_transformer_depth": 0,
            "num_bands": config["num_bands"],
            "dim_head": config["dim_head"],
            "heads": config["heads"],
            "mlp_expansion_factor": config["mlp_expansion_factor"],
            # The local MLP counts linear layers; the release config counts hidden layers.
            "mask_estimator_depth": config["mask_estimator_depth"] + 1,
            "stft_n_fft": config["n_fft"],
            "stft_hop_length": config["hop_length"],
            "stft_win_length": config["win_length"],
        },
        "inference": {
            "chunk_size": config["chunk_size"],
            "num_overlap": config["num_overlap"],
        },
        "training": {
            "instruments": ["Vocals", "Instrumental"],
            "target_instrument": "Vocals",
        },
    }


def download_model(entry: dict, model_file_dir: str | Path) -> Path:
    """Download the pinned Hub files into the normal separator model directory."""
    model_dir = Path(model_file_dir) / entry["filename"]
    model_dir.mkdir(parents=True, exist_ok=True)
    weight_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"

    def valid_weights() -> bool:
        return weight_path.is_file() and weight_path.stat().st_size == entry["weight_size_bytes"]

    def valid_config() -> bool:
        if not config_path.is_file():
            return False
        try:
            load_model_data(model_dir)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return True

    for filename, valid in (("model.safetensors", valid_weights), ("config.json", valid_config)):
        if valid():
            continue
        destination = model_dir / filename
        url = f"https://huggingface.co/{entry['repo_id']}/resolve/{entry['revision']}/{filename}"
        temporary = None
        try:
            with NamedTemporaryFile(dir=model_dir, prefix=f".{filename}.", delete=False) as handle:
                temporary = Path(handle.name)
                with requests.get(url, stream=True, timeout=(10, 300)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if not valid():
            raise RuntimeError(f"Downloaded ZFTurbo model file is incomplete or invalid: {destination}")

    return model_dir


class _LengthPreservingModel:
    """Match the release's iSTFT behavior by zero-padding its short tail."""

    def __init__(self, model):
        self.model = model

    def __call__(self, audio):
        import mlx.core as mx

        output = self.model(audio)
        missing = audio.shape[-1] - output.shape[-1]
        if missing > 0:
            output = mx.pad(output, [(0, 0), (0, 0), (0, missing)])
        return output[..., : audio.shape[-1]]


def load_local_model(model_dir: str | Path, model_data: dict):
    """Load the published weights into this project's MelBand-RoFormer port."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from mlx_audio_separator.separator.models.roformer.loader import (
        convert_torch_to_mlx_weights,
        create_mel_band_roformer_mlx,
    )

    model = create_mel_band_roformer_mlx(model_data)
    # This release has no final normalization after the transformer stack.
    model.final_norm = nn.Identity()
    source_weights = mx.load(str(Path(model_dir) / "model.safetensors"))
    weights = convert_torch_to_mlx_weights(source_weights)

    # The local attention module packs Q, K and V into one projection.
    for key in list(weights):
        if key.endswith("attn.to_q.weight"):
            prefix = key[: -len("to_q.weight")]
            weights[prefix + "to_qkv.weight"] = mx.concatenate(
                [weights.pop(prefix + name + ".weight") for name in ("to_q", "to_k", "to_v")],
                axis=0,
            )

    expected = dict(tree_flatten(model.parameters()))
    generated_buffers = {"freq_indices", "num_freqs_per_band", "num_bands_per_freq"}
    missing = set(expected) - set(weights) - generated_buffers
    extra = set(weights) - set(expected)
    wrong_shape = [key for key in expected.keys() & weights.keys() if expected[key].shape != weights[key].shape]
    if missing or extra or wrong_shape:
        raise ValueError(
            f"ZFTurbo weights do not match the local RoFormer: "
            f"missing={sorted(missing)}, extra={sorted(extra)}, wrong_shape={sorted(wrong_shape)}"
        )

    model.load_weights(list(weights.items()), strict=False)
    model.eval()
    return _LengthPreservingModel(model)
