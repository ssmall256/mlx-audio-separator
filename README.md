# mlx-audio-separator

MLX-native stem separation for Apple Silicon Macs.

This project ports the inference paths from [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) to MLX so separation runs on Apple Silicon without requiring PyTorch or ONNX Runtime at inference time.
Core runtime components are powered by [mlx-audio-io](https://github.com/ssmall256/mlx-audio-io) (audio I/O) and [mlx-spectro](https://github.com/ssmall256/mlx-spectro) (spectral transforms).

## Requirements

- macOS 13+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.10+

## Installation

```bash
pip install mlx-audio-separator
```

If you need first-run conversion from upstream checkpoints (`.ckpt`/`.onnx`/Demucs weights), install conversion extras:

```bash
pip install "mlx-audio-separator[convert]"
```

## Quick Start

### CLI

```bash
# Separate with default model
mlx-audio-separator song.mp3

# Use a specific model
mlx-audio-separator song.mp3 -m htdemucs_ft.yaml

# List supported models
mlx-audio-separator --list_models
```

### Python

```python
from mlx_audio_separator import Separator

sep = Separator()
sep.load_model()
outputs = sep.separate("song.mp3")
print(outputs)
```

## Supported Architectures

- Roformer (BS-Roformer and MelBand-Roformer families)
- MDXC (including MDX23C-style checkpoints)
- MDX
- VR
- Demucs

## Validation Snapshot

Release validation snapshot (2026-02-24 to 2026-02-26):

| Check | Result |
|---|---|
| Full-catalog benchmark gate | 163/163 models `ok` (0 failures) |
| Unit tests | 167 passed, 1 skipped |
| MLX vs `audio-separator` parity smoke | 4/4 models passed (`rel L2 <= 5e-2`) |

Scope: Apple Silicon (`M4 mini`), MUSDB18-HQ test subset, release gate + parity smoke model set.

Detailed evidence and provenance: [`docs/release-validation.md`](docs/release-validation.md).

## Performance Snapshot

MLX vs `audio-separator` (ABBA, 12-song MUSDB18-HQ test subset, M4 mini):

| Model | MLX speedup vs PAS |
|---|---:|
| `htdemucs_ft.yaml` | 1.40x |
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | 2.16x |
| `mel_band_roformer_instrumental_instv7n_gabox.ckpt` | 2.50x |
| `UVR-MDX-NET-Inst_HQ_3.onnx` | 1.53x |

Median speedup across the 4-model overlap set: **1.847x**.

These numbers are scoped to the benchmark settings above and are not universal guarantees for all machines, models, or audio inputs.

## Stable Runtime Tuning

Release-facing stable controls:

- `--speed_mode {default,latency_safe,latency_safe_v2}`
- `--cache_clear_policy {aggressive,deferred}`
- `--write_workers <int>`

Example:

```bash
mlx-audio-separator song.mp3 \
  --speed_mode latency_safe \
  --cache_clear_policy deferred \
  --write_workers 2
```

Basic benchmark command:

```bash
mlx-audio-separator \
  --benchmark song.mp3 \
  --benchmark_warmup 1 \
  --benchmark_repeats 3 \
  --benchmark_profile
```

## Documentation

| Document | Description |
|---|---|
| [`docs/release-validation.md`](docs/release-validation.md) | Release evidence snapshot |
| [`docs/release-first.md`](docs/release-first.md) | Release execution playbook |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Reproducibility guide |
| [`docs/wave4-opt-in.md`](docs/wave4-opt-in.md) | Wave 4 opt-in/experimental roadmap |
| [`CHANGELOG.md`](CHANGELOG.md) | Changelog |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | Third-party attribution and license notices |

## License

This project is MIT licensed.

## Acknowledgments

`mlx-audio-separator` is derived from [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) (MIT) by [beveradb](https://github.com/beveradb) and the [nomadkaraoke](https://github.com/nomadkaraoke) community. Substantial portions of the architecture, model loading, and separation logic are adapted from that project. If you find this package useful, please also star and support the upstream project.

The models used by this project were trained by the [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) community, primarily [@Anjok07](https://github.com/Anjok07) and [@aufr33](https://github.com/aufr33). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for full attribution and license details.

Additional references:

- [mlx-audio-io](https://github.com/ssmall256/mlx-audio-io)
- [mlx-spectro](https://github.com/ssmall256/mlx-spectro)
- [BS-Roformer](https://arxiv.org/abs/2309.02612)
- [Demucs (Meta Research)](https://github.com/facebookresearch/demucs)
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
