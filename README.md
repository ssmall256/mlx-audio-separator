# mlx-audio-separator

An MLX-native port of [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) for Apple Silicon Macs — no PyTorch or ONNX runtime required at inference time.

## Background

[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) by [beveradb](https://github.com/beveradb) (and the [nomadkaraoke](https://github.com/nomadkaraoke) community) is the standard Python package for audio stem separation, built on the incredible model zoo from [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) by [@Anjok07](https://github.com/Anjok07) and [@aufr33](https://github.com/aufr33). It supports a wide range of architectures — MDX-Net, VR, Demucs, MDXC, Roformer — and handles everything from vocal/instrumental splits to multi-stem extraction (drums, bass, piano, guitar, and more).

**mlx-audio-separator** re-implements the inference paths of that project using Apple's [MLX](https://github.com/ml-explore/mlx) framework, so that separation runs natively on the Apple Silicon GPU/Neural Engine with no dependency on PyTorch or ONNX Runtime. The model weights are converted once from the original upstream checkpoints; after that, inference is pure MLX.

This project owes its existence to the upstream work. If you find it useful, please also star and support [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) and [UVR](https://github.com/Anjok07/ultimatevocalremovergui).

## Supported Architectures

| Architecture | Notes |
|---|---|
| Roformer | BS-Roformer and MelBand-Roformer families |
| MDXC | MDX23C-style checkpoints |
| MDX | ConvTDFNet-style checkpoints |
| VR | UVR VR-family checkpoints |
| Demucs | Hybrid transformer Demucs variants |

Use `mlx-audio-separator --list_models` for the full model catalog.

## Requirements

- macOS 13+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.10+

## Installation

```bash
pip install mlx-audio-separator
```

If you need first-run conversion from upstream checkpoints (`.ckpt`/`.onnx`/Demucs weights), install conversion extras (`torch`, `onnx`, `demucs`):

```bash
pip install "mlx-audio-separator[convert]"
```

## Quick Start

### CLI

```bash
# Separate with the default model
mlx-audio-separator song.mp3

# Use a specific model
mlx-audio-separator song.mp3 -m htdemucs_ft.yaml

# List available models
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

## Performance Controls (Opt-In)

All defaults are behavior-compatible with the standard separation path. Speed and latency controls are strictly opt-in:

```bash
mlx-audio-separator song.mp3 \
  --speed_mode latency_safe \
  --cache_clear_policy deferred \
  --write_workers 2 \
  --perf_trace \
  --perf_trace_path ./perf_trace.jsonl
```

### Available Flags

| Flag | Values / Type |
|---|---|
| `--speed_mode` | `default`, `latency_safe`, `latency_safe_v2` |
| `--auto_tune_batch` | *(flag)* |
| `--tune_probe_seconds` | seconds |
| `--cache_clear_policy` | `aggressive`, `deferred` |
| `--write_workers` | int |
| `--experimental_vectorized_chunking` | *(flag, MDXC only)* |
| `--experimental_compile_model_forward` | *(flag, MDX23C only)* |
| `--experimental_compile_shapeless` | *(flag, accepted but currently ignored)* |
| `--experimental_roformer_static_compiled_demix` | *(flag, accepted but currently ignored)* |
| `--perf_trace` | *(flag)* |
| `--perf_trace_path` | path |

### Latency Preset Batch Sizes

| Architecture | `latency_safe` | `latency_safe_v2` |
|---|---|---|
| Demucs | 8 | 12 |
| MDXC | 1 | 1 |
| MDX | 1 | 1 |
| VR | 1 | 2 |

### Determinism Notes

Deterministic equivalence checks default to strict pass/fail gating for MDXC, MDX, and VR. Demucs strict gating is opt-in; deterministic Demucs mode retains fused GroupNorm+GELU while enforcing deterministic evaluation controls.

## Benchmarking

Single-file benchmark with warmup, repeats, and optional phase profiling:

```bash
mlx-audio-separator \
  --benchmark song.mp3 \
  --benchmark_warmup 1 \
  --benchmark_repeats 3 \
  --benchmark_profile
```

### Cross-Runtime Comparison

Compare MLX latency against python-audio-separator (ABBA pattern) on overlapping models:

```bash
PATH="/usr/local/bin:/opt/homebrew/bin:$PATH" \
uv run --with audio-separator --with onnxruntime python scripts/perf/mlx_vs_pas_abba.py \
  --corpus-file /path/to/corpus.txt \
  --models htdemucs_ft.yaml,model_bs_roformer_ep_317_sdr_12.9755.ckpt,UVR-MDX-NET-Inst_HQ_3.onnx \
  --model-file-dir /tmp/audio-separator-models \
  --mlx-config '{"output_format":"WAV","performance_params":{"speed_mode":"latency_safe","cache_clear_policy":"deferred"}}' \
  --pas-config '{"output_format":"WAV"}'
```

### Parity Check

Verify output equivalence between MLX and python-audio-separator:

```bash
PATH="/usr/local/bin:/opt/homebrew/bin:$PATH" \
uv run --with audio-separator --with onnxruntime python scripts/perf/mlx_vs_pas_parity.py \
  --corpus-file /path/to/corpus.txt \
  --models htdemucs_ft.yaml,model_bs_roformer_ep_317_sdr_12.9755.ckpt,mel_band_roformer_instrumental_instv7n_gabox.ckpt,UVR-MDX-NET-Inst_HQ_3.onnx \
  --model-file-dir /tmp/audio-separator-models \
  --mlx-config '{"output_format":"WAV","performance_params":{"speed_mode":"latency_safe","cache_clear_policy":"deferred"}}' \
  --pas-config '{"output_format":"WAV"}' \
  --threshold-rel-l2 5e-2 \
  --fail-fast
```

**Parity thresholds:**

- MLX internal (same backend, same config): `1e-5`
- MLX vs python-audio-separator (cross-runtime): `5e-2`

### Unified Optimization Report

Generate a combined latency + parity + quality report:

```bash
uv run python scripts/perf/run_optimization_report.py \
  --corpus-file /path/to/corpus.txt \
  --baseline-config /path/to/baseline.json \
  --candidate-config /path/to/candidate.json \
  --models htdemucs.yaml,model_bs_roformer_ep_317_sdr_12.9755.ckpt,UVR-MDX-NET-Inst_HQ_3.onnx \
  --output-json /path/to/optimization_report.json \
  --output-markdown /path/to/optimization_report.md
```

Optional report flags: `--quality-reference-manifest`, `--parity-strict-demucs`, `--python-mps-latency`, `--python-mps-parity`.

Reference manifest scaffold helper:

```bash
python scripts/perf/generate_reference_manifest.py \
  --corpus-file /path/to/corpus.txt \
  --output-json /path/to/reference_manifest.json \
  --search-dir /path/to/reference_stems
```

## Reproducibility

For publication and release-quality reproducibility guidance, see [`docs/reproducibility.md`](docs/reproducibility.md).

## Documentation

| Document | Description |
|---|---|
| [`docs/reproducibility.md`](docs/reproducibility.md) | Reproducibility and release evidence checklist |
| [`docs/release-first.md`](docs/release-first.md) | Release-first execution playbook |
| [`docs/wave4-opt-in.md`](docs/wave4-opt-in.md) | Wave 4 opt-in performance roadmap |
| [`docs/release-notes-0.1.1.md`](docs/release-notes-0.1.1.md) | Release notes |
| [`CHANGELOG.md`](CHANGELOG.md) | Changelog |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | Third-party attribution and license notices |

## License

This project is licensed under the MIT License.

## Acknowledgments

This project is a port of [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) (MIT) by [beveradb](https://github.com/beveradb) and contributors. Substantial portions of the architecture, model loading, and separation logic are adapted from that project. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for full attribution and license details.

The models used by this project were trained by the [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) community, primarily [@Anjok07](https://github.com/Anjok07) and [@aufr33](https://github.com/aufr33).

Additional references:

- [BS-Roformer](https://arxiv.org/abs/2309.02612)
- [Demucs (Meta Research)](https://github.com/facebookresearch/demucs)
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
