# mlx-audio-separator

MLX-native music stem separation for Apple Silicon (macOS), with no PyTorch or ONNX runtime required at inference time.

## What This Project Is

- Fast local stem separation on Apple Silicon using MLX.
- Multi-architecture support: Roformer, MDXC, MDX, VR, and Demucs.
- Opt-in latency controls, benchmarking, and reproducibility tooling.

## What This Project Is Not

- A model-training framework.
- A cross-platform inference package (this project targets Apple Silicon + MLX).

## Documentation Map

- Reproducibility and release evidence checklist: `docs/reproducibility.md`
- Release-first execution playbook: `docs/release-first.md`
- Wave 4 opt-in performance roadmap: `docs/wave4-opt-in.md`
- Release notes: `docs/release-notes-0.1.1.md`
- Changelog: `CHANGELOG.md`
- Third-party attribution and license notices: `THIRD_PARTY_NOTICES.md`

## Installation

```bash
pip install mlx-audio-separator
```

Optional conversion extras for converting some upstream checkpoints:

```bash
pip install "mlx-audio-separator[convert]"
```

## Quick Start (CLI)

```bash
# Default separation (default model)
mlx-audio-separator song.mp3

# Use a specific model
mlx-audio-separator song.mp3 -m htdemucs_ft.yaml

# List available models
mlx-audio-separator --list_models
```

## Quick Start (Python)

```python
from mlx_audio_separator import Separator

sep = Separator()
sep.load_model()
outputs = sep.separate("song.mp3")
print(outputs)
```

## Supported Architectures

| Architecture | Notes |
|---|---|
| Roformer | BS-Roformer and MelBand-Roformer families |
| MDXC | MDX23C-style checkpoints |
| MDX | ConvTDFNet-style ONNX checkpoints |
| VR | UVR VR-family checkpoints |
| Demucs | Hybrid transformer Demucs variants |

Use `mlx-audio-separator --list_models` for the current catalog.

## Performance Controls (Opt-In)

Defaults remain behavior-compatible. Enable speed controls explicitly:

```bash
mlx-audio-separator song.mp3 \
  --speed_mode latency_safe \
  --cache_clear_policy deferred \
  --write_workers 2 \
  --perf_trace \
  --perf_trace_path ./perf_trace.jsonl
```

Available flags:

- `--speed_mode {default,latency_safe,latency_safe_v2}`
- `--auto_tune_batch`
- `--tune_probe_seconds <seconds>`
- `--cache_clear_policy {aggressive,deferred}`
- `--write_workers <int>`
- `--experimental_vectorized_chunking`
- `--experimental_compile_model_forward`
- `--experimental_compile_shapeless`
- `--experimental_roformer_static_compiled_demix`
- `--perf_trace`
- `--perf_trace_path <path>`

`--experimental_vectorized_chunking` currently enables an MDXC-only experimental chunk scheduler path.
`--experimental_compile_model_forward` currently provides a real acceleration path for MDX23C checkpoints.
Roformer compile paths (shapeless and static compiled demix) are currently disabled by policy; those flags are accepted for compatibility but ignored at runtime.

Current `latency_safe` preset recommendations:

- `Demucs`: `batch_size=8`
- `MDXC`: `batch_size=1`
- `MDX`: `batch_size=1`
- `VR`: `batch_size=1`

Wave 4 opt-in experimental profile `latency_safe_v2`:

- `Demucs`: `batch_size=12`
- `MDXC`: `batch_size=1`
- `MDX`: `batch_size=1`
- `VR`: `batch_size=2`

Deterministic equivalence checks default to strict pass/fail gating for `MDXC`, `MDX`, and `VR`. `Demucs` strict gating remains opt-in; deterministic Demucs mode still keeps fused `GroupNorm+GELU` while enforcing deterministic evaluation controls.

## Benchmarking

Single-file benchmark with warmup/repeats and optional phase profiling:

```bash
mlx-audio-separator \
  --benchmark song.mp3 \
  --benchmark_warmup 1 \
  --benchmark_repeats 3 \
  --benchmark_profile
```

Unified optimization report (latency + parity + quality):

```bash
python scripts/perf/run_optimization_report.py \
  --corpus-file /path/to/corpus.txt \
  --baseline-config /path/to/baseline.json \
  --candidate-config /path/to/candidate.json \
  --models htdemucs.yaml,model_bs_roformer_ep_317_sdr_12.9755.ckpt,UVR-MDX-NET-Inst_HQ_3.onnx \
  --output-json /path/to/optimization_report.json \
  --output-markdown /path/to/optimization_report.md
```

MLX vs python-audio-separator ABBA latency comparison on overlapping models:

```bash
python scripts/perf/mlx_vs_pas_abba.py \
  --corpus-file /path/to/corpus.txt \
  --models htdemucs_ft.yaml,model_bs_roformer_ep_317_sdr_12.9755.ckpt,UVR-MDX-NET-Inst_HQ_3.onnx \
  --model-file-dir /tmp/audio-separator-models \
  --mlx-config '{"output_format":"WAV","performance_params":{"speed_mode":"latency_safe","cache_clear_policy":"deferred"}}' \
  --pas-config '{"output_format":"WAV"}'
```

MLX vs `audio-separator` parity check (fail-fast, one command):

```bash
python scripts/perf/mlx_vs_pas_parity.py \
  --corpus-file /path/to/corpus.txt \
  --models htdemucs_ft.yaml,model_bs_roformer_ep_317_sdr_12.9755.ckpt,mel_band_roformer_instrumental_instv7n_gabox.ckpt,UVR-MDX-NET-Inst_HQ_3.onnx \
  --model-file-dir /tmp/audio-separator-models \
  --mlx-config '{"output_format":"WAV","performance_params":{"speed_mode":"latency_safe","cache_clear_policy":"deferred"}}' \
  --pas-config '{"output_format":"WAV"}' \
  --threshold-rel-l2 5e-2 \
  --fail-fast
```

Parity interpretation:

- MLX internal deterministic parity (same backend/config): use strict threshold `1e-5`.
- MLX vs `audio-separator` (cross-runtime parity): use threshold `5e-2`.
- Demucs cross-backend parity runs with strict MLX Demucs kernel settings by default in parity tooling.

Optional report flags:

- `--quality-reference-manifest /path/to/reference_manifest.json`
- `--parity-strict-demucs`
- `--python-mps-latency`
- `--python-mps-parity`

Reference manifest scaffold helper:

```bash
python scripts/perf/generate_reference_manifest.py \
  --corpus-file /path/to/corpus.txt \
  --output-json /path/to/reference_manifest.json \
  --search-dir /path/to/reference_stems
```

## Reproducibility

For publication and release-quality reproducibility guidance, see `docs/reproducibility.md`.

## Requirements

- macOS 13+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.10+

## License and Attribution

This project is MIT licensed.

Portions are adapted from [`python-audio-separator`](https://github.com/nomadkaraoke/python-audio-separator) (MIT). See `THIRD_PARTY_NOTICES.md` for attribution and license notice details.

## Credits

- [BS-Roformer](https://arxiv.org/abs/2309.02612)
- [Demucs](https://github.com/facebookresearch/demucs)
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
