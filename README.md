# mlx-audio-separator

MLX-native stem separation for Apple Silicon Macs.

This project ports the inference paths from [audio-separator](https://pypi.org/project/audio-separator/) (upstream repo: [nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)) to MLX so separation runs on Apple Silicon without requiring PyTorch or ONNX Runtime at inference time.
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

### Demucs cache security and migration

Demucs conversion stores MLX weights as `<model>.safetensors` with a validated
`<model>_config.json` sidecar. The loader verifies that the two files match
before constructing a model. Legacy `<model>_mlx.pkl` caches are never
deserialized. When automatic conversion is enabled they are left untouched and
replaced by newly generated safe files from the official Demucs source.

To regenerate a cache explicitly:

```bash
pip install "mlx-audio-separator[convert]"
python -m mlx_audio_separator.demucs_mlx.mlx_convert htdemucs \
  --output-dir ~/.cache/mlx-audio-separator/demucs
```

Converted Demucs weights live in `~/.cache/mlx-audio-separator/demucs`.
`MLX_AUDIO_SEPARATOR_DEMUCS_CACHE_DIR` points that elsewhere. Caches left in the
older `~/.cache/demucs-mlx` by a release before 0.1.12 are still read, so
upgrading does not cost a reconversion; nothing is written there any more,
because the demucs-mlx package uses that directory for its own differently
shaped cache.

Official Demucs downloads retain their filename hash checks and are loaded with
PyTorch's restricted weight-only deserializer. This trusts the installed
PyTorch, Demucs, NumPy, and optional DiffQ implementations plus the official
model registry; arbitrary checkpoint globals and local pickle caches are not
trusted.

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

**The defaults are the recommended configuration.** Every inference option ships
at the value that measured best for both quality and speed, so you should not
need to set anything to get good results. See [docs/tuning.md](docs/tuning.md)
for what those values are, the measurements behind them, and the levers that
remain for benchmarking or exact-parity work.

Release-facing stable controls:

- `--cache_clear_policy {aggressive,deferred}` — defaults to `deferred`
- `--write_workers <int>` — defaults to `2`

Both already ship at the value that measured fastest, so the example that used
to live here (passing them explicitly) is no longer needed.

`--speed_mode` is **deprecated and ignored**. Its `latency_safe*` profiles now
resolve to the defaults, and it will be removed in the next major version.

Basic benchmark command:

```bash
mlx-audio-separator \
  --benchmark song.mp3 \
  --benchmark_warmup 1 \
  --benchmark_repeats 3 \
  --benchmark_profile
```

## BS-Roformer-SW Performance (Opt-In)

For `BS-Roformer-SW.ckpt`, no extra flags are needed:

```bash
mlx-audio-separator song.mp3 \
  -m BS-Roformer-SW.ckpt \
  --output_format FLAC
```

What `--speed_mode latency_safe_v3` used to enable -- `deferred` cache clearing
plus async stem writes -- is now the default. Measured on a 195 s track to
FLAC it runs ~6-17% faster than the old defaults with bit-identical output.

To avoid repeated checkpoint conversion overhead, pre-convert once to
`*.safetensors` and exit:

```bash
mlx-audio-separator \
  -m BS-Roformer-SW.ckpt \
  --save_converted_safetensors \
  --preconvert_only
```

`safetensors` primarily improves model load/startup time. It is not expected to
materially change per-file inference latency.

Validation command (latency + deterministic equivalence):

```bash
uv run --with torch python scripts/perf/compare_latency.py \
  --corpus-file /tmp/corpus_one.txt \
  --baseline-config scripts/perf/configs/bs_roformer_sw_default_baseline.json \
  --candidate-config scripts/perf/configs/bs_roformer_sw_latency_safe_v3_candidate.json \
  --model-file-dir ~/.cache/mlx-audio-separator/models \
  --allow-speed-mode-mismatch \
  --target-improvement-demucs-mdxc 10.0 \
  --equivalence-check \
  --equivalence-threshold-rel-l2 1e-6 \
  --output-json /tmp/bs_roformer_sw_latency_safe_v3_compare.json \
  --output-markdown /tmp/bs_roformer_sw_latency_safe_v3_compare.md
```

## BS-Roformer-SW Optimization Program (Opt-In Tracks)

As of 0.1.8, what `latency_safe_v3` promoted -- `deferred` cache clearing and
async stem writes -- is the default for every model, and `--speed_mode` is
deprecated and ignored. Measure candidates against the defaults. All
experimental tracks below remain parked pending new evidence.

Candidate configs for staged exploration live under `scripts/perf/configs/`:

- `bs_roformer_sw_cand_grouped_bandmask.json`
- `bs_roformer_sw_cand_fused_ola.json`
- `bs_roformer_sw_cand_stream_pipeline.json`
- `bs_roformer_sw_cand_compile_fullgraph.json`
- `bs_roformer_sw_cand_flac_fastwrite.json`

Corpus manifest templates:

- Quick gate (3 files): `scripts/perf/corpora/bs_roformer_sw_quick.txt`
- Full gate (12 files): `scripts/perf/corpora/bs_roformer_sw_full.txt`

Quick-gate example:

```bash
uv run --with torch python scripts/perf/compare_latency.py \
  --corpus-file scripts/perf/corpora/bs_roformer_sw_quick.txt \
  --baseline-config scripts/perf/configs/bs_roformer_sw_latency_safe_v3_baseline.json \
  --candidate-config scripts/perf/configs/bs_roformer_sw_cand_grouped_bandmask.json \
  --model-file-dir ~/.cache/mlx-audio-separator/models \
  --target-improvement-demucs-mdxc 3.0 \
  --equivalence-check \
  --equivalence-threshold-rel-l2 1e-6 \
  --equivalence-max-files 1 \
  --output-json /tmp/bs_roformer_sw_quick_gate.json \
  --output-markdown /tmp/bs_roformer_sw_quick_gate.md
```

Full-gate example:

```bash
uv run --with torch python scripts/perf/compare_latency.py \
  --corpus-file scripts/perf/corpora/bs_roformer_sw_full.txt \
  --baseline-config scripts/perf/configs/bs_roformer_sw_latency_safe_v3_baseline.json \
  --candidate-config scripts/perf/configs/bs_roformer_sw_cand_grouped_bandmask.json \
  --model-file-dir ~/.cache/mlx-audio-separator/models \
  --target-improvement-demucs-mdxc 5.0 \
  --equivalence-check \
  --equivalence-threshold-rel-l2 1e-6 \
  --equivalence-max-files 0 \
  --output-json /tmp/bs_roformer_sw_full_gate.json \
  --output-markdown /tmp/bs_roformer_sw_full_gate.md
```

## Documentation

| Document | Description |
|---|---|
| [`docs/release-validation.md`](docs/release-validation.md) | Release evidence snapshot |
| [`docs/release-first.md`](docs/release-first.md) | Release execution playbook |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Reproducibility guide |
| [`docs/wave4-opt-in.md`](docs/wave4-opt-in.md) | Wave 4 opt-in/experimental roadmap |
| [`docs/bs-roformer-sw-optimization-program.md`](docs/bs-roformer-sw-optimization-program.md) | BS-Roformer-SW candidate gating and promotion table |
| [`CHANGELOG.md`](CHANGELOG.md) | Changelog |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | Third-party attribution and license notices |

## License

This project is MIT licensed.

## Acknowledgments

`mlx-audio-separator` is derived from [audio-separator](https://pypi.org/project/audio-separator/) (upstream repo: [nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator), MIT) by [beveradb](https://github.com/beveradb) and the [nomadkaraoke](https://github.com/nomadkaraoke) community. Substantial portions of the architecture, model loading, and separation logic are adapted from that project. If you find this package useful, please also star and support the upstream project.

The models used by this project were trained by the [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) community, primarily [@Anjok07](https://github.com/Anjok07) and [@aufr33](https://github.com/aufr33). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for full attribution and license details.

Additional references:

- [mlx-audio-io](https://github.com/ssmall256/mlx-audio-io)
- [mlx-spectro](https://github.com/ssmall256/mlx-spectro)
- [BS-Roformer](https://arxiv.org/abs/2309.02612)
- [Demucs (Meta Research)](https://github.com/facebookresearch/demucs)
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-separator](https://pypi.org/project/audio-separator/) / [nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
