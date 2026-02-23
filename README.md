# mlx-audio-separator

MLX-native audio stem separation for Apple Silicon. Separate vocals, instruments, drums, bass, and more from any audio file — no PyTorch or ONNX runtime required.

## Installation

```bash
pip install mlx-audio-separator
```

For weight conversion from PyTorch/ONNX checkpoints (first-time setup only):

```bash
pip install mlx-audio-separator[convert]
```

## Quick Start

### CLI

```bash
# Separate vocals from a song (default: BS-Roformer)
mlx-audio-separator song.mp3

# Use a specific model
mlx-audio-separator song.mp3 -m htdemucs_ft.yaml

# List all available models
mlx-audio-separator --list_models

# Opt-in latency-safe mode with perf trace output
mlx-audio-separator song.mp3 --speed_mode latency_safe --perf_trace

# Auto-tune batch size for current model/audio (first run probes, then caches)
mlx-audio-separator song.mp3 --auto_tune_batch --tune_probe_seconds 8
```

### Python API

```python
from mlx_audio_separator import Separator

sep = Separator()
sep.load_model()
stems = sep.separate("song.mp3")
print(stems)  # list of output file paths
```

## Supported Models

| Architecture | Models | Description |
|---|---|---|
| Roformer | 78 | BS-Roformer, MelBand-Roformer |
| MDXC | 2 | MDX23C |
| MDX | 1 | ConvTDFNet |
| VR | 2 | CascadedASPPNet, CascadedNet |
| **Total** | **83** | |

Run `mlx-audio-separator --list_models` to see the full list.

## Performance Tuning

The default behavior is unchanged. Use opt-in flags to tune latency:

```bash
mlx-audio-separator song.mp3 \
  --speed_mode latency_safe \
  --cache_clear_policy deferred \
  --write_workers 2 \
  --perf_trace \
  --perf_trace_path ./perf_trace.jsonl
```

Available performance options:

- `--speed_mode {default,latency_safe}`
- `--auto_tune_batch`
- `--tune_probe_seconds <seconds>`
- `--cache_clear_policy {aggressive,deferred}`
- `--write_workers <int>`
- `--perf_trace`
- `--perf_trace_path <path>`

Current `latency_safe` preset recommendations (strict output-equivalent path, opt-in):

- `Demucs`: `batch_size=8`
- `MDXC` (Roformer): `batch_size=1`
- `MDX`: `batch_size=1`
- `VR`: `batch_size=1`

Deterministic equivalence checks use strict pass/fail gating for `MDXC`, `MDX`, and `VR` by default.
`Demucs` is reported as informational unless strict Demucs gating is explicitly enabled.
When equivalence utilities run, they automatically enable deterministic fused-kernel mode for reproducible Demucs comparisons.
In deterministic mode, Demucs keeps fused `GroupNorm+GELU`, enables strict forward eval barriers, disables fused iSTFT and Wiener `vmap` parallelization, and forces deterministic overlap-add accumulation to reduce run-to-run drift.

## Benchmarking

Benchmark now supports warmup/repeat medians and optional phase profiling:

```bash
mlx-audio-separator \
  --benchmark song.mp3 \
  --benchmark_warmup 1 \
  --benchmark_repeats 3 \
  --benchmark_profile
```

Unified optimization report (latency + deterministic parity + fast-mode quality):

```bash
python /Users/sam/Code/mlx-audio-separator/scripts/perf/run_optimization_report.py \
  --corpus-file /tmp/corpus.txt \
  --baseline-config /tmp/baseline.json \
  --candidate-config /tmp/candidate.json \
  --models htdemucs.yaml,model_bs_roformer_ep_317_sdr_12.9755.ckpt,UVR-MDX-NET-Inst_HQ_3.onnx \
  --parity-strict-demucs \
  --parity-max-files 2
```

Optional `--quality-reference-manifest /path/to/references.json` enables SI-SDR/SDR delta quality gates against reference stems.
Without reference stems, proxy quality metrics are emitted as informational by default (use `--quality-enforce-proxy-gate` to make them pass/fail).
Use `--python-mps-latency` to include optional python-audio-separator (MPS) latency deltas in the same report.
Report outputs include a reproducibility appendix (command, git commit/dirty state, platform versions, corpus/model manifests with optional SHA256 hashes).
For a full release checklist, see `/Users/sam/Code/mlx-audio-separator/docs/reproducibility.md`.

Reference manifest scaffold helper:

```bash
python /Users/sam/Code/mlx-audio-separator/scripts/perf/generate_reference_manifest.py \
  --corpus-file /tmp/corpus.txt \
  --output-json /tmp/reference_manifest.json \
  --search-dir /path/to/reference_stems
```

## Requirements

- macOS 13+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.10+

## License

MIT

## Credits

- [BS-Roformer](https://arxiv.org/abs/2309.02612) — Band-Split Roformer for music source separation
- [Demucs](https://github.com/facebookresearch/demucs) — Facebook Research hybrid transformer model
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) — VR and MDX model architectures and pretrained weights
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) — Original Python implementation
