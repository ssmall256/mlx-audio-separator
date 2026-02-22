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

## Benchmarking

Benchmark now supports warmup/repeat medians and optional phase profiling:

```bash
mlx-audio-separator \
  --benchmark song.mp3 \
  --benchmark_warmup 1 \
  --benchmark_repeats 3 \
  --benchmark_profile
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
