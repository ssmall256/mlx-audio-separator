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
