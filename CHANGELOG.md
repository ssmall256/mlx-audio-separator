# Changelog

All notable changes to this project are documented in this file.

## 0.1.4 - 2026-03-06

### Changed

- All audio loading paths use `mac.load(path, sr=target)` for resampling, which now auto-selects `soxr_vhq` quality when available (via upstream mlx-audio-io 1.3.9).
- Demucs CLI (`separate.py`) no longer raises `ValueError` when input sample rate differs from model sample rate — it resamples automatically.
- Demucs audio pipeline (`separate.py`, `api.py`) now uses native MLX arrays end-to-end instead of numpy intermediaries.
- `mlx_backend.resample_mx()` uses direct `mac.resample()` instead of writing/reading a temp file.
- Removed unnecessary MLX→numpy→MLX round-trips in `audio_chunking` and `VRSeparator`.
- Bumped minimum `mlx-audio-io` to `>=1.3.9`.

## 0.1.1 - 2026-02-24

### Added

- Benchmark reliability hardening:
  - Invalid-output detection (`stems == 0`, missing files, empty files).
  - Narrow one-time redownload retry for corrupted archive signatures.
  - Demucs benchmark preflight skip behavior when conversion dependency is missing.
- Loader recovery and compatibility updates across VR/MDX/MDXC families.
- Strict benchmark diagnostics path to preserve inner exception context in benchmark outputs.
- MLX-native MDX23C (`tfc_tdf_v3`) support and checkpoint conversion/cache path.
- Release/readiness and reproducibility documentation improvements.

### Changed

- Compile policy update:
  - MDX23C compiled forward remains available behind opt-in runtime flags.
  - Roformer compile paths (shapeless/static compiled demix) are currently disabled by policy.
- CLI help text and README now document current compile-path behavior explicitly.

### Fixed

- Multiple benchmark false-positive success cases.
- Model-loader mismatch families in VR, MDX, and MDXC routing.
- Silent/opaque benchmark failure surfaces now report actionable diagnostics.
