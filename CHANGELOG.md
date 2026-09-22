# Changelog

All notable changes to this project are documented in this file.

## 0.1.8 - 2026-09-22

### Fixed

- Demucs models produced near-silent stems on 0.1.7 (issue #4). The safetensors cache
  introduced in 0.1.7 was written from `tree_flatten(model.state_dict())` (MLX attribute
  names) but read back through the PyTorch-style key walker used by the conversion path,
  which collapses the `conv`/`layers` wrapper segments. 360 of 573 tensors never matched
  and were left at their random initialization, with no error and no warning, so every
  Demucs model emitted noise-floor output. Cache loading now validates keys and shapes
  against the constructed model and fails loudly on any mismatch.
  **Existing caches do not need to be regenerated** — only the reader was wrong.
- `--sample_rate` no longer truncates Demucs output. Demucs runs at its trained 44100 Hz
  rate, but the stems were written with the requested rate in the header and no resample,
  so `--sample_rate 48000` produced files 8.8% short. Stems are now resampled to the
  requested rate before writing; the default 44100 path is unchanged and does no work.

### Changed

- Always use `mx.slice_update()` for overlap-add accumulation instead of gating on
  `mlx >= 0.31.2`. The declared floor was `mlx >= 0.31.0`, so 0.31.0 and 0.31.1 silently
  took the corrupting `array.at[...].add()` path. MLX fixed the underlying strided
  scatter-add bug in 0.32.0; `mx.slice_update()` is correct on every supported version.
  Set `MLX_AUDIO_SEPARATOR_UNSAFE_SLICE_ADD=1` to benchmark the legacy path.
- Raise the minimum MLX version to 0.31.2 to match what is actually tested.

## 0.1.7 - 2026-08-12

### Changed

- Run Ruff across the full repository in CI and clean the existing lint baseline.
- Update GitHub-hosted workflow actions to Node.js 24-native major versions.

### Security

- Load official Demucs packages with PyTorch's restricted weight-only deserializer and a narrow compatibility allowlist instead of the unrestricted upstream repository loader.
- Replace Demucs MLX pickle caches with versioned safetensors plus validated JSON metadata; legacy `*_mlx.pkl` files are ignored and safely regenerated rather than deserialized.
- Fail closed on incomplete, malformed, or digest-mismatched Demucs caches without downgrading to pickle.

## 0.1.6 - 2026-08-12

### Security

- Use restricted weight-only deserialization for direct RoFormer, MDX23C, and VR PyTorch checkpoint conversion paths.
- Reject PyTorch versions older than 2.6 before reading checkpoints because CVE-2025-32434 affects weight-only loading in earlier releases.

### Changed

- Require PyTorch 2.6 or newer for the optional `convert` dependency group.
- Continue supporting plain tensor state dictionaries and `state_dict`/`model` wrappers while intentionally rejecting checkpoints containing arbitrary Python objects.

## 0.1.5 - 2026-06-13

### Fixed

- Work around MLX 0.31.2 contiguous slice-update corruption in split-mode and chunked overlap-add accumulators by using explicit `mx.slice_update()` accumulation on affected MLX versions.

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
