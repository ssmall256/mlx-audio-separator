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

- **Demucs defaults now match the parity configuration out of the box.** Fused
  GroupNorm/GLU Metal kernels are off by default. Measured on a 45 s clip through
  `htdemucs`, enabling them costs ~20 dB SNR against the unfused path (19.7 dB on drums,
  23.7 dB on other) -- audible, not float noise -- because the kernel uses an
  erf-approximation GELU and threadgroup reductions whose width varies with tensor shape.
  They are not faster either (0.783 s fused vs 0.776 s unfused, median of five runs).
  Out-of-the-box SNR against the parity configuration goes from ~20 dB to ~130 dB.
  `MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE=all` restores them for benchmarking.
- **Demucs batch size default is 2, was 8.** Measured: 0.872 s vs 1.870 s on a 45 s clip
  and 3.901 s vs 5.524 s on a 195 s one, at 4.75 GB vs 9.20 GB peak memory. The old
  `latency_safe_v2` value of 12 was ~10x slower than 2. All speed-mode profiles and the
  auto-tune candidate list now use the measured optimum, and the value lives in one
  place (`demucs_mlx/defaults.py`) so the CLI, API and `apply_model` cannot drift apart.
- **Experimental flags no longer clobber the environment.** Ten `MLX_AUDIO_SEPARATOR_*`
  variables were written unconditionally on every run, silently overriding anything a
  user exported -- including `MLX_AUDIO_SEPARATOR_GN_GLU_MULTIGROUP`, which the docs told
  users to set and which therefore could never work. Six of the ten have no CLI flag, so
  the environment was the only way to reach them. A variable is now only written when the
  caller actually supplied the corresponding `performance_params` key.
- **VR batch size default is 2, was 1.** Measured through `metalq` on
  `UVR-BVE-4B_SN-44100-2`: 2.975 s vs 3.488 s on a 45 s clip and 16.364 s vs 17.946 s on
  a 195 s one, for 5.00 GB vs 3.24 GB peak. Batch 1 is never the fastest option. Batch 4
  is marginally quicker on long inputs (15.073 s) but costs another 2.5 GB, which is the
  wrong trade for a default on a 16 GB machine; batch 8 is slower than 1. Output across
  every batch size differs by at most 3.052e-05, exactly one pcm16 LSB -- encoding
  rounding, not divergence. The retired `latency_safe_v2` profile had used 2 here, but it
  was never measured and never the default.
- **`deferred` cache clearing and 2 stem-writer threads are now the defaults.** Measured
  on a 195 s track to FLAC: 21.0 s against 22.4-25.3 s, bit-identical output (max abs
  diff 0.000e+00 on every stem), ~70 MB more peak RSS. This was the only thing
  `--speed_mode latency_safe_v3` actually did. Note that a writer thread is now spawned
  by default, which matters if you embed the library in a constrained process; pass
  `--write_workers 1` to restore serial writes.
- **`--speed_mode` is deprecated, accepted and ignored**, and will be removed in the next
  major version. `latency_safe` never did anything -- it set the values that were already
  the defaults the day it shipped. `latency_safe_v2` raised the Demucs batch to 12, which
  measures ~10x slower than 2. `latency_safe_v3`'s settings are now the defaults. Passing
  any of them logs a warning and changes nothing; the nine `scripts/perf/configs/*.json`
  that pinned a profile now say `default`, which is what they resolve to.
- **The library no longer calls `warnings.filterwarnings("ignore")`.** `Separator()` was
  silencing every warning in the host process at any log level above DEBUG. That is not a
  library's call to make, and it was hiding real defects: three leaked file handles in
  `core.py` (now closed) and the `FutureWarning` telling users their legacy Demucs pickle
  cache would not be loaded.
- **`--demucs_seed` defaults to a fixed seed**, so repeated runs on the same input
  reproduce. Pass `--demucs_seed random` for the previous per-run variation.
- **New `--precision {auto,bf16,fp32}` flag.** bf16 was already enabled for
  Roformer/MDXC transformers with no flag and no documentation. Measured on BS-Roformer
  it is ~15% faster and differs from fp32 by ~70 dB SNR (max abs diff 6.1e-05, around the
  16-bit LSB), so it stays the default -- but it is now discoverable and switchable.
- A missing `demucs` package during conversion now reports the missing `[convert]` extra
  instead of "Refusing to load Demucs checkpoint with unrestricted deserialization",
  which pointed at a security problem that did not exist.
- Always use `mx.slice_update()` for overlap-add accumulation instead of gating on
  `mlx >= 0.31.2`. The declared floor was `mlx >= 0.31.0`, so 0.31.0 and 0.31.1 silently
  took the corrupting `array.at[...].add()` path. MLX fixed the underlying strided
  scatter-add bug in 0.32.0; `mx.slice_update()` is correct on every supported version.
  Set `MLX_AUDIO_SEPARATOR_UNSAFE_SLICE_ADD=1` to benchmark the legacy path.
- Raise the minimum MLX version to 0.31.2 and allow 0.32.x (`mlx>=0.31.2,<0.33`).
  Verified end to end against MLX 0.32.2: all suites pass and Demucs output is
  bit-identical to 0.31.2 (max abs diff 0.000e+00 on every stem). Note that MLX fixed
  the strided scatter-add bug in 0.32.0, so the `mx.slice_update` accumulation this
  release makes unconditional is now belt-and-braces on 0.32.x rather than load-bearing.

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
