# Changelog

All notable changes to this project are documented in this file.

## 0.1.12 - 2026-09-23

### Fixed

- **Converted Demucs weights no longer share a directory with demucs-mlx.**
  Both packages wrote `<model>_config.json` and `<model>.safetensors` into
  `~/.cache/demucs-mlx` under config schemas that reject each other, and as of
  0.1.10 and demucs-mlx 1.4.8 both rebuild a cache they cannot read -- so with
  the two installed side by side (which is the default, since demucs-mlx does
  not depend on mlx-weights and falls back to that directory) every alternating
  run reconverted, each time needing torch and the upstream checkpoint. This
  package now writes to `~/.cache/mlx-audio-separator/demucs`, and never to
  demucs-mlx's directory. A cache left in the old location by an earlier release
  is still read, so upgrading costs no reconversion.
  `MLX_AUDIO_SEPARATOR_DEMUCS_CACHE_DIR` overrides the location.

## 0.1.11 - 2026-09-23

### Fixed

- **The CLI exited 0 when every file failed.** `separate()` logs a file it
  could not process and moves on, so one bad input does not abort a batch --
  but the CLI then printed `Separation complete! Output file(s):` with an empty
  list and exited 0 regardless, so a run that produced nothing looked like a
  success to any script wrapping it. It now reports each failure and exits 1,
  whether some files succeeded or none did. `Separator.failed_files` lists
  `(path, message)` for the last `separate()` call, so library callers can make
  the same distinction.

## 0.1.10 - 2026-09-23

### Fixed

- **Demucs produced no output on a machine with no mlx-spectro tuning cache
  yet.** 0.1.9 compiles the Demucs forward, and the STFT inside it picks its
  Metal threadgroup size by timing candidates -- which needs `mx.eval`, and MLX
  forbids `mx.eval` inside a compile trace. On a first run the separation ended
  with `no usable threadgroup size for n_fft=4096, hop=1024` and wrote no stems;
  once any uncompiled call had populated the cache, the same command worked, so
  it did not reproduce on a machine that had already run one. The first call at
  each input shape now runs eagerly and the next one compiles. Its output is
  used, so the warm-up costs nothing, and compiled output still tracks the eager
  path at 120-138 dB SNR across `htdemucs`, `htdemucs_6s` and `hdemucs_mmi`.
  Requires mlx-spectro 0.9.2, which also stops tuning from raising under a
  trace.
- **An unreadable Demucs cache now regenerates instead of raising.**
  `get_mlx_model` caught only `FileNotFoundError`, so a cache that existed but
  failed validation raised `SafeCacheError` to the caller -- with nothing saying
  that regenerating was the fix. Caches written before 0.1.8 lack fields the
  hardened loader requires, and demucs-mlx writes a differently-shaped config
  into the same `~/.cache/demucs-mlx` directory, so either upgrading or having
  both packages installed could produce it. An unusable cache is now rebuilt,
  which is also the right response to a digest mismatch. Errors that are not
  about the cache still propagate.

## 0.1.9 - 2026-09-23

### Added

- **The Demucs forward is compiled.** `mx.compile` on the model forward is worth
  **+15.9%** end to end on a 60 s track, or **+16.8%** together with the
  overlap-add change below, measured on an idle M4 against a same-config control
  (noise floor 1.04%). It is faster on the first call too — 0.671 s against
  0.776 s on a 20 s clip with no warmup — so there is no cold-start cost.
  Compilation reassociates floating-point adds, so output is 107–115 dB SNR from
  the eager path, verified on `htdemucs`, `htdemucs_6s` and `hdemucs_mmi`.
  `MLX_AUDIO_SEPARATOR_DEMUCS_COMPILE=0` restores the eager forward.
- `scripts/perf/ab_harness.py`, an A/B harness that runs one process per arm,
  rotates the arm order, and computes a noise floor from same-config control
  arms. On an idle machine it resolves to 0.04%.

### Changed

- **Demucs overlap-add evaluates after every update rather than every 8.**
  **+4.3%** end to end on a 60 s track, output bit-identical. Sweeping the
  interval on an idle M4 against a same-config control (noise floor 0.62%):
  1 → +4.3%, 2 → +3.5%, 4 → +1.7%, 16 → +1.0%.
- **Roformer/MDXC precision now defaults to fp32.** bf16 measures no faster on
  current hardware — dead on a same-config control at a 0.04% noise floor — and
  costs ~78 dB SNR, so fp32 is the better default. `--precision bf16` and
  `MLX_ENABLE_AMP=1` remain available.
- Release workflows set the version from the workflow input before building,
  wait up to 10 minutes for a new file to reach every CDN mirror, and retry the
  install.

### Fixed

- **Fused GroupNorm Metal kernels: added a missing threadgroup barrier.** The
  three-pass reduction shares one `shared_sums` array, and pass 2 could overwrite
  the group mean before every simdgroup had read it. With the barrier, relative
  error at real htdemucs shapes drops from 1.6e-02 to **2.0e-07**, output becomes
  run-to-run deterministic, and end-to-end SNR against the unfused path is
  **118.2 dB**. The kernels stay opt-in via
  `MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE=all`, since the unfused path is the
  same speed. New tests cover parity and determinism at the real shapes.
- Loading a Demucs cache written by a different MLX version no longer suggests
  reconverting. One cache produces bit-identical stems under 0.31.2 and 0.32.2,
  and the strict loader already rejects a genuinely incompatible cache.

## 0.1.8 - 2026-09-22

### Fixed

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
  Timed on an otherwise idle Mac mini (M4) with two `uv` environments identical except
  for MLX, arms alternating each round: 6.757 s median on 0.31.2 vs 6.647 s on 0.32.2
  over 16 timed separations each (+/-2.2% and +/-1.9% spread), i.e. **1.02x** with
  median and best agreeing. Peak memory 4.67 GB on both. Upgrading is safe on speed.
- A Demucs cache written by a different MLX version no longer prints
  `Consider reconverting`. The check compared version strings exactly and advised a
  torch-requiring, multi-minute reconversion on any difference. It is wrong: one cache
  produces bit-identical stems (0.000e+00, and deterministic run to run) under both
  0.31.2 and 0.32.2, and the strict loader added this release already raises on a
  genuinely incompatible cache. A warning now fires only when a version is outside the
  supported range, and goes through `warnings.warn` rather than `print`.

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
