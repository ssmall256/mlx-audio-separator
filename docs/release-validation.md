# Release Validation Snapshot

This document captures the fixed release evidence snapshot referenced by `README.md`.

## Snapshot Scope

- Hardware: Apple Silicon `M4 mini`
- Dataset: `MUSDB18-HQ` test subset
- Comparison model set (MLX vs `audio-separator` overlap):
  - `htdemucs_ft.yaml`
  - `model_bs_roformer_ep_317_sdr_12.9755.ckpt`
  - `mel_band_roformer_instrumental_instv7n_gabox.ckpt`
  - `UVR-MDX-NET-Inst_HQ_3.onnx`
- Command families:
  - full benchmark + release gate (`mlx_audio_separator.utils.cli --benchmark`, `scripts/release/check_release_readiness.py`)
  - cross-runtime ABBA (`scripts/perf/mlx_vs_pas_abba.py`)
  - parity smoke (`scripts/perf/mlx_vs_pas_parity.py`)

## Validation Results

| Check | Result |
|---|---|
| Full-catalog benchmark release gate | `152 / 152` models `ok`, `0` failures |
| Unit tests | `167 passed`, `1 skipped` |
| MLX vs `audio-separator` parity smoke | `4 / 4` models passed (`rel L2 <= 5e-2`) |

Notes:
- Benchmark gate count reflects the validated release snapshot run.
- Unit test summary reflects the release prep test run in this repository.

## Performance Results (ABBA, 12-song subset)

| Model | Arch | MLX speedup vs PAS | Delta % (MLX vs PAS) |
|---|---|---:|---:|
| `htdemucs_ft.yaml` | Demucs | `1.40x` | `-28.34%` |
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | MDXC | `2.16x` | `-53.77%` |
| `mel_band_roformer_instrumental_instv7n_gabox.ckpt` | MDXC | `2.50x` | `-60.01%` |
| `UVR-MDX-NET-Inst_HQ_3.onnx` | MDX | `1.53x` | `-34.68%` |

Median speedup across the 4-model overlap set: **`1.847x`**.

## Artifact Provenance

Primary artifacts used for this snapshot:

1. `/Users/sam/Downloads/release_gate_summary-20260224.json`
2. `/Users/sam/Downloads/mlx_vs_pas_abba_12.json`
3. `/Users/sam/Downloads/mlx_vs_pas_abba_12.md`
4. `/Users/sam/Documents/release-run-20260225-230249/mlx_vs_pas_parity_smoke.json` (parity smoke source run)

Supplemental context:

1. `/Users/sam/Downloads/optimization_report.json`
2. `/Users/sam/Downloads/optimization_report.md`

Snapshot timeline:

- 2026-02-24: full benchmark release gate snapshot
- 2026-02-26: ABBA cross-runtime snapshot and parity smoke confirmation

Snapshot revision metadata:

- Commit SHA (from optimization report metadata): `81d78c82490d7ff3ec59a4613e1ee7f5d906f804`
- Optimization report timestamp (UTC): `2026-02-26T17:10:16.628937+00:00`
