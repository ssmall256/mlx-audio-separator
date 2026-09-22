# Tuning

**You should not need this page.** The defaults are chosen to give the best
results out of the box: every option below is already set to the value that
measured best, and where accuracy and speed conflicted, accuracy won unless the
difference was inaudible.

This page exists so the remaining levers are discoverable in one place rather
than by reading source, and so the measurements behind each default are on the
record.

## What the defaults are, and why

| Setting | Default | Why |
|---|---|---|
| Demucs fused GroupNorm/GLU kernels | **off** | Cost ~20 dB SNR vs the unfused path (19.7 dB on drums) and are not faster (0.783 s vs 0.776 s). Strictly worse on both axes. |
| Demucs batch size | **2** | Fastest *and* smallest: 0.872 s / 4.75 GB vs 1.870 s / 9.20 GB at batch 8 on a 45 s clip. Batch 12 is ~10x slower. |
| Demucs shifts | **2** | Matches python-audio-separator. Upstream `demucs` uses 1; each shift costs a full pass, so `--demucs_shifts 1` roughly halves runtime at some quality cost. |
| Demucs shift seed | **fixed** | Repeated runs on the same input reproduce. `--demucs_seed random` restores per-run variation. |
| Roformer/MDXC precision | **bf16** | ~15% faster, and ~70 dB SNR from fp32 (max abs diff 6.1e-05, about the 16-bit LSB) -- inaudible. `--precision fp32` for exact parity work. |
| Overlap-add accumulation | `mx.slice_update` | Correct on every supported MLX version. MLX before 0.32.0 corrupts strided slice scatter-add. |

All numbers measured on Apple silicon, 128 GB, macOS 27, mlx 0.31.2, using
`htdemucs` and `model_bs_roformer_ep_317_sdr_12.9755`.

## Environment variables

Every variable below defaults to off/unset. None of them is needed for good
results; they exist for benchmarking, parity investigations and debugging.

### Demucs

| Variable | Default | Effect |
|---|---|---|
| `MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE` | `off` | `all`, `glu_only`, `gelu_only` or `off`. Re-enables the fused kernels. Costs parity; see above. |
| `MLX_AUDIO_SEPARATOR_DEMUCS_USE_FUSED_GN_GLU` | — | Equivalent switch in standalone `demucs-mlx` (`DEMUCS_MLX_USE_FUSED_GN_GLU`). |
| `MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED` | off | Caps reduction threadgroups at 256 and disables fused GroupNorm. |
| `MLX_AUDIO_SEPARATOR_DETERMINISTIC_ACCUMULATION` | off | Forces ordered overlap-add accumulation. Slower; for reproducibility checks. |
| `MLX_AUDIO_SEPARATOR_DEMUCS_STRICT_EVAL` | off | Inserts `mx.eval` barriers through the forward pass. |
| `MLX_AUDIO_SEPARATOR_DEMUCS_ISTFT_ALLOW_FUSED` | on | Fused iSTFT. Measured effect on output is negligible. |
| `MLX_AUDIO_SEPARATOR_DEMUCS_WIENER_USE_VMAP` | on | vmap-parallel Wiener filtering. `hdemucs`-family only. |
| `MLX_AUDIO_SEPARATOR_UNSAFE_SLICE_ADD` | off | Restores the legacy `at[].add()` accumulator. **Produces wrong output on MLX < 0.32**; benchmarking only. |

### Roformer / MDXC

| Variable | Default | Effect |
|---|---|---|
| `MLX_ENABLE_AMP` | `1` | bf16 transformer stack. Prefer `--precision`. |
| `MLX_USE_FAST_SDP` | `1` | `mx.fast.scaled_dot_product_attention`. |
| `MLX_ENABLE_COMPILE` | `1` | `mx.compile` on the transformer subgraph. |

Experimental Roformer/MDXC toggles are reached through `performance_params`
(API) or the matching `--experimental_*` CLI flags, not by exporting variables.
A variable you export is now respected: the separator only publishes a value
for a flag you actually asked about.

## Exact-parity configuration

To compare against the reference PyTorch implementation, the shipped defaults
are already the parity configuration for Demucs. For Roformer add:

```bash
mlx-audio-separator input.wav --precision fp32
```

`scripts/perf/mlx_vs_pas_parity.py` additionally pins
`MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED=1`,
`MLX_AUDIO_SEPARATOR_DEMUCS_ISTFT_ALLOW_FUSED=0`,
`MLX_AUDIO_SEPARATOR_DEMUCS_WIENER_USE_VMAP=0` and
`MLX_AUDIO_SEPARATOR_DEMUCS_STRICT_EVAL=1`. Measured against the current
defaults, those four together move the result by well under 1 dB -- the ~20 dB
gap they used to close came entirely from the fused GroupNorm kernels, which
are now off by default.
