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
| Demucs compiled forward | **on** | `mx.compile` on the model forward: +15.9% end to end, +16.8% with the flush below. Faster on the first call too, so no cold-start cost. Output 107-115 dB SNR from eager (fusion reassociates adds), verified on htdemucs, htdemucs_6s and hdemucs_mmi. `MLX_AUDIO_SEPARATOR_DEMUCS_COMPILE=0` disables. |
| Demucs overlap-add flush | **every update** | Was every 8. +4.3% on 60 s, output bit-identical. Sweep: 1 -> +4.3%, 2 -> +3.5%, 4 -> +1.7%, 16 -> +1.0%. |
| Demucs fused GroupNorm/GLU kernels | **off** | Had a missing threadgroup barrier: the mean in `shared_sums[0]` could be clobbered by pass 2 before every simdgroup had read it, worst at small `elems_per_group` — the freq-branch DConv shapes. Fixed, which took end-to-end SNR from ~20 dB to **118.2 dB** and relative error from 1.6e-02 to 2.0e-07. Still off, now only because they are not *faster*: 1.7331 s against a 1.7270 s control at a 1.76% noise floor. |
| Demucs batch size | **2** | Fastest *and* smallest: 0.872 s / 4.75 GB vs 1.870 s / 9.20 GB at batch 8 on a 45 s clip. Batch 12 is ~10x slower. |
| Demucs shifts | **2** | Matches python-audio-separator. Upstream `demucs` uses 1; each shift costs a full pass, so `--demucs_shifts 1` roughly halves runtime at some quality cost. |
| Demucs shift seed | **fixed** | Repeated runs on the same input reproduce. `--demucs_seed random` restores per-run variation. |
| VR batch size | **2** | Batch 1 is never fastest: 2.975 s vs 3.488 s on a 45 s clip, 16.364 s vs 17.946 s on a 195 s one. Batch 4 edges it on long inputs but costs another 2.5 GB. |
| Roformer/MDXC precision | **fp32** | bf16 was the default on a "~15% faster, ~70 dB" claim. The precision half is right for BS-Roformer (78.1 dB). The speed half is not: on an idle M4 mini with three identical fp32 control arms (noise floor **0.04%**), bf16 landed dead on the control -- 7.341 s against 7.340/7.341/7.343 s. Casting weights too is 0.3% slower. And for mel-band models the switch is never read at all, so output is byte-identical to fp32. Mechanically it cannot be large: the cast covers activations only, and MLX promotes `bf16 @ fp32` back to fp32, so no matmul runs in half precision. `--precision bf16` still available. |
| Cache clear policy | **`deferred`** | ~6-17% faster end to end with bit-identical output, for ~70 MB more peak RSS. |
| Stem writer threads | **2** | Same measurement: overlaps encoding with inference. |
| Overlap-add accumulation | `mx.slice_update` | Correct on every supported MLX version. MLX before 0.32.0 corrupts strided slice scatter-add. |

### `--speed_mode` is deprecated

It is accepted and ignored, and will be removed in the next major version.

`latency_safe` never did anything: it set Demucs 8 / MDXC 1 / MDX 1 / VR 1,
which were already the defaults the day it shipped. `latency_safe_v2` raised
the Demucs batch to 12, which measures ~10x slower than the default of 2.
`latency_safe_v3` was the only profile that did something useful, and its
`deferred` cache clearing and two writer threads are now simply the defaults.

The names described a lineage rather than a behaviour, which is the opposite of
a discoverable option. Use `--cache_clear_policy` and `--write_workers`
directly if you need to change either.

All numbers measured on Apple silicon, 128 GB, macOS 27, mlx 0.31.2, using
`htdemucs`, `model_bs_roformer_ep_317_sdr_12.9755` and
`UVR-BVE-4B_SN-44100-2`. Benchmarks run through `metalq`, which serializes
GPU jobs and waits for thermal cooldown between them, so timings are
comparable rather than whatever the machine happened to be doing.

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
| `MLX_ENABLE_AMP` | `0` | bf16 transformer stack. Off by default since it measured slower, and is a no-op for mel-band models entirely. If half precision is revisited, use fp16 rather than bf16: 78.5 dB against 59.1 dB SNR from fp32 for identical matmul cost. |
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

### BS-Roformer already has its compile win

`MLX_ENABLE_COMPILE=1` (set by the Roformer loader) compiles
`_forward_transformers`, and that is worth **23%**: turning it off measures
9.038 s against a 7.343/7.352/7.351 s control at a 0.12% noise floor.
Extending compilation to the whole `__call__` adds **0.2%** — nothing.

This is the opposite of Demucs, which had no compiled graph at all until 0.1.8
and gained 15.9% from one. If you are looking for headroom, look where nothing
is compiled, not where something already is.

## Measuring a change here

Use `scripts/perf/ab_harness.py`. It runs one process per arm, rotates the arm
order, and computes a noise floor from two same-config control arms; anything
smaller than that floor has not been shown to do anything. On a loaded laptop it
reports a ~40% floor and will happily show you five "wins" of 46-56%; on an idle
M4 the same run gives 0.4-1.0%. If the floor comes back above a few percent the
measurement is void — move machines rather than reading the table.

Two Demucs "wins" of -16% and -20% in `roformer-kernel-fusion-followup.md` were
discarded because a control arm moved. Re-measured with this harness, the -16%
one (`DEMUCS_APPLY_CONCAT_BATCHING`) is +0.7%, inside the noise.
