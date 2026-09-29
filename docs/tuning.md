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
| Demucs compiled forward | **on** | `mx.compile` on the model forward: **+15.9%** end to end in the original comparison, and faster on the first call too. Output is 107-115 dB SNR from eager (compilation reassociates float adds). `MLX_AUDIO_SEPARATOR_DEMUCS_COMPILE=0` disables. |
| Demucs overlap-add evaluation | **per batch for up to 8 offsets; per update from 9 at batch two** | True per-update evaluation was 3.4% slower on 30 s and 4.3% faster on 60 s; outputs were bit-identical. |
| Demucs GroupNorm | **fast MLX LayerNorm** | Flattens each channel group and uses `mx.fast.layer_norm`. `MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE=all` opts into the older custom GroupNorm+activation modules. |
| Demucs phased decoders | **on for FP32 kernel-8/stride-4** | Computes four two-tap phases and interleaves them; other shapes and dtypes retain the ordinary MLX transposed convolution. |
| Demucs DConv compile | **off** | The existing whole-model compile makes separate DConv compilation redundant. It remains available for experiments. |
| Demucs batch size | **2** | Fastest *and* smallest: 0.872 s / 4.75 GB vs 1.870 s / 9.20 GB at batch 8 on a 45 s clip. Batch 12 is ~10x slower. |
| Demucs shifts | **1** | Matches upstream Demucs and the standalone project. Use `--demucs_shifts 2` to spend a second full pass for the shift-averaging quality tradeoff. |
| Demucs shift seed | **fixed** | Repeated runs on the same input reproduce. `--demucs_seed random` restores per-run variation. |
| VR batch size | **2** | Batch 1 is never fastest: 2.975 s vs 3.488 s on a 45 s clip, 16.364 s vs 17.946 s on a 195 s one. Batch 4 edges it on long inputs but costs another 2.5 GB. |
| Roformer/MDXC precision | **fp32** | bf16 measures no faster on current hardware (within a 0.04% noise floor) and costs ~78 dB SNR, so fp32 is the better default. `--precision bf16` remains available. |
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
| `MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE` | `off` | `all`, `glu_only`, `gelu_only` or `off`. Selects the custom fused modules when the model is loaded. |
| `MLX_AUDIO_SEPARATOR_DEMUCS_DCONV_COMPILE` | `0` | Enables separate DConv block compilation. The standalone `DEMUCS_MLX_COMPILE_DCONV` spelling is accepted as a fallback. |
| `MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED` | off | Caps reduction threadgroups at 256 and disables fused GroupNorm. |
| `MLX_AUDIO_SEPARATOR_DETERMINISTIC_ACCUMULATION` | off | Forces ordered overlap-add accumulation. Slower; for reproducibility checks. |
| `MLX_AUDIO_SEPARATOR_DEMUCS_STRICT_EVAL` | off | Inserts `mx.eval` barriers through the forward pass. |
| `MLX_AUDIO_SEPARATOR_DEMUCS_ISTFT_ALLOW_FUSED` | on | Fused iSTFT. Measured effect on output is negligible. |
| `MLX_AUDIO_SEPARATOR_DEMUCS_WIENER_USE_VMAP` | on | vmap-parallel Wiener filtering. `hdemucs`-family only. |
| `MLX_AUDIO_SEPARATOR_UNSAFE_SLICE_ADD` | off | Restores the legacy `at[].add()` accumulator. **Produces wrong output on MLX < 0.32**; benchmarking only. |

### Roformer / MDXC

| Variable | Default | Effect |
|---|---|---|
| `MLX_ENABLE_AMP` | `0` | bf16 transformer stack, read by BS-Roformer. Off by default: it measures no faster on current hardware and costs ~78 dB SNR. If half precision is revisited, fp16 is the better choice — 78.5 dB against 59.1 dB from fp32 at the same matmul cost. |
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

## Demucs improvements ported on 2026-09-29

The embedded default `htdemucs` path now uses fast GroupNorm and phased
waveform/frequency decoders. A same-process benchmark loaded the old and new
module variants independently, warmed each path, and alternated their order
on identical synthetic stereo inputs (`mq-19a2bb`) on an M4 Max with MLX
0.32.2. The existing whole-model compile remained enabled in both paths.
The benchmark used this separator's **then-default two shifts**, 25% overlap, batch
two, and materialized all four stems; it excluded model loading and file I/O.

| Input | Previous | New | Less wall time | More audio per second | Minimum stem SNR | Peak error |
|---:|---:|---:|---:|---:|---:|---:|
| 30 s | 1.037 s | **0.931 s** | **10.2%** | **11.4%** | 103.15 dB | 1.10e-7 |
| 60 s | 1.957 s | **1.742 s** | **11.0%** | **12.3%** | 104.81 dB | 7.45e-8 |

These are means of two warmed pairs. Other processes used the GPU during
15.3% of this job's wall time; no other MetalQ job ran concurrently. Both
pairs at each length agreed closely. A one-shift paired run (`mq-7ae170`)
measured 10.7% and 18.5% less wall time at 30 and 60 seconds, with larger
pair-to-pair variation and 27.8% other-process GPU activity. Compare paths
within a job, not absolute times across jobs.
The six-source checkpoint also converted through this repository's restricted
loader and wrote six 44,100-frame outputs from a one-second smoke input
(`mq-d40bc5`, `mq-31ba30`).

Separate DConv compilation did not help this copy of Demucs. With the
whole-model forward compiled, eager versus separately compiled DConv was a
timing tie and output was exactly equal (`mq-d599f4`). With whole-model
compilation disabled, separate DConv was 2.9% slower at 30 seconds and 5.8%
slower at 60 seconds (`mq-f7666b`). It is therefore opt-in here.

For `htdemucs_ft`, `--single_stem vocals`, `--single_stem drums`,
`--single_stem bass`, or `--single_stem other` now runs only its contributing
specialist model. Previously the separator computed all four and discarded
three after inference. A real-model alternating benchmark (`mq-b67a01`)
with one shift measured **3.83–4.26×** faster vocals-only inference at 30
and 60 seconds;
the selected samples matched the full bag exactly. This changes the requested
output to one stem and is separate from the default four-stem gain. The job
had 37.6% other-process GPU activity, so its absolute times are not directly
comparable to the default-model table.

The standalone project's ANE option was not carried over: its current offload
is a throughput tie, and its compiled asset is validated against that
project's separate weight cache. The larger ANE waveform stages did not pass
stem-fidelity checks. The spectral frontend was also left as is; its measured
share of a segment was about 1.7% in the standalone project.

### Overlap-add evaluation boundary and standard shift count

The old overlap-add comment said "every update," but its `maybe_eval()` call
was after the loop over a batch. Its measured `eval_flush_interval=1` therefore
meant one evaluation **per batch of two**, not per segment update. Job
`mq-440839` compared that actual boundary with true per-update evaluation on
an M4 Max with MLX 0.32.2, using the current compiled HTDemucs path, one shift,
batch two and 25% overlap. Three
alternating pairs measured per update **3.4% slower at 30 s** and **4.3% faster
at 60 s**; all stems were bit-identical. A crossover sweep (`mq-e38e87`)
measured per update 1.1% slower at 40 s, 1.3% slower at 45 s, and 2.1% faster
at 50 s. The default now evaluates per batch for up to eight segment offsets
and per update from nine offsets, which starts around 50 seconds at the
default segment, overlap and batch-two settings. Other batch sizes keep the
per-batch boundary. Deterministic accumulation continues to force per-update
evaluation at any length.

The main CLI and `DemucsSeparator` wrapper now default to **one shift**, as
the embedded Demucs API and standalone project already did. Older benchmark
figures above that explicitly used two shifts remain valid for that setting;
the new default spends about one model pass instead of two. A fixed seed is
still used for reproducibility.

Reproduce the evaluation-boundary comparison through MetalQ:

```bash
metalq submit -w --no-env-sync --queue-exclusive -n demucs-ola-boundary -- python scripts/perf/demucs_overlap_eval.py
```

Reproduce the paired comparisons and fine-tuned stem benchmark through MetalQ:

```bash
metalq submit -w --no-env-sync --queue-exclusive -n demucs-port-paired -- python scripts/perf/demucs_port_paired.py --shifts 2
metalq submit -w --no-env-sync --queue-exclusive -n demucs-dconv-paired -- python scripts/perf/demucs_port_paired.py --comparison dconv
metalq submit -w --no-env-sync --queue-exclusive -n demucs-ft-stem -- python scripts/perf/bench_ft_single_stem.py
```

### BS-Roformer already has its compile win

`MLX_ENABLE_COMPILE=1` (set by the Roformer loader) compiles
`_forward_transformers`, and that is worth **23%**: turning it off measures
9.038 s against a 7.343/7.352/7.351 s control at a 0.12% noise floor.
Extending compilation to the whole `__call__` adds **0.2%** — nothing.

This is the opposite of Demucs, which had no compiled graph at all until 0.1.8
and gained 15.9% from one. If you are looking for headroom, look where nothing
is compiled, not where something already is.

## Where converted Demucs weights live

`~/.cache/mlx-audio-separator/demucs`, or wherever
`MLX_AUDIO_SEPARATOR_DEMUCS_CACHE_DIR` points. A cache left in the older
`~/.cache/demucs-mlx` by a release before 0.1.12 is still read, so upgrading
costs no reconversion. Nothing is written there any more: the demucs-mlx package
keeps its own cache in that directory under a different schema, and with both
packages installed each used to rebuild what the other had just written.

## Where downloaded models live

`~/.cache/mlx-audio-separator/models`, or `--model_file_dir` /
`AUDIO_SEPARATOR_MODEL_DIR`. This was `/tmp/audio-separator-models/` before
0.1.13: macOS clears `/tmp` on boot, so every reboot cost a re-download, and
`/tmp` is world-writable, which is the wrong place for a cache the downloader
reuses without checking. Files already in the old directory — including the
converted `.safetensors` the loaders write beside a checkpoint — are hard-linked
across on first use, so nothing is downloaded or converted twice and nothing
extra is stored.

## Measuring a change here

Use `scripts/perf/ab_harness.py`. It runs one process per arm, rotates the arm
order, and computes a noise floor from two same-config control arms; anything
smaller than that floor has not been shown to do anything. On a loaded laptop it
reports a ~40% floor and will happily show you five "wins" of 46-56%; on an idle
M4 the same run gives 0.4-1.0%. If the floor comes back above a few percent the
measurement is void — move machines rather than reading the table.
