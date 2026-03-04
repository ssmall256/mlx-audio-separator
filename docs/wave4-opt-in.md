# Wave 4 Plan (Opt-In Only)

Wave 4 is performance work after release stabilization. All new behavior remains opt-in until promotion criteria are met.

## Experimental Controls and Status

| Control | Scope | Status | Notes |
|---|---|---|---|
| `--speed_mode latency_safe_v2` | Runtime profile | Active (opt-in) | Experimental release profile; defaults unchanged. |
| `--speed_mode latency_safe_v3` | Runtime profile | Active (opt-in) | FLAC-focused no-drift profile (`deferred` cache clears + `write_workers=2`). |
| `--auto_tune_batch` | Runtime profile | Active (opt-in) | Per-arch auto-tuning probe path. |
| `--experimental_vectorized_chunking` | MDXC | Active (opt-in) | Experimental MDXC chunk scheduler path. |
| `--experimental_roformer_fast_norm` | Roformer (MDXC path) | Active (opt-in) | Uses `mx.fast.rms_norm` in Roformer `L2Norm` blocks. |
| `--experimental_roformer_grouped_band_split` | Roformer (MDXC path) | Active (opt-in) | Grouped same-width band projection path in Roformer band split. |
| `--experimental_roformer_grouped_mask_estimator` | Roformer (MDXC path) | Active (opt-in) | Grouped same-width mask-estimator path for per-band MLP blocks. |
| `--experimental_roformer_fused_overlap_add` | Roformer (MDXC path) | Active (opt-in) | Fused overlap-add accumulation path with Metal-kernel fallback. |
| `--experimental_mlx_stream_pipeline` | MDXC | Active (opt-in) | MLX stream scheduling path for chunk inference. |
| `--experimental_roformer_compile_fullgraph` | Roformer (MDXC path) | Active (opt-in) | Shape-keyed full-graph compile cache for Roformer forward model. |
| `--experimental_flac_fast_write` | All arches (FLAC output) | Active (opt-in) | Requests FLAC fast-write mode when backend support is available. |
| `--experimental_compile_model_forward` | MDX23C (MDXC path) | Active (opt-in) | Compiled forward path where supported. |
| `MLX_AUDIO_SEPARATOR_GN_GLU_MULTIGROUP=1` | Demucs fused GroupNorm+GLU | Active (opt-in, env) | Enables multigroup hybrid GN+GLU fast path (`num_groups > 1`) for Demucs experiments. |
| `--experimental_compile_shapeless` | Roformer compile path | Inactive (compat-only) | Accepted for compatibility; currently inactive by policy. |
| `--experimental_roformer_static_compiled_demix` | Roformer static demix | Inactive (compat-only) | Accepted for compatibility; currently inactive by policy. |

## Guardrails

1. No default behavior changes in Wave 4.
2. Output-equivalent path remains available and documented.
3. Every optimization must be measurable and reversible.

## Workstreams

## 1) Adaptive Runtime Profiles

1. Add optional profile variants under `performance_params` and CLI:
   - `speed_mode=latency_safe_v2` (opt-in)
   - `auto_tune_batch=true` with per-arch candidate constraints
   - `experimental_vectorized_chunking=true` (MDXC-only experimental data path)
2. Keep current defaults unchanged.

Acceptance:

1. Median latency improvement vs current release on target corpus.
2. No architecture with >5% regression unless explicitly documented and accepted.

## 2) Demucs Deep Optimization (Deterministic Path Intact)

1. Investigate and remove avoidable eval barriers in Demucs inference path.
2. Preserve strict deterministic controls for equivalence checks.
3. Keep strict Demucs equivalence gating opt-in until parity evidence is stable.

Current status:

1. Correctness-first hybrid GN+GLU path is implemented (fp32 GN+affine + fp32 GLU + final cast).
2. Deterministic mode now defaults fused GroupNorm mode to `off` (stable parity path).
3. Multigroup GN+GLU fast path is available only as opt-in via `MLX_AUDIO_SEPARATOR_GN_GLU_MULTIGROUP=1`.

Acceptance:

1. Latency gain on Demucs models with no quality-gate regression.
2. Deterministic equivalence report remains reproducible under strict mode.

## 3) MDXC Chunk Scheduler and Write Pipeline

1. Refine chunk scheduling to reduce tail overhead.
2. Expand async write overlap safely.
3. Preserve output-file contract (all expected stems materialize).

Acceptance:

1. Lower p90/p95 latency on MDXC-heavy corpus.
2. Zero missing/empty-output validation failures.

## 4) Cross-Backend Evidence

1. Re-run MLX vs python-audio-separator (MPS) comparisons with randomized backend order and repeats.
2. Publish only stable, repeatable positive deltas.

Acceptance:

1. Positive or neutral deltas on prioritized model set across repeated runs.
2. Thermal/warmup conditions documented in report metadata.

## Promotion Policy

A Wave 4 feature can move from opt-in to default only after:

1. Two clean benchmark passes on at least two Apple Silicon machines.
2. Equivalence/quality gates pass for affected architectures.
3. Release-readiness gate remains green (`0` failures).

## BS-Roformer-SW Candidate Corpus Manifests

- Quick gate corpus template: `scripts/perf/corpora/bs_roformer_sw_quick.txt` (3 files).
- Full gate corpus template: `scripts/perf/corpora/bs_roformer_sw_full.txt` (12 files).

## Roformer Fast-Norm Evaluation

Use the provided configs to run latency + parity checks before considering default promotion:

```bash
uv run --with torch python scripts/perf/compare_latency.py \
  --corpus-file /tmp/corpus_one.txt \
  --baseline-config scripts/perf/configs/roformer_fast_norm_baseline.json \
  --candidate-config scripts/perf/configs/roformer_fast_norm_candidate.json \
  --model-file-dir /tmp/audio-separator-models \
  --target-improvement-demucs-mdxc 2.0 \
  --equivalence-check \
  --equivalence-threshold-rel-l2 1e-3 \
  --equivalence-max-files 1 \
  --output-json /tmp/roformer_fast_norm_compare.json \
  --output-markdown /tmp/roformer_fast_norm_compare.md
```
