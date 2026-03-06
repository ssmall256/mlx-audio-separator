# Follow-Up Issue: Roformer Kernel Fusion R&D

## Objective

Investigate additional no-drift runtime reductions for `BS-Roformer-SW.ckpt`
via deeper kernel fusion in MDXC/Roformer inference.

## Scope

- Explore custom overlap-add fusion opportunities in MDXC chunk accumulation.
- Explore mask-estimator path fusion where MLX compiler currently leaves
  repeated Python/module boundaries.
- Keep changes opt-in until evidence is stable.

## Non-Goals

- No default behavior changes in this track.
- No quality-degrading context reductions (for example, forcing
  `override_model_segment_size=True` on full-length audio).

## Success Gate

Proceed only if the candidate demonstrates:

1. `> 5%` additional median end-to-end speedup over `latency_safe_v3`
   on `BS-Roformer-SW.ckpt` FLAC workflow.
2. Output parity within strict tolerance (target `max rel L2 <= 1e-6`).
3. No regressions in existing MDXC correctness tests.

## Suggested Validation Command

Use `scripts/perf/compare_latency.py` with baseline/candidate configs and
equivalence checks enabled to generate reproducible evidence.

## Pass 1 Outcome (Roformer + Demucs Priority)

Date: 2026-03-05

- Promoted:
  - none in pass 1 (kept default-off due measurement instability)
- Parked:
  - `experimental_mdxc_defer_batch_eval` (R1) as promising but unproven
  - `experimental_mdxc_precompute_gather_idx` (R2)
  - `R1+R2` combined path
  - Demucs D2 (`.item()` sync removal) pending a measurable, isolated gain

### Roformer (MDXC) Evidence

- Single-file (`/tmp/f.wav`) order-balanced result for R1 (signal only):
  - baseline median `total_s`: `14.4171`
  - candidate median `total_s`: `13.8888`
  - delta: `-3.66%`
- Corpus (`/tmp/f.wav`, `/tmp/f_short.wav`) order-balanced result for R1 (signal only):
  - baseline median `total_s`: `10.1896`
  - candidate median `total_s`: `8.1249`
  - delta: `-20.26%`
- Equivalence:
  - strict threshold `1e-6`
  - `max_rel_l2 = 0.0` (pass)
- Control noise check (same-config baseline vs baseline) showed large thermal/order drift:
  - Roformer `total_s` delta: `+14.12%`
  - Demucs `total_s` delta: `+10.18%`
  - Conclusion: this session cannot support a trustworthy `>=3%` promotion decision.

Artifacts:

- `perf_reports/roformer_demucs/pass1/r1/roformer_single_compare.json`
- `perf_reports/roformer_demucs/pass1/r1/roformer_single_compare_swapped.json`
- `perf_reports/roformer_demucs/pass1/r1/roformer_corpus_compare.json`
- `perf_reports/roformer_demucs/pass1/r1/roformer_corpus_compare_swapped.json`
- `perf_reports/roformer_demucs/pass1/r1/roformer_single_equiv.json`

### Demucs Notes

- D1 compatibility fix (`mx.angle` fallback to `mx.arctan2(imag, real)`) is kept.
- Demucs cross-target checks were strongly order/thermal sensitive in this session.
- D2 is not promoted in pass 1.

## Pass 2 Outcome (#2 -> #3 -> #5)

Date: 2026-03-05

- #2 grouped packing cache: parity pass, ABBA regression (`+4.17%`), parked.
- #3 chunk gather batching (`separate_audio_chunked`): parity pass, ABBA regression (`+0.71%`), parked.
- #5 Demucs Wiener prealloc output: parity pass, ABBA regression (`+0.48%`), parked.

Detailed artifacts:
- `perf_reports/roformer_demucs/pass2/summary.md`

## Pass 3 Outcome (#4 -> #6 -> #8)

Date: 2026-03-05

- #4 (Roformer overlap-add Metal kernel tuning): parity pass, ABBA regression (`+10.33%` on Roformer), parked.
- #6 (Demucs apply concat batching): parity pass, large apparent speedup on Demucs but implausible Roformer shift on untouched path (`-30.94%`), parked as thermally contaminated.
- #8 (Demucs GN+GLU multigroup coverage): parity pass, large apparent speedup on Demucs but non-zero Roformer shift on untouched path (`-6.65%`), parked as thermally contaminated.

Detailed artifacts:
- `perf_reports/roformer_demucs/pass3/summary.md`
