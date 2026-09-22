# BS-Roformer-SW Optimization Program

## Objective

Explore, apply, and measure opt-in no-drift optimizations for
`BS-Roformer-SW.ckpt`, with a strict promotion process relative to the
shipped defaults.

Current decision status: what `latency_safe_v3` used to promote -- `deferred`
cache clearing and async stem writes -- became the default in 0.1.8, and
`speed_mode` is deprecated and ignored. Measure against the defaults; the
baseline configs in `scripts/perf/configs/` now say `speed_mode: "default"`,
which is what every profile resolves to. See `docs/tuning.md`.

## Gating

1. Quick gate:
- Corpus: `scripts/perf/corpora/bs_roformer_sw_quick.txt`
- Improvement target: `>= 3%` median
- Equivalence: `max rel L2 <= 1e-6` on `--equivalence-max-files 1`

2. Full gate:
- Corpus: `scripts/perf/corpora/bs_roformer_sw_full.txt`
- Improvement target: `>= 5%` median
- Equivalence: `max rel L2 <= 1e-6` on full corpus

3. Integration gate:
- Combined promoted profile target: `>= 10%` median total speedup
- Keep defaults unchanged; opt-in only.

## Candidate Configs

- `scripts/perf/configs/bs_roformer_sw_default_baseline.json`
- `scripts/perf/configs/bs_roformer_sw_latency_safe_v3_baseline.json`
- `scripts/perf/configs/bs_roformer_sw_cand_grouped_bandmask.json`
- `scripts/perf/configs/bs_roformer_sw_cand_fused_ola.json`
- `scripts/perf/configs/bs_roformer_sw_cand_stream_pipeline.json`
- `scripts/perf/configs/bs_roformer_sw_cand_compile_fullgraph.json`
- `scripts/perf/configs/bs_roformer_sw_cand_flac_fastwrite.json`

## Promotion Decision Table

| Candidate | Flag(s) | Quick Gate | Full Gate | Decision | Notes |
|---|---|---|---|---|---|
| Grouped band/mask | `experimental_roformer_grouped_band_split`, `experimental_roformer_grouped_mask_estimator` | Fail (`-2.63%` aligned primary, `+2.07%` aligned swapped, both `max rel L2=0.0`) | Skipped | Parked | Earlier apparent gain came from invalid cross-speed-mode comparison. |
| Fused overlap-add | `experimental_roformer_fused_overlap_add` | Fail (`-1.07%` primary, `+0.06%` swapped, `max rel L2=0.0`) | Skipped | Parked | No reliable latency gain. |
| Stream pipeline | `experimental_mlx_stream_pipeline` | Fail (`-0.01%` primary, `+0.06%` swapped, `max rel L2=0.0`) | Skipped | Parked | Neutral to slightly worse. |
| Compile fullgraph | `experimental_roformer_compile_fullgraph` | Fail (`-0.31%` primary, `+0.41%` swapped, `max rel L2=0.0`) | Skipped | Parked | Warm-path benefit not observed. |
| FLAC fast write | `experimental_flac_fast_write` | Fail (`+0.03%` primary, `+0.77%` swapped, `max rel L2=0.0`) | Skipped | Parked | No measurable write-stage win in this implementation. |

## Report Paths

Store candidate reports under:

- `perf_reports/bs_roformer_sw/<candidate>_quick_gate.{json,md}`
- `perf_reports/bs_roformer_sw/<candidate>_full_gate.{json,md}`
- `perf_reports/bs_roformer_sw/integration_profile_candidate.{json,md}`
- Aligned grouped-bandmask thermal checks:
  - `perf_reports/bs_roformer_sw/quick/bs_roformer_sw_cand_grouped_bandmask_quick_thermal_aligned_primary.{json,md}`
  - `perf_reports/bs_roformer_sw/quick/bs_roformer_sw_cand_grouped_bandmask_quick_thermal_aligned_swapped.{json,md}`

## Notes

`compare_latency.py` now records model-level and per-file stage medians:
`decode_s`, `preprocess_s`, `inference_s`, `postprocess_s`, `write_s`, and
`total_s`, plus p95 total.

The benchmark harness guards against accidental baseline/candidate
`speed_mode` mismatch unless `--allow-speed-mode-mismatch` is explicitly set.
Since 0.1.8 every `speed_mode` resolves to the same configuration, so this
guard no longer distinguishes anything; it stays until the option is removed.
