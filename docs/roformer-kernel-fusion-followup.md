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
