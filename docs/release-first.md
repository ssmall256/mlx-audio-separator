# Release-First Playbook

This playbook is the default release path: stabilize and ship reliability first, then iterate on speed behind explicit opt-in controls.

## Goals

1. Ship a release candidate with predictable behavior and trustworthy benchmark reporting.
2. Require objective gates before publishing.
3. Keep new performance work in opt-in mode until evidence is strong across machines.

## Release Gate Inputs

1. Full benchmark JSON from a clean run.
2. Unit test pass in target environment.
3. Optimization report artifacts (JSON/Markdown) for publication-ready evidence.

## Commands

Run tests:

```bash
uv run pytest -q
```

Run full benchmark:

```bash
uv run python -m mlx_audio_separator.utils.cli \
  --benchmark "/path/to/mixture.wav" \
  --model_file_dir /tmp/audio-separator-models \
  --benchmark_skip_download \
  --benchmark_warmup 1 \
  --benchmark_repeats 2 \
  --benchmark_cooldown 5 \
  --output_dir /tmp/bench-release-$(date +%Y%m%d-%H%M%S)
```

Evaluate benchmark gate:

```bash
uv run python scripts/release/check_release_readiness.py \
  --benchmark-json /tmp/bench-release-YYYYMMDD-HHMMSS/benchmark_results.json \
  --require-total-models 152 \
  --max-failures 0 \
  --min-success-rate 1.0 \
  --require-model-ok htdemucs_ft.yaml \
  --require-model-ok model_bs_roformer_ep_317_sdr_12.9755.ckpt \
  --require-model-ok UVR-MDX-NET-Inst_HQ_3.onnx \
  --output-json /tmp/bench-release-YYYYMMDD-HHMMSS/release_gate_summary.json
```

Generate optimization report:

```bash
uv run python scripts/perf/run_optimization_report.py \
  --corpus-file /path/to/corpus.txt \
  --baseline-config /path/to/baseline.json \
  --candidate-config /path/to/candidate.json \
  --models htdemucs_ft.yaml,model_bs_roformer_ep_317_sdr_12.9755.ckpt,UVR-MDX-NET-Inst_HQ_3.onnx \
  --output-json /tmp/optimization_report.json \
  --output-markdown /tmp/optimization_report.md
```

## Go / No-Go

Go only if all are true:

1. `uv run pytest -q` passes.
2. Release readiness gate script exits `0`.
3. Optimization report contains no architecture-level regressions you consider release blockers.

No-Go if any are false. Fix and re-run from benchmark stage.

## RC Then Stable

1. Publish RC to TestPyPI using `.github/workflows/release-testpypi.yml`.
2. Validate smoke install and a short benchmark sanity pass on a second Apple Silicon machine.
3. Publish stable to PyPI using `.github/workflows/release-pypi.yml`.
