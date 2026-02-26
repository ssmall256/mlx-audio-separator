# Reproducibility Guide (No Audio Redistribution)

This project supports reproducible performance and quality claims without shipping copyrighted audio.
For a concise product overview, use `README.md`; this document keeps full methodology and reporting details.

## What To Publish

1. Dataset metadata: dataset name, version, split, and license terms (example: `MUSDB18-HQ`, `test` split).
2. Track manifest: track IDs or filenames included in evaluation; do not include audio files in this repository.
3. Reference stem manifest: JSON map of `mix_path -> {stem_name: reference_stem_path}`.
4. Reproduction commands: exact commands used for final report generation, including parity, quality, and latency flags.
5. Runtime and revision metadata: git commit SHA/branch/dirty state, Python/package versions, and model hashes.
6. Result artifacts: report JSON and Markdown outputs from `scripts/perf/run_optimization_report.py`.
7. Third-party attribution: include upstream license notices for adapted code paths (`THIRD_PARTY_NOTICES.md`).

## Minimum Reproducibility Packet

For a minimally credible external review, publish at least:

- Dataset and split statement with license terms.
- Track manifest (`tracks.txt`) with IDs/filenames only.
- Baseline and candidate config files.
- One exact report command line.
- Final report artifacts (`optimization_report.json` and `optimization_report.md`).

## Step 1: Prepare Corpus List

Create a text file with one mix path per line:

```text
/abs/path/to/mix_001.wav
/abs/path/to/mix_002.wav
...
```

## Step 2: Build Reference Manifest Scaffold

```bash
uv run python scripts/perf/generate_reference_manifest.py \
  --corpus-file /path/to/corpus.txt \
  --output-json /path/to/reference_manifest.json \
  --search-dir /path/to/reference_stems \
  --require-existing
```

Notes:
- Stem keys should match model outputs (for Demucs: `vocals`, `drums`, `bass`, `other`).
- If your naming convention differs, pass `--template` multiple times.

## Step 3: Run Unified Optimization Report

```bash
PATH="/usr/local/bin:/opt/homebrew/bin:$PATH" \
uv run --with audio-separator --with onnxruntime python scripts/perf/run_optimization_report.py \
  --corpus-file /path/to/corpus.txt \
  --baseline-config /path/to/baseline.json \
  --candidate-config /path/to/candidate.json \
  --models htdemucs.yaml,model_bs_roformer_ep_317_sdr_12.9755.ckpt,UVR-MDX-NET-Inst_HQ_3.onnx \
  --quality-reference-manifest /path/to/reference_manifest.json \
  --parity-strict-demucs \
  --parity-max-files 0 \
  --python-mps-latency \
  --python-mps-parity \
  --python-mps-parity-max-files 0 \
  --python-mps-parity-threshold-rel-l2 5e-2 \
  --output-json /path/to/optimization_report.json \
  --output-markdown /path/to/optimization_report.md
```

Parity tolerance policy:

- MLX-vs-MLX deterministic parity: strict `relative L2 <= 1e-5`.
- MLX-vs-`audio-separator` parity: `relative L2 <= 5e-2` for cross-runtime comparisons.
- Demucs MLX parity runs should use strict Demucs kernel parity settings (enabled by default in `mlx_vs_pas_parity.py`).

## Recommended Evaluation Sizes

- Initial release evidence (practical): `~12` songs with diversified genres/instrumentation.
- Strong reviewer confidence: `20-25` songs.
- Extended stress evidence (optional): `~50` songs.

## Reviewer Packet Checklist

- [ ] Dataset/split/license statement.
- [ ] Track list (IDs only).
- [ ] Baseline/candidate config files.
- [ ] Reference manifest schema and generation method.
- [ ] Exact report commands.
- [ ] Report JSON + Markdown artifacts.
- [ ] Commit SHA and model hashes.
