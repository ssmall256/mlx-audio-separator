# Reproducibility Guide (No Audio Redistribution)

This project supports reproducible performance and quality claims without shipping copyrighted audio.

## What To Publish

1. Dataset metadata:
- Dataset name, version, split(s), and license terms.
- Example: `MUSDB18-HQ`, `test split`.

2. Track manifest (no audio files):
- Track IDs / filenames included in your evaluation.
- Do not include raw audio in this repository.

3. Reference stem manifest format:
- JSON map of `mix_path -> {stem_name: reference_stem_path}`.
- Paths are local in your environment and are not required to exist for external readers.

4. Reproduction commands:
- Command line used to generate the final report(s).
- Include all flags (especially parity/quality/latency settings).

5. Runtime and revision metadata:
- Git commit SHA / branch / dirty state.
- Python + package versions.
- Model file hashes.

6. Result artifacts:
- Report JSON and Markdown outputs from `scripts/perf/run_optimization_report.py`.
7. Third-party attribution:
- Include third-party notices and upstream license references for adapted code paths.
- In this repo, see `/Users/sam/Code/mlx-audio-separator/THIRD_PARTY_NOTICES.md`.

## Step 1: Prepare Corpus List

Create a text file with one mix path per line:

```text
/abs/path/to/mix_001.wav
/abs/path/to/mix_002.wav
...
```

## Step 2: Build Reference Manifest Scaffold

```bash
python /Users/sam/Code/mlx-audio-separator/scripts/perf/generate_reference_manifest.py \
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
python /Users/sam/Code/mlx-audio-separator/scripts/perf/run_optimization_report.py \
  --corpus-file /path/to/corpus.txt \
  --baseline-config /path/to/baseline.json \
  --candidate-config /path/to/candidate.json \
  --models htdemucs.yaml,model_bs_roformer_ep_317_sdr_12.9755.ckpt,UVR-MDX-NET-Inst_HQ_3.onnx \
  --quality-reference-manifest /path/to/reference_manifest.json \
  --parity-strict-demucs \
  --parity-max-files 0 \
  --python-mps-latency \
  --output-json /path/to/optimization_report.json \
  --output-markdown /path/to/optimization_report.md
```

## Recommended Evaluation Sizes

- Minimum publishable claim: `20-25` songs.
- Strong reviewer confidence: `~50` songs.
- High-confidence release claim: `60-70` songs, including out-of-domain tracks.

## Reviewer Packet Checklist

- [ ] Dataset/split/license statement.
- [ ] Track list (IDs only).
- [ ] Baseline/candidate config files.
- [ ] Reference manifest schema and generation method.
- [ ] Exact report commands.
- [ ] Report JSON + Markdown artifacts.
- [ ] Commit SHA and model hashes.
