#!/usr/bin/env python3
"""Deterministic output-equivalence check across a corpus and model set."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlx_audio_separator.utils.equivalence import run_equivalence_suite


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a JSON object.")
    return data


def _load_corpus(path: Path, max_files: int) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        audio_path = os.path.abspath(os.path.expanduser(line))
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Corpus entry does not exist: {audio_path}")
        out.append(audio_path)
    if not out:
        raise ValueError(f"Corpus file {path} contains no valid entries.")
    if max_files > 0:
        out = out[:max_files]
    return out


def _resolve_models(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    models_arg: str | None,
) -> list[str]:
    if models_arg:
        models = [x.strip() for x in models_arg.split(",") if x.strip()]
        if not models:
            raise ValueError("--models provided but no valid model names found.")
        return models

    base_models = baseline_payload.get("models")
    cand_models = candidate_payload.get("models")
    if not isinstance(base_models, list) or not all(isinstance(x, str) for x in base_models):
        raise ValueError("Baseline config must include string list 'models' when --models is not set.")
    if not isinstance(cand_models, list) or not all(isinstance(x, str) for x in cand_models):
        raise ValueError("Candidate config must include string list 'models' when --models is not set.")
    if base_models != cand_models:
        raise ValueError(
            "Baseline and candidate 'models' lists must match. "
            "Use --models to override explicitly."
        )
    return [x.strip() for x in base_models if x.strip()]


def _write_markdown(path: Path, payload: dict[str, Any]):
    summary = payload["summary"]
    threshold = payload["threshold_rel_l2"]
    lines = [
        "# Deterministic Equivalence",
        "",
        f"- Threshold (relative L2): `{threshold}`",
        f"- Seed: `{payload['seed']}`",
        f"- Demucs shifts forced to zero: `{payload['demucs_shifts_zero']}`",
        f"- Gated architectures: `{', '.join(payload['gated_arches'])}`",
        "",
        "| Model | Arch | Max rel L2 | Gated | Pass | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['model']}` | {row['arch']} | {row['max_rel_l2']:.8f} | "
            f"{'yes' if row['gated'] else 'no'} | {'yes' if row['pass'] else 'no'} | {row['status']} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic equivalence check for baseline vs candidate configs.")
    parser.add_argument("--corpus-file", required=True, help="Text file with one audio path per line.")
    parser.add_argument("--baseline-config", required=True, help="Baseline JSON config path.")
    parser.add_argument("--candidate-config", required=True, help="Candidate JSON config path.")
    parser.add_argument("--models", default=None, help="Comma-separated model filenames (overrides config files).")
    parser.add_argument("--model-file-dir", default=None, help="Override model_file_dir for both runs.")
    parser.add_argument("--threshold-rel-l2", type=float, default=1e-5, help="Equivalence threshold for relative L2 drift.")
    parser.add_argument("--seed", type=int, default=12345, help="Deterministic seed used for Python/NumPy/MLX.")
    parser.add_argument(
        "--demucs-shifts-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force Demucs shifts=0 during equivalence runs (default: true).",
    )
    parser.add_argument(
        "--strict-demucs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include Demucs in strict pass/fail gating (default: false; Demucs informational).",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Limit corpus files checked (0 = all).")
    parser.add_argument("--output-json", default=None, help="Output JSON path.")
    parser.add_argument("--output-markdown", default=None, help="Output Markdown path.")
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_payload = _load_json(Path(args.baseline_config))
    candidate_payload = _load_json(Path(args.candidate_config))
    corpus = _load_corpus(Path(args.corpus_file), max_files=int(args.max_files))
    models = _resolve_models(baseline_payload, candidate_payload, args.models)

    baseline_separator_kwargs = dict(baseline_payload.get("separator", baseline_payload.get("separator_kwargs", {})))
    candidate_separator_kwargs = dict(candidate_payload.get("separator", candidate_payload.get("separator_kwargs", {})))

    payload = run_equivalence_suite(
        corpus=corpus,
        models=models,
        baseline_separator_kwargs=baseline_separator_kwargs,
        candidate_separator_kwargs=candidate_separator_kwargs,
        threshold_rel_l2=args.threshold_rel_l2,
        seed=args.seed,
        demucs_shifts_zero=args.demucs_shifts_zero,
        model_file_dir_override=args.model_file_dir,
        gated_arches={"Demucs", "MDXC", "MDX", "VR"} if args.strict_demucs else {"MDXC", "MDX", "VR"},
    )
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_json = Path(args.output_json) if args.output_json else Path(f"/Users/sam/Code/mlx-audio-separator/perf_reports/equivalence_{now}.json")
    output_md = Path(args.output_markdown) if args.output_markdown else output_json.with_suffix(".md")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_md, payload)

    total = len(payload["summary"])
    passed = sum(1 for row in payload["summary"] if row["pass"])
    failed = total - passed
    print(f"Compared {total} models: {passed} pass, {failed} fail")
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")


if __name__ == "__main__":
    main()
