#!/usr/bin/env python3
"""Compare baseline vs candidate latency across a corpus and model set.

The script runs end-to-end separation timings (decode -> inference -> write) for
the same corpus/models under two configurations and emits:
1) machine-readable JSON with raw runs and medians
2) Markdown summary with pass/fail against configured criteria

Config file format (JSON):
{
  "models": ["htdemucs.yaml", "UVR-MDX-NET-Inst_HQ_3.onnx"],
  "warmup": 1,
  "repeats": 3,
  "separator": {
    "model_file_dir": "/tmp/audio-separator-models",
    "output_format": "WAV",
    "performance_params": {
      "speed_mode": "default"
    }
  }
}
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ARCH_TARGETED_IMPROVEMENT = {"Demucs", "MDXC"}
ARCH_REGRESSION_GUARD = {"MDX", "VR"}


@dataclass
class RunConfig:
    label: str
    separator_kwargs: dict[str, Any]
    warmup: int
    repeats: int


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a JSON object.")
    return data


def _load_corpus(path: Path) -> list[str]:
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
    return out


def _build_run_config(label: str, payload: dict[str, Any]) -> RunConfig:
    sep_kwargs = payload.get("separator", payload.get("separator_kwargs", {}))
    if not isinstance(sep_kwargs, dict):
        raise ValueError(f"{label} config must include object field 'separator'.")

    warmup = int(payload.get("warmup", 1))
    repeats = int(payload.get("repeats", 3))
    if warmup < 0:
        raise ValueError(f"{label}.warmup must be >= 0")
    if repeats <= 0:
        raise ValueError(f"{label}.repeats must be >= 1")

    return RunConfig(
        label=label,
        separator_kwargs=dict(sep_kwargs),
        warmup=warmup,
        repeats=repeats,
    )


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


def _timed_separate(sep, audio_path: str, repeats: int) -> list[float]:
    run_times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sep.separate(audio_path)
        run_times.append(time.perf_counter() - t0)
    return run_times


def run_config(
    cfg: RunConfig,
    corpus: list[str],
    models: list[str],
    workspace_temp: Path,
    model_file_dir_override: str | None = None,
) -> dict[str, dict[str, Any]]:
    from mlx_audio_separator.core import Separator

    results: dict[str, dict[str, Any]] = {}
    run_root = workspace_temp / cfg.label
    run_root.mkdir(parents=True, exist_ok=True)

    for model in models:
        model_out_dir = run_root / model.replace("/", "_")
        model_out_dir.mkdir(parents=True, exist_ok=True)

        model_result: dict[str, Any] = {
            "status": "error",
            "arch": None,
            "load_s": 0.0,
            "median_s": 0.0,
            "runs_s": [],
            "per_file": {},
            "error": None,
        }

        sep_kwargs = dict(cfg.separator_kwargs)
        sep_kwargs.setdefault("output_format", "WAV")
        sep_kwargs["output_dir"] = str(model_out_dir)
        if model_file_dir_override:
            sep_kwargs["model_file_dir"] = model_file_dir_override

        try:
            sep = Separator(**sep_kwargs)

            t0 = time.perf_counter()
            sep.load_model(model_filename=model)
            model_result["load_s"] = round(time.perf_counter() - t0, 6)
            model_result["arch"] = sep.model_type

            all_runs: list[float] = []
            for audio_path in corpus:
                file_entry: dict[str, Any] = {"warmup_runs_s": [], "runs_s": [], "median_s": 0.0}

                for _ in range(cfg.warmup):
                    t0 = time.perf_counter()
                    sep.separate(audio_path)
                    file_entry["warmup_runs_s"].append(round(time.perf_counter() - t0, 6))

                timed_runs = _timed_separate(sep, audio_path, cfg.repeats)
                timed_runs = [round(v, 6) for v in timed_runs]
                file_entry["runs_s"] = timed_runs
                file_entry["median_s"] = round(median(timed_runs), 6)

                model_result["per_file"][audio_path] = file_entry
                all_runs.extend(timed_runs)

            model_result["runs_s"] = all_runs
            model_result["median_s"] = round(median(all_runs), 6) if all_runs else 0.0
            model_result["status"] = "ok"
        except Exception as exc:  # pragma: no cover - runtime-side failure path
            model_result["status"] = f"error: {exc}"
            model_result["error"] = str(exc)

        results[model] = model_result

    return results


def compare_results(
    models: list[str],
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    target_improvement_demucs_mdxc: float,
    max_regression_mdx_vr: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in models:
        b = baseline.get(model, {})
        c = candidate.get(model, {})
        arch = c.get("arch") or b.get("arch") or "unknown"
        b_status = b.get("status", "missing")
        c_status = c.get("status", "missing")
        b_med = float(b.get("median_s", 0.0) or 0.0)
        c_med = float(c.get("median_s", 0.0) or 0.0)

        delta_pct: float | None = None
        if b_status == "ok" and c_status == "ok" and b_med > 0:
            delta_pct = ((c_med - b_med) / b_med) * 100.0

        criterion = "n/a"
        passed = False
        if delta_pct is not None:
            if arch in ARCH_TARGETED_IMPROVEMENT:
                criterion = f"<= -{target_improvement_demucs_mdxc:.1f}%"
                passed = delta_pct <= -float(target_improvement_demucs_mdxc)
            elif arch in ARCH_REGRESSION_GUARD:
                criterion = f"<= +{max_regression_mdx_vr:.1f}%"
                passed = delta_pct <= float(max_regression_mdx_vr)
            else:
                criterion = "informational"
                passed = True

        if b_status != "ok" or c_status != "ok":
            passed = False

        out.append(
            {
                "model": model,
                "arch": arch,
                "baseline_status": b_status,
                "candidate_status": c_status,
                "baseline_median_s": round(b_med, 6),
                "candidate_median_s": round(c_med, 6),
                "delta_pct": round(delta_pct, 3) if delta_pct is not None else None,
                "criterion": criterion,
                "pass": passed,
            }
        )
    return out


def _write_markdown(path: Path, summary_rows: list[dict[str, Any]], baseline_label: str, candidate_label: str):
    lines = [
        "# Latency Comparison",
        "",
        f"| Model | Arch | {baseline_label} (s) | {candidate_label} (s) | Delta % | Criterion | Pass |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in summary_rows:
        delta = "n/a" if row["delta_pct"] is None else f"{row['delta_pct']:+.2f}%"
        lines.append(
            f"| `{row['model']}` | {row['arch']} | {row['baseline_median_s']:.3f} | "
            f"{row['candidate_median_s']:.3f} | {delta} | {row['criterion']} | "
            f"{'yes' if row['pass'] else 'no'} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs candidate separation latency.")
    parser.add_argument("--corpus-file", required=True, help="Text file with one audio path per line.")
    parser.add_argument("--baseline-config", required=True, help="Baseline JSON config path.")
    parser.add_argument("--candidate-config", required=True, help="Candidate JSON config path.")
    parser.add_argument("--models", default=None, help="Comma-separated model filenames (overrides config files).")
    parser.add_argument("--model-file-dir", default=None, help="Override model_file_dir for both runs.")
    parser.add_argument("--output-json", default=None, help="Output JSON path.")
    parser.add_argument("--output-markdown", default=None, help="Output Markdown path.")
    parser.add_argument("--target-improvement-demucs-mdxc", type=float, default=20.0, help="Required improvement percentage.")
    parser.add_argument("--max-regression-mdx-vr", type=float, default=5.0, help="Allowed regression percentage.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary benchmark output directories.")
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_payload = _load_json(Path(args.baseline_config))
    candidate_payload = _load_json(Path(args.candidate_config))
    corpus = _load_corpus(Path(args.corpus_file))
    models = _resolve_models(baseline_payload, candidate_payload, args.models)

    baseline_cfg = _build_run_config("baseline", baseline_payload)
    candidate_cfg = _build_run_config("candidate", candidate_payload)

    run_temp = Path(tempfile.mkdtemp(prefix="latency-compare-"))
    try:
        baseline_results = run_config(
            baseline_cfg,
            corpus=corpus,
            models=models,
            workspace_temp=run_temp,
            model_file_dir_override=args.model_file_dir,
        )
        candidate_results = run_config(
            candidate_cfg,
            corpus=corpus,
            models=models,
            workspace_temp=run_temp,
            model_file_dir_override=args.model_file_dir,
        )
    finally:
        if not args.keep_temp:
            for root, dirs, files in os.walk(run_temp, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except OSError:
                        pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except OSError:
                        pass
            try:
                os.rmdir(run_temp)
            except OSError:
                pass

    summary_rows = compare_results(
        models=models,
        baseline=baseline_results,
        candidate=candidate_results,
        target_improvement_demucs_mdxc=args.target_improvement_demucs_mdxc,
        max_regression_mdx_vr=args.max_regression_mdx_vr,
    )

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_json = Path(args.output_json) if args.output_json else Path(f"/Users/sam/Code/mlx-audio-separator/perf_reports/compare_latency_{now}.json")
    output_md = Path(args.output_markdown) if args.output_markdown else output_json.with_suffix(".md")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus,
        "models": models,
        "baseline": {
            "config": baseline_payload,
            "results": baseline_results,
        },
        "candidate": {
            "config": candidate_payload,
            "results": candidate_results,
        },
        "summary": summary_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_md, summary_rows, baseline_cfg.label, candidate_cfg.label)

    total = len(summary_rows)
    passed = sum(1 for row in summary_rows if row["pass"])
    failed = total - passed
    print(f"Compared {total} models: {passed} pass, {failed} fail")
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")


if __name__ == "__main__":
    main()

