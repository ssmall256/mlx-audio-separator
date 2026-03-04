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

from mlx_audio_separator.utils.equivalence import run_equivalence_suite


ARCH_TARGETED_IMPROVEMENT = {"Demucs", "MDXC"}
ARCH_REGRESSION_GUARD = {"MDX", "VR"}
STAGE_KEYS = (
    "decode_s",
    "preprocess_s",
    "inference_s",
    "postprocess_s",
    "write_s",
    "cleanup_s",
    "total_s",
)


@dataclass
class RunConfig:
    label: str
    separator_kwargs: dict[str, Any]
    warmup: int
    repeats: int


def _sleep_if_needed(seconds: float, *, reason: str):
    if seconds <= 0:
        return
    print(f"[cooldown] sleeping {seconds:.1f}s ({reason})")
    time.sleep(seconds)


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


def _extract_speed_mode(payload: dict[str, Any]) -> str:
    sep_kwargs = payload.get("separator", payload.get("separator_kwargs", {}))
    if not isinstance(sep_kwargs, dict):
        return "default"
    performance_params = sep_kwargs.get("performance_params", {})
    if not isinstance(performance_params, dict):
        return "default"
    mode = performance_params.get("speed_mode", "default")
    if mode is None:
        return "default"
    return str(mode)


def _validate_speed_mode_alignment(
    baseline_mode: str,
    candidate_mode: str,
    *,
    allow_mismatch: bool,
):
    if baseline_mode == candidate_mode:
        return
    message = (
        "Baseline/candidate speed_mode mismatch detected: "
        f"baseline={baseline_mode!r}, candidate={candidate_mode!r}."
    )
    if allow_mismatch:
        print(f"[warning] {message} Continuing due to --allow-speed-mode-mismatch.")
        return
    raise ValueError(f"{message} Use --allow-speed-mode-mismatch for intentional cross-mode comparisons.")


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


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(float(v) for v in values)
    idx = int(round((len(sorted_vals) - 1) * 0.95))
    idx = min(max(idx, 0), len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def _timed_separate(sep, audio_path: str, repeats: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sep.separate(audio_path)
        total = float(time.perf_counter() - t0)
        metrics = dict(getattr(sep, "last_perf_metrics", {}) or {})
        stages = {k: float(metrics.get(k, 0.0)) for k in STAGE_KEYS}
        if stages["total_s"] <= 0:
            stages["total_s"] = total
        out.append({"total_s": total, "stages": stages})
    return out


def run_config(
    cfg: RunConfig,
    corpus: list[str],
    models: list[str],
    workspace_temp: Path,
    model_file_dir_override: str | None = None,
    cooldown_seconds_after_file: float = 0.0,
    cooldown_seconds_after_model: float = 0.0,
) -> dict[str, dict[str, Any]]:
    from mlx_audio_separator.core import Separator

    results: dict[str, dict[str, Any]] = {}
    run_root = workspace_temp / cfg.label
    run_root.mkdir(parents=True, exist_ok=True)

    for model_idx, model in enumerate(models):
        model_out_dir = run_root / model.replace("/", "_")
        model_out_dir.mkdir(parents=True, exist_ok=True)

        model_result: dict[str, Any] = {
            "status": "error",
            "arch": None,
            "load_s": 0.0,
            "median_s": 0.0,
            "p95_s": 0.0,
            "stage_medians_s": {k: 0.0 for k in STAGE_KEYS},
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
            stage_runs: dict[str, list[float]] = {k: [] for k in STAGE_KEYS}
            for file_idx, audio_path in enumerate(corpus):
                file_entry: dict[str, Any] = {
                    "warmup_runs_s": [],
                    "runs_s": [],
                    "median_s": 0.0,
                    "p95_s": 0.0,
                    "stage_medians_s": {k: 0.0 for k in STAGE_KEYS},
                }

                for _ in range(cfg.warmup):
                    t0 = time.perf_counter()
                    sep.separate(audio_path)
                    file_entry["warmup_runs_s"].append(round(time.perf_counter() - t0, 6))

                timed_runs = _timed_separate(sep, audio_path, cfg.repeats)
                totals = [round(float(item["total_s"]), 6) for item in timed_runs]
                file_entry["runs_s"] = totals
                file_entry["median_s"] = round(median(totals), 6)
                file_entry["p95_s"] = round(_p95(totals), 6)
                for stage in STAGE_KEYS:
                    stage_vals = [float(item["stages"][stage]) for item in timed_runs]
                    file_entry["stage_medians_s"][stage] = round(median(stage_vals), 6)
                    stage_runs[stage].extend(stage_vals)

                model_result["per_file"][audio_path] = file_entry
                all_runs.extend(totals)
                if file_idx < len(corpus) - 1:
                    _sleep_if_needed(
                        cooldown_seconds_after_file,
                        reason=f"{cfg.label}:{model} next-file",
                    )

            model_result["runs_s"] = all_runs
            model_result["median_s"] = round(median(all_runs), 6) if all_runs else 0.0
            model_result["p95_s"] = round(_p95(all_runs), 6) if all_runs else 0.0
            model_result["stage_medians_s"] = {
                stage: round(median(vals), 6) if vals else 0.0
                for stage, vals in stage_runs.items()
            }
            model_result["status"] = "ok"
        except Exception as exc:  # pragma: no cover - runtime-side failure path
            model_result["status"] = f"error: {exc}"
            model_result["error"] = str(exc)

        results[model] = model_result
        if model_idx < len(models) - 1:
            _sleep_if_needed(
                cooldown_seconds_after_model,
                reason=f"{cfg.label}:{model} next-model",
            )

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
        b_p95 = float(b.get("p95_s", 0.0) or 0.0)
        c_p95 = float(c.get("p95_s", 0.0) or 0.0)

        delta_pct: float | None = None
        p95_delta_pct: float | None = None
        if b_status == "ok" and c_status == "ok" and b_med > 0:
            delta_pct = ((c_med - b_med) / b_med) * 100.0
        if b_status == "ok" and c_status == "ok" and b_p95 > 0:
            p95_delta_pct = ((c_p95 - b_p95) / b_p95) * 100.0

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
                "baseline_p95_s": round(b_p95, 6),
                "candidate_p95_s": round(c_p95, 6),
                "delta_pct": round(delta_pct, 3) if delta_pct is not None else None,
                "p95_delta_pct": round(p95_delta_pct, 3) if p95_delta_pct is not None else None,
                "baseline_stage_medians_s": b.get("stage_medians_s", {}),
                "candidate_stage_medians_s": c.get("stage_medians_s", {}),
                "criterion": criterion,
                "pass": passed,
            }
        )
    return out


def _write_markdown(
    path: Path,
    summary_rows: list[dict[str, Any]],
    baseline_label: str,
    candidate_label: str,
    equivalence_summary: list[dict[str, Any]] | None = None,
    equivalence_threshold_rel_l2: float | None = None,
    equivalence_gated_arches: list[str] | None = None,
):
    lines = [
        "# Latency Comparison",
        "",
        f"| Model | Arch | {baseline_label} median (s) | {candidate_label} median (s) | Delta % | "
        f"{baseline_label} p95 (s) | {candidate_label} p95 (s) | p95 Delta % | Criterion | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in summary_rows:
        delta = "n/a" if row["delta_pct"] is None else f"{row['delta_pct']:+.2f}%"
        p95_delta = "n/a" if row["p95_delta_pct"] is None else f"{row['p95_delta_pct']:+.2f}%"
        lines.append(
            f"| `{row['model']}` | {row['arch']} | {row['baseline_median_s']:.3f} | "
            f"{row['candidate_median_s']:.3f} | {delta} | "
            f"{row['baseline_p95_s']:.3f} | {row['candidate_p95_s']:.3f} | {p95_delta} | "
            f"{row['criterion']} | "
            f"{'yes' if row['pass'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Stage Medians (Seconds)",
            "",
            "| Model | Arch | Stage | Baseline | Candidate | Delta % |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        b_stage = row.get("baseline_stage_medians_s", {}) or {}
        c_stage = row.get("candidate_stage_medians_s", {}) or {}
        for stage in STAGE_KEYS:
            b_val = float(b_stage.get(stage, 0.0) or 0.0)
            c_val = float(c_stage.get(stage, 0.0) or 0.0)
            if b_val > 0:
                d = ((c_val - b_val) / b_val) * 100.0
                d_fmt = f"{d:+.2f}%"
            else:
                d_fmt = "n/a"
            lines.append(
                f"| `{row['model']}` | {row['arch']} | `{stage}` | {b_val:.3f} | {c_val:.3f} | {d_fmt} |"
            )

    if equivalence_summary:
        lines.extend(
            [
                "",
                "## Deterministic Equivalence",
                "",
                f"- Threshold (relative L2): `{equivalence_threshold_rel_l2}`",
                f"- Gated architectures: `{', '.join(equivalence_gated_arches or [])}`",
                "",
                "| Model | Arch | Max rel L2 | Gated | Pass | Status |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in equivalence_summary:
            lines.append(
                f"| `{row['model']}` | {row['arch']} | {row['max_rel_l2']:.8f} | "
                f"{'yes' if row['gated'] else 'no'} | {'yes' if row['pass'] else 'no'} | {row['status']} |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs candidate separation latency.")
    parser.add_argument("--corpus-file", required=True, help="Text file with one audio path per line.")
    parser.add_argument("--baseline-config", required=True, help="Baseline JSON config path.")
    parser.add_argument("--candidate-config", required=True, help="Candidate JSON config path.")
    parser.add_argument("--models", default=None, help="Comma-separated model filenames (overrides config files).")
    parser.add_argument("--model-file-dir", default=None, help="Override model_file_dir for both runs.")
    parser.add_argument(
        "--allow-speed-mode-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow baseline/candidate speed_mode mismatch (default: false).",
    )
    parser.add_argument("--warmup-override", type=int, default=None, help="Override warmup count for both runs.")
    parser.add_argument("--repeats-override", type=int, default=None, help="Override repeat count for both runs.")
    parser.add_argument("--output-json", default=None, help="Output JSON path.")
    parser.add_argument("--output-markdown", default=None, help="Output Markdown path.")
    parser.add_argument("--target-improvement-demucs-mdxc", type=float, default=20.0, help="Required improvement percentage.")
    parser.add_argument("--max-regression-mdx-vr", type=float, default=5.0, help="Allowed regression percentage.")
    parser.add_argument("--equivalence-check", action="store_true", help="Run deterministic output-equivalence checks.")
    parser.add_argument("--equivalence-threshold-rel-l2", type=float, default=1e-5, help="Relative L2 threshold for equivalence.")
    parser.add_argument("--equivalence-seed", type=int, default=12345, help="Deterministic seed for equivalence checks.")
    parser.add_argument("--equivalence-max-files", type=int, default=1, help="Number of corpus files for equivalence (0=all).")
    parser.add_argument(
        "--equivalence-strict-demucs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include Demucs in strict equivalence pass/fail gating (default: false).",
    )
    parser.add_argument(
        "--equivalence-demucs-shifts-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force Demucs shifts=0 in equivalence runs (default: true).",
    )
    parser.add_argument(
        "--cooldown-seconds-after-file",
        type=float,
        default=0.0,
        help="Optional sleep between corpus files within a config/model run.",
    )
    parser.add_argument(
        "--cooldown-seconds-after-model",
        type=float,
        default=0.0,
        help="Optional sleep between models within a config run.",
    )
    parser.add_argument(
        "--cooldown-seconds-between-configs",
        type=float,
        default=0.0,
        help="Optional sleep between baseline and candidate runs.",
    )
    parser.add_argument(
        "--cooldown-seconds-before-equivalence",
        type=float,
        default=0.0,
        help="Optional sleep before equivalence checks.",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary benchmark output directories.")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    baseline_payload = _load_json(Path(args.baseline_config))
    candidate_payload = _load_json(Path(args.candidate_config))
    corpus = _load_corpus(Path(args.corpus_file))
    models = _resolve_models(baseline_payload, candidate_payload, args.models)

    baseline_speed_mode = _extract_speed_mode(baseline_payload)
    candidate_speed_mode = _extract_speed_mode(candidate_payload)
    _validate_speed_mode_alignment(
        baseline_speed_mode,
        candidate_speed_mode,
        allow_mismatch=bool(args.allow_speed_mode_mismatch),
    )

    baseline_cfg = _build_run_config("baseline", baseline_payload)
    candidate_cfg = _build_run_config("candidate", candidate_payload)
    if args.warmup_override is not None:
        if int(args.warmup_override) < 0:
            raise ValueError("--warmup-override must be >= 0")
        baseline_cfg.warmup = int(args.warmup_override)
        candidate_cfg.warmup = int(args.warmup_override)
    if args.repeats_override is not None:
        if int(args.repeats_override) < 1:
            raise ValueError("--repeats-override must be >= 1")
        baseline_cfg.repeats = int(args.repeats_override)
        candidate_cfg.repeats = int(args.repeats_override)

    run_temp = Path(tempfile.mkdtemp(prefix="latency-compare-"))
    try:
        baseline_results = run_config(
            baseline_cfg,
            corpus=corpus,
            models=models,
            workspace_temp=run_temp,
            model_file_dir_override=args.model_file_dir,
            cooldown_seconds_after_file=float(args.cooldown_seconds_after_file),
            cooldown_seconds_after_model=float(args.cooldown_seconds_after_model),
        )
        _sleep_if_needed(
            float(args.cooldown_seconds_between_configs),
            reason="between baseline and candidate",
        )
        candidate_results = run_config(
            candidate_cfg,
            corpus=corpus,
            models=models,
            workspace_temp=run_temp,
            model_file_dir_override=args.model_file_dir,
            cooldown_seconds_after_file=float(args.cooldown_seconds_after_file),
            cooldown_seconds_after_model=float(args.cooldown_seconds_after_model),
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

    equivalence_payload = None
    if args.equivalence_check:
        _sleep_if_needed(
            float(args.cooldown_seconds_before_equivalence),
            reason="before equivalence",
        )
        eq_corpus = corpus if int(args.equivalence_max_files) == 0 else corpus[: int(args.equivalence_max_files)]
        equivalence_payload = run_equivalence_suite(
            corpus=eq_corpus,
            models=models,
            baseline_separator_kwargs=baseline_cfg.separator_kwargs,
            candidate_separator_kwargs=candidate_cfg.separator_kwargs,
            threshold_rel_l2=float(args.equivalence_threshold_rel_l2),
            seed=int(args.equivalence_seed),
            demucs_shifts_zero=bool(args.equivalence_demucs_shifts_zero),
            model_file_dir_override=args.model_file_dir,
            gated_arches={"Demucs", "MDXC", "MDX", "VR"} if args.equivalence_strict_demucs else {"MDXC", "MDX", "VR"},
        )

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_json = Path(args.output_json) if args.output_json else (Path("perf_reports") / f"compare_latency_{now}.json")
    output_md = Path(args.output_markdown) if args.output_markdown else output_json.with_suffix(".md")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus,
        "models": models,
        "speed_mode_alignment": {
            "baseline": baseline_speed_mode,
            "candidate": candidate_speed_mode,
            "allow_mismatch": bool(args.allow_speed_mode_mismatch),
            "mismatch": baseline_speed_mode != candidate_speed_mode,
        },
        "baseline": {
            "config": baseline_payload,
            "results": baseline_results,
        },
        "candidate": {
            "config": candidate_payload,
            "results": candidate_results,
        },
        "summary": summary_rows,
        "equivalence": equivalence_payload,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(
        output_md,
        summary_rows,
        baseline_cfg.label,
        candidate_cfg.label,
        equivalence_summary=equivalence_payload["summary"] if equivalence_payload else None,
        equivalence_threshold_rel_l2=equivalence_payload["threshold_rel_l2"] if equivalence_payload else None,
        equivalence_gated_arches=equivalence_payload["gated_arches"] if equivalence_payload else None,
    )

    total = len(summary_rows)
    passed = sum(1 for row in summary_rows if row["pass"])
    failed = total - passed
    print(f"Compared {total} models: {passed} pass, {failed} fail")
    if equivalence_payload:
        eq_total = len(equivalence_payload["summary"])
        eq_passed = sum(1 for row in equivalence_payload["summary"] if row["pass"])
        print(f"Equivalence: {eq_passed}/{eq_total} pass")
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")


if __name__ == "__main__":
    main()
