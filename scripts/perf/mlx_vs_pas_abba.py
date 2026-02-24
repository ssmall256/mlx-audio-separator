#!/usr/bin/env python3
"""Run an MLX vs python-audio-separator ABBA latency comparison report.

This script is designed for release evidence on overlapping models that load and
run on both backends:
  - MLX backend: mlx_audio_separator
  - Python baseline: python-audio-separator on MPS
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def _load_corpus(path: Path) -> list[str]:
    if not path.is_file():
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
        raise ValueError(f"Corpus file contains no valid entries: {path}")
    return out


def _parse_models(models_arg: str) -> list[str]:
    models = [m.strip() for m in str(models_arg).split(",") if m.strip()]
    if not models:
        raise ValueError("No models provided. Use --models with comma-separated model filenames.")
    return models


def _resolve_config_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    candidate = Path(str(value)).expanduser()
    if candidate.is_file():
        return _load_json(candidate)
    payload = json.loads(str(value))
    if not isinstance(payload, dict):
        raise ValueError("Config argument must be a JSON object or a path to one.")
    return payload


def _filter_kwargs_for_ctor(ctor: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(ctor)
        allowed = set(sig.parameters.keys())
        allowed.discard("self")
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return dict(kwargs)


def _cleanup_outputs(paths: list[str] | None):
    for p in paths or []:
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass


def _validate_outputs(paths: list[str] | None) -> tuple[bool, str | None]:
    if not paths:
        return False, "no output files returned"
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        return False, f"missing output files: {len(missing)}"
    empty = [p for p in paths if os.path.getsize(p) <= 0]
    if empty:
        return False, f"empty output files: {len(empty)}"
    return True, None


def _build_leg_sequence(order_template: str, order_repeats: int, randomize_abba: bool, rng: random.Random) -> list[str]:
    seq: list[str] = []
    template = str(order_template).strip().upper()
    if not template:
        raise ValueError("order_template cannot be empty")
    for _ in range(int(max(1, order_repeats))):
        if randomize_abba:
            template = "ABBA" if rng.random() < 0.5 else "BAAB"
        for c in template:
            if c not in {"A", "B"}:
                raise ValueError(f"Invalid order character '{c}'. Use only A/B.")
            seq.append(c)
    return seq


def _write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]):
    lines = [
        "# MLX vs python-audio-separator (ABBA)",
        "",
        "| Model | Arch (MLX) | MLX median (s) | PAS median (s) | MLX speedup | Delta % (MLX vs PAS) | Pass | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        mlx_med = row.get("mlx_median_s")
        pas_med = row.get("pas_median_s")
        speedup = row.get("mlx_speedup_vs_pas")
        delta = row.get("delta_pct_mlx_vs_pas")
        lines.append(
            f"| `{row.get('model')}` | {row.get('arch_mlx') or 'unknown'} | "
            f"{'n/a' if mlx_med is None else f'{mlx_med:.3f}'} | "
            f"{'n/a' if pas_med is None else f'{pas_med:.3f}'} | "
            f"{'n/a' if speedup is None else f'{speedup:.2f}x'} | "
            f"{'n/a' if delta is None else f'{delta:+.2f}%'} | "
            f"{'yes' if row.get('pass') else 'no'} | {row.get('status')} |"
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Total models: `{summary['total_models']}`",
            f"- OK models: `{summary['ok_models']}`",
            f"- Failed models: `{summary['failed_models']}`",
            f"- Models passing speed gate: `{summary['pass_models']}`",
            f"- Median speedup (OK rows): `"
            + ("n/a" if summary.get("median_speedup_ok") is None else f"{summary['median_speedup_ok']:.3f}x")
            + "`",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_model(
    *,
    model: str,
    corpus: list[str],
    model_file_dir: str | None,
    mlx_kwargs: dict[str, Any],
    pas_kwargs: dict[str, Any],
    output_root: Path,
    warmup: int,
    order_template: str,
    order_repeats: int,
    randomize_abba: bool,
    rng: random.Random,
) -> dict[str, Any]:
    from mlx_audio_separator.core import Separator as MLXSeparator

    try:
        from audio_separator.separator import Separator as PASSeparator
    except Exception as exc:
        return {
            "model": model,
            "status": "error",
            "error": f"python-audio-separator unavailable: {exc}",
            "pass": False,
        }

    model_safe = model.replace("/", "_")
    mlx_out = output_root / "mlx" / model_safe
    pas_out = output_root / "pas" / model_safe
    mlx_out.mkdir(parents=True, exist_ok=True)
    pas_out.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {
        "model": model,
        "status": "error",
        "arch_mlx": None,
        "arch_pas": None,
        "load_mlx_s": None,
        "load_pas_s": None,
        "runs_mlx_s": [],
        "runs_pas_s": [],
        "mlx_median_s": None,
        "pas_median_s": None,
        "mlx_speedup_vs_pas": None,
        "delta_pct_mlx_vs_pas": None,
        "pass": False,
        "error": None,
    }

    try:
        mk = dict(mlx_kwargs)
        mk.setdefault("output_format", "WAV")
        mk["output_dir"] = str(mlx_out)
        if model_file_dir:
            mk["model_file_dir"] = model_file_dir
        mlx_sep = MLXSeparator(**mk)

        pk = dict(pas_kwargs)
        pk.setdefault("output_format", "WAV")
        pk["output_dir"] = str(pas_out)
        if model_file_dir:
            pk["model_file_dir"] = model_file_dir
        pk = _filter_kwargs_for_ctor(PASSeparator.__init__, pk)
        pas_sep = PASSeparator(**pk)

        t0 = time.perf_counter()
        mlx_sep.load_model(model_filename=model)
        row["load_mlx_s"] = round(time.perf_counter() - t0, 6)
        row["arch_mlx"] = getattr(mlx_sep, "model_type", None)

        t0 = time.perf_counter()
        pas_sep.load_model(model_filename=model)
        row["load_pas_s"] = round(time.perf_counter() - t0, 6)
        row["arch_pas"] = getattr(pas_sep, "model_type", None)
    except Exception as exc:
        row["error"] = f"load error: {exc}"
        row["status"] = f"error: {exc}"
        return row

    backends = {"A": ("mlx", mlx_sep), "B": ("pas", pas_sep)}
    try:
        for audio_path in corpus:
            # Warm both backends once per file for fairness before timed ABBA legs.
            for leg in ("A", "B"):
                _, sep = backends[leg]
                for _ in range(int(max(0, warmup))):
                    outs = sep.separate(audio_path)
                    _cleanup_outputs(outs)

            leg_sequence = _build_leg_sequence(
                order_template=order_template,
                order_repeats=order_repeats,
                randomize_abba=randomize_abba,
                rng=rng,
            )
            for leg in leg_sequence:
                backend_name, sep = backends[leg]
                t0 = time.perf_counter()
                outs = sep.separate(audio_path)
                dt = time.perf_counter() - t0
                ok, reason = _validate_outputs(outs)
                _cleanup_outputs(outs)
                if not ok:
                    raise RuntimeError(f"{backend_name} invalid outputs: {reason}")
                if backend_name == "mlx":
                    row["runs_mlx_s"].append(round(dt, 6))
                else:
                    row["runs_pas_s"].append(round(dt, 6))
    except Exception as exc:
        row["error"] = f"separate error: {exc}"
        row["status"] = f"error: {exc}"
        return row

    if row["runs_mlx_s"] and row["runs_pas_s"]:
        mlx_med = float(median(row["runs_mlx_s"]))
        pas_med = float(median(row["runs_pas_s"]))
        row["mlx_median_s"] = round(mlx_med, 6)
        row["pas_median_s"] = round(pas_med, 6)
        if mlx_med > 0:
            row["mlx_speedup_vs_pas"] = round(pas_med / mlx_med, 6)
        if pas_med > 0:
            row["delta_pct_mlx_vs_pas"] = round(((mlx_med - pas_med) / pas_med) * 100.0, 3)
        row["status"] = "ok"
    else:
        row["status"] = "error: missing timed runs"
        row["error"] = "missing timed runs"

    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLX vs python-audio-separator ABBA latency comparison.")
    p.add_argument("--corpus-file", required=True, help="Text file with one audio path per line.")
    p.add_argument("--models", required=True, help="Comma-separated model filenames to compare.")
    p.add_argument("--model-file-dir", default=None, help="Shared model directory for both backends.")
    p.add_argument("--mlx-config", default=None, help="JSON object or JSON file path for MLX Separator kwargs.")
    p.add_argument("--pas-config", default=None, help="JSON object or JSON file path for PAS Separator kwargs.")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs per backend per file (default: %(default)s).")
    p.add_argument("--order-template", default="ABBA", help="Leg order template with A/B chars (default: %(default)s).")
    p.add_argument("--order-repeats", type=int, default=1, help="How many times to repeat order template per file.")
    p.add_argument(
        "--randomize-abba",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Per file, randomly choose ABBA/BAAB each repeat (default: true).",
    )
    p.add_argument("--seed", type=int, default=12345, help="Random seed for backend order randomization.")
    p.add_argument("--min-speedup", type=float, default=1.0, help="Pass threshold for mlx_speedup_vs_pas (default: %(default)s).")
    p.add_argument("--output-json", default=None, help="Output JSON path.")
    p.add_argument("--output-markdown", default=None, help="Output markdown path.")
    p.add_argument("--keep-temp", action="store_true", help="Keep temporary output directories.")
    return p.parse_args()


def main():
    args = parse_args()
    corpus = _load_corpus(Path(args.corpus_file).expanduser())
    models = _parse_models(args.models)
    mlx_kwargs = _resolve_config_arg(args.mlx_config)
    pas_kwargs = _resolve_config_arg(args.pas_config)
    rng = random.Random(int(args.seed))

    output_root = Path(tempfile.mkdtemp(prefix="mlx-vs-pas-abba-"))
    try:
        rows: list[dict[str, Any]] = []
        for model in models:
            row = run_model(
                model=model,
                corpus=corpus,
                model_file_dir=args.model_file_dir,
                mlx_kwargs=mlx_kwargs,
                pas_kwargs=pas_kwargs,
                output_root=output_root,
                warmup=int(args.warmup),
                order_template=str(args.order_template),
                order_repeats=int(args.order_repeats),
                randomize_abba=bool(args.randomize_abba),
                rng=rng,
            )
            if row.get("status") == "ok":
                speedup = row.get("mlx_speedup_vs_pas")
                row["pass"] = bool(speedup is not None and float(speedup) >= float(args.min_speedup))
            rows.append(row)
    finally:
        if not args.keep_temp:
            for root, dirs, files in os.walk(output_root, topdown=False):
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
                os.rmdir(output_root)
            except OSError:
                pass

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    speedups = [float(r["mlx_speedup_vs_pas"]) for r in ok_rows if r.get("mlx_speedup_vs_pas") is not None]
    summary = {
        "total_models": len(rows),
        "ok_models": len(ok_rows),
        "failed_models": len(rows) - len(ok_rows),
        "pass_models": sum(1 for r in rows if bool(r.get("pass"))),
        "median_speedup_ok": (float(median(speedups)) if speedups else None),
    }

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus,
        "models": models,
        "settings": {
            "warmup": int(args.warmup),
            "order_template": str(args.order_template),
            "order_repeats": int(args.order_repeats),
            "randomize_abba": bool(args.randomize_abba),
            "seed": int(args.seed),
            "min_speedup": float(args.min_speedup),
            "model_file_dir": args.model_file_dir,
            "mlx_config": mlx_kwargs,
            "pas_config": pas_kwargs,
        },
        "results": rows,
        "summary": summary,
    }

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_json = Path(args.output_json) if args.output_json else Path(f"perf_reports/mlx_vs_pas_abba_{now}.json")
    output_md = Path(args.output_markdown) if args.output_markdown else output_json.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(output_md, rows=rows, summary=summary)

    print(
        f"Compared {summary['total_models']} models: "
        f"{summary['ok_models']} ok, {summary['failed_models']} failed, "
        f"{summary['pass_models']} passed speed gate"
    )
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")


if __name__ == "__main__":
    main()

