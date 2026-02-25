#!/usr/bin/env python3
"""Run strict MLX vs audio-separator parity checks with fail-fast diagnostics."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlx_audio_separator.utils.equivalence import compare_stem_maps, read_stem_map, set_deterministic_seeds


class ParityError(RuntimeError):
    """Raised for strict parity failures that should surface directly."""


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def _load_corpus(path: Path, max_files: int) -> list[str]:
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
    if int(max_files) > 0:
        out = out[: int(max_files)]
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


def _normalize_output_paths(paths: list[str] | None, output_dir: str | Path | None) -> list[str]:
    out: list[str] = []
    base_dir = Path(output_dir).expanduser().resolve() if output_dir else None
    cwd = Path.cwd().resolve()
    for raw in paths or []:
        p = Path(str(raw))
        candidates: list[Path] = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append((cwd / p).resolve())
            if base_dir is not None:
                candidates.append((base_dir / p).resolve())
                candidates.append((base_dir / p.name).resolve())
        chosen = None
        for c in candidates:
            if c.exists():
                chosen = c
                break
        out.append(str((chosen or candidates[0]) if candidates else p))
    return out


def _validate_outputs(paths: list[str] | None, output_dir: str | Path | None, wait_seconds: float) -> tuple[bool, str | None, list[str]]:
    resolved = _normalize_output_paths(paths, output_dir=output_dir)
    if not paths:
        return False, "no output files returned", resolved

    deadline = time.perf_counter() + max(0.0, float(wait_seconds))
    while True:
        missing = [p for p in resolved if not os.path.isfile(p)]
        if not missing:
            break
        if time.perf_counter() >= deadline:
            return False, f"missing output files: {len(missing)}", resolved
        time.sleep(0.05)

    missing = [p for p in resolved if not os.path.isfile(p)]
    if missing:
        return False, f"missing output files: {len(missing)}", resolved
    empty = [p for p in resolved if os.path.getsize(p) <= 0]
    if empty:
        return False, f"empty output files: {len(empty)}", resolved
    return True, None, resolved


def _cleanup_outputs(paths: list[str] | None):
    for p in paths or []:
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass


def _force_demucs_determinism(sep: Any, *, shifts_zero: bool):
    if not shifts_zero:
        return
    inst = getattr(sep, "model_instance", None)
    if inst is None:
        return
    if hasattr(inst, "shifts"):
        inst.shifts = 0
    demucs_sep = getattr(inst, "_demucs_separator", None)
    if demucs_sep is not None and hasattr(demucs_sep, "update_parameter"):
        overlap = getattr(demucs_sep, "overlap", 0.25)
        split = getattr(demucs_sep, "split", True)
        try:
            demucs_sep.update_parameter(shifts=0, overlap=overlap, split=split)
        except Exception:
            pass


def _backend_run(
    *,
    sep: Any,
    backend_name: str,
    audio_path: str,
    output_dir: Path,
    wait_seconds: float,
) -> tuple[dict[str, tuple[Any, int]], dict[str, Any]]:
    t0 = time.perf_counter()
    outs = sep.separate(audio_path)
    elapsed_s = round(time.perf_counter() - t0, 6)

    ok, reason, resolved = _validate_outputs(outs, output_dir=output_dir, wait_seconds=wait_seconds)
    if not ok:
        raise ParityError(f"{backend_name} invalid outputs: {reason}")

    try:
        stems = read_stem_map(resolved)
    except Exception as exc:
        raise ParityError(f"{backend_name} failed reading outputs: {exc}") from exc

    return stems, {
        "elapsed_s": elapsed_s,
        "outputs": resolved,
        "output_count": len(resolved),
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], threshold_rel_l2: float):
    lines = [
        "# MLX vs audio-separator Parity",
        "",
        f"- Relative L2 threshold: `{threshold_rel_l2}`",
        f"- Fail-fast used: `{summary['fail_fast']}`",
        "",
        "| Model | Arch | Files checked | Max rel L2 | Pass | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        max_rel = row.get("max_rel_l2")
        lines.append(
            f"| `{row.get('model')}` | {row.get('arch_mlx') or 'unknown'} | {row.get('files_checked', 0)} | "
            f"{'n/a' if max_rel is None else f'{max_rel:.8f}'} | "
            f"{'yes' if row.get('pass') else 'no'} | {row.get('status', 'error')} |"
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Total models: `{summary['total_models']}`",
            f"- OK models: `{summary['ok_models']}`",
            f"- Failed models: `{summary['failed_models']}`",
            f"- Passed models: `{summary['pass_models']}`",
            f"- Terminated early: `{summary['terminated_early']}`",
            f"- Stop reason: `{summary.get('stop_reason') or 'n/a'}`",
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
    threshold_rel_l2: float,
    seed: int,
    demucs_shifts_zero: bool,
    output_root: Path,
) -> dict[str, Any]:
    from mlx_audio_separator.core import Separator as MLXSeparator

    try:
        from audio_separator.separator import Separator as PASSeparator
    except Exception as exc:
        return {
            "model": model,
            "status": f"error: audio-separator unavailable: {exc}",
            "pass": False,
            "error": f"audio-separator unavailable: {exc}",
        }

    model_safe = model.replace("/", "_")
    mlx_out = output_root / "mlx" / model_safe
    pas_out = output_root / "pas" / model_safe
    mlx_out.mkdir(parents=True, exist_ok=True)
    pas_out.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {
        "model": model,
        "status": "running",
        "pass": False,
        "arch_mlx": None,
        "arch_pas": None,
        "load_mlx_s": None,
        "load_pas_s": None,
        "files_checked": 0,
        "files_passed": 0,
        "max_rel_l2": None,
        "file_results": [],
        "error": None,
        "diagnostic": None,
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

        _force_demucs_determinism(mlx_sep, shifts_zero=demucs_shifts_zero)
        _force_demucs_determinism(pas_sep, shifts_zero=demucs_shifts_zero)
    except Exception as exc:
        row["status"] = f"error: {exc}"
        row["error"] = f"load error: {exc}"
        return row

    max_rel_l2 = 0.0
    for i, audio_path in enumerate(corpus, start=1):
        resolved_mlx: list[str] = []
        resolved_pas: list[str] = []
        try:
            set_deterministic_seeds(seed)
            mlx_stems, mlx_meta = _backend_run(
                sep=mlx_sep,
                backend_name="mlx",
                audio_path=audio_path,
                output_dir=mlx_out,
                wait_seconds=0.5,
            )
            resolved_mlx = list(mlx_meta["outputs"])

            set_deterministic_seeds(seed)
            pas_stems, pas_meta = _backend_run(
                sep=pas_sep,
                backend_name="pas",
                audio_path=audio_path,
                output_dir=pas_out,
                wait_seconds=2.0,
            )
            resolved_pas = list(pas_meta["outputs"])

            compared = compare_stem_maps(
                baseline=pas_stems,
                candidate=mlx_stems,
                threshold_rel_l2=float(threshold_rel_l2),
            )
            max_rel_l2 = max(max_rel_l2, float(compared.get("max_rel_l2", 0.0)))
            file_ok = bool(compared.get("pass", False))

            row["files_checked"] += 1
            if file_ok:
                row["files_passed"] += 1

            row["file_results"].append(
                {
                    "audio_path": audio_path,
                    "index": i,
                    "mlx_elapsed_s": mlx_meta["elapsed_s"],
                    "pas_elapsed_s": pas_meta["elapsed_s"],
                    "pass": file_ok,
                    "max_rel_l2": float(compared.get("max_rel_l2", 0.0)),
                    "detail": compared,
                }
            )

            if not file_ok:
                raise ParityError(
                    f"parity drift on file {i}/{len(corpus)}: max_rel_l2={compared.get('max_rel_l2', None)}"
                )
        except Exception as exc:
            row["status"] = f"error: {exc}"
            row["error"] = f"separate error: {exc}"
            row["diagnostic"] = {
                "audio_path": audio_path,
                "file_index": i,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "mlx_outputs": resolved_mlx,
                "pas_outputs": resolved_pas,
                "mlx_output_dir": str(mlx_out),
                "pas_output_dir": str(pas_out),
            }
            break
        finally:
            _cleanup_outputs(resolved_mlx)
            _cleanup_outputs(resolved_pas)

    row["max_rel_l2"] = round(max_rel_l2, 10)
    if row.get("error"):
        return row

    all_ok = row["files_checked"] == len(corpus) and row["files_passed"] == len(corpus)
    row["pass"] = bool(all_ok)
    row["status"] = "ok" if all_ok else "error: incomplete parity checks"
    return row


def _build_summary(rows: list[dict[str, Any]], fail_fast: bool, terminated_early: bool, stop_reason: str | None) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    return {
        "total_models": len(rows),
        "ok_models": len(ok_rows),
        "failed_models": len(rows) - len(ok_rows),
        "pass_models": sum(1 for r in rows if bool(r.get("pass"))),
        "fail_fast": bool(fail_fast),
        "terminated_early": bool(terminated_early),
        "stop_reason": stop_reason,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLX vs audio-separator strict parity check.")
    p.add_argument("--corpus-file", required=True, help="Text file with one audio path per line.")
    p.add_argument("--models", required=True, help="Comma-separated model filenames to compare.")
    p.add_argument("--model-file-dir", default=None, help="Shared model directory for both backends.")
    p.add_argument("--mlx-config", default=None, help="JSON object or JSON file path for MLX Separator kwargs.")
    p.add_argument("--pas-config", default=None, help="JSON object or JSON file path for PAS Separator kwargs.")
    p.add_argument("--max-files", type=int, default=0, help="Max number of corpus files to evaluate (0=all).")
    p.add_argument("--threshold-rel-l2", type=float, default=1e-5, help="Relative L2 pass threshold.")
    p.add_argument("--seed", type=int, default=12345, help="Deterministic seed for both backends.")
    p.add_argument(
        "--demucs-shifts-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force Demucs shifts=0 during parity checks (default: true).",
    )
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop after first model failure (default: true).",
    )
    p.add_argument("--output-json", default=None, help="Output JSON path.")
    p.add_argument("--output-markdown", default=None, help="Output markdown path.")
    p.add_argument("--keep-temp", action="store_true", help="Keep temporary output directories.")
    return p.parse_args()


def main():
    args = parse_args()
    corpus = _load_corpus(Path(args.corpus_file).expanduser(), max_files=int(args.max_files))
    models = _parse_models(args.models)
    mlx_kwargs = _resolve_config_arg(args.mlx_config)
    pas_kwargs = _resolve_config_arg(args.pas_config)

    output_root = Path(tempfile.mkdtemp(prefix="mlx-vs-pas-parity-"))
    rows: list[dict[str, Any]] = []
    terminated_early = False
    stop_reason = None

    try:
        for i, model in enumerate(models, start=1):
            print(f"[{i}/{len(models)}] model={model}")
            row = run_model(
                model=model,
                corpus=corpus,
                model_file_dir=args.model_file_dir,
                mlx_kwargs=mlx_kwargs,
                pas_kwargs=pas_kwargs,
                threshold_rel_l2=float(args.threshold_rel_l2),
                seed=int(args.seed),
                demucs_shifts_zero=bool(args.demucs_shifts_zero),
                output_root=output_root,
            )
            rows.append(row)
            print(f"    status={row.get('status')} files={row.get('files_passed', 0)}/{row.get('files_checked', 0)}")
            if bool(args.fail_fast) and row.get("status") != "ok":
                terminated_early = True
                stop_reason = f"{model}: {row.get('status')}"
                break
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

    summary = _build_summary(rows, fail_fast=bool(args.fail_fast), terminated_early=terminated_early, stop_reason=stop_reason)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus,
        "models": models,
        "settings": {
            "model_file_dir": args.model_file_dir,
            "threshold_rel_l2": float(args.threshold_rel_l2),
            "seed": int(args.seed),
            "demucs_shifts_zero": bool(args.demucs_shifts_zero),
            "fail_fast": bool(args.fail_fast),
            "max_files": int(args.max_files),
            "mlx_config": mlx_kwargs,
            "pas_config": pas_kwargs,
        },
        "results": rows,
        "summary": summary,
    }

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_json = Path(args.output_json) if args.output_json else Path(f"perf_reports/mlx_vs_pas_parity_{now}.json")
    output_md = Path(args.output_markdown) if args.output_markdown else output_json.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(output_md, rows=rows, summary=summary, threshold_rel_l2=float(args.threshold_rel_l2))

    print(
        f"Compared {summary['total_models']} models: "
        f"{summary['ok_models']} ok, {summary['failed_models']} failed, "
        f"{summary['pass_models']} passed parity"
    )
    if summary["terminated_early"]:
        print(f"Terminated early: {summary.get('stop_reason')}")
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")


if __name__ == "__main__":
    main()
