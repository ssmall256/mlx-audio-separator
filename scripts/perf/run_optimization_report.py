#!/usr/bin/env python3
"""Run unified parity + quality + latency report for optimization rollout."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import mlx_audio_io as mac
import numpy as np

from compare_latency import (
    _build_run_config,
    _load_corpus,
    _load_json,
    _resolve_models,
    compare_results,
    run_config,
)
from mlx_audio_separator.core import Separator
from mlx_audio_separator.utils.equivalence import read_stem_map, run_equivalence_suite
from mlx_vs_pas_parity import run_model as run_python_mps_parity_model


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_audio_2d(path: str) -> tuple[np.ndarray, int]:
    """Load audio as float32 NumPy array with shape (frames, channels)."""
    audio_mx, sample_rate = mac.load(str(path), dtype="float32")
    audio = np.array(audio_mx, copy=False)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    return audio, int(sample_rate)


def _run_cmd(cmd: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout).strip()
        return True, (proc.stdout or "").strip()
    except Exception as exc:  # pragma: no cover - runtime-side failure path
        return False, str(exc)


def collect_repro_metadata(
    *,
    corpus: list[str],
    models: list[str],
    model_file_dir: str | None,
    capture_corpus_hashes: bool,
    capture_model_hashes: bool,
) -> dict[str, Any]:
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    ok_commit, commit = _run_cmd(["git", "rev-parse", "HEAD"])
    ok_branch, branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    ok_dirty, dirty_out = _run_cmd(["git", "status", "--porcelain"])
    ok_remote, remote = _run_cmd(["git", "config", "--get", "remote.origin.url"])

    versions: dict[str, str | None] = {"python": py_ver, "numpy": np.__version__}
    try:
        import mlx

        versions["mlx"] = getattr(mlx, "__version__", None)
    except Exception:
        versions["mlx"] = None

    corpus_manifest: list[dict[str, Any]] = []
    for entry in corpus:
        p = Path(entry)
        row: dict[str, Any] = {
            "path": str(p),
            "exists": p.is_file(),
        }
        if p.is_file():
            st = p.stat()
            row["size_bytes"] = int(st.st_size)
            try:
                audio, sr = _load_audio_2d(str(p))
                row["sample_rate"] = int(sr)
                row["channels"] = int(audio.shape[1])
                row["frames"] = int(audio.shape[0])
                row["duration_s"] = float(audio.shape[0] / sr) if sr else None
            except Exception as exc:
                row["audio_read_error"] = str(exc)
            if capture_corpus_hashes:
                try:
                    row["sha256"] = _sha256_file(p)
                except Exception as exc:
                    row["sha256_error"] = str(exc)
        corpus_manifest.append(row)

    model_manifest: list[dict[str, Any]] = []
    model_dir_path = Path(model_file_dir).expanduser().resolve() if model_file_dir else None
    for model in models:
        row: dict[str, Any] = {"model": model, "resolved_path": None, "exists": False}
        if model_dir_path is not None:
            candidate = model_dir_path / model
            row["resolved_path"] = str(candidate)
            row["exists"] = candidate.is_file()
            if candidate.is_file():
                row["size_bytes"] = int(candidate.stat().st_size)
                if capture_model_hashes:
                    try:
                        row["sha256"] = _sha256_file(candidate)
                    except Exception as exc:
                        row["sha256_error"] = str(exc)
        model_manifest.append(row)

    return {
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "host": socket.gethostname(),
        "git": {
            "commit": commit if ok_commit else None,
            "branch": branch if ok_branch else None,
            "dirty": bool(dirty_out) if ok_dirty else None,
            "remote_origin": remote if ok_remote else None,
        },
        "environment": {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "versions": versions,
        },
        "corpus_manifest": corpus_manifest,
        "model_manifest": model_manifest,
    }


def _timed_separate(sep: Any, audio_path: str, repeats: int) -> list[float]:
    import time

    out: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sep.separate(audio_path)
        out.append(time.perf_counter() - t0)
    return out


def _filtered_kwargs_for_ctor(ctor: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(ctor)
        allowed = set(sig.parameters.keys())
        if "self" in allowed:
            allowed.remove("self")
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return dict(kwargs)


def run_python_mps_latency(
    *,
    corpus: list[str],
    models: list[str],
    output_root: Path,
    model_file_dir: str | None,
    warmup: int,
    repeats: int,
    separator_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from audio_separator.separator import Separator as PySeparator
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc), "results": {}}

    raw_kwargs = dict(separator_kwargs or {})
    raw_kwargs.setdefault("output_format", "WAV")
    if model_file_dir:
        raw_kwargs.setdefault("model_file_dir", model_file_dir)

    results: dict[str, Any] = {}
    for model in models:
        model_out_dir = output_root / model.replace("/", "_")
        model_out_dir.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {
            "status": "error",
            "arch": None,
            "device": "mps",
            "load_s": 0.0,
            "runs_s": [],
            "runs_by_file_s": {},
            "median_s": 0.0,
            "error": None,
        }
        try:
            kwargs = dict(raw_kwargs)
            kwargs["output_dir"] = str(model_out_dir)
            kwargs = _filtered_kwargs_for_ctor(PySeparator.__init__, kwargs)
            sep = PySeparator(**kwargs)

            import time

            t0 = time.perf_counter()
            sep.load_model(model_filename=model)
            row["load_s"] = float(time.perf_counter() - t0)
            row["arch"] = getattr(sep, "model_type", None)

            all_runs: list[float] = []
            for audio_path in corpus:
                for _ in range(int(max(0, warmup))):
                    sep.separate(audio_path)
                timed_runs = _timed_separate(sep, audio_path, int(max(1, repeats)))
                row["runs_by_file_s"][audio_path] = timed_runs
                all_runs.extend(timed_runs)
            row["runs_s"] = all_runs
            row["median_s"] = float(median(all_runs)) if all_runs else 0.0
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        results[model] = row

    return {"status": "ok", "results": results}


def run_python_mps_parity(
    *,
    corpus: list[str],
    models: list[str],
    output_root: Path,
    model_file_dir: str | None,
    mlx_separator_kwargs: dict[str, Any],
    pas_separator_kwargs: dict[str, Any] | None,
    threshold_rel_l2: float,
    seed: int,
    demucs_shifts_zero: bool,
    demucs_mlx_strict_kernels: bool,
    max_files: int,
    fail_fast: bool,
) -> dict[str, Any]:
    eval_corpus = list(corpus if int(max_files) <= 0 else corpus[: int(max_files)])
    rows: list[dict[str, Any]] = []
    terminated_early = False
    stop_reason = None

    for model in models:
        row = run_python_mps_parity_model(
            model=model,
            corpus=eval_corpus,
            model_file_dir=model_file_dir,
            mlx_kwargs=dict(mlx_separator_kwargs),
            pas_kwargs=dict(pas_separator_kwargs or {}),
            threshold_rel_l2=float(threshold_rel_l2),
            seed=int(seed),
            demucs_shifts_zero=bool(demucs_shifts_zero),
            demucs_mlx_strict_kernels=bool(demucs_mlx_strict_kernels),
            output_root=output_root,
        )
        rows.append(row)
        if bool(fail_fast) and row.get("status") != "ok":
            terminated_early = True
            stop_reason = f"{model}: {row.get('status')}"
            break

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    summary = {
        "total_models": len(rows),
        "ok_models": len(ok_rows),
        "failed_models": len(rows) - len(ok_rows),
        "pass_models": sum(1 for r in rows if bool(r.get("pass"))),
        "terminated_early": bool(terminated_early),
        "stop_reason": stop_reason,
        "threshold_rel_l2": float(threshold_rel_l2),
        "seed": int(seed),
        "demucs_shifts_zero": bool(demucs_shifts_zero),
        "demucs_mlx_strict_kernels": bool(demucs_mlx_strict_kernels),
        "fail_fast": bool(fail_fast),
        "max_files": int(max_files),
    }

    all_pass = (
        len(rows) == len(models)
        and all(r.get("status") == "ok" and bool(r.get("pass")) for r in rows)
    )

    return {
        "status": "ok",
        "corpus": eval_corpus,
        "models": models,
        "results": rows,
        "summary": summary,
        "all_pass": bool(all_pass),
    }


def _safe_percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _rel_l2(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = reference.reshape(-1).astype(np.float64, copy=False)
    est = estimate.reshape(-1).astype(np.float64, copy=False)
    denom = float(np.linalg.norm(ref))
    if denom == 0.0:
        return 0.0 if float(np.linalg.norm(est)) == 0.0 else float("inf")
    return float(np.linalg.norm(ref - est) / denom)


def _si_sdr_db(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-12) -> float:
    ref = reference.reshape(-1).astype(np.float64, copy=False)
    est = estimate.reshape(-1).astype(np.float64, copy=False)
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    ref_energy = float(np.dot(ref, ref))
    if ref_energy <= eps:
        return float("inf") if float(np.dot(est, est)) <= eps else -float("inf")
    scale = float(np.dot(est, ref) / (ref_energy + eps))
    target = scale * ref
    noise = est - target
    target_energy = float(np.dot(target, target))
    noise_energy = float(np.dot(noise, noise))
    return float(10.0 * np.log10((target_energy + eps) / (noise_energy + eps)))


def _sdr_db(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-12) -> float:
    ref = reference.reshape(-1).astype(np.float64, copy=False)
    est = estimate.reshape(-1).astype(np.float64, copy=False)
    num = float(np.dot(ref, ref))
    den = float(np.dot(ref - est, ref - est))
    return float(10.0 * np.log10((num + eps) / (den + eps)))


def _align_audio_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Expected 2D audio arrays (frames, channels).")
    frames = min(int(a.shape[0]), int(b.shape[0]))
    chans = min(int(a.shape[1]), int(b.shape[1]))
    aligned = (a.shape != (frames, chans)) or (b.shape != (frames, chans))
    return (
        a[:frames, :chans].astype(np.float32, copy=False),
        b[:frames, :chans].astype(np.float32, copy=False),
        aligned,
    )


def _load_reference_manifest(path: Path) -> dict[str, dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("quality-reference-manifest must be a JSON object.")
    out: dict[str, dict[str, str]] = {}
    for raw_audio, stems in data.items():
        if not isinstance(raw_audio, str):
            raise ValueError("Manifest keys must be audio paths.")
        if not isinstance(stems, dict):
            raise ValueError(f"Manifest entry for {raw_audio} must be an object.")
        audio_key = os.path.abspath(os.path.expanduser(raw_audio))
        out_stems: dict[str, str] = {}
        for raw_stem, raw_path in stems.items():
            if not isinstance(raw_stem, str) or not isinstance(raw_path, str):
                raise ValueError(f"Invalid stem entry for {raw_audio}.")
            stem = raw_stem.strip().lower()
            stem_path = os.path.abspath(os.path.expanduser(raw_path))
            if not os.path.isfile(stem_path):
                raise FileNotFoundError(f"Reference stem not found: {stem_path}")
            out_stems[stem] = stem_path
        out[audio_key] = out_stems
    return out


def _read_reference_stems(reference_paths: dict[str, str]) -> dict[str, tuple[np.ndarray, int]]:
    out: dict[str, tuple[np.ndarray, int]] = {}
    for stem, path in reference_paths.items():
        audio, sample_rate = _load_audio_2d(path)
        out[stem] = (audio, int(sample_rate))
    return out


def run_quality_suite(
    corpus: list[str],
    models: list[str],
    baseline_separator_kwargs: dict[str, Any],
    candidate_separator_kwargs: dict[str, Any],
    workspace_temp: Path,
    model_file_dir_override: str | None = None,
    reference_manifest: dict[str, dict[str, str]] | None = None,
    max_files: int = 0,
    threshold_median_abs_delta_db: float = 0.05,
    threshold_p95_abs_delta_db: float = 0.2,
    threshold_proxy_p05_sisdr_db: float = 35.0,
    threshold_proxy_p95_rel_l2: float = 1e-3,
    enforce_proxy_gate: bool = False,
) -> dict[str, Any]:
    eval_corpus = list(corpus if max_files <= 0 else corpus[:max_files])
    run_root = workspace_temp / "quality"
    run_root.mkdir(parents=True, exist_ok=True)

    mode = "reference_delta" if reference_manifest is not None else "proxy_similarity"
    results: dict[str, Any] = {}
    summary: list[dict[str, Any]] = []

    for model in models:
        base_out_dir = run_root / "baseline" / model.replace("/", "_")
        cand_out_dir = run_root / "candidate" / model.replace("/", "_")
        base_out_dir.mkdir(parents=True, exist_ok=True)
        cand_out_dir.mkdir(parents=True, exist_ok=True)

        row: dict[str, Any] = {
            "model": model,
            "status": "error",
            "arch": None,
            "mode": mode,
            "sample_count": 0,
            "missing_count": 0,
            "aligned_count": 0,
            "errors": [],
            "samples": [],
            "pass": False,
        }

        try:
            base_kwargs = dict(baseline_separator_kwargs)
            cand_kwargs = dict(candidate_separator_kwargs)
            base_kwargs.setdefault("output_format", "WAV")
            cand_kwargs.setdefault("output_format", "WAV")
            # Quality comparison requires both runs to emit comparable full stem sets.
            base_kwargs["output_single_stem"] = None
            cand_kwargs["output_single_stem"] = None
            base_kwargs["output_dir"] = str(base_out_dir)
            cand_kwargs["output_dir"] = str(cand_out_dir)
            if model_file_dir_override:
                base_kwargs["model_file_dir"] = model_file_dir_override
                cand_kwargs["model_file_dir"] = model_file_dir_override

            sep_base = Separator(**base_kwargs)
            sep_cand = Separator(**cand_kwargs)
            sep_base.load_model(model_filename=model)
            sep_cand.load_model(model_filename=model)

            row["arch"] = sep_cand.model_type or sep_base.model_type

            for audio_path in eval_corpus:
                base_outputs = sep_base.separate(audio_path)
                cand_outputs = sep_cand.separate(audio_path)
                base_stems = read_stem_map(base_outputs)
                cand_stems = read_stem_map(cand_outputs)

                if reference_manifest is not None:
                    ref_paths = reference_manifest.get(audio_path)
                    if ref_paths is None:
                        row["errors"].append(f"Missing reference entry for {audio_path}")
                        continue
                    ref_stems = _read_reference_stems(ref_paths)
                    stem_keys = sorted(set(base_stems) & set(cand_stems) & set(ref_stems))
                else:
                    stem_keys = sorted(set(base_stems) & set(cand_stems))

                expected_keys = set(base_stems) | set(cand_stems)
                if reference_manifest is not None:
                    expected_keys = expected_keys | set(reference_manifest.get(audio_path, {}).keys())
                missing = sorted(expected_keys - set(stem_keys))
                row["missing_count"] += len(missing)

                for stem in stem_keys:
                    base_audio, base_sr = base_stems[stem]
                    cand_audio, cand_sr = cand_stems[stem]
                    if base_sr != cand_sr:
                        row["errors"].append(f"Sample-rate mismatch for {audio_path} stem={stem}: {base_sr} vs {cand_sr}")
                        continue

                    base_aligned, cand_aligned, aligned = _align_audio_pair(base_audio, cand_audio)
                    if aligned:
                        row["aligned_count"] += 1

                    sample_row: dict[str, Any] = {
                        "file": audio_path,
                        "stem": stem,
                        "rel_l2_candidate_vs_baseline": _rel_l2(base_aligned, cand_aligned),
                        "si_sdr_candidate_vs_baseline_db": _si_sdr_db(base_aligned, cand_aligned),
                    }

                    if reference_manifest is not None:
                        ref_audio, ref_sr = ref_stems[stem]
                        if ref_sr != base_sr:
                            row["errors"].append(
                                f"Reference sample-rate mismatch for {audio_path} stem={stem}: {ref_sr} vs {base_sr}"
                            )
                            continue
                        ref_for_base, base_for_ref, aligned_ref_base = _align_audio_pair(ref_audio, base_audio)
                        ref_for_cand, cand_for_ref, aligned_ref_cand = _align_audio_pair(ref_audio, cand_audio)
                        if aligned_ref_base or aligned_ref_cand:
                            row["aligned_count"] += int(aligned_ref_base) + int(aligned_ref_cand)

                        baseline_si_sdr = _si_sdr_db(ref_for_base, base_for_ref)
                        candidate_si_sdr = _si_sdr_db(ref_for_cand, cand_for_ref)
                        baseline_sdr = _sdr_db(ref_for_base, base_for_ref)
                        candidate_sdr = _sdr_db(ref_for_cand, cand_for_ref)
                        sample_row.update(
                            {
                                "baseline_si_sdr_db": baseline_si_sdr,
                                "candidate_si_sdr_db": candidate_si_sdr,
                                "delta_si_sdr_db": candidate_si_sdr - baseline_si_sdr,
                                "baseline_sdr_db": baseline_sdr,
                                "candidate_sdr_db": candidate_sdr,
                                "delta_sdr_db": candidate_sdr - baseline_sdr,
                            }
                        )

                    row["samples"].append(sample_row)

            row["sample_count"] = len(row["samples"])
            if row["sample_count"] == 0:
                row["status"] = "error"
                row["pass"] = False
            else:
                row["status"] = "ok"
                if reference_manifest is not None:
                    delta_si_sdr = [float(x["delta_si_sdr_db"]) for x in row["samples"]]
                    abs_delta = [abs(x) for x in delta_si_sdr]
                    row["median_delta_si_sdr_db"] = float(median(delta_si_sdr))
                    row["median_abs_delta_si_sdr_db"] = float(median(abs_delta))
                    row["p95_abs_delta_si_sdr_db"] = _safe_percentile(abs_delta, 95.0)
                    row["criterion"] = {
                        "median_abs_delta_si_sdr_db<=": float(threshold_median_abs_delta_db),
                        "p95_abs_delta_si_sdr_db<=": float(threshold_p95_abs_delta_db),
                    }
                    row["pass"] = (
                        row["median_abs_delta_si_sdr_db"] <= float(threshold_median_abs_delta_db)
                        and row["p95_abs_delta_si_sdr_db"] <= float(threshold_p95_abs_delta_db)
                        and row["missing_count"] == 0
                        and not row["errors"]
                    )
                else:
                    proxy_si_sdr = [float(x["si_sdr_candidate_vs_baseline_db"]) for x in row["samples"]]
                    proxy_rel_l2 = [float(x["rel_l2_candidate_vs_baseline"]) for x in row["samples"]]
                    row["median_proxy_si_sdr_db"] = float(median(proxy_si_sdr))
                    row["p05_proxy_si_sdr_db"] = _safe_percentile(proxy_si_sdr, 5.0)
                    row["p95_proxy_rel_l2"] = _safe_percentile(proxy_rel_l2, 95.0)
                    row["criterion"] = {
                        "enforced": bool(enforce_proxy_gate),
                        "p05_proxy_si_sdr_db>=": float(threshold_proxy_p05_sisdr_db),
                        "p95_proxy_rel_l2<=": float(threshold_proxy_p95_rel_l2),
                    }
                    proxy_rule_pass = (
                        row["p05_proxy_si_sdr_db"] >= float(threshold_proxy_p05_sisdr_db)
                        and row["p95_proxy_rel_l2"] <= float(threshold_proxy_p95_rel_l2)
                    )
                    row["pass"] = (
                        proxy_rule_pass if bool(enforce_proxy_gate) else True
                    ) and row["missing_count"] == 0 and not row["errors"]
        except Exception as exc:  # pragma: no cover - runtime-side failure path
            row["status"] = "error"
            row["errors"].append(str(exc))
            row["pass"] = False

        results[model] = row
        summary.append(
            {
                "model": model,
                "arch": row.get("arch", "unknown"),
                "status": row["status"],
                "mode": row["mode"],
                "sample_count": row["sample_count"],
                "missing_count": row["missing_count"],
                "pass": bool(row["pass"]),
                "median_abs_delta_si_sdr_db": row.get("median_abs_delta_si_sdr_db"),
                "p95_abs_delta_si_sdr_db": row.get("p95_abs_delta_si_sdr_db"),
                "p05_proxy_si_sdr_db": row.get("p05_proxy_si_sdr_db"),
                "p95_proxy_rel_l2": row.get("p95_proxy_rel_l2"),
            }
        )

    all_pass = all(bool(x.get("pass", False)) for x in summary)
    return {
        "mode": mode,
        "corpus": eval_corpus,
        "models": models,
        "thresholds": {
            "median_abs_delta_si_sdr_db": float(threshold_median_abs_delta_db),
            "p95_abs_delta_si_sdr_db": float(threshold_p95_abs_delta_db),
            "p05_proxy_si_sdr_db": float(threshold_proxy_p05_sisdr_db),
            "p95_proxy_rel_l2": float(threshold_proxy_p95_rel_l2),
            "proxy_gate_enforced": bool(enforce_proxy_gate),
        },
        "results": results,
        "summary": summary,
        "all_pass": bool(all_pass),
    }


def _write_markdown(path: Path, payload: dict[str, Any]):
    latency_rows = payload["latency"]["summary"]
    parity_rows = payload["parity"]["summary"]
    quality_rows = payload["quality"]["summary"]

    lines = [
        "# Optimization Report",
        "",
        f"- Timestamp (UTC): `{payload['timestamp']}`",
        f"- Corpus files: `{len(payload['corpus'])}`",
        f"- Models: `{len(payload['models'])}`",
        f"- Overall pass: `{'yes' if payload['overall_pass'] else 'no'}`",
        "",
        "## Latency Gate",
        "",
        "| Model | Arch | Baseline (s) | Candidate (s) | Delta % | Criterion | Pass |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in latency_rows:
        delta = "n/a" if row["delta_pct"] is None else f"{row['delta_pct']:+.2f}%"
        lines.append(
            f"| `{row['model']}` | {row['arch']} | {row['baseline_median_s']:.3f} | "
            f"{row['candidate_median_s']:.3f} | {delta} | {row['criterion']} | "
            f"{'yes' if row['pass'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Parity Gate (Deterministic)",
            "",
            f"- Relative L2 threshold: `{payload['parity']['threshold_rel_l2']}`",
            f"- Gated arches: `{', '.join(payload['parity']['gated_arches'])}`",
            "",
            "| Model | Arch | Max rel L2 | Gated | Pass | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in parity_rows:
        lines.append(
            f"| `{row['model']}` | {row['arch']} | {row['max_rel_l2']:.8f} | "
            f"{'yes' if row['gated'] else 'no'} | {'yes' if row['pass'] else 'no'} | {row['status']} |"
        )

    lines.extend(
        [
            "",
            "## Quality Gate (Fast Mode)",
            "",
            f"- Mode: `{payload['quality']['mode']}`",
            f"- Proxy gate enforced: `{payload['quality']['thresholds'].get('proxy_gate_enforced', False)}`",
            "",
        ]
    )
    if payload["quality"]["mode"] == "reference_delta":
        lines.extend(
            [
                "| Model | Arch | Median |delta SI-SDR| (dB) | P95 |delta SI-SDR| (dB) | Missing | Pass |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in quality_rows:
            med = row["median_abs_delta_si_sdr_db"]
            p95 = row["p95_abs_delta_si_sdr_db"]
            lines.append(
                f"| `{row['model']}` | {row['arch']} | "
                f"{'n/a' if med is None else f'{med:.4f}'} | "
                f"{'n/a' if p95 is None else f'{p95:.4f}'} | "
                f"{row['missing_count']} | {'yes' if row['pass'] else 'no'} |"
            )
    else:
        lines.extend(
            [
                "| Model | Arch | P05 proxy SI-SDR (dB) | P95 proxy rel L2 | Missing | Pass |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in quality_rows:
            p05 = row["p05_proxy_si_sdr_db"]
            p95 = row["p95_proxy_rel_l2"]
            lines.append(
                f"| `{row['model']}` | {row['arch']} | "
                f"{'n/a' if p05 is None else f'{p05:.3f}'} | "
                f"{'n/a' if p95 is None else f'{p95:.6f}'} | "
                f"{row['missing_count']} | {'yes' if row['pass'] else 'no'} |"
            )

    py_mps = payload.get("python_mps_latency")
    if isinstance(py_mps, dict):
        lines.extend(["", "## Python MPS Baseline", ""])
        if py_mps.get("status") != "ok":
            lines.append(f"- Status: `unavailable` ({py_mps.get('error', 'unknown error')})")
        else:
            lines.extend(
                [
                    "| Model | MLX Candidate (s) | Python MPS (s) | MLX Speedup | Delta % (MLX vs Py) |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            rows = py_mps.get("summary", [])
            for row in rows:
                mlx = row.get("mlx_candidate_median_s")
                py = row.get("python_mps_median_s")
                speedup = row.get("mlx_speedup_vs_python_mps")
                delta = row.get("delta_pct_mlx_vs_py")
                lines.append(
                    f"| `{row.get('model')}` | "
                    f"{'n/a' if mlx is None else f'{mlx:.3f}'} | "
                    f"{'n/a' if py is None else f'{py:.3f}'} | "
                    f"{'n/a' if speedup is None else f'{speedup:.2f}x'} | "
                    f"{'n/a' if delta is None else f'{delta:+.2f}%'} |"
                )

    py_parity = payload.get("python_mps_parity")
    if isinstance(py_parity, dict):
        lines.extend(["", "## Python MPS Parity", ""])
        rows = py_parity.get("results", [])
        summary = py_parity.get("summary", {})
        lines.extend(
            [
                f"- Relative L2 threshold: `{summary.get('threshold_rel_l2')}`",
                f"- Fail-fast: `{summary.get('fail_fast')}`",
                f"- Max files: `{summary.get('max_files')}`",
                f"- Terminated early: `{summary.get('terminated_early')}`",
                f"- Stop reason: `{summary.get('stop_reason') or 'n/a'}`",
                "",
                "| Model | Arch | Files checked | Max rel L2 | Pass | Status |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            max_rel = row.get("max_rel_l2")
            lines.append(
                f"| `{row.get('model')}` | {row.get('arch_mlx') or 'unknown'} | "
                f"{row.get('files_checked', 0)} | "
                f"{'n/a' if max_rel is None else f'{max_rel:.8f}'} | "
                f"{'yes' if row.get('pass') else 'no'} | {row.get('status')} |"
            )

    repro = payload.get("reproducibility", {})
    if isinstance(repro, dict):
        git = repro.get("git", {})
        env = repro.get("environment", {})
        versions = env.get("versions", {}) if isinstance(env, dict) else {}
        lines.extend(
            [
                "",
                "## Reproducibility",
                "",
                f"- Command: `{repro.get('command', '')}`",
                f"- Working directory: `{repro.get('cwd', '')}`",
                f"- Git commit: `{git.get('commit')}`",
                f"- Git branch: `{git.get('branch')}`",
                f"- Git dirty: `{git.get('dirty')}`",
                f"- Git remote: `{git.get('remote_origin')}`",
                f"- Platform: `{env.get('platform')}`",
                f"- Python: `{versions.get('python')}`",
                f"- NumPy: `{versions.get('numpy')}`",
                f"- MLX: `{versions.get('mlx')}`",
                "",
                "| Corpus File | Size (bytes) | SHA256 | Duration (s) |",
                "|---|---:|---|---:|",
            ]
        )
        for row in repro.get("corpus_manifest", []):
            duration = row.get("duration_s")
            duration_str = "n/a" if duration is None else f"{float(duration):.3f}"
            lines.append(
                f"| `{row.get('path')}` | "
                f"{row.get('size_bytes', 'n/a')} | "
                f"{row.get('sha256', row.get('sha256_error', 'n/a'))} | "
                f"{duration_str} |"
            )
        lines.extend(["", "| Model | Resolved Path | Size (bytes) | SHA256 |", "|---|---|---:|---|"])
        for row in repro.get("model_manifest", []):
            lines.append(
                f"| `{row.get('model')}` | `{row.get('resolved_path')}` | "
                f"{row.get('size_bytes', 'n/a')} | "
                f"{row.get('sha256', row.get('sha256_error', 'n/a'))} |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cleanup_dir(path: Path):
    if not path.exists():
        return
    for root, dirs, files in os.walk(path, topdown=False):
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
        os.rmdir(path)
    except OSError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one-command parity + quality + latency optimization report.")
    parser.add_argument("--corpus-file", required=True, help="Text file with one audio path per line.")
    parser.add_argument("--baseline-config", required=True, help="Baseline JSON config path.")
    parser.add_argument("--candidate-config", required=True, help="Candidate JSON config path.")
    parser.add_argument("--models", default=None, help="Comma-separated model filenames (overrides config files).")
    parser.add_argument("--model-file-dir", default=None, help="Override model_file_dir for both runs.")
    parser.add_argument("--output-json", default=None, help="Output JSON path.")
    parser.add_argument("--output-markdown", default=None, help="Output Markdown path.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary output directories.")
    parser.add_argument(
        "--capture-corpus-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include SHA256 for corpus files in reproducibility metadata (default: true).",
    )
    parser.add_argument(
        "--capture-model-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include SHA256 for resolved model files in reproducibility metadata (default: true).",
    )

    # Latency gate
    parser.add_argument("--target-improvement-demucs-mdxc", type=float, default=20.0, help="Required improvement percentage.")
    parser.add_argument("--max-regression-mdx-vr", type=float, default=5.0, help="Allowed regression percentage.")

    # Parity gate
    parser.add_argument("--parity-max-files", type=int, default=1, help="Number of corpus files for deterministic parity (0=all).")
    parser.add_argument("--parity-threshold-rel-l2", type=float, default=1e-5, help="Relative L2 threshold for parity.")
    parser.add_argument("--parity-seed", type=int, default=12345, help="Deterministic seed for parity checks.")
    parser.add_argument(
        "--parity-demucs-shifts-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force Demucs shifts=0 in parity runs (default: true).",
    )
    parser.add_argument(
        "--parity-strict-demucs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Demucs in strict parity pass/fail gating (default: true).",
    )

    # Quality gate
    parser.add_argument(
        "--quality-reference-manifest",
        default=None,
        help="Optional JSON map: input audio path -> {stem_name: reference_stem_path}. "
        "When set, quality gate compares baseline/candidate SI-SDR against references.",
    )
    parser.add_argument("--quality-max-files", type=int, default=0, help="Number of corpus files for quality check (0=all).")
    parser.add_argument("--quality-threshold-median-abs-delta-db", type=float, default=0.05, help="Reference mode quality gate threshold.")
    parser.add_argument("--quality-threshold-p95-abs-delta-db", type=float, default=0.2, help="Reference mode quality gate threshold.")
    parser.add_argument("--quality-threshold-proxy-p05-sisdr-db", type=float, default=35.0, help="Proxy mode quality gate threshold.")
    parser.add_argument("--quality-threshold-proxy-p95-rel-l2", type=float, default=1e-3, help="Proxy mode quality gate threshold.")
    parser.add_argument(
        "--quality-enforce-proxy-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enforce strict pass/fail thresholds in proxy mode (default: false; proxy is informational).",
    )

    # Optional Python baseline latency (e.g., python-audio-separator on MPS).
    parser.add_argument(
        "--python-mps-latency",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include optional python-audio-separator latency results on MPS.",
    )
    parser.add_argument(
        "--python-mps-config",
        default=None,
        help="Optional JSON object of separator kwargs for python-audio-separator baseline.",
    )
    parser.add_argument(
        "--python-mps-warmup",
        type=int,
        default=None,
        help="Warmup runs for python MPS baseline (default: use candidate warmup).",
    )
    parser.add_argument(
        "--python-mps-repeats",
        type=int,
        default=None,
        help="Timed repeats for python MPS baseline (default: use candidate repeats).",
    )
    parser.add_argument(
        "--python-mps-parity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include optional MLX-vs-python deterministic parity checks.",
    )
    parser.add_argument(
        "--python-mps-parity-max-files",
        type=int,
        default=1,
        help="Number of corpus files for python MPS parity checks (0=all).",
    )
    parser.add_argument(
        "--python-mps-parity-threshold-rel-l2",
        type=float,
        default=1e-5,
        help="Relative L2 threshold for python MPS parity checks.",
    )
    parser.add_argument(
        "--python-mps-parity-seed",
        type=int,
        default=12345,
        help="Deterministic seed for python MPS parity checks.",
    )
    parser.add_argument(
        "--python-mps-parity-demucs-shifts-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force Demucs shifts=0 in python MPS parity checks (default: true).",
    )
    parser.add_argument(
        "--python-mps-parity-fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop python MPS parity checks after first model failure (default: true).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_payload = _load_json(Path(args.baseline_config))
    candidate_payload = _load_json(Path(args.candidate_config))
    corpus = _load_corpus(Path(args.corpus_file))
    models = _resolve_models(baseline_payload, candidate_payload, args.models)
    baseline_cfg = _build_run_config("baseline", baseline_payload)
    candidate_cfg = _build_run_config("candidate", candidate_payload)
    effective_model_file_dir = (
        args.model_file_dir
        or baseline_cfg.separator_kwargs.get("model_file_dir")
        or candidate_cfg.separator_kwargs.get("model_file_dir")
    )

    reference_manifest = None
    if args.quality_reference_manifest:
        reference_manifest = _load_reference_manifest(Path(args.quality_reference_manifest))

    python_mps_kwargs: dict[str, Any] | None = None
    if args.python_mps_config:
        cfg_path = Path(str(args.python_mps_config)).expanduser()
        if cfg_path.is_file():
            payload = _load_json(cfg_path)
        else:
            payload = json.loads(str(args.python_mps_config))
        if not isinstance(payload, dict):
            raise ValueError("--python-mps-config must be a JSON object or path to one.")
        python_mps_kwargs = payload

    repro_meta = collect_repro_metadata(
        corpus=corpus,
        models=models,
        model_file_dir=effective_model_file_dir,
        capture_corpus_hashes=bool(args.capture_corpus_hashes),
        capture_model_hashes=bool(args.capture_model_hashes),
    )

    run_temp = Path(tempfile.mkdtemp(prefix="optimization-report-"))
    try:
        baseline_latency = run_config(
            baseline_cfg,
            corpus=corpus,
            models=models,
            workspace_temp=run_temp,
            model_file_dir_override=args.model_file_dir,
        )
        candidate_latency = run_config(
            candidate_cfg,
            corpus=corpus,
            models=models,
            workspace_temp=run_temp,
            model_file_dir_override=args.model_file_dir,
        )

        latency_summary = compare_results(
            models=models,
            baseline=baseline_latency,
            candidate=candidate_latency,
            target_improvement_demucs_mdxc=float(args.target_improvement_demucs_mdxc),
            max_regression_mdx_vr=float(args.max_regression_mdx_vr),
        )
        latency_pass = all(bool(row.get("pass", False)) for row in latency_summary)

        parity_corpus = corpus if int(args.parity_max_files) == 0 else corpus[: int(args.parity_max_files)]
        parity_payload = run_equivalence_suite(
            corpus=parity_corpus,
            models=models,
            baseline_separator_kwargs=baseline_cfg.separator_kwargs,
            candidate_separator_kwargs=candidate_cfg.separator_kwargs,
            threshold_rel_l2=float(args.parity_threshold_rel_l2),
            seed=int(args.parity_seed),
            demucs_shifts_zero=bool(args.parity_demucs_shifts_zero),
            model_file_dir_override=args.model_file_dir,
            gated_arches={"Demucs", "MDXC", "MDX", "VR"} if args.parity_strict_demucs else {"MDXC", "MDX", "VR"},
        )
        parity_pass = bool(parity_payload.get("all_pass", False))

        quality_payload = run_quality_suite(
            corpus=corpus,
            models=models,
            baseline_separator_kwargs=baseline_cfg.separator_kwargs,
            candidate_separator_kwargs=candidate_cfg.separator_kwargs,
            workspace_temp=run_temp,
            model_file_dir_override=args.model_file_dir,
            reference_manifest=reference_manifest,
            max_files=int(args.quality_max_files),
            threshold_median_abs_delta_db=float(args.quality_threshold_median_abs_delta_db),
            threshold_p95_abs_delta_db=float(args.quality_threshold_p95_abs_delta_db),
            threshold_proxy_p05_sisdr_db=float(args.quality_threshold_proxy_p05_sisdr_db),
            threshold_proxy_p95_rel_l2=float(args.quality_threshold_proxy_p95_rel_l2),
            enforce_proxy_gate=bool(args.quality_enforce_proxy_gate),
        )
        quality_pass = bool(quality_payload.get("all_pass", False))

        python_mps_payload: dict[str, Any] | None = None
        python_mps_parity_payload: dict[str, Any] | None = None
        if bool(args.python_mps_latency):
            py_results = run_python_mps_latency(
                corpus=corpus,
                models=models,
                output_root=run_temp / "python_mps",
                model_file_dir=effective_model_file_dir,
                warmup=int(args.python_mps_warmup) if args.python_mps_warmup is not None else int(candidate_cfg.warmup),
                repeats=int(args.python_mps_repeats) if args.python_mps_repeats is not None else int(candidate_cfg.repeats),
                separator_kwargs=python_mps_kwargs,
            )
            summary: list[dict[str, Any]] = []
            if py_results.get("status") == "ok":
                py_rows = py_results.get("results", {})
                for model in models:
                    c_row = candidate_latency.get(model, {})
                    p_row = py_rows.get(model, {})
                    c_med = float(c_row.get("median_s", 0.0) or 0.0) if c_row.get("status") == "ok" else None
                    p_med = float(p_row.get("median_s", 0.0) or 0.0) if p_row.get("status") == "ok" else None
                    speedup = None
                    delta_pct = None
                    if c_med is not None and p_med is not None and c_med > 0:
                        speedup = p_med / c_med
                        delta_pct = ((c_med - p_med) / p_med) * 100.0
                    summary.append(
                        {
                            "model": model,
                            "mlx_candidate_median_s": c_med,
                            "python_mps_median_s": p_med,
                            "mlx_speedup_vs_python_mps": speedup,
                            "delta_pct_mlx_vs_py": delta_pct,
                            "python_status": p_row.get("status"),
                            "mlx_status": c_row.get("status"),
                        }
                    )
            python_mps_payload = dict(py_results)
            python_mps_payload["summary"] = summary if py_results.get("status") == "ok" else []

        if bool(args.python_mps_parity):
            python_mps_parity_payload = run_python_mps_parity(
                corpus=corpus,
                models=models,
                output_root=run_temp / "python_mps_parity",
                model_file_dir=effective_model_file_dir,
                mlx_separator_kwargs=candidate_cfg.separator_kwargs,
                pas_separator_kwargs=python_mps_kwargs,
                threshold_rel_l2=float(args.python_mps_parity_threshold_rel_l2),
                seed=int(args.python_mps_parity_seed),
                demucs_shifts_zero=bool(args.python_mps_parity_demucs_shifts_zero),
                demucs_mlx_strict_kernels=True,
                max_files=int(args.python_mps_parity_max_files),
                fail_fast=bool(args.python_mps_parity_fail_fast),
            )
    finally:
        if not args.keep_temp:
            _cleanup_dir(run_temp)

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_json = Path(args.output_json) if args.output_json else Path(f"/Users/sam/Code/mlx-audio-separator/perf_reports/optimization_report_{now}.json")
    output_md = Path(args.output_markdown) if args.output_markdown else output_json.with_suffix(".md")

    python_mps_parity_pass = True
    if bool(args.python_mps_parity):
        python_mps_parity_pass = bool((python_mps_parity_payload or {}).get("all_pass", False))

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus,
        "models": models,
        "reproducibility": repro_meta,
        "baseline_config": baseline_payload,
        "candidate_config": candidate_payload,
        "latency": {
            "baseline_results": baseline_latency,
            "candidate_results": candidate_latency,
            "summary": latency_summary,
            "all_pass": bool(latency_pass),
            "target_improvement_demucs_mdxc": float(args.target_improvement_demucs_mdxc),
            "max_regression_mdx_vr": float(args.max_regression_mdx_vr),
        },
        "parity": parity_payload,
        "quality": quality_payload,
        "python_mps_latency": python_mps_payload if bool(args.python_mps_latency) else None,
        "python_mps_parity": python_mps_parity_payload if bool(args.python_mps_parity) else None,
        "overall_pass": bool(latency_pass and parity_pass and quality_pass and python_mps_parity_pass),
        "gate_results": {
            "latency_pass": bool(latency_pass),
            "parity_pass": bool(parity_pass),
            "quality_pass": bool(quality_pass),
            "python_mps_parity_pass": bool(python_mps_parity_pass),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_md, payload)

    print(
        "Optimization gates: "
        f"latency={'pass' if latency_pass else 'fail'}, "
        f"parity={'pass' if parity_pass else 'fail'}, "
        f"quality={'pass' if quality_pass else 'fail'}, "
        f"python_mps_parity={'pass' if python_mps_parity_pass else 'fail'}"
    )
    if bool(args.python_mps_latency):
        status = payload.get("python_mps_latency", {}).get("status")
        print(f"Python MPS baseline: {status}")
    print(f"Overall: {'pass' if payload['overall_pass'] else 'fail'}")
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")


if __name__ == "__main__":
    main()
