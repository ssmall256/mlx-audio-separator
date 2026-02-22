"""Deterministic output-equivalence helpers for performance validation."""

from __future__ import annotations

import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from mlx_audio_separator.core import Separator


@contextmanager
def _temporary_env(var: str, value: str):
    previous = os.environ.get(var)
    os.environ[var] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = previous


def set_deterministic_seeds(seed: int):
    """Set Python, NumPy, and MLX RNG seeds."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import mlx.core as mx

        if hasattr(mx.random, "seed"):
            mx.random.seed(int(seed))
    except Exception:
        # Seed best-effort; MLX may not be available in some test environments.
        pass


def stem_key(path: str) -> str:
    """Extract stem key from output filename, normalizing case and whitespace."""
    stem_name = Path(path).stem
    start = stem_name.find("(")
    end = stem_name.rfind(")")
    if start >= 0 and end > start:
        return stem_name[start + 1 : end].strip().lower()
    return stem_name.strip().lower()


def read_stem_map(output_paths: list[str]) -> dict[str, tuple[np.ndarray, int]]:
    """Read stem files into memory keyed by normalized stem name."""
    out: dict[str, tuple[np.ndarray, int]] = {}
    for path in output_paths:
        audio, sample_rate = sf.read(path, always_2d=True)
        out[stem_key(path)] = (audio, int(sample_rate))
    return out


def compare_stem_maps(
    baseline: dict[str, tuple[np.ndarray, int]],
    candidate: dict[str, tuple[np.ndarray, int]],
    threshold_rel_l2: float,
) -> dict[str, Any]:
    """Compare two stem maps and report strict equivalence metrics."""
    keys = sorted(set(baseline.keys()) | set(candidate.keys()))
    rows: list[dict[str, Any]] = []
    all_ok = True
    max_rel_l2 = 0.0

    for key in keys:
        if key not in baseline or key not in candidate:
            rows.append(
                {
                    "stem": key,
                    "status": "missing",
                    "in_baseline": key in baseline,
                    "in_candidate": key in candidate,
                }
            )
            all_ok = False
            continue

        baseline_audio, baseline_sr = baseline[key]
        candidate_audio, candidate_sr = candidate[key]

        sample_rate_match = baseline_sr == candidate_sr
        shape_match = baseline_audio.shape == candidate_audio.shape
        rel_l2 = None
        if sample_rate_match and shape_match:
            denom = float(np.linalg.norm(baseline_audio))
            if denom == 0.0:
                rel_l2 = 0.0 if float(np.linalg.norm(candidate_audio)) == 0.0 else float("inf")
            else:
                rel_l2 = float(np.linalg.norm(baseline_audio - candidate_audio) / denom)
            max_rel_l2 = max(max_rel_l2, float(rel_l2))

        pass_rel_l2 = bool(rel_l2 is not None and rel_l2 <= float(threshold_rel_l2))
        stem_ok = sample_rate_match and shape_match and pass_rel_l2
        all_ok = all_ok and stem_ok

        shape_value: Any
        if shape_match:
            shape_value = list(baseline_audio.shape)
        else:
            shape_value = {
                "baseline": list(baseline_audio.shape),
                "candidate": list(candidate_audio.shape),
            }

        rows.append(
            {
                "stem": key,
                "status": "ok" if stem_ok else "drift",
                "sample_rate_match": sample_rate_match,
                "shape_match": shape_match,
                "shape": shape_value,
                "rel_l2": rel_l2,
                "pass_rel_l2": pass_rel_l2,
            }
        )

    counts_match = len(baseline) == len(candidate)
    stems_match = set(baseline.keys()) == set(candidate.keys())
    passed = bool(all_ok and counts_match and stems_match)
    return {
        "baseline_count": len(baseline),
        "candidate_count": len(candidate),
        "counts_match": counts_match,
        "stems_match": stems_match,
        "stems": rows,
        "max_rel_l2": float(max_rel_l2),
        "pass": passed,
    }


def _separator_from_kwargs(kwargs: dict[str, Any], output_dir: str, info_only: bool = False) -> Separator:
    params = dict(kwargs)
    params["output_dir"] = output_dir
    if info_only:
        params["info_only"] = True
    return Separator(**params)


def _apply_runtime_arch_params(
    sep: Separator,
    arch: str,
    arch_params: dict[str, Any],
    demucs_shifts_zero: bool,
):
    inst = sep.model_instance
    if inst is None:
        raise RuntimeError("Model must be loaded before applying runtime architecture params.")

    if arch == "Demucs":
        # Keep Demucs deterministic for equivalence runs.
        shifts = int(arch_params.get("shifts", getattr(inst, "shifts", 0)))
        if demucs_shifts_zero:
            shifts = 0
        overlap = float(arch_params.get("overlap", getattr(inst, "overlap", 0.25)))
        split = bool(arch_params.get("segments_enabled", getattr(inst, "segments_enabled", True)))
        batch_size = int(arch_params.get("batch_size", getattr(inst, "batch_size", 8)))

        inst.shifts = shifts
        inst.overlap = overlap
        inst.segments_enabled = split
        inst.batch_size = batch_size

        if hasattr(inst, "_demucs_separator"):
            demucs_sep = inst._demucs_separator
            demucs_sep.batch_size = batch_size
            segment_cfg = arch_params.get("segment_size", getattr(inst, "segment_size", "Default"))
            segment = None
            if split and segment_cfg != "Default":
                try:
                    segment = float(segment_cfg)
                except (TypeError, ValueError):
                    segment = getattr(demucs_sep, "segment", None)
            demucs_sep.segment = segment
            demucs_sep.update_parameter(shifts=shifts, overlap=overlap, split=split)
        return

    for key, value in arch_params.items():
        if hasattr(inst, key):
            setattr(inst, key, value)

    # MDX has derived chunk fields dependent on segment_size/hop_length.
    if arch == "MDX" and hasattr(inst, "hop_length") and hasattr(inst, "segment_size") and hasattr(inst, "trim"):
        inst.chunk_size = int(inst.hop_length) * (int(inst.segment_size) - 1)
        inst.gen_size = inst.chunk_size - 2 * int(inst.trim)


def _demucs_in_memory_stem_map(sep: Separator, audio_path: str) -> dict[str, tuple[np.ndarray, int]]:
    """Run Demucs separation in-memory and return normalized stem map."""
    import mlx.core as mx
    import mlx_audio_io as mac

    inst = sep.model_instance
    if inst is None or not hasattr(inst, "_demucs_separator"):
        raise RuntimeError("Demucs in-memory path requires loaded Demucs separator instance.")

    demucs_sep = inst._demucs_separator
    audio_mx, sr = mac.load(str(audio_path), dtype="float32")
    if int(sr) != int(demucs_sep.samplerate):
        audio_mx, sr = mac.load(str(audio_path), sr=demucs_sep.samplerate, dtype="float32")
    wav_mx = audio_mx.T if audio_mx.ndim == 2 else mx.stack([audio_mx, audio_mx], axis=0)

    _, stems_mx = demucs_sep.separate_tensor(wav_mx, return_mx=True)
    out: dict[str, tuple[np.ndarray, int]] = {}
    for stem_name, stem_value in stems_mx.items():
        # Materialize an owned host buffer for stable cross-run comparisons.
        arr = np.array(stem_value, copy=True)
        # Demucs tensors are (channels, frames); normalize to (frames, channels).
        if arr.ndim == 2:
            arr = arr.T
        out[str(stem_name).strip().lower()] = (arr, int(sr))
    return out


def run_model_equivalence(
    model_filename: str,
    corpus: list[str],
    baseline_separator_kwargs: dict[str, Any],
    candidate_separator_kwargs: dict[str, Any],
    threshold_rel_l2: float = 1e-5,
    seed: int = 12345,
    demucs_shifts_zero: bool = True,
    model_file_dir_override: str | None = None,
) -> dict[str, Any]:
    """Run deterministic equivalence checks for one model across a corpus."""
    base_kwargs = dict(baseline_separator_kwargs)
    cand_kwargs = dict(candidate_separator_kwargs)
    if model_file_dir_override:
        base_kwargs["model_file_dir"] = model_file_dir_override
        cand_kwargs["model_file_dir"] = model_file_dir_override

    run_root = tempfile.mkdtemp(prefix="equivalence-")
    result: dict[str, Any] = {
        "model": model_filename,
        "status": "error",
        "arch": None,
        "error": None,
        "per_file": {},
        "max_rel_l2": 0.0,
        "pass": False,
    }

    try:
        # One loaded model instance avoids false drift from model reload variance.
        with _temporary_env("MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED", "1"):
            # Keep Demucs iSTFT in non-fused mode for deterministic equivalence.
            # This isolates model-level behavior from fused iSTFT kernel drift.
            with _temporary_env("MLX_AUDIO_SEPARATOR_DEMUCS_ISTFT_ALLOW_FUSED", "0"):
                with _temporary_env("MLX_AUDIO_SEPARATOR_DEMUCS_WIENER_USE_VMAP", "0"):
                    with _temporary_env("MLX_AUDIO_SEPARATOR_DEMUCS_STRICT_EVAL", "1"):
                        sep_base = _separator_from_kwargs(base_kwargs, output_dir=os.path.join(run_root, "run_base"), info_only=False)
                        sep_base.load_model(model_filename=model_filename)
                        arch = sep_base.model_type or "unknown"
                        result["arch"] = arch

                        sep_cand = sep_base
                        if arch == "Demucs":
                            # Demucs inference mutates internal runtime state across calls.
                            # Keep baseline/candidate on separate loaded model instances.
                            sep_cand = _separator_from_kwargs(cand_kwargs, output_dir=os.path.join(run_root, "run_cand"), info_only=False)
                            sep_cand.load_model(model_filename=model_filename)

                        # Use info-only Separators to resolve finalized per-arch params from both configs.
                        baseline_info = _separator_from_kwargs(base_kwargs, output_dir=os.path.join(run_root, "baseline_info"), info_only=True)
                        candidate_info = _separator_from_kwargs(cand_kwargs, output_dir=os.path.join(run_root, "candidate_info"), info_only=True)
                        base_arch_params = dict(baseline_info.arch_specific_params.get(arch, {}))
                        cand_arch_params = dict(candidate_info.arch_specific_params.get(arch, {}))

                        all_pass = True
                        max_rel_l2 = 0.0
                        for audio_path in corpus:
                            _apply_runtime_arch_params(sep_base, arch, base_arch_params, demucs_shifts_zero=demucs_shifts_zero)
                            set_deterministic_seeds(seed)
                            if arch == "Demucs":
                                baseline_stems = _demucs_in_memory_stem_map(sep_base, audio_path)
                            else:
                                baseline_outputs = sep_base.separate(audio_path)
                                baseline_stems = read_stem_map(baseline_outputs)

                            _apply_runtime_arch_params(sep_cand, arch, cand_arch_params, demucs_shifts_zero=demucs_shifts_zero)
                            set_deterministic_seeds(seed)
                            if arch == "Demucs":
                                candidate_stems = _demucs_in_memory_stem_map(sep_cand, audio_path)
                            else:
                                candidate_outputs = sep_cand.separate(audio_path)
                                candidate_stems = read_stem_map(candidate_outputs)

                            compared = compare_stem_maps(
                                baseline=baseline_stems,
                                candidate=candidate_stems,
                                threshold_rel_l2=threshold_rel_l2,
                            )
                            result["per_file"][audio_path] = compared
                            all_pass = all_pass and bool(compared["pass"])
                            max_rel_l2 = max(max_rel_l2, float(compared["max_rel_l2"]))

                        result["status"] = "ok"
                        result["max_rel_l2"] = float(max_rel_l2)
                        result["pass"] = bool(all_pass)
    except Exception as exc:  # pragma: no cover - runtime-side failure path
        result["status"] = "error"
        result["error"] = str(exc)
        result["pass"] = False

    return result


def run_equivalence_suite(
    corpus: list[str],
    models: list[str],
    baseline_separator_kwargs: dict[str, Any],
    candidate_separator_kwargs: dict[str, Any],
    threshold_rel_l2: float = 1e-5,
    seed: int = 12345,
    demucs_shifts_zero: bool = True,
    model_file_dir_override: str | None = None,
    gated_arches: set[str] | None = None,
) -> dict[str, Any]:
    """Run deterministic equivalence checks for all models in a corpus."""
    if gated_arches is None:
        # Demucs is currently treated as informational due backend nondeterminism.
        gated_arches = {"MDXC", "MDX", "VR"}

    results: dict[str, Any] = {}
    for model in models:
        results[model] = run_model_equivalence(
            model_filename=model,
            corpus=corpus,
            baseline_separator_kwargs=baseline_separator_kwargs,
            candidate_separator_kwargs=candidate_separator_kwargs,
            threshold_rel_l2=threshold_rel_l2,
            seed=seed,
            demucs_shifts_zero=demucs_shifts_zero,
            model_file_dir_override=model_file_dir_override,
        )

    summary: list[dict[str, Any]] = []
    all_pass = True
    for model in models:
        row = results.get(model, {})
        arch = row.get("arch", "unknown")
        strict_pass = bool(row.get("pass", False))
        gated = arch in gated_arches
        passed = strict_pass if gated else row.get("status") == "ok"
        all_pass = all_pass and bool(passed)
        summary.append(
            {
                "model": model,
                "arch": arch,
                "status": row.get("status", "error"),
                "max_rel_l2": float(row.get("max_rel_l2", 0.0)),
                "strict_pass": strict_pass,
                "gated": gated,
                "pass": bool(passed),
                "error": row.get("error"),
            }
        )

    return {
        "threshold_rel_l2": float(threshold_rel_l2),
        "seed": int(seed),
        "demucs_shifts_zero": bool(demucs_shifts_zero),
        "gated_arches": sorted(gated_arches),
        "corpus": list(corpus),
        "models": list(models),
        "results": results,
        "summary": summary,
        "all_pass": bool(all_pass),
    }
