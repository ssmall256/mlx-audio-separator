#!/usr/bin/env python3
"""Bisect Demucs MLX nondeterminism by tracing layer outputs across two runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx_audio_io as mac
import numpy as np

from mlx_audio_separator.demucs_mlx.api import Separator as DemucsMLXSeparator
from mlx_audio_separator.utils.equivalence import set_deterministic_seeds


def _clear_cache_compat():
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
            return
    except Exception:
        pass
    try:
        metal = getattr(mx, "metal", None)
        if metal is not None and hasattr(metal, "clear_cache"):
            metal.clear_cache()
    except Exception:
        pass


def _iter_arrays(value: Any, prefix: str = "out"):
    if isinstance(value, mx.array):
        yield prefix, value
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            yield from _iter_arrays(item, f"{prefix}[{idx}]")
        return
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            yield from _iter_arrays(value[key], f"{prefix}.{key}")


def _sample_rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    denom = float(np.linalg.norm(a64))
    if denom == 0.0:
        return 0.0 if float(np.linalg.norm(b64)) == 0.0 else float("inf")
    return float(np.linalg.norm(a64 - b64) / denom)


def _fingerprint_array(arr: mx.array, sample_size: int) -> dict[str, Any]:
    np_arr = np.asarray(arr)
    if np.iscomplexobj(np_arr):
        flat = np.stack([np_arr.real, np_arr.imag], axis=-1).reshape(-1)
    else:
        flat = np_arr.reshape(-1)
    if flat.size == 0:
        sample = np.empty((0,), dtype=np.float32)
    else:
        step = max(1, int(np.ceil(flat.size / max(1, sample_size))))
        sample = flat[::step][:sample_size].astype(np.float32, copy=False)
    digest = hashlib.blake2b(sample.tobytes(), digest_size=16).hexdigest()
    sample64 = sample.astype(np.float64, copy=False)
    return {
        "shape": [int(x) for x in np_arr.shape],
        "dtype": str(np_arr.dtype),
        "size": int(np_arr.size),
        "sample_hash": digest,
        "sample_mean": float(np.mean(sample64)) if sample64.size else 0.0,
        "sample_std": float(np.std(sample64)) if sample64.size else 0.0,
        "sample_l2": float(np.linalg.norm(sample64)) if sample64.size else 0.0,
        "_sample": sample.copy(),
    }


@dataclass
class TraceConfig:
    sample_size: int
    max_events: int


class ModuleTracer:
    def __init__(self, module_paths: dict[int, str], cfg: TraceConfig):
        self.module_paths = module_paths
        self.cfg = cfg
        self.call_counts: dict[int, int] = {}
        self.events: list[dict[str, Any]] = []
        self.dropped_events = 0
        self.record_errors = 0

    def record(self, module: Any, method: str, output: Any):
        if len(self.events) >= self.cfg.max_events:
            self.dropped_events += 1
            return
        mid = id(module)
        path = self.module_paths.get(mid)
        if path is None:
            return
        call_index = self.call_counts.get(mid, 0)
        self.call_counts[mid] = call_index + 1
        event = {
            "module_path": path,
            "module_class": module.__class__.__name__,
            "method": method,
            "call_index": int(call_index),
            "event_key": f"{path}.{method}#{call_index}",
            "arrays": [],
        }
        try:
            for slot, arr in _iter_arrays(output):
                fp = _fingerprint_array(arr, sample_size=self.cfg.sample_size)
                fp["slot"] = slot
                event["arrays"].append(fp)
        except Exception:
            self.record_errors += 1
        self.events.append(event)


def _collect_modules(root: Any) -> dict[int, tuple[str, Any]]:
    out: dict[int, tuple[str, Any]] = {}
    seen: set[int] = set()

    def walk(obj: Any, path: str):
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        module_name = getattr(obj.__class__, "__module__", "")
        if isinstance(obj, nn.Module):
            # Use named_modules() as the canonical child traversal for MLX nn.Module.
            try:
                named = obj.named_modules()
            except Exception:
                named = [("", obj)]
            for child_name, child_module in named:
                child_path = path if child_name == "" else f"{path}.{child_name}"
                out[id(child_module)] = (child_path, child_module)
            return

        if isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                walk(item, f"{path}[{idx}]")
            return

        if isinstance(obj, dict):
            for key in sorted(obj.keys(), key=lambda x: str(x)):
                walk(obj[key], f"{path}.{key}")
            return

        # BagOfModelsMLX and related wrappers are regular Python objects.
        # Recurse through Demucs-local object graphs to discover nested modules.
        if module_name.startswith("mlx_audio_separator.demucs_mlx") and hasattr(obj, "__dict__"):
            for key, value in vars(obj).items():
                walk(value, f"{path}.{key}")

    walk(root, "model")
    return out


@contextmanager
def _trace_model_calls(model: Any, tracer: ModuleTracer):
    modules = _collect_modules(model)
    module_paths = {mid: path for mid, (path, _) in modules.items()}
    tracer.module_paths = module_paths

    classes: dict[type, None] = {}
    for _, instance in modules.values():
        cls = instance.__class__
        module_name = getattr(cls, "__module__", "")
        if module_name.startswith("mlx_audio_separator.demucs_mlx"):
            classes[cls] = None

    patches: list[tuple[type, str, Any]] = []

    def patch_method(cls: type, method_name: str):
        original = getattr(cls, method_name, None)
        if not callable(original):
            return

        def wrapped(self, *args, __orig=original, __m=method_name, **kwargs):
            out = __orig(self, *args, **kwargs)
            tracer.record(self, __m, out)
            return out

        setattr(cls, method_name, wrapped)
        patches.append((cls, method_name, original))

    try:
        for cls in classes.keys():
            patch_method(cls, "__call__")
            for method_name in ("_spec", "_magnitude", "_mask", "_wiener", "_ispec"):
                patch_method(cls, method_name)
        yield
    finally:
        for cls, method_name, original in reversed(patches):
            setattr(cls, method_name, original)


def _json_safe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        event_row = {
            "module_path": event["module_path"],
            "module_class": event["module_class"],
            "method": event["method"],
            "call_index": event["call_index"],
            "event_key": event["event_key"],
            "arrays": [],
        }
        for arr in event["arrays"]:
            event_row["arrays"].append(
                {
                    "slot": arr["slot"],
                    "shape": arr["shape"],
                    "dtype": arr["dtype"],
                    "size": arr["size"],
                    "sample_hash": arr["sample_hash"],
                    "sample_mean": arr["sample_mean"],
                    "sample_std": arr["sample_std"],
                    "sample_l2": arr["sample_l2"],
                }
            )
        out.append(event_row)
    return out


def _compare_traces(run_a: list[dict[str, Any]], run_b: list[dict[str, Any]]) -> dict[str, Any]:
    min_len = min(len(run_a), len(run_b))
    for idx in range(min_len):
        a = run_a[idx]
        b = run_b[idx]
        if a["event_key"] != b["event_key"]:
            return {
                "status": "diverged",
                "event_index": idx,
                "reason": "event_key_mismatch",
                "event_a": a["event_key"],
                "event_b": b["event_key"],
            }

        if len(a["arrays"]) != len(b["arrays"]):
            return {
                "status": "diverged",
                "event_index": idx,
                "reason": "array_count_mismatch",
                "event_key": a["event_key"],
                "array_count_a": len(a["arrays"]),
                "array_count_b": len(b["arrays"]),
            }

        for arr_idx, (arr_a, arr_b) in enumerate(zip(a["arrays"], b["arrays"])):
            if arr_a["slot"] != arr_b["slot"]:
                return {
                    "status": "diverged",
                    "event_index": idx,
                    "reason": "array_slot_mismatch",
                    "event_key": a["event_key"],
                    "slot_a": arr_a["slot"],
                    "slot_b": arr_b["slot"],
                }
            if arr_a["shape"] != arr_b["shape"] or arr_a["dtype"] != arr_b["dtype"]:
                return {
                    "status": "diverged",
                    "event_index": idx,
                    "reason": "array_metadata_mismatch",
                    "event_key": a["event_key"],
                    "slot": arr_a["slot"],
                    "shape_a": arr_a["shape"],
                    "shape_b": arr_b["shape"],
                    "dtype_a": arr_a["dtype"],
                    "dtype_b": arr_b["dtype"],
                }
            if arr_a["sample_hash"] != arr_b["sample_hash"]:
                sample_rel = _sample_rel_l2(arr_a["_sample"], arr_b["_sample"])
                return {
                    "status": "diverged",
                    "event_index": idx,
                    "reason": "array_sample_hash_mismatch",
                    "event_key": a["event_key"],
                    "slot": arr_a["slot"],
                    "array_index": arr_idx,
                    "sample_rel_l2": float(sample_rel),
                    "sample_hash_a": arr_a["sample_hash"],
                    "sample_hash_b": arr_b["sample_hash"],
                    "sample_mean_a": float(arr_a["sample_mean"]),
                    "sample_mean_b": float(arr_b["sample_mean"]),
                    "sample_std_a": float(arr_a["sample_std"]),
                    "sample_std_b": float(arr_b["sample_std"]),
                }

    if len(run_a) != len(run_b):
        return {
            "status": "diverged",
            "event_index": min_len,
            "reason": "event_count_mismatch",
            "event_count_a": len(run_a),
            "event_count_b": len(run_b),
        }

    return {"status": "equal", "event_count": len(run_a)}


def _compare_stems(stems_a: dict[str, np.ndarray], stems_b: dict[str, np.ndarray]) -> dict[str, Any]:
    keys = sorted(set(stems_a.keys()) | set(stems_b.keys()))
    rows: list[dict[str, Any]] = []
    max_rel_l2 = 0.0
    for key in keys:
        if key not in stems_a or key not in stems_b:
            rows.append({"stem": key, "status": "missing"})
            continue
        a = stems_a[key]
        b = stems_b[key]
        if a.shape != b.shape:
            rows.append({"stem": key, "status": "shape_mismatch", "shape_a": list(a.shape), "shape_b": list(b.shape)})
            continue
        rel = _sample_rel_l2(a.astype(np.float32, copy=False).reshape(-1), b.astype(np.float32, copy=False).reshape(-1))
        max_rel_l2 = max(max_rel_l2, rel)
        rows.append({"stem": key, "status": "ok", "rel_l2": float(rel)})
    return {"stems": rows, "max_rel_l2": float(max_rel_l2)}


def _run_once(
    separator: DemucsMLXSeparator,
    wav_mx: mx.array,
    seed: int,
    clear_cache: bool,
) -> dict[str, np.ndarray]:
    set_deterministic_seeds(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    if clear_cache:
        _clear_cache_compat()
    _, stems_mx = separator.separate_tensor(wav_mx, return_mx=True)
    # Own host-side buffers so subsequent runs cannot alias and overwrite references.
    return {name: np.array(value, copy=True) for name, value in stems_mx.items()}


def _write_markdown(path: Path, payload: dict[str, Any]):
    lines = [
        "# Demucs Nondeterminism Bisector",
        "",
        f"- Audio: `{payload['audio_file']}`",
        f"- Model: `{payload['model']}`",
        f"- Fused GroupNorm mode: `{payload['fused_groupnorm_mode']}`",
        f"- Shifts: `{payload['shifts']}`",
        f"- Split: `{payload['split']}`",
        f"- Segment: `{payload['segment']}`",
        f"- Batch size: `{payload['batch_size']}`",
        "",
        "## Summary",
        "",
        f"- First divergence status: `{payload['trace_compare']['status']}`",
        f"- Stem max rel L2: `{payload['stem_compare']['max_rel_l2']}`",
        f"- Trace events run1: `{payload['run1']['event_count']}`",
        f"- Trace events run2: `{payload['run2']['event_count']}`",
    ]

    if payload["trace_compare"]["status"] == "diverged":
        d = payload["trace_compare"]
        lines.extend(
            [
                "",
                "## First Divergence",
                "",
                f"- Reason: `{d.get('reason')}`",
                f"- Event index: `{d.get('event_index')}`",
                f"- Event key: `{d.get('event_key', d.get('event_a', 'n/a'))}`",
            ]
        )
        if "sample_rel_l2" in d:
            lines.append(f"- Sample rel L2: `{d['sample_rel_l2']}`")
        if "slot" in d:
            lines.append(f"- Output slot: `{d['slot']}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bisect Demucs MLX nondeterminism layer-by-layer.")
    parser.add_argument("--audio-file", required=True, help="Input audio file path.")
    parser.add_argument("--model", default="htdemucs", help="Demucs model name in demucs_mlx registry.")
    parser.add_argument("--fused-groupnorm-mode", default="auto", choices=["auto", "all", "glu_only", "gelu_only", "off"])
    parser.add_argument("--deterministic-fused", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--shifts", type=int, default=0)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clear-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-size", type=int, default=4096)
    parser.add_argument("--max-events", type=int, default=20000)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    audio_file = os.path.abspath(os.path.expanduser(args.audio_file))
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    if str(args.fused_groupnorm_mode) == "auto":
        os.environ.pop("MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE", None)
        fused_mode_effective = "auto"
    else:
        os.environ["MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE"] = str(args.fused_groupnorm_mode)
        fused_mode_effective = str(args.fused_groupnorm_mode)
    if bool(args.deterministic_fused):
        os.environ["MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED"] = "1"

    wav_mx, sr = mac.load(audio_file, dtype="float32")
    wav_mx = wav_mx.T if wav_mx.ndim == 2 else mx.stack([wav_mx, wav_mx], axis=0)

    separator = DemucsMLXSeparator(
        model=str(args.model),
        shifts=int(args.shifts),
        overlap=float(args.overlap),
        split=bool(args.split),
        segment=args.segment,
        batch_size=int(args.batch_size),
        progress=False,
    )
    if int(sr) != separator.samplerate:
        wav_mx, _ = mac.load(audio_file, sr=separator.samplerate, dtype="float32")
        wav_mx = wav_mx.T if wav_mx.ndim == 2 else mx.stack([wav_mx, wav_mx], axis=0)

    cfg = TraceConfig(sample_size=int(args.sample_size), max_events=int(args.max_events))
    tracer1 = ModuleTracer(module_paths={}, cfg=cfg)
    tracer2 = ModuleTracer(module_paths={}, cfg=cfg)

    t0 = time.perf_counter()
    with _trace_model_calls(separator.model, tracer1):
        stems1 = _run_once(separator, wav_mx, seed=int(args.seed), clear_cache=bool(args.clear_cache))
    run1_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    with _trace_model_calls(separator.model, tracer2):
        stems2 = _run_once(separator, wav_mx, seed=int(args.seed), clear_cache=bool(args.clear_cache))
    run2_s = time.perf_counter() - t0

    trace_compare = _compare_traces(tracer1.events, tracer2.events)
    stem_compare = _compare_stems(stems1, stems2)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audio_file": audio_file,
        "model": str(args.model),
        "sample_rate": int(separator.samplerate),
        "fused_groupnorm_mode": fused_mode_effective,
        "deterministic_fused": bool(args.deterministic_fused),
        "seed": int(args.seed),
        "shifts": int(args.shifts),
        "overlap": float(args.overlap),
        "split": bool(args.split),
        "segment": None if args.segment is None else float(args.segment),
        "batch_size": int(args.batch_size),
        "clear_cache": bool(args.clear_cache),
        "run1": {
            "duration_s": float(run1_s),
            "event_count": len(tracer1.events),
            "dropped_events": int(tracer1.dropped_events),
            "record_errors": int(tracer1.record_errors),
            "events": _json_safe_events(tracer1.events),
        },
        "run2": {
            "duration_s": float(run2_s),
            "event_count": len(tracer2.events),
            "dropped_events": int(tracer2.dropped_events),
            "record_errors": int(tracer2.record_errors),
            "events": _json_safe_events(tracer2.events),
        },
        "trace_compare": trace_compare,
        "stem_compare": stem_compare,
    }

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_json = Path(args.output_json) if args.output_json else (Path("perf_reports") / f"demucs_bisect_{now}.json")
    output_md = Path(args.output_markdown) if args.output_markdown else output_json.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_md, payload)

    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")
    print(f"Trace compare: {trace_compare}")
    print(f"Stem max rel L2: {stem_compare['max_rel_l2']}")


if __name__ == "__main__":
    main()
