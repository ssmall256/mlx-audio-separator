"""Performance utilities for inference tuning and tracing."""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


DEFAULT_PERFORMANCE_PARAMS = {
    "speed_mode": "default",
    "auto_tune_batch": False,
    "tune_probe_seconds": 8.0,
    "cache_clear_policy": "aggressive",
    "write_workers": 1,
    "experimental_vectorized_chunking": False,
    "experimental_roformer_fast_norm": False,
    "experimental_roformer_grouped_band_split": False,
    "experimental_roformer_grouped_mask_estimator": False,
    "experimental_roformer_grouped_weight_cache": False,
    "experimental_roformer_chunk_gather_batching": False,
    "experimental_roformer_fused_overlap_add": False,
    "experimental_roformer_ola_simd_tuning": False,
    "experimental_mlx_stream_pipeline": False,
    "experimental_roformer_compile_fullgraph": False,
    "experimental_compile_model_forward": False,
    "experimental_mdxc_defer_batch_eval": False,
    "experimental_mdxc_precompute_gather_idx": False,
    "experimental_demucs_apply_concat_batching": False,
    "experimental_demucs_wiener_preallocate_output": False,
    "experimental_demucs_gn_glu_multigroup": False,
    "experimental_vr_device_residency": False,
    "experimental_compile_shapeless": False,
    "experimental_roformer_static_compiled_demix": False,
    "experimental_flac_fast_write": False,
    "perf_trace": False,
    "perf_trace_path": None,
}


def normalize_performance_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize performance parameters."""
    out = dict(DEFAULT_PERFORMANCE_PARAMS)
    if params:
        out.update(params)

    out["speed_mode"] = str(out["speed_mode"]).strip().lower()
    if out["speed_mode"] not in {"default", "latency_safe", "latency_safe_v2", "latency_safe_v3"}:
        raise ValueError(
            "performance_params.speed_mode must be one of: "
            "default, latency_safe, latency_safe_v2, latency_safe_v3"
        )

    out["cache_clear_policy"] = str(out["cache_clear_policy"]).strip().lower()
    if out["cache_clear_policy"] not in {"aggressive", "deferred"}:
        raise ValueError("performance_params.cache_clear_policy must be one of: aggressive, deferred")

    out["auto_tune_batch"] = bool(out["auto_tune_batch"])
    out["perf_trace"] = bool(out["perf_trace"])
    out["experimental_vectorized_chunking"] = bool(out["experimental_vectorized_chunking"])
    out["experimental_roformer_fast_norm"] = bool(out["experimental_roformer_fast_norm"])
    out["experimental_roformer_grouped_band_split"] = bool(out["experimental_roformer_grouped_band_split"])
    out["experimental_roformer_grouped_mask_estimator"] = bool(out["experimental_roformer_grouped_mask_estimator"])
    out["experimental_roformer_grouped_weight_cache"] = bool(out["experimental_roformer_grouped_weight_cache"])
    out["experimental_roformer_chunk_gather_batching"] = bool(out["experimental_roformer_chunk_gather_batching"])
    out["experimental_roformer_fused_overlap_add"] = bool(out["experimental_roformer_fused_overlap_add"])
    out["experimental_roformer_ola_simd_tuning"] = bool(out["experimental_roformer_ola_simd_tuning"])
    out["experimental_mlx_stream_pipeline"] = bool(out["experimental_mlx_stream_pipeline"])
    out["experimental_roformer_compile_fullgraph"] = bool(out["experimental_roformer_compile_fullgraph"])
    out["experimental_compile_model_forward"] = bool(out["experimental_compile_model_forward"])
    out["experimental_mdxc_defer_batch_eval"] = bool(out["experimental_mdxc_defer_batch_eval"])
    out["experimental_mdxc_precompute_gather_idx"] = bool(out["experimental_mdxc_precompute_gather_idx"])
    out["experimental_demucs_apply_concat_batching"] = bool(out["experimental_demucs_apply_concat_batching"])
    out["experimental_demucs_wiener_preallocate_output"] = bool(out["experimental_demucs_wiener_preallocate_output"])
    out["experimental_demucs_gn_glu_multigroup"] = bool(out["experimental_demucs_gn_glu_multigroup"])
    out["experimental_vr_device_residency"] = bool(out["experimental_vr_device_residency"])
    out["experimental_compile_shapeless"] = bool(out["experimental_compile_shapeless"])
    out["experimental_roformer_static_compiled_demix"] = bool(out["experimental_roformer_static_compiled_demix"])
    out["experimental_flac_fast_write"] = bool(out["experimental_flac_fast_write"])

    out["tune_probe_seconds"] = float(out["tune_probe_seconds"])
    if out["tune_probe_seconds"] <= 0:
        raise ValueError("performance_params.tune_probe_seconds must be > 0")

    out["write_workers"] = int(out["write_workers"])
    if out["write_workers"] <= 0:
        raise ValueError("performance_params.write_workers must be >= 1")

    if out["perf_trace_path"] is not None:
        out["perf_trace_path"] = str(out["perf_trace_path"])

    return out


def clear_mlx_cache(logger=None):
    """Clear MLX cache in a compatibility-safe way."""
    try:
        import mlx.core as mx
    except Exception:
        return

    clear_fn = getattr(mx, "clear_cache", None)
    if callable(clear_fn):
        clear_fn()
        if logger:
            logger.debug("Cleared MLX cache via mx.clear_cache().")
        return

    metal = getattr(mx, "metal", None)
    if metal is not None:
        legacy_clear = getattr(metal, "clear_cache", None)
        if callable(legacy_clear):
            legacy_clear()
            if logger:
                logger.debug("Cleared MLX cache via mx.metal.clear_cache().")


class PerfTraceWriter:
    """Append-only JSONL writer for per-file performance trace records."""

    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def write(self, record: dict[str, Any]):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def tuning_cache_path() -> Path:
    base = Path.home() / ".cache" / "mlx-audio-separator"
    base.mkdir(parents=True, exist_ok=True)
    return base / "tuning.json"


def load_tuning_cache(path: Path | None = None) -> dict[str, Any]:
    path = path or tuning_cache_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_tuning_cache(data: dict[str, Any], path: Path | None = None):
    path = path or tuning_cache_path()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def select_best_candidate(timings_by_candidate: dict[int, list[float]], tie_ratio: float = 0.03) -> int:
    """Select best candidate by median timing, preferring smaller values on ties."""
    if not timings_by_candidate:
        raise ValueError("No candidates to select from.")

    med = {k: median(v) for k, v in timings_by_candidate.items() if v}
    if not med:
        raise ValueError("No timing samples to select from.")

    best_time = min(med.values())
    threshold = best_time * (1.0 + float(tie_ratio))
    eligible = sorted([k for k, t in med.items() if t <= threshold])
    return eligible[0]


@dataclass
class _SaveTask:
    stem_path: str
    stem_source: np.ndarray
    sample_rate: int
    encoding: str
    bitrate: str
    flac_fast_write: bool = False


class AsyncStemWriter:
    """Concurrent audio writer used by separators when write_workers > 1."""

    def __init__(self, workers: int = 2):
        if workers <= 0:
            raise ValueError("workers must be >= 1")
        self._workers = int(workers)
        self._queue: "queue.Queue[_SaveTask | None]" = queue.Queue(maxsize=max(4, workers * 2))
        self._error: BaseException | None = None
        self._threads = [
            threading.Thread(target=self._run, daemon=True, name=f"stem-writer-{idx}")
            for idx in range(self._workers)
        ]
        for thread in self._threads:
            thread.start()

    def _run(self):
        import mlx_audio_io as mac

        while True:
            task = self._queue.get()
            try:
                if task is None:
                    self._queue.task_done()
                    return
                mac.save(
                    str(task.stem_path),
                    task.stem_source,
                    task.sample_rate,
                    encoding=task.encoding,
                    bitrate=task.bitrate,
                    flac_compression=("fast" if task.flac_fast_write else "default"),
                )
            except TypeError:
                # Backward-compatible fallback for mlx-audio-io versions without flac_compression.
                mac.save(
                    str(task.stem_path),
                    task.stem_source,
                    task.sample_rate,
                    encoding=task.encoding,
                    bitrate=task.bitrate,
                )
            except BaseException as exc:
                self._error = exc
            finally:
                if task is not None:
                    self._queue.task_done()

    def submit(
        self,
        stem_path: str,
        stem_source: np.ndarray,
        sample_rate: int,
        encoding: str,
        bitrate: str,
        flac_fast_write: bool = False,
    ):
        if self._error is not None:
            raise self._error
        task = _SaveTask(
            stem_path=stem_path,
            stem_source=np.ascontiguousarray(stem_source),
            sample_rate=int(sample_rate),
            encoding=str(encoding),
            bitrate=str(bitrate),
            flac_fast_write=bool(flac_fast_write),
        )
        self._queue.put(task)

    def flush(self):
        self._queue.join()
        if self._error is not None:
            raise self._error

    def close(self):
        for _ in range(self._workers):
            self._queue.put(None)
        self._queue.join()
        for thread in self._threads:
            thread.join()
        if self._error is not None:
            raise self._error
