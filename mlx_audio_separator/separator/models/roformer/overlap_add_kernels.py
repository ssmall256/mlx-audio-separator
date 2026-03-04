"""Roformer overlap-add helpers for experimental fused accumulation paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import mlx.core as mx


_METAL_WEIGHTED_ACCUM_SOURCE = r'''
int num_chunks = params[0];
int safe_len = params[1];
int span_len = params[2];
int num_sc = params[3];

int t = (int)thread_position_in_grid.x;
int sc = (int)thread_position_in_grid.y;
if (t >= span_len || sc >= num_sc) return;

float acc = 0.0f;
for (int k = 0; k < num_chunks; ++k) {
    int rel_start = starts[k];
    int off = t - rel_start;
    if (off >= 0 && off < safe_len) {
        int idx = (k * num_sc + sc) * safe_len + off;
        acc += (float)weighted[idx];
    }
}

out[sc * span_len + t] = acc;
'''


_METAL_COUNTER_ACCUM_SOURCE = r'''
int num_chunks = params[0];
int safe_len = params[1];
int span_len = params[2];

int t = (int)thread_position_in_grid.x;
if (t >= span_len) return;

float acc = 0.0f;
for (int k = 0; k < num_chunks; ++k) {
    int rel_start = starts[k];
    int off = t - rel_start;
    if (off >= 0 && off < safe_len) {
        acc += (float)window[off];
    }
}

out[t] = acc;
'''


def _accumulate_span_python(
    weighted: mx.array,
    rel_starts: list[int],
    span_len: int,
    safe_len: int,
    window_safe: mx.array,
    num_stems: int,
    channels: int,
) -> tuple[mx.array, mx.array]:
    span_result = mx.zeros((int(num_stems), int(channels), int(span_len)), dtype=mx.float32)
    span_counter = mx.zeros((int(span_len),), dtype=mx.float32)
    for local_idx, rel_start in enumerate(rel_starts):
        write_start = int(rel_start)
        write_end = write_start + int(safe_len)
        span_result = span_result.at[:, :, write_start:write_end].add(weighted[local_idx])
        span_counter = span_counter.at[write_start:write_end].add(window_safe)
    return span_result, span_counter


class _MetalKernelCache:
    """Singleton cache for Metal overlap-add kernels."""

    _weighted_kernel = None
    _counter_kernel = None
    _lock = Lock()

    @classmethod
    def get_weighted(cls):
        if cls._weighted_kernel is not None:
            return cls._weighted_kernel
        with cls._lock:
            if cls._weighted_kernel is not None:
                return cls._weighted_kernel
            try:
                cls._weighted_kernel = mx.fast.metal_kernel(
                    name="roformer_ola_weighted_accum",
                    input_names=["weighted", "starts", "params"],
                    output_names=["out"],
                    source=_METAL_WEIGHTED_ACCUM_SOURCE,
                    ensure_row_contiguous=True,
                )
            except Exception:
                cls._weighted_kernel = False
            return cls._weighted_kernel

    @classmethod
    def get_counter(cls):
        if cls._counter_kernel is not None:
            return cls._counter_kernel
        with cls._lock:
            if cls._counter_kernel is not None:
                return cls._counter_kernel
            try:
                cls._counter_kernel = mx.fast.metal_kernel(
                    name="roformer_ola_counter_accum",
                    input_names=["window", "starts", "params"],
                    output_names=["out"],
                    source=_METAL_COUNTER_ACCUM_SOURCE,
                    ensure_row_contiguous=True,
                )
            except Exception:
                cls._counter_kernel = False
            return cls._counter_kernel


@dataclass
class OverlapAddFusionCache:
    """Shape-keyed compile cache for overlap-add span accumulation."""

    compiled: dict[tuple[Any, ...], Any] = field(default_factory=dict)

    def _accumulate_span_metal(
        self,
        weighted: mx.array,
        rel_starts: list[int],
        span_len: int,
        safe_len: int,
        window_safe: mx.array,
        num_stems: int,
        channels: int,
    ) -> tuple[mx.array, mx.array] | None:
        kernel_weighted = _MetalKernelCache.get_weighted()
        kernel_counter = _MetalKernelCache.get_counter()
        if kernel_weighted is False or kernel_counter is False:
            return None

        num_chunks = int(weighted.shape[0])
        num_sc = int(num_stems) * int(channels)
        weighted_sc = mx.reshape(weighted, (num_chunks, num_sc, int(safe_len)))
        starts_mx = mx.array(rel_starts, dtype=mx.int32)
        params = mx.array([num_chunks, int(safe_len), int(span_len), num_sc], dtype=mx.int32)

        tgx = 256 if int(span_len) >= 256 else max(1, int(span_len))
        weighted_out = kernel_weighted(
            inputs=[weighted_sc, starts_mx, params],
            grid=(int(span_len), int(num_sc), 1),
            threadgroup=(int(tgx), 1, 1),
            output_shapes=[(int(num_sc), int(span_len))],
            output_dtypes=[mx.float32],
            init_value=0,
        )[0]
        counter_out = kernel_counter(
            inputs=[window_safe, starts_mx, params],
            grid=(int(span_len), 1, 1),
            threadgroup=(int(tgx), 1, 1),
            output_shapes=[(int(span_len),)],
            output_dtypes=[mx.float32],
            init_value=0,
        )[0]
        span_result = mx.reshape(weighted_out, (int(num_stems), int(channels), int(span_len)))
        return span_result, counter_out

    def accumulate_span(
        self,
        weighted: mx.array,
        starts_batch: list[int],
        span_start: int,
        safe_len: int,
        window_safe: mx.array,
        num_stems: int,
        channels: int,
        use_compiled: bool,
    ) -> tuple[mx.array, mx.array]:
        if not starts_batch:
            return (
                mx.zeros((int(num_stems), int(channels), 0), dtype=mx.float32),
                mx.zeros((0,), dtype=mx.float32),
            )

        span_end = int(starts_batch[-1]) + int(safe_len)
        span_len = max(0, int(span_end) - int(span_start))
        rel_starts = [int(start) - int(span_start) for start in starts_batch]

        # Experimental fused path is strictly opt-in.
        if not use_compiled:
            return _accumulate_span_python(
                weighted=weighted,
                rel_starts=rel_starts,
                span_len=span_len,
                safe_len=int(safe_len),
                window_safe=window_safe,
                num_stems=int(num_stems),
                channels=int(channels),
            )

        metal_out = self._accumulate_span_metal(
            weighted=weighted,
            rel_starts=rel_starts,
            span_len=span_len,
            safe_len=int(safe_len),
            window_safe=window_safe,
            num_stems=int(num_stems),
            channels=int(channels),
        )
        if metal_out is not None:
            return metal_out

        key = (
            tuple(rel_starts),
            int(span_len),
            int(safe_len),
            int(num_stems),
            int(channels),
        )
        compiled_fn = self.compiled.get(key)
        if compiled_fn is None:
            compile_fn = getattr(mx, "compile", None)
            if not callable(compile_fn):
                return _accumulate_span_python(
                    weighted=weighted,
                    rel_starts=rel_starts,
                    span_len=span_len,
                    safe_len=int(safe_len),
                    window_safe=window_safe,
                    num_stems=int(num_stems),
                    channels=int(channels),
                )

            rel_const = list(rel_starts)

            def _accumulate(weighted_in: mx.array, window_in: mx.array):
                span_result = mx.zeros((int(num_stems), int(channels), int(span_len)), dtype=mx.float32)
                span_counter = mx.zeros((int(span_len),), dtype=mx.float32)
                for local_idx, rel_start in enumerate(rel_const):
                    write_start = int(rel_start)
                    write_end = write_start + int(safe_len)
                    span_result = span_result.at[:, :, write_start:write_end].add(weighted_in[local_idx])
                    span_counter = span_counter.at[write_start:write_end].add(window_in)
                return span_result, span_counter

            try:
                compiled_fn = compile_fn(_accumulate, shapeless=False)
                self.compiled[key] = compiled_fn
            except Exception:
                return _accumulate_span_python(
                    weighted=weighted,
                    rel_starts=rel_starts,
                    span_len=span_len,
                    safe_len=int(safe_len),
                    window_safe=window_safe,
                    num_stems=int(num_stems),
                    channels=int(channels),
                )

        try:
            return compiled_fn(weighted, window_safe)
        except Exception:
            self.compiled.pop(key, None)
            return _accumulate_span_python(
                weighted=weighted,
                rel_starts=rel_starts,
                span_len=span_len,
                safe_len=int(safe_len),
                window_safe=window_safe,
                num_stems=int(num_stems),
                channels=int(channels),
            )
