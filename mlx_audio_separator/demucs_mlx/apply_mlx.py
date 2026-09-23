"""
MLX inference apply_model equivalent.
"""
from __future__ import annotations

import os
import random
import typing as tp
import warnings
import weakref

import mlx.core as mx

from .defaults import DEFAULT_BATCH_SIZE
from .mlx_utils import center_trim

_WEIGHT_CACHE: dict[tuple[int, float, str], mx.array] = {}
# MLX < 0.32.0 corrupts strided (non-leading-axis) slice scatter-add: the
# Metal slice_update kernel linearizes a 2-D/3-D dispatch grid incorrectly, so
# `arr.at[..., a:b].add(x)` silently accumulates into aliased cells. Fixed in
# MLX 0.32.0 (backend/metal/kernels/indexing/scatter.h). `mx.slice_update` with
# an mx.array start takes the DynamicSliceUpdate path and is correct on every
# supported version, so it is used unconditionally. Set
# MLX_AUDIO_SEPARATOR_UNSAFE_SLICE_ADD=1 to benchmark the legacy path.
_USE_SAFE_SLICE_ACCUMULATION = os.getenv(
    "MLX_AUDIO_SEPARATOR_UNSAFE_SLICE_ADD", ""
).strip().lower() not in {"1", "true", "yes", "on"}


def _deterministic_accumulation_enabled() -> bool:
    """Enable strict ordered overlap-add accumulation for reproducibility checks."""
    raw = os.environ.get("MLX_AUDIO_SEPARATOR_DETERMINISTIC_ACCUMULATION")
    if raw is not None:
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    # Default to strict accumulation in deterministic-fused mode.
    fused = os.environ.get("MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED")
    return str(fused).strip().lower() in {"1", "true", "yes", "on"}


def _demucs_apply_concat_batching_enabled() -> bool:
    """Enable concat-based split batching to avoid temporary 4D stack tensors."""
    raw = os.environ.get("MLX_AUDIO_SEPARATOR_DEMUCS_APPLY_CONCAT_BATCHING", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


class TensorChunk:
    def __init__(self, tensor: mx.array, offset=0, length=None):
        total_length = tensor.shape[-1]
        if offset < 0 or offset >= total_length:
            raise ValueError("Invalid offset.")
        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)
        if isinstance(tensor, TensorChunk):
            self.tensor = tensor.tensor
            self.offset = offset + tensor.offset
        else:
            self.tensor = tensor
            self.offset = offset
        self.length = length

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        if delta < 0:
            raise ValueError("target_length must be >= length.")
        total_length = self.tensor.shape[-1]
        start = self.offset - delta // 2
        end = start + target_length
        correct_start = max(0, start)
        correct_end = min(total_length, end)
        pad_left = correct_start - start
        pad_right = end - correct_end
        out = self.tensor[..., correct_start:correct_end]
        if pad_left or pad_right:
            out = mx.pad(out, [(0, 0), (0, 0), (pad_left, pad_right)], mode="constant")
        return out


def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    if not isinstance(tensor_or_chunk, mx.array):
        raise TypeError("Expected mx.array.")
    return TensorChunk(tensor_or_chunk)


# MLX_AUDIO_SEPARATOR_DEMUCS_COMPILE=0 falls back to the eager forward.
_DEMUCS_COMPILE_ENV = "MLX_AUDIO_SEPARATOR_DEMUCS_COMPILE"
# MLX modules are not hashable, so this is keyed on id() with a weakref held
# alongside: the weakref both proves the entry still refers to the model we
# were handed and lets a dead one be evicted, which id() alone cannot do.
_COMPILED_FORWARDS: "dict[int, tuple[tp.Any, dict]]" = {}


def _demucs_compile_enabled() -> bool:
    raw = os.getenv(_DEMUCS_COMPILE_ENV, "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _forward(model: tp.Any, x: mx.array) -> mx.array:
    """Run the model, compiling the graph once per input shape.

    The Demucs path otherwise runs entirely uncompiled -- `mx.compile` appears
    only in `wiener_mlx.py`, which htdemucs never reaches. That matters more
    than it sounds: ablation shows this path is not arithmetic-bound (deleting
    the cross-transformer, ~75% of the FLOPs, saves 21% of wall clock, and
    bf16 on it saves nothing), so the win is in fused dispatch, not faster math.

    Measured on an idle M4 through `scripts/perf/ab_harness.py` with a
    same-config control (noise floor 1.04%), htdemucs, 60 s of audio:
    **+15.9%**, or +16.8% together with the per-update overlap-add flush.
    Faster on the very first call too -- 0.671 s against 0.776 s on a 20 s clip
    with no warmup -- so compilation repays itself inside one separation. The
    first call at each shape still runs eagerly, because mlx-spectro cannot tune
    its STFT threadgroup sizes inside a trace; that call's output is used, so
    the only cost is one chunk going uncompiled.

    Fusion reassociates floating-point adds, so output is not bit-identical:
    107-114 dB SNR against the eager path across htdemucs, htdemucs_6s and
    hdemucs_mmi, i.e. error near -76 dBFS. The cache is keyed on shape and
    `apply_model` emits at most two (full batches plus a trailing partial),
    so this cannot churn.
    """
    if not _demucs_compile_enabled():
        return model(x)
    try:
        ref = weakref.ref(model)
    except TypeError:          # not weak-referenceable; compile nothing
        return model(x)

    slot = _COMPILED_FORWARDS.get(id(model))
    if slot is None or slot[0]() is not model:
        slot = (ref, {})
        _COMPILED_FORWARDS[id(model)] = slot
    per_shape = slot[1]

    key = (tuple(x.shape), str(x.dtype))
    if key not in per_shape:
        # The first call at a shape runs eagerly, and the next one compiles.
        # mlx-spectro picks its STFT/iSTFT threadgroup sizes by timing
        # candidates, which needs mx.eval -- and MLX forbids mx.eval inside a
        # compile trace. On a machine with no tuning cache yet, compiling the
        # very first call therefore leaves it no way to tune. Running that call
        # eagerly populates the cache at no cost, because its output is used.
        per_shape[key] = None
        return model(x)

    fn = per_shape[key]
    if fn is None:
        fn = mx.compile(lambda t, _m=model: _m(t))
        per_shape[key] = fn
    return fn(x)


def apply_model(
    model,
    mix: tp.Union[mx.array, "TensorChunk"],
    shifts: int = 1,
    split: bool = True,
    overlap: float = 0.25,
    transition_power: float = 1.0,
    progress: bool = False,
    num_workers: int = 0,
    segment: tp.Optional[float] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: tp.Optional[int] = None,
    _rng: tp.Optional[random.Random] = None,
):
    if _rng is None:
        _rng = random if seed is None else random.Random(int(seed))

    progress_enabled = bool(progress)
    if num_workers > 0:
        warnings.warn("num_workers > 0 ignored on MLX.", RuntimeWarning)
        num_workers = 0

    # --- BagOfModels Handling ---
    from .mlx_convert import BagOfModelsMLX
    if isinstance(model, BagOfModelsMLX):
        totals = [0.0] * len(model.sources)
        estimates = None
        min_length = None

        for sub_model, model_weights in zip(model.models, model.weights):
            res = apply_model(
                sub_model, mix, shifts, split, overlap, transition_power,
                progress, num_workers, segment, batch_size, seed, _rng
            )
            out = mx.array(res)

            # Vectorized per-source weighting: (1, S, 1, 1)
            w = mx.array(model_weights, dtype=out.dtype).reshape(1, -1, 1, 1)
            out = out * w

            # Track totals in Python (tiny), keep math vectorized above.
            for k, inst_weight in enumerate(model_weights):
                totals[k] += float(inst_weight)

            if min_length is None:
                min_length = out.shape[-1]
                estimates = out
            else:
                if out.shape[-1] < min_length:
                    min_length = out.shape[-1]
                    estimates = estimates[..., :min_length]
                elif out.shape[-1] > min_length:
                    out = out[..., :min_length]
                estimates = estimates + out

        # Vectorized normalization by totals.
        denom = mx.array(totals, dtype=estimates.dtype).reshape(1, -1, 1, 1)
        estimates = estimates / denom
        return estimates

    # --- Standard Inference ---
    if isinstance(mix, TensorChunk):
        # Materialize the chunk view (respecting offset and length)
        mix_array = mix.padded(mix.length)
    else:
        mix_array = mix
    batch, channels, length = mix_array.shape

    if shifts:
        max_shift = int(0.5 * model.samplerate)
        mix_chunk = TensorChunk(mix_array)
        padded_mix = mix_chunk.padded(length + 2 * max_shift)
        out = 0.0
        for _ in range(shifts):
            offset = _rng.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            shifted_out = apply_model(
                model, shifted, 0, split, overlap, transition_power,
                False, num_workers, segment, batch_size, seed, _rng
            )
            out = out + shifted_out[..., max_shift - offset:]
        out = out / shifts
        return out

    if split:
        deterministic_accum = _deterministic_accumulation_enabled()
        out = mx.zeros((batch, len(model.sources), channels, length), dtype=mix_array.dtype)
        sum_weight = mx.zeros((length,), dtype=mix_array.dtype)

        if segment is None:
            segment = model.segment
        segment_length = int(model.samplerate * segment)
        stride = int((1 - overlap) * segment_length)
        offsets = list(range(0, length, stride))

        # Prepare Weight
        cache_key = (segment_length, float(transition_power), mix_array.dtype)
        weight = _WEIGHT_CACHE.get(cache_key)
        if weight is None:
            weight = mx.concatenate([
                mx.arange(1, segment_length // 2 + 1),
                mx.arange(segment_length - segment_length // 2, 0, -1),
            ], axis=0)
            weight = (weight / mx.max(weight)) ** transition_power
            _WEIGHT_CACHE[cache_key] = weight
        weight_view = weight.reshape(1, 1, 1, -1)

        progress_bar = None
        if progress_enabled:
            from tqdm import tqdm
            progress_bar = tqdm(total=len(offsets), desc="segments", unit="seg", leave=False)

        # --- BATCHING STATE ---
        batch_inputs = []
        batch_indices = []
        # Evaluate after every overlap-add update rather than letting a
        # batch-sized run of them accumulate. Counterintuitive -- this is
        # strictly more synchronization -- but the accumulator is the largest
        # tensor in the job and deferring its updates builds a lazy graph whose
        # working set grows with the interval. Measured on an idle M4 through
        # the ABBA harness with a same-config control (noise floor 0.62%),
        # htdemucs, 60 s of audio, against the previous value of 8:
        #     interval 1 -> +4.3%   2 -> +3.5%   4 -> +1.7%   16 -> +1.0%
        # Output is bit-identical at every interval.
        eval_flush_interval = 1
        pending_updates = 0
        if hasattr(model, "valid_length"):
            std_valid_len = model.valid_length(segment_length)
        else:
            std_valid_len = segment_length

        def maybe_eval(force=False):
            nonlocal pending_updates
            if force or pending_updates >= eval_flush_interval:
                mx.eval(out, sum_weight)
                pending_updates = 0

        def flush_batch():
            nonlocal batch_inputs, batch_indices, out, sum_weight, pending_updates
            if not batch_inputs:
                return

            b_seg = len(batch_inputs)
            if _demucs_apply_concat_batching_enabled():
                # Concat directly to flattened (Batch_Segments * Audio_Batch, Channels, Time).
                b_audio, channels, length = batch_inputs[0].shape
                batch_tensor_flat = mx.concatenate(batch_inputs, axis=0)
            else:
                # 1. Stack: (Batch_Segments, Audio_Batch, Channels, Time)
                batch_tensor = mx.stack(batch_inputs)
                # 2. Reshape: Flatten Segments into Batch -> (Total_Batch, Channels, Time)
                _, b_audio, channels, length = batch_tensor.shape
                batch_tensor_flat = batch_tensor.reshape(b_seg * b_audio, channels, length)

            # 3. Run Model (Standard 3D Input)
            batch_out_flat = _forward(model, batch_tensor_flat)

            # 4. Unflatten: (Batch_Segments, Audio_Batch, Sources, Channels, Time)
            _, sources, out_c, out_t = batch_out_flat.shape
            batch_out = batch_out_flat.reshape(b_seg, b_audio, sources, out_c, out_t)

            for i, idx in enumerate(batch_indices):
                chunk_out = center_trim(batch_out[i], segment_length)

                offset = offsets[idx]
                end = offset + segment_length

                update = weight_view * chunk_out
                if _USE_SAFE_SLICE_ACCUMULATION:
                    start = mx.array([offset])
                    out = mx.slice_update(
                        out,
                        out[:, :, :, offset:end] + update,
                        start,
                        axes=(3,),
                    )
                    sum_weight = mx.slice_update(
                        sum_weight,
                        sum_weight[offset:end] + weight,
                        start,
                        axes=(0,),
                    )
                else:
                    out = out.at[:, :, :, offset:end].add(update)
                    sum_weight = sum_weight.at[offset:end].add(weight)
                pending_updates += 1
                if deterministic_accum:
                    # Preserve update order for deterministic equivalence runs.
                    maybe_eval(force=True)

                if progress_bar is not None:
                    progress_bar.update(1)

            batch_inputs = []
            batch_indices = []
            maybe_eval()

        try:
            for i, offset in enumerate(offsets):
                this_chunk_len = min(segment_length, length - offset)
                chunk = TensorChunk(mix_array, offset, this_chunk_len)

                # Batch only standard-sized chunks
                if this_chunk_len == segment_length:
                    padded = chunk.padded(std_valid_len)
                    batch_inputs.append(padded)
                    batch_indices.append(i)

                    if len(batch_inputs) >= batch_size:
                        flush_batch()
                else:
                    # Flush pending batch
                    flush_batch()

                    # Run odd-sized chunk individually
                    if hasattr(model, "valid_length"):
                        valid_len = model.valid_length(this_chunk_len)
                    else:
                        valid_len = this_chunk_len
                    padded = chunk.padded(valid_len)

                    # FIX: Pass 'padded' directly. It is already (Batch, Channels, Time).
                    chunk_out = _forward(model, padded)
                    chunk_out = center_trim(chunk_out, this_chunk_len)

                    end = offset + this_chunk_len
                    weight_slice = weight[:this_chunk_len]
                    w = weight_slice.reshape(1, 1, 1, -1)
                    update = w * chunk_out
                    if _USE_SAFE_SLICE_ACCUMULATION:
                        start = mx.array([offset])
                        out = mx.slice_update(
                            out,
                            out[:, :, :, offset:end] + update,
                            start,
                            axes=(3,),
                        )
                        sum_weight = mx.slice_update(
                            sum_weight,
                            sum_weight[offset:end] + weight_slice,
                            start,
                            axes=(0,),
                        )
                    else:
                        out = out.at[:, :, :, offset:end].add(update)
                        sum_weight = sum_weight.at[offset:end].add(weight_slice)
                    pending_updates += 1
                    maybe_eval(force=deterministic_accum)
                    if progress_bar is not None:
                        progress_bar.update(1)

            flush_batch()
            maybe_eval(force=True)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if bool(mx.any(sum_weight == 0).item()):
            raise ValueError("sum_weight has zeros; check segment and overlap settings")

        out = out / sum_weight
        return out

    # No split path
    valid_length = model.valid_length(length) if hasattr(model, "valid_length") else length
    mix_chunk = TensorChunk(mix_array)
    padded_mix = mix_chunk.padded(valid_length)
    out = _forward(model, padded_mix)
    return center_trim(out, length)
