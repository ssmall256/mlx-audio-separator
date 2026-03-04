"""MDXC/Roformer stem separator using MLX backend."""

import gc
import os
import time

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mlx_audio_separator.separator.common_separator import CommonSeparator, match_array_shapes, normalize
from mlx_audio_separator.separator.models.roformer.overlap_add_kernels import OverlapAddFusionCache


class MDXCSeparator(CommonSeparator):
    """
    MDXC architecture separator using MLX acceleration.

    Supports BS-Roformer (and eventually MelBand-Roformer) models
    with chunked overlap-add inference on Apple Silicon.
    """

    def __init__(self, common_config, arch_config):
        super().__init__(config=common_config)

        self.segment_size = arch_config.get("segment_size", 256)
        self.override_model_segment_size = arch_config.get("override_model_segment_size", False)
        self.batch_size = arch_config.get("batch_size", 1)
        self.overlap = arch_config.get("overlap", 8)
        self.pitch_shift = arch_config.get("pitch_shift", 0)

        self.logger.debug(f"MDXC params: segment_size={self.segment_size}, overlap={self.overlap}, batch_size={self.batch_size}")
        self.logger.debug(f"MDXC params: override_model_segment_size={self.override_model_segment_size}, pitch_shift={self.pitch_shift}")

        self._mlx_window_cache = {}
        self._np_window_cache = {}
        self.experimental_vectorized_chunking = bool(
            self.performance_params.get("experimental_vectorized_chunking", False)
        )
        self.experimental_roformer_fast_norm = bool(
            self.performance_params.get("experimental_roformer_fast_norm", False)
        )
        self.experimental_roformer_grouped_band_split = bool(
            self.performance_params.get("experimental_roformer_grouped_band_split", False)
        )
        self.experimental_roformer_grouped_mask_estimator = bool(
            self.performance_params.get("experimental_roformer_grouped_mask_estimator", False)
        )
        self.experimental_roformer_fused_overlap_add = bool(
            self.performance_params.get("experimental_roformer_fused_overlap_add", False)
        )
        self.experimental_mlx_stream_pipeline = bool(
            self.performance_params.get("experimental_mlx_stream_pipeline", False)
        )
        self.experimental_roformer_compile_fullgraph = bool(
            self.performance_params.get("experimental_roformer_compile_fullgraph", False)
        )
        self.experimental_compile_model_forward = bool(
            self.performance_params.get("experimental_compile_model_forward", False)
        )
        self.experimental_compile_shapeless = bool(
            self.performance_params.get("experimental_compile_shapeless", False)
        )
        self.experimental_roformer_static_compiled_demix = bool(
            self.performance_params.get("experimental_roformer_static_compiled_demix", False)
        )
        self._compiled_model_run = None
        self._fixed_batch_compiled_forward = False
        self._compiled_demix_fn_cache = {}
        self._compiled_demix_shapeless_disabled = set()
        self._logged_static_shapeless_disable = False
        self._overlap_add_fusion_cache = OverlapAddFusionCache()
        self._pipeline_stream = None
        if self.experimental_mlx_stream_pipeline:
            try:
                self._pipeline_stream = mx.new_stream(mx.default_device())
            except Exception as exc:
                self.logger.warning("Failed to create MLX stream pipeline; falling back to default stream: %s", exc)
                self.experimental_mlx_stream_pipeline = False

        # Load model
        self._load_model()

        self.logger.info("MDXC Separator initialisation complete")

    def _load_model(self):
        """Load MDXC model (Roformer or MDX23C) using MLX."""
        from mlx_audio_separator.separator.models.mdxc.loader import load_mdxc_model

        # Controls L2Norm implementation in Roformer model construction.
        os.environ["MLX_AUDIO_SEPARATOR_ROFORMER_FAST_NORM"] = (
            "1" if self.experimental_roformer_fast_norm else "0"
        )
        os.environ["MLX_AUDIO_SEPARATOR_ROFORMER_GROUPED_BAND_SPLIT"] = (
            "1" if getattr(self, "experimental_roformer_grouped_band_split", False) else "0"
        )
        os.environ["MLX_AUDIO_SEPARATOR_ROFORMER_GROUPED_MASK_ESTIMATOR"] = (
            "1" if getattr(self, "experimental_roformer_grouped_mask_estimator", False) else "0"
        )
        os.environ["MLX_AUDIO_SEPARATOR_ROFORMER_COMPILE_FULLGRAPH"] = (
            "1" if getattr(self, "experimental_roformer_compile_fullgraph", False) else "0"
        )
        if self.experimental_roformer_fast_norm:
            self.logger.info("Enabled experimental Roformer fast norm path (mx.fast.rms_norm).")

        self.logger.debug("Loading MDXC model with MLX...")
        self.model_run, self.model_type = load_mdxc_model(
            model_path=self.model_path,
            config=self.model_data,
        )
        if self.experimental_compile_model_forward:
            compile_fn = getattr(mx, "compile", None)
            if callable(compile_fn):
                try:
                    if self.model_type == "mdx23c_tfc_tdf_v3":
                        compile_kwargs = {"shapeless": False}
                        if self.experimental_compile_shapeless:
                            self.logger.info(
                                "Disabling shapeless compile for MDX23C due CustomKernel shape-inference limitations."
                            )
                        self.model_run = compile_fn(self.model_run, **compile_kwargs)
                        self.logger.info(
                            "Enabled experimental compiled MDX23C forward path"
                            f" (shapeless={compile_kwargs['shapeless']})."
                        )
                    elif "roformer" in str(self.model_type).lower():
                        if self.experimental_roformer_static_compiled_demix or self.experimental_compile_shapeless:
                            self.logger.info(
                                "Roformer compile paths are disabled by policy; "
                                "ignoring Roformer static/shapeless compile options."
                            )
                        self.experimental_roformer_static_compiled_demix = False
                        self._compiled_model_run = None
                        self._fixed_batch_compiled_forward = False
                    else:
                        self.logger.info(f"Skipping experimental compiled forward for MDXC model_type={self.model_type}.")
                except Exception as exc:
                    self.logger.warning(f"Failed to compile MDXC forward path, continuing uncompiled: {exc}")
                    self._compiled_model_run = None
                    self._fixed_batch_compiled_forward = False
            else:
                self.logger.warning("MLX compile() unavailable; experimental compiled MDXC forward path disabled.")
        self.logger.info(f"Loaded {self.model_type} model with MLX")

    def _run_model_callable(self, run_fn, batch):
        use_pipeline = bool(getattr(self, "experimental_mlx_stream_pipeline", False))
        stream = getattr(self, "_pipeline_stream", None)
        if use_pipeline and stream is not None:
            with mx.stream(stream):
                out = run_fn(batch)
                if out.ndim == 3:
                    out = mx.expand_dims(out, axis=1)
                mx.eval(out)
            mx.synchronize(stream)
            return out

        out = run_fn(batch)
        if out.ndim == 3:
            out = mx.expand_dims(out, axis=1)
        mx.eval(out)
        return out

    def _run_fixed_compiled_batch(self, mix_mx, starts, start_idx, current_batch_size, chunk_size, arange_chunk):
        starts_batch = starts[start_idx : start_idx + current_batch_size]
        if not starts_batch:
            return None, []
        batch_size = max(1, int(self.batch_size))
        padded_starts = list(starts_batch)
        if current_batch_size < batch_size:
            padded_starts.extend([starts_batch[-1]] * (batch_size - current_batch_size))

        starts_mx = mx.array(padded_starts, dtype=mx.int32)
        gather_idx = starts_mx[:, None] + arange_chunk[None, :]
        batch = mx.transpose(mix_mx[:, gather_idx], (1, 0, 2))

        if current_batch_size < batch_size:
            valid_mask = np.zeros((batch_size, 1, 1), dtype=np.float32)
            valid_mask[:current_batch_size] = 1.0
            batch = batch * mx.array(valid_mask, dtype=mx.float32)

        x = self._run_model_callable(self._compiled_model_run, batch)
        return x, starts_batch

    def _run_roformer_static_compiled_demix(
        self,
        mix_mx: mx.array,
        starts: list[int],
        chunk_size: int,
        window_mx: mx.array,
        num_stems: int,
        shapeless_override: bool | None = None,
    ) -> np.ndarray:
        """Run Roformer with a static chunk plan compiled end-to-end."""
        channels, total_samples = int(mix_mx.shape[0]), int(mix_mx.shape[1])
        num_chunks = len(starts)
        if num_chunks <= 0:
            raise ValueError("Static compiled demix requires at least one chunk.")

        # Guard against pathological memory for very long files in this experimental path.
        if num_chunks > 256:
            raise ValueError(f"Static compiled demix disabled for num_chunks={num_chunks} (>256).")

        # Probe output shape once so scatter/index dimensions are fixed for compilation.
        probe = mx.expand_dims(mix_mx[:, starts[0] : starts[0] + chunk_size], axis=0)
        probe_out = self.model_run(probe)
        if probe_out.ndim == 3:
            probe_out = mx.expand_dims(probe_out, axis=1)
        mx.eval(probe_out)
        safe_len = min(int(chunk_size), int(probe_out.shape[-1]), int(window_mx.shape[0]))
        if safe_len <= 0:
            raise ValueError("Static compiled demix produced invalid safe_len <= 0.")

        starts_arr = np.asarray(starts, dtype=np.int32)
        starts_mx = mx.array(starts_arr, dtype=mx.int32)
        arange_chunk = mx.arange(int(chunk_size), dtype=mx.int32)
        window_safe = window_mx[:safe_len]

        use_shapeless = bool(self.experimental_compile_shapeless) if shapeless_override is None else bool(shapeless_override)

        plan_shape_key = (
            int(total_samples),
            int(num_chunks),
            int(chunk_size),
            int(safe_len),
            int(num_stems),
            int(channels),
        )
        if use_shapeless and plan_shape_key in self._compiled_demix_shapeless_disabled:
            use_shapeless = False

        plan_key = plan_shape_key + (int(use_shapeless),)
        compiled_demix = self._compiled_demix_fn_cache.get(plan_key)
        if compiled_demix is None:
            compile_fn = getattr(mx, "compile", None)
            if not callable(compile_fn):
                raise RuntimeError("MLX compile() unavailable for static compiled demix path.")

            def _demix_fn(mix_in):
                gather_idx = starts_mx[:, None] + arange_chunk[None, :]
                batch = mx.transpose(mix_in[:, gather_idx], (1, 0, 2))
                out = self.model_run(batch)
                if out.ndim == 3:
                    out = mx.expand_dims(out, axis=1)
                out = out[..., :safe_len]
                weighted = out * window_safe[None, None, None, :]

                result = mx.zeros((num_stems, channels, total_samples), dtype=mx.float32)
                counter = mx.zeros((total_samples,), dtype=mx.float32)
                for chunk_idx, write_start in enumerate(starts_arr.tolist()):
                    write_end = int(write_start) + int(safe_len)
                    result = result.at[:, :, int(write_start) : write_end].add(weighted[chunk_idx])
                    counter = counter.at[int(write_start) : write_end].add(window_safe)
                return result / mx.maximum(counter[None, None, :], mx.array(1e-10, dtype=mx.float32))

            compiled_demix = compile_fn(_demix_fn, shapeless=use_shapeless)
            self._compiled_demix_fn_cache[plan_key] = compiled_demix

        try:
            out = compiled_demix(mix_mx)
            mx.eval(out)
            return np.array(out, dtype=np.float32, copy=False)
        except Exception as exc:
            if use_shapeless:
                self._compiled_demix_shapeless_disabled.add(plan_shape_key)
                self._compiled_demix_fn_cache.pop(plan_key, None)
                self.logger.warning(
                    "Shapeless static compiled demix failed; retrying with shaped compile: %s",
                    exc,
                )
                return self._run_roformer_static_compiled_demix(
                    mix_mx=mix_mx,
                    starts=starts,
                    chunk_size=chunk_size,
                    window_mx=window_mx,
                    num_stems=num_stems,
                    shapeless_override=False,
                )
            raise

    def separate(self, audio_file_path, custom_output_names=None):
        """Separate audio file into stems using MDXC/Roformer MLX."""
        self.reset_perf_metrics()
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        self.logger.debug(f"Preparing mix for {self.audio_file_path}...")
        t0 = time.perf_counter()
        mix = self.prepare_mix(self.audio_file_path)
        self.add_perf_time("decode_s", time.perf_counter() - t0)

        # Auto-enable segment size override for short audio for this run only.
        effective_override_model_segment_size = bool(self.override_model_segment_size)
        audio_duration_seconds = mix.shape[1] / self.sample_rate
        if audio_duration_seconds < 10.0 and not effective_override_model_segment_size:
            effective_override_model_segment_size = True
            self.logger.warning(
                f"Audio duration ({audio_duration_seconds:.2f}s) < 10s, "
                "auto-enabling override_model_segment_size for this run"
            )

        self.logger.debug("Normalizing mix before demixing...")
        t0 = time.perf_counter()
        mix = normalize(wave=mix, max_peak=self.normalization_threshold, min_peak=self.amplification_threshold)
        self.add_perf_time("preprocess_s", time.perf_counter() - t0)

        t0 = time.perf_counter()
        source = self._demix_mlx(
            mix,
            override_model_segment_size=effective_override_model_segment_size,
        )
        self.add_perf_time("inference_s", time.perf_counter() - t0)
        self.logger.debug("Demixing completed.")

        # Build output files
        output_files = []

        if isinstance(source, dict):
            # Always emit all stems returned by inference unless single-stem output is requested.
            # For single-target models, _demix_mlx can synthesize a complementary secondary stem.
            stem_list = list(source.keys())

            for stem_name in stem_list:
                if self.output_single_stem and self.output_single_stem.lower() != stem_name.lower():
                    continue

                t0 = time.perf_counter()
                stem_source = source[stem_name]
                # Transpose to (frames, channels) for write_audio
                if stem_source.ndim == 2 and stem_source.shape[0] == 2 and stem_source.shape[1] > 2:
                    stem_source = stem_source.T
                self.add_perf_time("postprocess_s", time.perf_counter() - t0)

                stem_output_path = self.get_stem_output_path(stem_name, custom_output_names)
                self.logger.info(f"Writing stem '{stem_name}' to {stem_output_path}")
                self.write_audio(stem_output_path, stem_source)

                if self.output_dir:
                    full_path = os.path.join(self.output_dir, stem_output_path)
                else:
                    full_path = stem_output_path
                output_files.append(full_path)
        else:
            # Single source output
            t0 = time.perf_counter()
            stem_source = source
            if stem_source.ndim == 2 and stem_source.shape[0] == 2 and stem_source.shape[1] > 2:
                stem_source = stem_source.T
            self.add_perf_time("postprocess_s", time.perf_counter() - t0)

            stem_output_path = self.get_stem_output_path(self.primary_stem_name, custom_output_names)
            self.logger.info(f"Writing stem '{self.primary_stem_name}' to {stem_output_path}")
            self.write_audio(stem_output_path, stem_source)

            if self.output_dir:
                full_path = os.path.join(self.output_dir, stem_output_path)
            else:
                full_path = stem_output_path
            output_files.append(full_path)

        return output_files

    @staticmethod
    def _chunk_starts(total_samples: int, chunk_size: int, step: int) -> list[int]:
        max_start = max(int(total_samples) - int(chunk_size), 0)
        starts = list(range(0, max_start + 1, int(step)))
        if not starts:
            return [0]
        if starts[-1] != max_start:
            starts.append(max_start)
        return starts

    def _run_chunked_model_vectorized(
        self,
        mix_mx: mx.array,
        starts: list[int],
        chunk_size: int,
        window_mx: mx.array,
        num_stems: int,
    ) -> np.ndarray:
        """Run chunked inference with vectorized gather and batched span overlap-add."""
        channels, total_samples = int(mix_mx.shape[0]), int(mix_mx.shape[1])
        batch_size = max(1, int(self.batch_size))
        arange_chunk = mx.arange(int(chunk_size), dtype=mx.int32)
        use_fixed_compiled_batch = bool(
            getattr(self, "_fixed_batch_compiled_forward", False)
            and getattr(self, "_compiled_model_run", None) is not None
        )
        overlap_add_cache = getattr(self, "_overlap_add_fusion_cache", None) or OverlapAddFusionCache()

        result_mx = mx.zeros((num_stems, channels, total_samples), dtype=mx.float32)
        counter_mx = mx.zeros((total_samples,), dtype=mx.float32)

        eval_flush_interval = max(8, batch_size * 2)
        pending_updates = 0
        use_fused_ola = bool(getattr(self, "experimental_roformer_fused_overlap_add", False))

        for start_idx in tqdm(range(0, len(starts), batch_size), desc="MLX inference (vectorized)"):
            current_batch_size = min(batch_size, len(starts) - start_idx)
            if current_batch_size <= 0:
                continue

            if use_fixed_compiled_batch:
                out_mx, starts_batch = self._run_fixed_compiled_batch(
                    mix_mx=mix_mx,
                    starts=starts,
                    start_idx=start_idx,
                    current_batch_size=current_batch_size,
                    chunk_size=chunk_size,
                    arange_chunk=arange_chunk,
                )
                if out_mx is None or not starts_batch:
                    continue
                out_mx = out_mx[: len(starts_batch)]
            else:
                starts_batch = starts[start_idx : start_idx + current_batch_size]
                if not starts_batch:
                    continue

                starts_mx = mx.array(starts_batch, dtype=mx.int32)
                gather_idx = starts_mx[:, None] + arange_chunk[None, :]
                batch = mx.transpose(mix_mx[:, gather_idx], (1, 0, 2))

                out_mx = self._run_model_callable(self.model_run, batch)

            safe_len = min(int(chunk_size), int(out_mx.shape[-1]), int(window_mx.shape[0]))
            if safe_len <= 0:
                continue

            window_safe = window_mx[:safe_len]
            weighted = out_mx[..., :safe_len] * window_safe[None, None, None, :]  # (B,S,C,L)
            span_start = int(starts_batch[0])
            span_end = int(starts_batch[-1]) + safe_len
            span_len = span_end - span_start

            span_result, span_counter = overlap_add_cache.accumulate_span(
                weighted=weighted,
                starts_batch=starts_batch,
                span_start=span_start,
                safe_len=safe_len,
                window_safe=window_safe,
                num_stems=num_stems,
                channels=channels,
                use_compiled=use_fused_ola,
            )

            result_mx = result_mx.at[:, :, span_start:span_end].add(span_result)
            counter_mx = counter_mx.at[span_start:span_end].add(span_counter)
            pending_updates += 1

            if pending_updates >= eval_flush_interval:
                mx.eval(result_mx, counter_mx)
                pending_updates = 0

        mx.eval(result_mx, counter_mx)
        out = result_mx / mx.maximum(counter_mx[None, None, :], mx.array(1e-10, dtype=mx.float32))
        mx.eval(out)
        return np.array(out, dtype=np.float32, copy=False)

    def _demix_mlx(self, mix: np.ndarray, override_model_segment_size: bool | None = None) -> dict:
        """Demix using MLX-accelerated Roformer inference with chunked overlap-add.

        Args:
            mix: Input audio, shape (channels, samples)
            override_model_segment_size: Optional per-run override flag.

        Returns:
            Dictionary of separated stems
        """
        self.logger.info("Processing audio with MLX-accelerated inference")
        orig_mix = mix

        # Get configuration
        training = self.model_data.get("training", {})
        inference = self.model_data.get("inference", {})
        audio_cfg = self.model_data.get("audio", {})
        model_cfg = self.model_data.get("model", {})

        target_instrument = training.get("target_instrument")
        instruments = training.get("instruments", [])

        if override_model_segment_size is None:
            effective_override_model_segment_size = bool(self.override_model_segment_size)
        else:
            effective_override_model_segment_size = bool(override_model_segment_size)

        if effective_override_model_segment_size:
            mdx_segment_size = self.segment_size
        else:
            mdx_segment_size = inference.get("dim_t", self.segment_size)

        num_stems = 1 if target_instrument else len(instruments)
        if num_stems <= 0:
            raise ValueError("Invalid model metadata: unable to determine output stems.")

        # Calculate chunk size
        stft_hop_len = model_cfg.get("stft_hop_length", audio_cfg.get("hop_length", 512))
        chunk_size = int(stft_hop_len) * (int(mdx_segment_size) - 1)
        self.logger.debug(f"Chunk size: {chunk_size} (stft_hop={stft_hop_len}, dim_t={mdx_segment_size})")

        # Calculate step size
        sample_rate = audio_cfg.get("sample_rate", self.sample_rate)
        desired_step = int(self.overlap * sample_rate)
        step = chunk_size if desired_step <= 0 else min(desired_step, chunk_size)
        step = max(1, int(step))

        # Create Hamming window (both MLX and NumPy for different accumulation paths)
        window_np = self._np_window_cache.get(chunk_size)
        window_mx = self._mlx_window_cache.get(chunk_size)
        if window_np is None or window_mx is None:
            window_np = np.hamming(chunk_size).astype(np.float32, copy=False)
            window_mx = mx.array(window_np, dtype=mx.float32)
            self._np_window_cache[chunk_size] = window_np
            self._mlx_window_cache[chunk_size] = window_mx

        mix_mlx = mx.array(mix, dtype=mx.float32)
        model_run = self.model_run

        if mix.shape[1] < chunk_size:
            # Initialize accumulators
            req_shape = (num_stems,) + tuple(mix.shape)
            result = mx.zeros(req_shape, dtype=mx.float32)
            counter = mx.zeros(req_shape, dtype=mx.float32)

            # Short audio: single chunk
            part = mx.expand_dims(mix_mlx, axis=0)
            x = model_run(part)
            if x.ndim == 3:
                x = mx.expand_dims(x, axis=1)
            x = x[0]
            mx.eval(x)
            safe_len = min(mix.shape[1], x.shape[-1], window_mx.shape[0])
            if safe_len > 0:
                weighted_chunk = x[..., :safe_len] * window_mx[:safe_len]
                result = result.at[..., :safe_len].add(weighted_chunk)
                counter = counter.at[..., :safe_len].add(window_mx[:safe_len])

            inferenced_outputs = result / mx.maximum(counter, mx.array(1e-10))
            inferenced_outputs_np = np.array(inferenced_outputs, dtype=np.float32, copy=False)
            del result, counter, inferenced_outputs
        else:
            # Chunked processing with overlap-add
            starts = self._chunk_starts(mix.shape[1], chunk_size, step)
            num_chunks = len(starts)
            self.logger.debug(f"Processing {num_chunks} chunks")

            used_static_path = False
            if (
                self.experimental_compile_model_forward
                and self.experimental_roformer_static_compiled_demix
                and "roformer" in str(self.model_type).lower()
            ):
                try:
                    self.logger.info("Using experimental Roformer static-plan compiled demix path.")
                    if self.experimental_compile_shapeless and not self._logged_static_shapeless_disable:
                        self.logger.info(
                            "Roformer static demix forcing shaped compile "
                            "(experimental_compile_shapeless ignored for this path)."
                        )
                        self._logged_static_shapeless_disable = True
                    inferenced_outputs_np = self._run_roformer_static_compiled_demix(
                        mix_mx=mix_mlx,
                        starts=starts,
                        chunk_size=chunk_size,
                        window_mx=window_mx,
                        num_stems=num_stems,
                        shapeless_override=False,
                    )
                    used_static_path = True
                except Exception as exc:
                    self.logger.warning(
                        "Experimental Roformer static compiled demix failed; "
                        "falling back to standard chunk loop: %s",
                        exc,
                    )

            if not used_static_path and self.experimental_vectorized_chunking:
                self.logger.info("Using experimental vectorized MDXC chunking path.")
                inferenced_outputs_np = self._run_chunked_model_vectorized(
                    mix_mx=mix_mlx,
                    starts=starts,
                    chunk_size=chunk_size,
                    window_mx=window_mx,
                    num_stems=num_stems,
                )
            elif not used_static_path:
                # Initialize accumulators
                req_shape = (num_stems,) + tuple(mix.shape)
                result = mx.zeros(req_shape, dtype=mx.float32)
                counter = mx.zeros(req_shape, dtype=mx.float32)

                batch_size = max(1, int(self.batch_size))
                eval_flush_interval = max(8, batch_size * 2)
                pending_updates = 0
                arange_chunk = mx.arange(int(chunk_size), dtype=mx.int32)

                def maybe_eval(force=False):
                    nonlocal pending_updates, result, counter
                    if force or pending_updates >= eval_flush_interval:
                        mx.eval(result, counter)
                        pending_updates = 0

                def run_batch(start_idx: int, current_batch_size: int):
                    nonlocal result, counter, pending_updates
                    if current_batch_size <= 0:
                        return
                    overlap_add_cache = getattr(self, "_overlap_add_fusion_cache", None) or OverlapAddFusionCache()

                    if self._fixed_batch_compiled_forward and self._compiled_model_run is not None:
                        x, starts_batch = self._run_fixed_compiled_batch(
                            mix_mx=mix_mlx,
                            starts=starts,
                            start_idx=start_idx,
                            current_batch_size=current_batch_size,
                            chunk_size=chunk_size,
                            arange_chunk=arange_chunk,
                        )
                    else:
                        starts_batch = starts[start_idx : start_idx + current_batch_size]
                        parts_batch = [
                            mix_mlx[:, write_start : write_start + chunk_size]
                            for write_start in starts_batch
                        ]
                        batch = mx.stack(parts_batch, axis=0)  # (B, channels, chunk_size)
                        x = self._run_model_callable(model_run, batch)

                    if not starts_batch:
                        return

                    safe_len = min(chunk_size, int(x.shape[-1]), int(window_mx.shape[0]))
                    if safe_len > 0:
                        window_safe = window_mx[:safe_len]
                        weighted = x[..., :safe_len] * window_safe[None, None, None, :]
                        span_start = int(starts_batch[0])
                        span_end = int(starts_batch[-1]) + int(safe_len)
                        span_result, span_counter = overlap_add_cache.accumulate_span(
                            weighted=weighted,
                            starts_batch=starts_batch,
                            span_start=span_start,
                            safe_len=safe_len,
                            window_safe=window_safe,
                            num_stems=num_stems,
                            channels=int(mix.shape[0]),
                            use_compiled=bool(getattr(self, "experimental_roformer_fused_overlap_add", False)),
                        )
                        result = result.at[..., span_start:span_end].add(span_result)
                        counter = counter.at[..., span_start:span_end].add(span_counter)
                        pending_updates += 1

                    maybe_eval()

                num_full_batches, tail_size = divmod(num_chunks, batch_size)
                total_batches = num_full_batches + (1 if tail_size else 0)
                for batch_id in tqdm(range(total_batches), desc="MLX inference"):
                    start_idx = batch_id * batch_size
                    current_batch_size = batch_size if batch_id < num_full_batches else tail_size
                    run_batch(start_idx, current_batch_size)

                maybe_eval(force=True)

                inferenced_outputs = result / mx.maximum(counter, mx.array(1e-10))
                inferenced_outputs_np = np.array(inferenced_outputs, dtype=np.float32, copy=False)
                del result, counter, inferenced_outputs

        del mix_mlx
        gc.collect()

        # Build output dictionary
        if num_stems > 1:
            sources = {}
            for key, value in zip(instruments, inferenced_outputs_np):
                sources[key] = value

            return sources
        else:
            # Single-source model output
            primary = inferenced_outputs_np[0]

            if target_instrument:
                if primary.shape[1] != orig_mix.shape[1]:
                    primary = match_array_shapes(primary, orig_mix)
                secondary = orig_mix - primary
                return {
                    self.primary_stem_name: primary,
                    self.secondary_stem_name: secondary,
                }

            return primary
