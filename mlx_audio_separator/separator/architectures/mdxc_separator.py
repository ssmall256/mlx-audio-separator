"""MDXC/Roformer stem separator using MLX backend."""

import gc
import os
import time

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mlx_audio_separator.separator.common_separator import CommonSeparator, match_array_shapes, normalize


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

        # Load model
        self._load_model()

        self.logger.info("MDXC Separator initialisation complete")

    def _load_model(self):
        """Load Roformer model using MLX."""
        from mlx_audio_separator.separator.models.roformer.loader import load_roformer_model

        self.logger.debug("Loading Roformer model with MLX...")
        self.model_run, self.model_type = load_roformer_model(
            model_path=self.model_path,
            config=self.model_data,
        )
        self.logger.info(f"Loaded {self.model_type} model with MLX")

    def separate(self, audio_file_path, custom_output_names=None):
        """Separate audio file into stems using MDXC/Roformer MLX."""
        self.reset_perf_metrics()
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        self.logger.debug(f"Preparing mix for {self.audio_file_path}...")
        t0 = time.perf_counter()
        mix = self.prepare_mix(self.audio_file_path)
        self.add_perf_time("decode_s", time.perf_counter() - t0)

        # Auto-enable segment size override for short audio
        audio_duration_seconds = mix.shape[1] / self.sample_rate
        if audio_duration_seconds < 10.0 and not self.override_model_segment_size:
            self.override_model_segment_size = True
            self.logger.warning(
                f"Audio duration ({audio_duration_seconds:.2f}s) < 10s, "
                "auto-enabling override_model_segment_size"
            )

        self.logger.debug("Normalizing mix before demixing...")
        t0 = time.perf_counter()
        mix = normalize(wave=mix, max_peak=self.normalization_threshold, min_peak=self.amplification_threshold)
        self.add_perf_time("preprocess_s", time.perf_counter() - t0)

        t0 = time.perf_counter()
        source = self._demix_mlx(mix)
        self.add_perf_time("inference_s", time.perf_counter() - t0)
        self.logger.debug("Demixing completed.")

        # Build output files
        output_files = []

        if isinstance(source, dict):
            # Determine stem list
            training = self.model_data.get("training", {})
            target_instrument = training.get("target_instrument")
            instruments = training.get("instruments", [])

            if target_instrument:
                stem_list = [target_instrument]
            else:
                stem_list = instruments

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

    def _demix_mlx(self, mix: np.ndarray) -> dict:
        """Demix using MLX-accelerated Roformer inference with chunked overlap-add.

        Args:
            mix: Input audio, shape (channels, samples)

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

        if self.override_model_segment_size:
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

        # Create Hamming window
        window = self._mlx_window_cache.get(chunk_size)
        if window is None:
            window = mx.array(np.hamming(chunk_size), dtype=mx.float32)
            self._mlx_window_cache[chunk_size] = window

        # Initialize accumulators
        req_shape = (num_stems,) + tuple(mix.shape)
        result = mx.zeros(req_shape, dtype=mx.float32)
        counter = mx.zeros(req_shape, dtype=mx.float32)

        mix_mlx = mx.array(mix, dtype=mx.float32)
        model_run = self.model_run

        if mix.shape[1] < chunk_size:
            # Short audio: single chunk
            part = mx.expand_dims(mix_mlx, axis=0)
            x = model_run(part)
            if x.ndim == 3:
                x = mx.expand_dims(x, axis=1)
            x = x[0]
            mx.eval(x)
            safe_len = min(mix.shape[1], x.shape[-1], window.shape[0])
            if safe_len > 0:
                weighted_chunk = x[..., :safe_len] * window[:safe_len]
                result = result.at[..., :safe_len].add(weighted_chunk)
                counter = counter.at[..., :safe_len].add(window[:safe_len])
        else:
            # Chunked processing with overlap-add
            max_start = max(mix.shape[1] - chunk_size, 0)
            starts = list(range(0, max_start + 1, step))
            if not starts:
                starts = [0]
            elif starts[-1] != max_start:
                starts.append(max_start)
            num_chunks = len(starts)
            self.logger.debug(f"Processing {num_chunks} chunks")

            batch_size = max(1, int(self.batch_size))
            for i in tqdm(range(0, num_chunks, batch_size), desc="MLX inference"):
                starts_batch = starts[i : i + batch_size]
                parts_batch = [mix_mlx[:, s : s + chunk_size] for s in starts_batch]
                batch = mx.stack(parts_batch, axis=0)  # (B, channels, chunk_size)
                x = model_run(batch)
                if x.ndim == 3:
                    x = mx.expand_dims(x, axis=1)
                mx.eval(x)

                for batch_idx, write_start in enumerate(starts_batch):
                    out = x[batch_idx]
                    safe_len = min(chunk_size, out.shape[-1], window.shape[0])
                    if safe_len > 0:
                        weighted_chunk = out[..., :safe_len] * window[:safe_len]
                        result = result.at[..., write_start : write_start + safe_len].add(weighted_chunk)
                        counter = counter.at[..., write_start : write_start + safe_len].add(window[:safe_len])

        # Normalize by overlap counter
        inferenced_outputs = result / mx.maximum(counter, mx.array(1e-10))
        inferenced_outputs_np = np.array(inferenced_outputs)

        del result, counter, inferenced_outputs, mix_mlx
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
