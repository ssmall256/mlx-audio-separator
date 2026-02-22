"""MDX stem separator using MLX backend."""

import os
import time

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mlx_audio_separator.separator.common_separator import CommonSeparator, normalize


class MDXSeparator(CommonSeparator):
    """
    MDX architecture separator using MLX acceleration.

    Handles ONNX-based MDX-Net models (ConvTDFNet U-Net) by converting
    weights to MLX format. Uses STFT-based spectral processing with
    overlap-add chunking.
    """

    def __init__(self, common_config, arch_config):
        super().__init__(config=common_config)

        self.segment_size = arch_config.get("segment_size", 256)
        self.overlap = arch_config.get("overlap", 0.25)
        self.batch_size = arch_config.get("batch_size", 1)
        self.hop_length = arch_config.get("hop_length", 1024)
        self.enable_denoise = arch_config.get("enable_denoise", False)

        self.logger.debug(f"MDX params: segment_size={self.segment_size}, overlap={self.overlap}")
        self.logger.debug(f"MDX params: batch_size={self.batch_size}, hop_length={self.hop_length}")

        # Model data parameters (from JSON hash lookup)
        self.compensate = self.model_data["compensate"]
        self.dim_f = self.model_data["mdx_dim_f_set"]
        self.dim_t = 2 ** self.model_data["mdx_dim_t_set"]
        self.n_fft = self.model_data["mdx_n_fft_scale_set"]

        self.logger.debug(f"MDX model params: compensate={self.compensate}, dim_f={self.dim_f}, dim_t={self.dim_t}, n_fft={self.n_fft}")

        # Derived parameters
        self.n_bins = self.n_fft // 2 + 1
        self.trim = self.n_fft // 2
        self.chunk_size = self.hop_length * (self.segment_size - 1)
        self.gen_size = self.chunk_size - 2 * self.trim

        # Load model
        self._load_model()

        # Create STFT processor
        from mlx_audio_separator.separator.models.mdx.stft import STFT
        self.stft = STFT(self.n_fft, self.hop_length, self.dim_f)
        self._window_cache = {}

        self.logger.info("MDX Separator initialisation complete")

    def _load_model(self):
        """Load MDX ConvTDFNet model with MLX weights."""
        from mlx_audio_separator.separator.models.mdx.loader import load_mdx_model

        self.logger.debug("Loading MDX model...")
        self.model_run, _ = load_mdx_model(
            model_path=self.model_path,
            model_data=self.model_data,
        )
        self.logger.info("MDX model loaded with MLX")

    def separate(self, audio_file_path, custom_output_names=None):
        """Separate audio into primary and secondary stems."""
        self.reset_perf_metrics()
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        self.logger.debug(f"Preparing mix for {self.audio_file_path}...")
        t0 = time.perf_counter()
        mix = self.prepare_mix(self.audio_file_path)
        self.add_perf_time("decode_s", time.perf_counter() - t0)

        self.logger.debug("Normalizing mix...")
        t0 = time.perf_counter()
        peak = np.abs(mix).max()
        mix = normalize(wave=mix, max_peak=self.normalization_threshold, min_peak=self.amplification_threshold)
        self.add_perf_time("preprocess_s", time.perf_counter() - t0)

        # Demix
        t0 = time.perf_counter()
        source = self.demix(mix) * peak
        self.add_perf_time("inference_s", time.perf_counter() - t0)

        self.primary_source = source.T

        output_files = []

        # Compute secondary source
        if self.invert_using_spec:
            self.logger.debug("Computing secondary stem using spectral inversion...")
            t0 = time.perf_counter()
            raw_mix = self.demix(mix, is_match_mix=True)
            self.add_perf_time("inference_s", time.perf_counter() - t0)
            t0 = time.perf_counter()
            self.secondary_source = self._invert_stem(raw_mix, self.primary_source * self.compensate)
            self.add_perf_time("postprocess_s", time.perf_counter() - t0)
        else:
            self.logger.debug("Computing secondary stem by subtraction...")
            t0 = time.perf_counter()
            self.secondary_source = (-self.primary_source * self.compensate) + mix.T
            self.add_perf_time("postprocess_s", time.perf_counter() - t0)

        # Write secondary stem
        if not self.output_single_stem or self.output_single_stem.lower() == self.secondary_stem_name.lower():
            stem_path = self.get_stem_output_path(self.secondary_stem_name, custom_output_names)
            self.logger.info(f"Writing {self.secondary_stem_name} stem to {stem_path}")
            self.final_process(stem_path, self.secondary_source, self.secondary_stem_name)
            output_files.append(os.path.join(self.output_dir, stem_path) if self.output_dir else stem_path)

        # Write primary stem
        if not self.output_single_stem or self.output_single_stem.lower() == self.primary_stem_name.lower():
            stem_path = self.get_stem_output_path(self.primary_stem_name, custom_output_names)
            self.logger.info(f"Writing {self.primary_stem_name} stem to {stem_path}")
            self.final_process(stem_path, self.primary_source, self.primary_stem_name)
            output_files.append(os.path.join(self.output_dir, stem_path) if self.output_dir else stem_path)

        return output_files

    def demix(self, mix, is_match_mix=False):
        """Demix using STFT-based spectral processing with overlap-add.

        Args:
            mix: Input audio (channels, samples)
            is_match_mix: If True, return raw STFT for spectral inversion

        Returns:
            Separated source array
        """
        self.logger.debug(f"Demixing (is_match_mix={is_match_mix})...")

        if is_match_mix:
            chunk_size = self.hop_length * (self.segment_size - 1)
            overlap = 0.02
        else:
            chunk_size = self.chunk_size
            overlap = self.overlap

        gen_size = chunk_size - 2 * self.trim

        # Pad the mix
        pad = gen_size + self.trim - (mix.shape[-1] % gen_size)
        mixture = np.concatenate(
            (np.zeros((2, self.trim), dtype="float32"), mix, np.zeros((2, pad), dtype="float32")),
            axis=1,
        )

        step = max(1, int((1 - overlap) * chunk_size))

        # Accumulator arrays
        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)

        starts = list(range(0, mixture.shape[-1], step))
        total_chunks = len(starts)
        batch_size = max(1, int(self.batch_size))

        window = None
        if overlap != 0:
            window = self._window_cache.get(chunk_size)
            if window is None:
                window = np.hanning(chunk_size).astype(np.float32)[None, None, :]
                window = np.tile(window, (1, 2, 1))
                self._window_cache[chunk_size] = window

        batch_parts = []
        batch_meta = []

        def flush_batch():
            nonlocal batch_parts, batch_meta, result, divider
            if not batch_parts:
                return

            batch_inputs = np.stack(batch_parts, axis=0).astype(np.float32, copy=False)
            batch_outputs = self._run_model_batch(batch_inputs, is_match_mix=is_match_mix)

            for idx, (start, valid_len) in enumerate(batch_meta):
                end = start + valid_len
                out = batch_outputs[idx : idx + 1, :, :valid_len]

                if window is not None:
                    if valid_len == chunk_size:
                        w = window
                    else:
                        w = self._window_cache.get(valid_len)
                        if w is None:
                            w = np.hanning(valid_len).astype(np.float32)[None, None, :]
                            w = np.tile(w, (1, 2, 1))
                            self._window_cache[valid_len] = w
                    result[..., start:end] += out * w
                    divider[..., start:end] += w
                else:
                    result[..., start:end] += out
                    divider[..., start:end] += 1

            batch_parts = []
            batch_meta = []

        for start in tqdm(starts, desc="MDX inference", total=total_chunks):
            end = min(start + chunk_size, mixture.shape[-1])
            valid_len = end - start

            mix_part = mixture[:, start:end]
            if valid_len < chunk_size:
                pad_size = chunk_size - valid_len
                mix_part = np.pad(mix_part, ((0, 0), (0, pad_size)), mode="constant")

            batch_parts.append(mix_part)
            batch_meta.append((start, valid_len))
            if len(batch_parts) >= batch_size:
                flush_batch()

        flush_batch()

        # Normalize by divider
        tar_waves = result / np.maximum(divider, 1e-8)

        # Trim padding
        tar_waves = tar_waves[:, :, self.trim : -self.trim]
        tar_waves = tar_waves[:, :, : mix.shape[-1]]

        source = tar_waves[0]
        return source

    def _run_model_batch(self, mix_np, is_match_mix=False):
        """Run the model on a batch of chunks.

        Args:
            mix_np: Input chunks (batch, 2, chunk_size) as numpy
            is_match_mix: If True, return STFT directly without model

        Returns:
            Processed chunks as numpy array
        """
        mix_mx = mx.array(mix_np, dtype=mx.float32)
        spek = self.stft(mix_mx)

        # Zero out first 3 frequency bins (reduce low-freq noise)
        spek = spek.at[:, :, :3, :].add(-spek[:, :, :3, :])

        if is_match_mix:
            spec_pred_mx = spek
        elif self.enable_denoise:
            spec_pred_mx = self.model_run(-spek) * -0.5 + self.model_run(spek) * 0.5
        else:
            spec_pred_mx = self.model_run(spek)

        result_mx = self.stft.inverse(spec_pred_mx)
        mx.eval(result_mx)
        return np.array(result_mx, dtype=np.float32, copy=False)

    def _invert_stem(self, raw_mix, primary_source):
        """Invert stem using spectral subtraction.

        Args:
            raw_mix: Raw STFT output from match_mix demix
            primary_source: Primary stem (time domain, transposed)

        Returns:
            Secondary source
        """
        # Simple subtraction in time domain as fallback
        # Full spectral inversion would require complex STFT manipulation
        return raw_mix.T - primary_source
