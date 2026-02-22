"""VR stem separator using MLX backend.

Handles VR-Net models (CascadedASPPNet / CascadedNet) by converting
PyTorch .pth weights to MLX format. Uses multi-band STFT-based spectral
processing with windowed inference.
"""

import os
import time

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mlx_audio_separator.separator.common_separator import CommonSeparator
from mlx_audio_separator.separator.models.vr import spec_utils


class VRSeparator(CommonSeparator):
    """VR architecture separator using MLX acceleration."""

    def __init__(self, common_config, arch_config):
        super().__init__(config=common_config)

        self.model_capacity = 32, 128
        self.is_vr_51_model = False

        if "nout" in self.model_data and "nout_lstm" in self.model_data:
            self.model_capacity = self.model_data["nout"], self.model_data["nout_lstm"]
            self.is_vr_51_model = True

        # Arch config
        self.enable_tta = arch_config.get("enable_tta", False)
        self.enable_post_process = arch_config.get("enable_post_process", False)
        self.post_process_threshold = arch_config.get("post_process_threshold", 0.2)
        self.batch_size = arch_config.get("batch_size", 1)
        self.window_size = arch_config.get("window_size", 512)
        self.high_end_process = arch_config.get("high_end_process", False)
        self.aggression = float(int(arch_config.get("aggression", 5)) / 100)

        self.input_high_end_h = None
        self.input_high_end = None

        # Load model
        self._load_model()

        self.aggressiveness = {
            "value": self.aggression,
            "split_bin": self.model_params.param["band"][1]["crop_stop"],
            "aggr_correction": self.model_params.param.get("aggr_correction"),
        }
        self.model_samplerate = self.model_params.param["sr"]

        self.logger.debug(f"VR params: window_size={self.window_size}, batch_size={self.batch_size}, tta={self.enable_tta}")
        self.logger.info("VR Separator initialisation complete")

    def _load_model(self):
        """Load VR model with MLX weights."""
        from mlx_audio_separator.separator.models.vr.loader import load_vr_model

        self.model_run, self.model_params, self.is_vr_51_model = load_vr_model(
            model_path=self.model_path,
            model_data=self.model_data,
        )
        self.logger.info("VR model loaded with MLX")

    def separate(self, audio_file_path, custom_output_names=None):
        """Separate audio into primary and secondary stems."""
        self.reset_perf_metrics()
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        self.logger.debug(f"Starting VR separation for {audio_file_path}")

        # Load and convert to spectrogram
        t0 = time.perf_counter()
        X_spec = self._loading_mix()
        self.add_perf_time("preprocess_s", time.perf_counter() - t0)

        # Run inference
        t0 = time.perf_counter()
        y_spec, v_spec = self._inference_vr(X_spec, self.aggressiveness)
        self.add_perf_time("inference_s", time.perf_counter() - t0)

        # Sanitize
        t0 = time.perf_counter()
        y_spec = np.nan_to_num(y_spec, nan=0.0, posinf=0.0, neginf=0.0)
        v_spec = np.nan_to_num(v_spec, nan=0.0, posinf=0.0, neginf=0.0)
        self.add_perf_time("postprocess_s", time.perf_counter() - t0)

        output_files = []

        # Save primary stem
        if not self.output_single_stem or self.output_single_stem.lower() == self.primary_stem_name.lower():
            t0 = time.perf_counter()
            primary_source = self._spec_to_wav(y_spec).T
            if self.model_samplerate != 44100:
                primary_source = self._resample(primary_source.T, self.model_samplerate, 44100).T
            self.add_perf_time("postprocess_s", time.perf_counter() - t0)

            stem_path = self.get_stem_output_path(self.primary_stem_name, custom_output_names)
            self.logger.info(f"Writing {self.primary_stem_name} stem to {stem_path}")
            self.final_process(stem_path, primary_source, self.primary_stem_name)
            output_files.append(os.path.join(self.output_dir, stem_path) if self.output_dir else stem_path)

        # Save secondary stem
        if not self.output_single_stem or self.output_single_stem.lower() == self.secondary_stem_name.lower():
            t0 = time.perf_counter()
            secondary_source = self._spec_to_wav(v_spec).T
            if self.model_samplerate != 44100:
                secondary_source = self._resample(secondary_source.T, self.model_samplerate, 44100).T
            self.add_perf_time("postprocess_s", time.perf_counter() - t0)

            stem_path = self.get_stem_output_path(self.secondary_stem_name, custom_output_names)
            self.logger.info(f"Writing {self.secondary_stem_name} stem to {stem_path}")
            self.final_process(stem_path, secondary_source, self.secondary_stem_name)
            output_files.append(os.path.join(self.output_dir, stem_path) if self.output_dir else stem_path)

        return output_files

    def _loading_mix(self):
        """Load audio and convert to multi-band spectrogram."""
        import mlx_audio_io as mac

        X_wave, X_spec_s = {}, {}
        bands_n = len(self.model_params.param["band"])

        for d in tqdm(range(bands_n, 0, -1), desc="Loading bands"):
            bp = self.model_params.param["band"][d]

            if d == bands_n:
                # Load highest band from file — keep as mlx for GPU STFT
                audio_data, sr = mac.load(str(self.audio_file_path), sr=bp["sr"], dtype="float32")
                # mac.load returns (frames, channels), transpose to (channels, frames)
                wave_mx = audio_data.T if audio_data.ndim == 2 else mx.stack([audio_data, audio_data])
                if wave_mx.ndim == 1:
                    wave_mx = mx.stack([wave_mx, wave_mx])
                # Numpy copy for resampling to lower bands
                X_wave[d] = np.array(wave_mx)
            else:
                # Resample from higher band
                X_wave[d] = self._resample(
                    X_wave[d + 1],
                    self.model_params.param["band"][d + 1]["sr"],
                    bp["sr"],
                )

            # Use mlx array directly for highest band (avoids numpy→mlx round-trip)
            wave_for_stft = wave_mx if d == bands_n else X_wave[d]
            X_spec_s[d] = spec_utils.wave_to_spectrogram(
                wave_for_stft, bp["hl"], bp["n_fft"], self.model_params, band=d,
                is_v51_model=self.is_vr_51_model,
            )

            if d == bands_n and self.high_end_process:
                self.input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (
                    self.model_params.param["pre_filter_stop"] - self.model_params.param["pre_filter_start"]
                )
                self.input_high_end = X_spec_s[d][:, bp["n_fft"] // 2 - self.input_high_end_h : bp["n_fft"] // 2, :]

        X_spec = spec_utils.combine_spectrograms(X_spec_s, self.model_params, is_v51_model=self.is_vr_51_model)
        return X_spec

    def _resample(self, audio, orig_sr, target_sr):
        """Resample audio using mlx-audio-io."""
        import mlx_audio_io as mac

        if orig_sr == target_sr:
            return audio

        # audio is numpy (channels, frames) — convert to (frames, channels) mlx for mac.resample
        audio_mx = mx.array(audio.T, dtype=mx.float32)
        resampled = mac.resample(audio_mx, int(orig_sr), int(target_sr))
        return np.array(resampled).T.astype(np.float32)

    def _inference_vr(self, X_spec, aggressiveness):
        """Run VR inference with windowed processing."""

        def _execute(X_mag_pad, roi_size):
            patches = (X_mag_pad.shape[2] - 2 * self.model_run.offset) // roi_size
            if patches <= 0:
                raise ValueError("Window size error: no patches could be processed")

            mask = None
            batch_size = max(1, int(self.batch_size))

            for i in tqdm(range(0, patches, batch_size), desc="VR inference"):
                batch_end = min(i + batch_size, patches)
                starts = [patch_idx * roi_size for patch_idx in range(i, batch_end)]
                X_batch = np.stack(
                    [X_mag_pad[:, :, start:start + self.window_size] for start in starts],
                    axis=0,
                ).astype(np.float32, copy=False)

                # Convert to NHWC for model input.
                X_mx = mx.array(np.transpose(X_batch, (0, 2, 3, 1)), dtype=mx.float32)
                pred = self.model_run.predict_mask(X_mx)  # (B, F, roi, 2)

                # Reorder and flatten batch dimension into time axis:
                # (B, F, roi, 2) -> (2, F, B * roi)
                pred = mx.transpose(pred, (3, 1, 0, 2))
                pred = mx.reshape(pred, (pred.shape[0], pred.shape[1], pred.shape[2] * pred.shape[3]))
                mx.eval(pred)
                pred_np = np.array(pred, dtype=np.float32, copy=False)

                if mask is None:
                    mask = np.empty((pred_np.shape[0], pred_np.shape[1], patches * roi_size), dtype=np.float32)

                write_start = i * roi_size
                write_end = write_start + pred_np.shape[2]
                mask[:, :, write_start:write_end] = pred_np

            return mask

        def postprocess(mask, X_mag, X_phase):
            is_non_accom_stem = self.primary_stem_name in CommonSeparator.NON_ACCOM_STEMS
            mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, aggressiveness)

            if self.enable_post_process:
                mask = spec_utils.merge_artifacts(mask, thres=self.post_process_threshold)

            y_spec = mask * X_mag * np.exp(1.0j * X_phase)
            v_spec = (1 - mask) * X_mag * np.exp(1.0j * X_phase)
            return y_spec, v_spec

        X_mag, X_phase = spec_utils.preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, self.window_size, self.model_run.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
        max_val = X_mag_pad.max()
        if max_val > 0:
            X_mag_pad /= max_val

        mask = _execute(X_mag_pad, roi_size)

        if self.enable_tta:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
            max_val = X_mag_pad.max()
            if max_val > 0:
                X_mag_pad /= max_val
            mask_tta = _execute(X_mag_pad, roi_size)
            mask_tta = mask_tta[:, :, roi_size // 2:]
            mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5
        else:
            mask = mask[:, :, :n_frame]

        return postprocess(mask, X_mag, X_phase)

    def _spec_to_wav(self, spec):
        """Convert spectrogram back to waveform."""
        if self.high_end_process and isinstance(self.input_high_end, np.ndarray) and self.input_high_end_h:
            input_high_end_ = spec_utils.mirroring("mirroring", spec, self.input_high_end, self.model_params)
            return spec_utils.cmb_spectrogram_to_wave(
                spec, self.model_params, self.input_high_end_h, input_high_end_,
                is_v51_model=self.is_vr_51_model,
            )
        else:
            return spec_utils.cmb_spectrogram_to_wave(
                spec, self.model_params, is_v51_model=self.is_vr_51_model,
            )
