"""Demucs stem separator using MLX backend via demucs_mlx."""

import os
import time

import mlx.core as mx
import numpy as np

from mlx_audio_separator.demucs_mlx.defaults import DEFAULT_BATCH_SIZE
from mlx_audio_separator.separator.common_separator import CommonSeparator
from mlx_audio_separator.utils.performance import apply_experimental_env


class DemucsSeparator(CommonSeparator):
    """
    Demucs architecture separator using MLX acceleration.

    Delegates to the demucs_mlx package for model loading and inference.
    """

    def __init__(self, common_config, arch_config):
        super().__init__(config=common_config)

        # Demucs-specific parameters
        self.segment_size = arch_config.get("segment_size", "Default")
        self.shifts = arch_config.get("shifts", 2)
        self.overlap = arch_config.get("overlap", 0.25)
        self.segments_enabled = arch_config.get("segments_enabled", True)
        self.batch_size = int(arch_config.get("batch_size", DEFAULT_BATCH_SIZE))
        self.seed = arch_config.get("seed")
        self.experimental_demucs_wiener_preallocate_output = bool(
            self.performance_params.get("experimental_demucs_wiener_preallocate_output", False)
        )
        self.experimental_demucs_apply_concat_batching = bool(
            self.performance_params.get("experimental_demucs_apply_concat_batching", False)
        )
        self.experimental_demucs_gn_glu_multigroup = bool(
            self.performance_params.get("experimental_demucs_gn_glu_multigroup", False)
        )
        # Only publish a flag the caller actually asked about; otherwise an
        # exported value would be silently overwritten before any kernel is
        # built (see apply_experimental_env).
        explicit = getattr(self, "performance_params_explicit_keys", frozenset())
        for env_var, key, enabled in (
            (
                "MLX_AUDIO_SEPARATOR_DEMUCS_WIENER_PREALLOC_OUTPUT",
                "experimental_demucs_wiener_preallocate_output",
                self.experimental_demucs_wiener_preallocate_output,
            ),
            (
                "MLX_AUDIO_SEPARATOR_DEMUCS_APPLY_CONCAT_BATCHING",
                "experimental_demucs_apply_concat_batching",
                self.experimental_demucs_apply_concat_batching,
            ),
            (
                "MLX_AUDIO_SEPARATOR_GN_GLU_MULTIGROUP",
                "experimental_demucs_gn_glu_multigroup",
                self.experimental_demucs_gn_glu_multigroup,
            ),
        ):
            apply_experimental_env(
                env_var, enabled, explicit=key in explicit, logger=self.logger
            )

        self.logger.debug(
            f"Demucs params: segment_size={self.segment_size}, shifts={self.shifts}, "
            f"overlap={self.overlap}, batch_size={self.batch_size}, seed={self.seed}"
        )

        # Determine demucs model name from the YAML model data
        self._demucs_model_name = self._resolve_demucs_model_name()
        self.logger.info(f"Demucs MLX model name resolved to: {self._demucs_model_name}")

        # Load the demucs_mlx Separator
        from mlx_audio_separator.demucs_mlx.api import Separator as DemucsMLXSeparator

        segment = None
        if self.segment_size != "Default" and self.segments_enabled:
            try:
                segment = float(self.segment_size)
            except (ValueError, TypeError):
                pass

        self._demucs_separator = DemucsMLXSeparator(
            model=self._demucs_model_name,
            shifts=self.shifts,
            overlap=self.overlap,
            split=self.segments_enabled,
            segment=segment,
            batch_size=self.batch_size,
            seed=self.seed,
            progress=self.log_level <= 10,  # Show progress for DEBUG level
        )

        self.logger.info(
            f"Demucs MLX model loaded: samplerate={self._demucs_separator.samplerate}, "
            f"channels={self._demucs_separator.audio_channels}"
        )

    def _resolve_demucs_model_name(self):
        """Resolve the demucs-mlx model name from the loaded model data/path."""
        from mlx_audio_separator.demucs_mlx.mlx_registry import MLX_MODEL_REGISTRY

        # Try to match from model_name (e.g., "htdemucs_ft")
        if self.model_name in MLX_MODEL_REGISTRY:
            return self.model_name

        # Try from model_path filename
        if self.model_path:
            basename = os.path.basename(self.model_path)
            name_without_ext = os.path.splitext(basename)[0]
            if name_without_ext in MLX_MODEL_REGISTRY:
                return name_without_ext

        # Try matching from model_data
        if self.model_data:
            # Demucs YAML files have model names in various places
            for key in ["name", "model", "model_name"]:
                if key in self.model_data and self.model_data[key] in MLX_MODEL_REGISTRY:
                    return self.model_data[key]

            # Check bag_of_models - if it's a bag, the YAML name is the model name
            if "models" in self.model_data:
                # This is a bag-of-models YAML like htdemucs_ft.yaml
                # The model name matches the yaml filename
                if self.model_path:
                    yaml_name = os.path.splitext(os.path.basename(self.model_path))[0]
                    if yaml_name in MLX_MODEL_REGISTRY:
                        return yaml_name

        # Default fallback
        self.logger.warning(f"Could not resolve demucs model name from '{self.model_name}', falling back to 'htdemucs'")
        return "htdemucs"

    def separate(self, audio_file_path, custom_output_names=None):
        """Separate audio file into stems using Demucs MLX."""
        import mlx_audio_io as mac

        self.reset_perf_metrics()
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        self.logger.info(f"Separating {audio_file_path} with Demucs MLX model {self._demucs_model_name}")

        # Decode (kept in MLX tensors to avoid host round-trips).
        t0 = time.perf_counter()
        audio_mx, sr = mac.load(str(audio_file_path), sr=self._demucs_separator.samplerate, dtype="float32")
        wav_mx = audio_mx.T if audio_mx.ndim == 2 else mx.stack([audio_mx, audio_mx], axis=0)
        self.add_perf_time("decode_s", time.perf_counter() - t0)

        # Inference.
        t0 = time.perf_counter()
        _, stems = self._demucs_separator.separate_tensor(wav_mx, return_mx=True)
        self.add_perf_time("inference_s", time.perf_counter() - t0)

        self.logger.info(f"Demucs separation complete. Stems: {list(stems.keys())}")

        # Demucs runs at the model's own rate; resample to the requested output
        # rate before writing, otherwise the header rate and the payload
        # disagree and the file plays back at the wrong length.
        model_sr = int(self._demucs_separator.samplerate)
        target_sr = int(self.sample_rate) if self.sample_rate else model_sr
        if target_sr != model_sr:
            self.logger.debug(f"Resampling stems from {model_sr} Hz to {target_sr} Hz for output.")

        # Write output files
        output_files = []
        for stem_name, stem_data in stems.items():
            # Skip conversion/write work for filtered stems.
            if self.output_single_stem is not None and stem_name.lower() != self.output_single_stem.lower():
                continue

            # Postprocess host transfer per emitted stem only.
            t0 = time.perf_counter()
            if target_sr != model_sr:
                stem_data = mac.resample(stem_data, model_sr, target_sr, layout="channels_first")
            stem_np = np.asarray(stem_data)
            if stem_np.ndim == 2:
                stem_np = stem_np.T
            self.add_perf_time("postprocess_s", time.perf_counter() - t0)

            stem_output_path = self.get_stem_output_path(stem_name, custom_output_names)
            self.logger.info(f"Writing stem '{stem_name}' to {stem_output_path}")
            self.write_audio(stem_output_path, stem_np)

            if self.output_dir:
                full_path = os.path.join(self.output_dir, stem_output_path)
            else:
                full_path = stem_output_path
            output_files.append(full_path)

        return output_files
