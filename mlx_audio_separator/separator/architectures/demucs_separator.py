"""Demucs stem separator using MLX backend via demucs_mlx."""

import os

import numpy as np

from mlx_audio_separator.separator.common_separator import CommonSeparator


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
        self.batch_size = int(arch_config.get("batch_size", 8))

        self.logger.debug(
            f"Demucs params: segment_size={self.segment_size}, shifts={self.shifts}, "
            f"overlap={self.overlap}, batch_size={self.batch_size}"
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
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        self.logger.info(f"Separating {audio_file_path} with Demucs MLX model {self._demucs_model_name}")

        # Use demucs_mlx to separate
        mix_wav, stems = self._demucs_separator.separate_audio_file(audio_file_path)

        self.logger.info(f"Demucs separation complete. Stems: {list(stems.keys())}")

        # Write output files
        output_files = []
        for stem_name, stem_data in stems.items():
            # stem_data shape is (channels, time) as numpy array
            stem_np = np.asarray(stem_data)

            # Transpose to (time, channels) for write_audio
            if stem_np.ndim == 2:
                stem_np = stem_np.T

            # Check if we should skip this stem
            if self.output_single_stem is not None:
                if stem_name.lower() != self.output_single_stem.lower():
                    continue

            stem_output_path = self.get_stem_output_path(stem_name, custom_output_names)
            self.logger.info(f"Writing stem '{stem_name}' to {stem_output_path}")
            self.write_audio(stem_output_path, stem_np)

            if self.output_dir:
                full_path = os.path.join(self.output_dir, stem_output_path)
            else:
                full_path = stem_output_path
            output_files.append(full_path)

        return output_files
