"""Base class common to all architecture-specific MLX separator classes."""

import gc
import os
import re
import time
from logging import Logger

import mlx_audio_io as mac
import numpy as np
from mlx_audio_separator.utils.performance import AsyncStemWriter, clear_mlx_cache


def normalize(wave, max_peak=1.0, min_peak=None):
    """Normalize (or amplify) audio waveform to a specified peak value."""
    maxv = np.abs(wave).max()
    if maxv > max_peak:
        wave *= max_peak / maxv
    elif min_peak is not None and maxv < min_peak:
        wave *= min_peak / maxv
    return wave


def match_array_shapes(array_1, array_2):
    """Match array_1's last dimension to array_2's last dimension."""
    if array_1.shape[-1] > array_2.shape[-1]:
        array_1 = array_1[..., :array_2.shape[-1]]
    elif array_1.shape[-1] < array_2.shape[-1]:
        padding = array_2.shape[-1] - array_1.shape[-1]
        array_1 = np.pad(array_1, [(0, 0)] * (array_1.ndim - 1) + [(0, padding)], "constant")
    return array_1


class CommonSeparator:
    """Base class with common methods and attributes for all MLX architecture-specific separators."""

    ALL_STEMS = "All Stems"
    VOCAL_STEM = "Vocals"
    INST_STEM = "Instrumental"
    OTHER_STEM = "Other"
    BASS_STEM = "Bass"
    DRUM_STEM = "Drums"
    GUITAR_STEM = "Guitar"
    PIANO_STEM = "Piano"
    SYNTH_STEM = "Synthesizer"
    STRINGS_STEM = "Strings"
    WOODWINDS_STEM = "Woodwinds"
    BRASS_STEM = "Brass"
    WIND_INST_STEM = "Wind Inst"
    NO_OTHER_STEM = "No Other"
    NO_BASS_STEM = "No Bass"
    NO_DRUM_STEM = "No Drums"
    NO_GUITAR_STEM = "No Guitar"
    NO_PIANO_STEM = "No Piano"
    NO_SYNTH_STEM = "No Synthesizer"
    NO_STRINGS_STEM = "No Strings"
    NO_WOODWINDS_STEM = "No Woodwinds"
    NO_WIND_INST_STEM = "No Wind Inst"
    NO_BRASS_STEM = "No Brass"
    PRIMARY_STEM = "Primary Stem"
    SECONDARY_STEM = "Secondary Stem"
    LEAD_VOCAL_STEM = "lead_only"
    BV_VOCAL_STEM = "backing_only"
    LEAD_VOCAL_STEM_I = "with_lead_vocals"
    BV_VOCAL_STEM_I = "with_backing_vocals"
    LEAD_VOCAL_STEM_LABEL = "Lead Vocals"
    BV_VOCAL_STEM_LABEL = "Backing Vocals"
    NO_STEM = "No "

    STEM_PAIR_MAPPER = {
        VOCAL_STEM: INST_STEM, INST_STEM: VOCAL_STEM,
        LEAD_VOCAL_STEM: BV_VOCAL_STEM, BV_VOCAL_STEM: LEAD_VOCAL_STEM,
        PRIMARY_STEM: SECONDARY_STEM,
    }

    NON_ACCOM_STEMS = (
        VOCAL_STEM, OTHER_STEM, BASS_STEM, DRUM_STEM, GUITAR_STEM,
        PIANO_STEM, SYNTH_STEM, STRINGS_STEM, WOODWINDS_STEM,
        BRASS_STEM, WIND_INST_STEM,
    )

    def __init__(self, config):
        self.logger: Logger = config.get("logger")
        self.log_level: int = config.get("log_level")

        self.model_name = config.get("model_name")
        self.model_path = config.get("model_path")
        self.model_data = config.get("model_data")

        self.output_dir = config.get("output_dir")
        self.output_format = config.get("output_format")
        self.output_bitrate = config.get("output_bitrate")

        self.normalization_threshold = config.get("normalization_threshold")
        self.amplification_threshold = config.get("amplification_threshold")
        self.enable_denoise = config.get("enable_denoise")
        self.output_single_stem = config.get("output_single_stem")
        self.invert_using_spec = config.get("invert_using_spec")
        self.sample_rate = config.get("sample_rate")
        self.performance_params = config.get("performance_params", {}) or {}
        self.cache_clear_policy = self.performance_params.get("cache_clear_policy", "aggressive")
        self.write_workers = int(self.performance_params.get("write_workers", 1))
        self.experimental_flac_fast_write = bool(self.performance_params.get("experimental_flac_fast_write", False))
        self._write_suppressed = False
        self._writer = None
        self.reset_perf_metrics()

        self.primary_stem_name = None
        self.secondary_stem_name = None

        self.input_encoding = None

        if "training" in self.model_data and "instruments" in self.model_data["training"]:
            instruments = self.model_data["training"]["instruments"]
            if instruments:
                self.primary_stem_name = instruments[0]
                self.secondary_stem_name = instruments[1] if len(instruments) > 1 else self.secondary_stem(self.primary_stem_name)

        if self.primary_stem_name is None:
            self.primary_stem_name = self.model_data.get("primary_stem", "Vocals")
            self.secondary_stem_name = self.secondary_stem(self.primary_stem_name)

        self.is_karaoke = self.model_data.get("is_karaoke", False)
        self.is_bv_model = self.model_data.get("is_bv_model", False)
        self.bv_model_rebalance = self.model_data.get("is_bv_model_rebalanced", 0)

        self.logger.debug(f"Common params: model_name={self.model_name}, model_path={self.model_path}")
        self.logger.debug(f"Common params: output_dir={self.output_dir}, output_format={self.output_format}")
        self.logger.debug(f"Common params: normalization_threshold={self.normalization_threshold}")
        self.logger.debug(f"Common params: primary_stem_name={self.primary_stem_name}, secondary_stem_name={self.secondary_stem_name}")

        self.audio_file_path = None
        self.audio_file_base = None
        self.primary_source = None
        self.secondary_source = None
        self.primary_stem_output_path = None
        self.secondary_stem_output_path = None
        self.cached_sources_map = {}

    def secondary_stem(self, primary_stem: str):
        primary_stem = primary_stem if primary_stem else self.NO_STEM
        if primary_stem in self.STEM_PAIR_MAPPER:
            return self.STEM_PAIR_MAPPER[primary_stem]
        return primary_stem.replace(self.NO_STEM, "") if self.NO_STEM in primary_stem else f"{self.NO_STEM}{primary_stem}"

    def separate(self, audio_file_path):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def final_process(self, stem_path, source, stem_name):
        self.logger.debug(f"Finalizing {stem_name} stem processing and writing audio...")
        self.write_audio(stem_path, source)
        return {stem_name: source}

    def cached_sources_clear(self):
        self.cached_sources_map = {}

    def reset_perf_metrics(self):
        self.perf_metrics = {
            "decode_s": 0.0,
            "preprocess_s": 0.0,
            "inference_s": 0.0,
            "postprocess_s": 0.0,
            "write_s": 0.0,
            "cleanup_s": 0.0,
            "total_s": 0.0,
        }

    def add_perf_time(self, key: str, delta: float):
        if key not in self.perf_metrics:
            self.perf_metrics[key] = 0.0
        self.perf_metrics[key] += float(max(delta, 0.0))

    def set_write_suppressed(self, enabled: bool):
        self._write_suppressed = bool(enabled)

    def get_perf_metrics(self):
        return dict(self.perf_metrics)

    def flush_pending_writes(self):
        if self._writer is None:
            return
        t0 = time.perf_counter()
        self._writer.flush()
        self.add_perf_time("write_s", time.perf_counter() - t0)

    def prepare_mix(self, mix):
        """Load and prepare audio mix using mlx-audio-io."""
        audio_path = mix

        if not isinstance(mix, np.ndarray):
            self.logger.debug(f"Loading audio from file: {mix}")

            # Get audio info for encoding detection
            try:
                audio_info = mac.info(str(mix))
                self.input_encoding = audio_info.subtype
                self.logger.info(f"Input audio encoding: {self.input_encoding}, sr: {audio_info.sample_rate}")
            except Exception as e:
                self.logger.warning(f"Could not read audio file info: {e}")
                self.input_encoding = "pcm16"

            # Load with mlx-audio-io, resample to target sample rate
            audio_mx, sr = mac.load(str(mix), sr=self.sample_rate, dtype="float32")

            # Convert to numpy and transpose to (channels, frames)
            mix = np.array(audio_mx, copy=False)
            if mix.ndim == 2:
                mix = mix.T  # (frames, channels) -> (channels, frames)
            self.logger.debug(f"Audio loaded. Sample rate: {sr}, Audio shape: {mix.shape}")
        else:
            self.logger.debug("Using provided mix array.")
            if self.input_encoding is None:
                self.input_encoding = "pcm16"
            if mix.ndim == 2 and mix.shape[0] > mix.shape[1]:
                # Looks like (frames, channels), transpose to (channels, frames)
                mix = mix.T

        # Validate audio content
        if isinstance(audio_path, str):
            if not np.any(mix):
                error_msg = f"Audio file {audio_path} is empty or not valid"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        # Ensure stereo
        if mix.ndim == 1:
            self.logger.debug("Mix is mono. Converting to stereo.")
            mix = np.stack([mix, mix], axis=0)

        return mix

    def write_audio(self, stem_path: str, stem_source):
        """Write audio using mlx-audio-io."""
        if self._write_suppressed:
            self.logger.debug("Write suppressed for tuning run.")
            return

        stem_source = normalize(wave=stem_source, max_peak=self.normalization_threshold, min_peak=self.amplification_threshold)

        if np.max(np.abs(stem_source)) < 1e-6:
            self.logger.warning("stem_source array is near-silent; writing silent stem to preserve output contract.")

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            stem_path = os.path.join(self.output_dir, stem_path)

        # Determine encoding based on input and output format
        output_encoding = "pcm16"
        if self.input_encoding in ("float32", "pcm32"):
            output_encoding = "float32"
        elif self.input_encoding == "pcm24":
            output_encoding = "pcm24"
        self.logger.debug(f"Output encoding: {output_encoding} (input was {self.input_encoding})")

        # stem_source expected shape: (frames, channels) or (channels, frames)
        # mlx-audio-io expects (frames, channels)
        if stem_source.ndim == 2:
            if stem_source.shape[0] == 2 and stem_source.shape[1] > 2:
                # Looks like (channels, frames), transpose
                stem_source = stem_source.T

        # Determine bitrate for lossy formats
        file_format = stem_path.lower().split(".")[-1]
        bitrate = "auto"
        if file_format == "mp3" and self.output_bitrate:
            bitrate = self.output_bitrate
        elif file_format == "mp3":
            bitrate = "320k"
        flac_fast_write = bool(self.experimental_flac_fast_write and file_format == "flac")

        try:
            if self.write_workers > 1:
                if self._writer is None:
                    self._writer = AsyncStemWriter(workers=self.write_workers)
                self._writer.submit(
                    stem_path=str(stem_path),
                    stem_source=stem_source,
                    sample_rate=self.sample_rate,
                    encoding=output_encoding,
                    bitrate=bitrate,
                    flac_fast_write=flac_fast_write,
                )
            else:
                t0 = time.perf_counter()
                save_kwargs = {
                    "encoding": output_encoding,
                    "bitrate": bitrate,
                }
                if file_format == "flac":
                    save_kwargs["flac_compression"] = "fast" if flac_fast_write else "default"
                try:
                    mac.save(str(stem_path), stem_source, self.sample_rate, **save_kwargs)
                except TypeError:
                    # Backward-compatible fallback for mlx-audio-io versions without flac_compression.
                    save_kwargs.pop("flac_compression", None)
                    mac.save(str(stem_path), stem_source, self.sample_rate, **save_kwargs)
                self.add_perf_time("write_s", time.perf_counter() - t0)
            self.logger.debug(f"Exported audio file successfully to {stem_path}")
        except Exception as e:
            self.logger.error(f"Error exporting audio file: {e}")
            raise

    def clear_gpu_cache(self):
        self.logger.debug("Running garbage collection...")
        gc.collect()
        try:
            clear_mlx_cache(self.logger)
        except Exception:
            pass

    def clear_file_specific_paths(self):
        self.logger.info("Clearing input audio file paths, sources and stems...")
        self.audio_file_path = None
        self.audio_file_base = None
        self.primary_source = None
        self.secondary_source = None
        self.primary_stem_output_path = None
        self.secondary_stem_output_path = None
        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None

    def sanitize_filename(self, filename):
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_. ')
        return sanitized

    def get_stem_output_path(self, stem_name, custom_output_names):
        if custom_output_names:
            custom_output_names_lower = {k.lower(): v for k, v in custom_output_names.items()}
            stem_name_lower = stem_name.lower()
            if stem_name_lower in custom_output_names_lower:
                sanitized_custom_name = self.sanitize_filename(custom_output_names_lower[stem_name_lower])
                return f"{sanitized_custom_name}.{self.output_format.lower()}"

        sanitized_audio_base = self.sanitize_filename(self.audio_file_base)
        sanitized_stem_name = self.sanitize_filename(stem_name)
        sanitized_model_name = self.sanitize_filename(self.model_name)

        return f"{sanitized_audio_base}_({sanitized_stem_name})_{sanitized_model_name}.{self.output_format.lower()}"
