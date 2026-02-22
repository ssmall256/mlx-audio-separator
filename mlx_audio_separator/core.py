"""Main Separator class for MLX-native audio stem separation."""

import hashlib
import importlib
import io
import json
import logging
import os
import platform
import re
import subprocess
import time
import warnings
from datetime import datetime, timezone
from importlib import metadata

import requests
import yaml
from tqdm import tqdm
from mlx_audio_separator.utils.performance import (
    PerfTraceWriter,
    clear_mlx_cache,
    load_tuning_cache,
    normalize_performance_params,
    save_tuning_cache,
    select_best_candidate,
)


class Separator:
    """
    MLX-native audio stem separator.

    Supports Demucs, MDXC/Roformer, MDX, and VR architectures using Apple Silicon
    acceleration via MLX. No PyTorch or ONNX runtime required.

    Common Attributes:
        log_level (int): The logging level.
        model_file_dir (str): The directory where model files are stored.
        output_dir (str): The directory where output files will be saved.
        output_format (str): The format of the output audio file.
        output_bitrate (str): The bitrate of the output audio file.
        normalization_threshold (float): The threshold for audio normalization.
        amplification_threshold (float): The threshold for audio amplification.
        output_single_stem (str): Option to output a single stem.
        invert_using_spec (bool): Flag to invert using spectrogram.
        sample_rate (int): The sample rate of the audio.

    Demucs Architecture Specific Attributes & Defaults:
        segment_size: "Default"
        shifts: 2
        overlap: 0.25
        segments_enabled: True
        batch_size: 8

    MDXC Architecture Specific Attributes & Defaults:
        segment_size: 256
        override_model_segment_size: False
        batch_size: 1
        overlap: 8
        pitch_shift: 0

    Performance Parameters (opt-in):
        speed_mode: "default" | "latency_safe"
        auto_tune_batch: False
        tune_probe_seconds: 8.0
        cache_clear_policy: "aggressive" | "deferred"
        write_workers: 1
        perf_trace: False
        perf_trace_path: None
    """

    def __init__(
        self,
        log_level=logging.INFO,
        log_formatter=None,
        model_file_dir="/tmp/audio-separator-models/",
        output_dir=None,
        output_format="WAV",
        output_bitrate=None,
        normalization_threshold=0.9,
        amplification_threshold=0.0,
        output_single_stem=None,
        invert_using_spec=False,
        sample_rate=44100,
        chunk_duration=None,
        demucs_params=None,
        mdxc_params=None,
        mdx_params=None,
        vr_params=None,
        performance_params=None,
        info_only=False,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.log_level = log_level
        self.log_formatter = log_formatter

        self.log_handler = logging.StreamHandler()

        if self.log_formatter is None:
            self.log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")

        self.log_handler.setFormatter(self.log_formatter)

        if not self.logger.hasHandlers():
            self.logger.addHandler(self.log_handler)

        if log_level > logging.DEBUG:
            warnings.filterwarnings("ignore")

        if not info_only:
            package_version = self.get_package_distribution("mlx-audio-separator")
            version_str = package_version.version if package_version else "dev"
            self.logger.info(
                f"MLX Audio Separator version {version_str} instantiating with "
                f"output_dir: {output_dir}, output_format: {output_format}"
            )

        if output_dir is None:
            output_dir = os.getcwd()
            if not info_only:
                self.logger.info("Output directory not specified. Using current working directory.")

        self.output_dir = output_dir

        env_model_dir = os.environ.get("AUDIO_SEPARATOR_MODEL_DIR")
        if env_model_dir:
            self.model_file_dir = env_model_dir
            self.logger.info(f"Using model directory from AUDIO_SEPARATOR_MODEL_DIR env var: {self.model_file_dir}")
            if not os.path.exists(self.model_file_dir):
                raise FileNotFoundError(f"The specified model directory does not exist: {self.model_file_dir}")
        else:
            self.logger.info(f"Using model directory from model_file_dir parameter: {model_file_dir}")
            self.model_file_dir = model_file_dir

        os.makedirs(self.model_file_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.output_format = output_format
        self.output_bitrate = output_bitrate

        if self.output_format is None:
            self.output_format = "WAV"

        self.normalization_threshold = normalization_threshold
        if normalization_threshold <= 0 or normalization_threshold > 1:
            raise ValueError("The normalization_threshold must be greater than 0 and less than or equal to 1.")

        self.amplification_threshold = amplification_threshold
        if amplification_threshold < 0 or amplification_threshold > 1:
            raise ValueError("The amplification_threshold must be greater than or equal to 0 and less than or equal to 1.")

        self.output_single_stem = output_single_stem
        if output_single_stem is not None:
            self.logger.debug(f"Single stem output requested, so only one output file ({output_single_stem}) will be written")

        self.invert_using_spec = invert_using_spec

        try:
            self.sample_rate = int(sample_rate)
            if self.sample_rate <= 0:
                raise ValueError(f"The sample rate setting is {self.sample_rate} but it must be a non-zero whole number.")
            if self.sample_rate > 12800000:
                raise ValueError(f"The sample rate setting is {self.sample_rate}. Enter something less ambitious.")
        except ValueError:
            raise ValueError("The sample rate must be a non-zero whole number. Please provide a valid integer.")

        self.chunk_duration = chunk_duration
        if chunk_duration is not None:
            if chunk_duration <= 0:
                raise ValueError("chunk_duration must be greater than 0")
        self.performance_params = normalize_performance_params(performance_params)

        if demucs_params is None:
            demucs_params = {
                "segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True, "batch_size": 8,
            }
        if mdxc_params is None:
            mdxc_params = {
                "segment_size": 256, "override_model_segment_size": False,
                "batch_size": 1, "overlap": 8, "pitch_shift": 0,
            }
        if mdx_params is None:
            mdx_params = {
                "hop_length": 1024, "segment_size": 256, "overlap": 0.25,
                "batch_size": 1, "enable_denoise": False,
            }
        if vr_params is None:
            vr_params = {
                "batch_size": 1, "window_size": 512, "aggression": 5,
                "enable_tta": False, "enable_post_process": False,
                "post_process_threshold": 0.2, "high_end_process": False,
            }

        self.arch_specific_params = {
            "Demucs": demucs_params, "MDXC": mdxc_params,
            "MDX": mdx_params, "VR": vr_params,
        }
        self._apply_speed_mode_overrides()

        self.model_instance = None
        self.model_type = None
        self.last_perf_metrics = None
        self.model_is_uvr_vip = False
        self.model_friendly_name = None
        self._files_since_cache_clear = 0
        self._tuning_cache = load_tuning_cache()
        self._perf_trace_writer = None
        self._skip_auto_tune = False
        if self.performance_params["perf_trace"]:
            trace_path = self.performance_params["perf_trace_path"]
            if trace_path is None:
                trace_path = os.path.join(self.output_dir, "perf_trace.jsonl")
            self._perf_trace_writer = PerfTraceWriter(trace_path)

        if not info_only:
            self._check_mlx_available()

    def _check_mlx_available(self):
        """Verify MLX is available on this system."""
        try:
            import mlx.core as mx
            self.logger.info(f"MLX available. Default device: {mx.default_device()}")
        except ImportError:
            raise RuntimeError("MLX is not installed. This package requires Apple Silicon with MLX. Install with: pip install mlx")

        system_info = platform.uname()
        self.logger.info(f"System: {system_info.system} {system_info.machine} Python: {platform.python_version()}")
        self.check_ffmpeg_installed()

    def check_ffmpeg_installed(self):
        try:
            ffmpeg_version_output = subprocess.check_output(["ffmpeg", "-version"], text=True)
            first_line = ffmpeg_version_output.splitlines()[0]
            self.logger.info(f"FFmpeg installed: {first_line}")
        except FileNotFoundError:
            self.logger.error("FFmpeg is not installed. Please install FFmpeg to use this package.")
            if "PYTEST_CURRENT_TEST" not in os.environ:
                raise

    def _apply_speed_mode_overrides(self):
        speed_mode = self.performance_params["speed_mode"]
        if speed_mode != "latency_safe":
            return
        self.logger.info("Applying latency_safe speed-mode presets.")
        self.arch_specific_params["Demucs"]["batch_size"] = 12
        self.arch_specific_params["MDXC"]["batch_size"] = 1
        self.arch_specific_params["MDX"]["batch_size"] = 1
        self.arch_specific_params["VR"]["batch_size"] = 2

    def _build_tuning_key(self, arch: str, model_name: str, sr: int, channels: int):
        device = "unknown"
        try:
            import mlx.core as mx
            device = str(mx.default_device())
        except Exception:
            pass
        return f"{arch}|{model_name}|sr={int(sr)}|ch={int(channels)}|device={device}"

    def _set_model_batch_size(self, batch_size: int):
        batch_size = int(batch_size)
        if self.model_instance is None:
            return
        if self.model_type == "Demucs":
            self.model_instance.batch_size = batch_size
            if hasattr(self.model_instance, "_demucs_separator"):
                self.model_instance._demucs_separator.batch_size = batch_size
        elif self.model_type in {"MDXC", "MDX", "VR"}:
            self.model_instance.batch_size = batch_size

    def _candidate_batch_sizes(self):
        return {
            "Demucs": [4, 8, 12],
            "MDXC": [1, 2, 4],
            "MDX": [1, 2, 4],
            "VR": [1, 2, 4],
        }

    def _get_model_batch_size(self):
        if self.model_instance is None:
            return None
        return int(getattr(self.model_instance, "batch_size", 1))

    def _auto_tune_batch_if_needed(self, audio_file_path):
        if not self.performance_params["auto_tune_batch"]:
            return
        if self.model_instance is None or self.model_type not in self._candidate_batch_sizes():
            return

        import tempfile
        import mlx_audio_io as mac
        try:
            info = mac.info(str(audio_file_path))
            channels = int(getattr(info, "channels", 2))
            key = self._build_tuning_key(self.model_type, self.model_name, self.sample_rate, channels)
            if key in self._tuning_cache:
                tuned = int(self._tuning_cache[key]["batch_size"])
                self.logger.info(f"Using cached auto-tuned batch size {tuned} for {self.model_type}.")
                self._set_model_batch_size(tuned)
                return

            probe_seconds = float(self.performance_params["tune_probe_seconds"])
            audio_mx, sr = mac.load(str(audio_file_path), sr=self.sample_rate, dtype="float32")
            probe_frames = max(1, int(round(probe_seconds * sr)))
            probe_audio = audio_mx[:probe_frames]

            with tempfile.TemporaryDirectory(prefix="audio-separator-tune-") as tune_dir:
                probe_path = os.path.join(tune_dir, "probe.wav")
                mac.save(probe_path, probe_audio, sr, encoding="float32")

                original_output_dir = self.output_dir
                original_model_output_dir = getattr(self.model_instance, "output_dir", original_output_dir)
                original_batch = self._get_model_batch_size()

                self.output_dir = tune_dir
                self.model_instance.output_dir = tune_dir
                self.model_instance.set_write_suppressed(True)

                timings = {}
                try:
                    for candidate in self._candidate_batch_sizes()[self.model_type]:
                        self._set_model_batch_size(candidate)
                        candidate_timings = []
                        for _ in range(2):
                            t0 = time.perf_counter()
                            self.model_instance.separate(probe_path)
                            self.model_instance.flush_pending_writes()
                            candidate_timings.append(time.perf_counter() - t0)
                            self.model_instance.clear_file_specific_paths()
                        timings[candidate] = candidate_timings
                finally:
                    self.model_instance.set_write_suppressed(False)
                    self.output_dir = original_output_dir
                    self.model_instance.output_dir = original_model_output_dir
                    if original_batch is not None:
                        self._set_model_batch_size(original_batch)

            best = select_best_candidate(timings, tie_ratio=0.03)
            self._set_model_batch_size(best)
            self._tuning_cache[key] = {"batch_size": int(best), "timings": timings}
            save_tuning_cache(self._tuning_cache)
            self.logger.info(f"Auto-tuned batch size for {self.model_type}: {best}")
        except Exception as exc:
            self.logger.warning(f"Auto-tune failed, keeping configured batch size: {exc}")

    def _clear_cache_now(self):
        if self.model_instance:
            self.model_instance.clear_gpu_cache()
        else:
            clear_mlx_cache(self.logger)
        self._files_since_cache_clear = 0

    def _apply_cache_policy_after_file(self):
        policy = self.performance_params["cache_clear_policy"]
        if policy == "aggressive":
            self._clear_cache_now()
            return
        self._files_since_cache_clear += 1
        if self._files_since_cache_clear >= 10:
            self.logger.debug("Deferred cache policy: periodic clear after 10 files.")
            self._clear_cache_now()

    def _finalize_deferred_cache(self):
        if self.performance_params["cache_clear_policy"] == "deferred":
            self._clear_cache_now()

    def _emit_perf_trace(self, audio_file_path, metrics):
        if not self._perf_trace_writer:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "file": os.path.abspath(audio_file_path),
            "model": self.model_name,
            "arch": self.model_type,
            "metrics": metrics,
            "params": {
                "arch_params": self.arch_specific_params.get(self.model_type, {}),
                "performance_params": self.performance_params,
            },
        }
        self._perf_trace_writer.write(record)

    def get_package_distribution(self, package_name):
        try:
            return metadata.distribution(package_name)
        except metadata.PackageNotFoundError:
            self.logger.debug(f"Python package: {package_name} not installed")
            return None

    def get_model_hash(self, model_path):
        self.logger.debug(f"Calculating hash of model file {model_path}")
        BYTES_TO_HASH = 10000 * 1024

        try:
            file_size = os.path.getsize(model_path)
            with open(model_path, "rb") as f:
                if file_size < BYTES_TO_HASH:
                    hash_value = hashlib.md5(f.read()).hexdigest()
                else:
                    f.seek(file_size - BYTES_TO_HASH, io.SEEK_SET)
                    hash_value = hashlib.md5(f.read()).hexdigest()

            self.logger.info(f"Hash of model file {model_path} is {hash_value}")
            return hash_value
        except FileNotFoundError:
            self.logger.error(f"Model file not found at {model_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error calculating hash for {model_path}: {e}")
            raise

    def download_file_if_not_exists(self, url, output_path):
        if os.path.isfile(output_path):
            self.logger.debug(f"File already exists at {output_path}, skipping download")
            return

        self.logger.debug(f"Downloading file from {url} to {output_path} with timeout 300s")
        response = requests.get(url, stream=True, timeout=300)

        if response.status_code == 200:
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
        else:
            raise RuntimeError(f"Failed to download file from {url}, response code: {response.status_code}")

    def _load_model_scores(self):
        """Load model performance scores from bundled models-scores.json."""
        scores_path = os.path.join(os.path.dirname(__file__), "models-scores.json")
        try:
            with open(scores_path, encoding="utf-8") as f:
                scores = json.load(f)
            self.logger.debug("Model scores loaded")
            return scores
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Failed to load model scores: {e}")
            return {}

    def _score_entry(self, model_scores, filename):
        """Extract scores/stems/target_stem from model_scores for a given filename."""
        entry = model_scores.get(filename, {})
        return {
            "scores": entry.get("median_scores", {}),
            "stems": entry.get("stems", []),
            "target_stem": entry.get("target_stem"),
        }

    def list_supported_model_files(self):
        download_checks_path = os.path.join(self.model_file_dir, "download_checks.json")

        self.download_file_if_not_exists(
            "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json",
            download_checks_path,
        )

        model_downloads_list = json.load(open(download_checks_path, encoding="utf-8"))
        self.logger.debug("UVR model download list loaded")

        model_scores = self._load_model_scores()

        # Only show Demucs v4 models
        filtered_demucs_v4 = {
            key: value for key, value in model_downloads_list["demucs_download_list"].items()
            if key.startswith("Demucs v4")
        }

        demucs_models = {}
        for name, files in filtered_demucs_v4.items():
            yaml_file = next((filename for filename in files.keys() if filename.endswith(".yaml")), None)
            if yaml_file:
                demucs_models[name] = {
                    "filename": yaml_file,
                    **self._score_entry(model_scores, yaml_file),
                    "download_files": list(files.values()),
                }

        # Load audio-separator models list
        models_json_path = os.path.join(os.path.dirname(__file__), "models.json")
        if os.path.exists(models_json_path):
            audio_separator_models_list = json.load(open(models_json_path, encoding="utf-8"))
        else:
            audio_separator_models_list = {
                "vr_download_list": {}, "mdx_download_list": {},
                "mdx23c_download_list": {}, "roformer_download_list": {},
            }

        model_files_grouped_by_type = {
            "VR": {
                name: {
                    "filename": filename,
                    **self._score_entry(model_scores, filename),
                    "download_files": [filename],
                }
                for name, filename in {
                    **model_downloads_list["vr_download_list"],
                    **audio_separator_models_list.get("vr_download_list", {}),
                }.items()
            },
            "MDX": {
                name: {
                    "filename": filename,
                    **self._score_entry(model_scores, filename),
                    "download_files": [filename],
                }
                for name, filename in {
                    **model_downloads_list["mdx_download_list"],
                    **model_downloads_list.get("mdx_download_vip_list", {}),
                    **audio_separator_models_list.get("mdx_download_list", {}),
                }.items()
            },
            "Demucs": demucs_models,
            "MDXC": {
                name: {
                    "filename": next(iter(files.keys())),
                    **self._score_entry(model_scores, next(iter(files.keys()))),
                    "download_files": list(files.keys()) + list(files.values()),
                }
                for name, files in {
                    **model_downloads_list["mdx23c_download_list"],
                    **model_downloads_list.get("mdx23c_download_vip_list", {}),
                    **model_downloads_list.get("roformer_download_list", {}),
                    **audio_separator_models_list.get("mdx23c_download_list", {}),
                    **audio_separator_models_list.get("roformer_download_list", {}),
                }.items()
            },
        }

        return model_files_grouped_by_type

    def get_simplified_model_list(self, filter_sort_by=None):
        """
        Returns a simplified, user-friendly dict of models with key metrics.

        :param filter_sort_by: "name" sorts by friendly name, "filename" sorts by filename,
                               any stem name (e.g. "vocals") filters to models with that stem
                               and sorts by SDR descending.
        """
        model_files = self.list_supported_model_files()
        simplified_list = {}

        for model_type, models in model_files.items():
            for name, data in models.items():
                filename = data["filename"]
                scores = data.get("scores") or {}
                stems = data.get("stems") or []
                target_stem = data.get("target_stem")

                stems_with_scores = []
                stem_sdr_dict = {}

                for stem in stems:
                    stem_scores = scores.get(stem, {})
                    stem_display = f"{stem}*" if stem == target_stem else stem

                    if isinstance(stem_scores, dict) and "SDR" in stem_scores:
                        sdr = round(stem_scores["SDR"], 1)
                        stems_with_scores.append(f"{stem_display} ({sdr})")
                        stem_sdr_dict[stem.lower()] = sdr
                    else:
                        stems_with_scores.append(stem_display)
                        stem_sdr_dict[stem.lower()] = None

                if not stems_with_scores:
                    stems_with_scores = ["Unknown"]
                    stem_sdr_dict["unknown"] = None

                simplified_list[filename] = {
                    "Name": name, "Type": model_type,
                    "Stems": stems_with_scores, "SDR": stem_sdr_dict,
                }

        if filter_sort_by:
            if filter_sort_by == "name":
                return dict(sorted(simplified_list.items(), key=lambda x: x[1]["Name"]))
            elif filter_sort_by == "filename":
                return dict(sorted(simplified_list.items()))
            else:
                sort_by_lower = filter_sort_by.lower()
                filtered_list = {k: v for k, v in simplified_list.items() if sort_by_lower in v["SDR"]}

                def sort_key(item):
                    sdr = item[1]["SDR"][sort_by_lower]
                    return (0 if sdr is None else 1, sdr if sdr is not None else float("-inf"))

                return dict(sorted(filtered_list.items(), key=sort_key, reverse=True))

        return simplified_list

    def print_uvr_vip_message(self):
        if self.model_is_uvr_vip:
            self.logger.warning(
                f"The model: '{self.model_friendly_name}' is a VIP model, "
                "intended by Anjok07 for access by paying subscribers only."
            )
            self.logger.warning(
                "If you are not already subscribed, please consider supporting "
                "the developer of UVR, Anjok07 by subscribing here: https://patreon.com/uvr"
            )

    def download_model_files(self, model_filename):
        model_path = os.path.join(self.model_file_dir, f"{model_filename}")

        supported_model_files_grouped = self.list_supported_model_files()
        public_model_repo_url_prefix = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"
        vip_model_repo_url_prefix = "https://github.com/Anjok0109/ai_magic/releases/download/v5"
        audio_separator_models_repo_url_prefix = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs"

        yaml_config_filename = None

        self.logger.debug(f"Searching for model_filename {model_filename} in supported_model_files_grouped")

        for model_type, models in supported_model_files_grouped.items():
            for model_friendly_name, model_info in models.items():
                self.model_is_uvr_vip = "VIP" in model_friendly_name
                model_repo_url_prefix = vip_model_repo_url_prefix if self.model_is_uvr_vip else public_model_repo_url_prefix

                if model_info["filename"] == model_filename or model_filename in model_info["download_files"]:
                    self.logger.debug(f"Found matching model: {model_friendly_name}")
                    self.model_friendly_name = model_friendly_name
                    self.print_uvr_vip_message()

                    for file_to_download in model_info["download_files"]:
                        if file_to_download.startswith("http"):
                            filename = file_to_download.split("/")[-1]
                            download_path = os.path.join(self.model_file_dir, filename)
                            self.download_file_if_not_exists(file_to_download, download_path)
                            continue

                        download_path = os.path.join(self.model_file_dir, file_to_download)

                        if model_type == "MDXC" and file_to_download.endswith(".yaml"):
                            yaml_config_filename = file_to_download
                            try:
                                yaml_url = f"{model_repo_url_prefix}/mdx_model_data/mdx_c_configs/{file_to_download}"
                                self.download_file_if_not_exists(yaml_url, download_path)
                            except RuntimeError:
                                self.logger.debug("YAML config not found in UVR repo, trying audio-separator models repo...")
                                yaml_url = f"{audio_separator_models_repo_url_prefix}/{file_to_download}"
                                self.download_file_if_not_exists(yaml_url, download_path)
                            continue

                        try:
                            download_url = f"{model_repo_url_prefix}/{file_to_download}"
                            self.download_file_if_not_exists(download_url, download_path)
                        except RuntimeError:
                            self.logger.debug("Model not found in UVR repo, trying audio-separator models repo...")
                            download_url = f"{audio_separator_models_repo_url_prefix}/{file_to_download}"
                            self.download_file_if_not_exists(download_url, download_path)

                    return model_filename, model_type, model_friendly_name, model_path, yaml_config_filename

        raise ValueError(f"Model file {model_filename} not found in supported model files")

    def load_model_data_from_yaml(self, yaml_config_filename):
        if not os.path.exists(yaml_config_filename):
            model_data_yaml_filepath = os.path.join(self.model_file_dir, yaml_config_filename)
        else:
            model_data_yaml_filepath = yaml_config_filename

        self.logger.debug(f"Loading model data from YAML at path {model_data_yaml_filepath}")
        model_data = yaml.load(open(model_data_yaml_filepath, encoding="utf-8"), Loader=yaml.FullLoader)
        self.logger.debug(f"Model data loaded from YAML file: {model_data}")

        if "roformer" in model_data_yaml_filepath.lower():
            model_data["is_roformer"] = True

        return model_data

    def load_model_data_using_hash(self, model_path):
        model_data_url_prefix = "https://raw.githubusercontent.com/TRvlvr/application_data/main"
        vr_model_data_url = f"{model_data_url_prefix}/vr_model_data/model_data_new.json"
        mdx_model_data_url = f"{model_data_url_prefix}/mdx_model_data/model_data_new.json"

        self.logger.debug("Calculating MD5 hash for model file to identify model parameters from UVR data...")
        model_hash = self.get_model_hash(model_path)

        vr_model_data_path = os.path.join(self.model_file_dir, "vr_model_data.json")
        self.download_file_if_not_exists(vr_model_data_url, vr_model_data_path)

        mdx_model_data_path = os.path.join(self.model_file_dir, "mdx_model_data.json")
        self.download_file_if_not_exists(mdx_model_data_url, mdx_model_data_path)

        vr_model_data_object = json.load(open(vr_model_data_path, encoding="utf-8"))
        mdx_model_data_object = json.load(open(mdx_model_data_path, encoding="utf-8"))

        # Merge local model data (bundled with package) for models not in upstream UVR data
        local_model_data_path = os.path.join(os.path.dirname(__file__), "model-data.json")
        if os.path.exists(local_model_data_path):
            local_data = json.load(open(local_model_data_path, encoding="utf-8"))
            vr_model_data_object = {**vr_model_data_object, **local_data.get("vr_model_data", {})}
            mdx_model_data_object = {**mdx_model_data_object, **local_data.get("mdx_model_data", {})}

        if model_hash in mdx_model_data_object:
            model_data = mdx_model_data_object[model_hash]
        elif model_hash in vr_model_data_object:
            model_data = vr_model_data_object[model_hash]
        else:
            raise ValueError(
                f"Unsupported Model File: parameters for MD5 hash {model_hash} "
                "could not be found in UVR model data file for MDX or VR arch."
            )

        self.logger.debug(f"Model data loaded using hash {model_hash}: {model_data}")
        return model_data

    def load_model(self, model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt"):
        """Load a separation model, downloading it first if necessary."""
        self.logger.info(f"Loading model {model_filename}...")

        load_model_start_time = time.perf_counter()

        model_filename, model_type, model_friendly_name, model_path, yaml_config_filename = self.download_model_files(model_filename)
        model_name = model_filename.split(".")[0]
        self.model_name = model_name
        self.logger.debug(f"Model downloaded, friendly name: {model_friendly_name}, model_path: {model_path}")

        if model_path.lower().endswith(".yaml"):
            yaml_config_filename = model_path

        if yaml_config_filename is not None:
            model_data = self.load_model_data_from_yaml(yaml_config_filename)
        else:
            model_data = self.load_model_data_using_hash(model_path)

        common_params = {
            "logger": self.logger,
            "log_level": self.log_level,
            "model_name": model_name,
            "model_path": model_path,
            "model_data": model_data,
            "output_format": self.output_format,
            "output_bitrate": self.output_bitrate,
            "output_dir": self.output_dir,
            "normalization_threshold": self.normalization_threshold,
            "amplification_threshold": self.amplification_threshold,
            "output_single_stem": self.output_single_stem,
            "invert_using_spec": self.invert_using_spec,
            "sample_rate": self.sample_rate,
            "performance_params": self.performance_params,
        }

        separator_classes = {
            "Demucs": "demucs_separator.DemucsSeparator",
            "MDXC": "mdxc_separator.MDXCSeparator",
            "MDX": "mdx_separator.MDXSeparator",
            "VR": "vr_separator.VRSeparator",
        }

        if model_type not in self.arch_specific_params or model_type not in separator_classes:
            raise ValueError(f"Model type not yet supported in MLX backend: {model_type}")

        self.logger.debug(f"Importing module for model type {model_type}: {separator_classes[model_type]}")

        module_name, class_name = separator_classes[model_type].split(".")
        module = importlib.import_module(f"mlx_audio_separator.separator.architectures.{module_name}")
        separator_class = getattr(module, class_name)

        self.logger.debug(f"Instantiating separator class for model type {model_type}: {separator_class}")
        self.model_instance = separator_class(common_config=common_params, arch_config=self.arch_specific_params[model_type])
        self.model_type = model_type

        self.logger.debug("Loading model completed.")
        self.logger.info(f'Load model duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - load_model_start_time)))}')

    def separate(self, audio_file_path, custom_output_names=None):
        """Separate audio file(s) into stems."""
        if self.model_instance is None:
            raise ValueError("No model loaded. Please call load_model() before attempting to separate.")

        if isinstance(audio_file_path, str):
            audio_file_path = [audio_file_path]

        output_files = []

        for path in audio_file_path:
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith((".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aiff", ".ac3")):
                            full_path = os.path.join(root, file)
                            self.logger.info(f"Processing file: {full_path}")
                            try:
                                files_output = self._separate_file(full_path, custom_output_names)
                                output_files.extend(files_output)
                            except Exception as e:
                                self.logger.error(f"Failed to process file {full_path}: {e}")
                                self._clear_cache_now()
            else:
                self.logger.info(f"Processing file: {path}")
                try:
                    files_output = self._separate_file(path, custom_output_names)
                    output_files.extend(files_output)
                except Exception as e:
                    self.logger.error(f"Failed to process file {path}: {e}")
                    self._clear_cache_now()

        self._finalize_deferred_cache()
        return output_files

    def _separate_file(self, audio_file_path, custom_output_names=None):
        if self.chunk_duration is not None:
            import mlx_audio_io as mac
            duration = mac.info(str(audio_file_path)).duration

            from mlx_audio_separator.separator.audio_chunking import AudioChunker
            chunker = AudioChunker(self.chunk_duration, self.logger)

            if chunker.should_chunk(duration):
                self.logger.info(f"File duration {duration:.1f}s exceeds chunk size {self.chunk_duration}s, using chunked processing")
                return self._process_with_chunking(audio_file_path, custom_output_names)

        self.logger.info(f"Starting separation process for audio_file_path: {audio_file_path}")
        separate_start_time = time.perf_counter()

        if not self._skip_auto_tune:
            self._auto_tune_batch_if_needed(audio_file_path)
        output_files = self.model_instance.separate(audio_file_path, custom_output_names)
        self.model_instance.flush_pending_writes()

        metrics = self.model_instance.get_perf_metrics()
        cleanup_start = time.perf_counter()
        self.model_instance.clear_file_specific_paths()
        self._apply_cache_policy_after_file()
        cleanup_time = time.perf_counter() - cleanup_start
        metrics["cleanup_s"] = float(metrics.get("cleanup_s", 0.0)) + cleanup_time
        self.print_uvr_vip_message()

        self.logger.debug("Separation process completed.")
        total_time = time.perf_counter() - separate_start_time
        metrics["total_s"] = total_time
        self.last_perf_metrics = dict(metrics)
        self._emit_perf_trace(audio_file_path, metrics)
        self.logger.info(f'Separation duration: {time.strftime("%H:%M:%S", time.gmtime(int(total_time)))}')

        return output_files

    def _process_with_chunking(self, audio_file_path, custom_output_names=None):
        import shutil
        import tempfile

        from mlx_audio_separator.separator.audio_chunking import AudioChunker

        temp_dir = tempfile.mkdtemp(prefix="audio-separator-chunks-")
        self.logger.debug(f"Created temporary directory for chunks: {temp_dir}")
        original_skip_auto_tune = self._skip_auto_tune

        try:
            chunker = AudioChunker(self.chunk_duration, self.logger)
            chunk_paths = chunker.split_audio(audio_file_path, temp_dir)

            processed_chunks_by_stem = {}
            self._skip_auto_tune = True

            for i, chunk_path in enumerate(chunk_paths):
                self.logger.info(f"Processing chunk {i+1}/{len(chunk_paths)}: {chunk_path}")

                original_chunk_duration = self.chunk_duration
                original_output_dir = self.output_dir
                self.chunk_duration = None
                self.output_dir = temp_dir

                if self.model_instance:
                    original_model_output_dir = self.model_instance.output_dir
                    self.model_instance.output_dir = temp_dir

                try:
                    output_files = self._separate_file(chunk_path, custom_output_names)

                    for stem_path in output_files:
                        filename = os.path.basename(stem_path)
                        match = re.search(r'_\(([^)]+)\)', filename)
                        if match:
                            stem_name = match.group(1)
                        else:
                            stem_index = len([k for k in processed_chunks_by_stem.keys() if k.startswith('stem_')])
                            stem_name = f"stem_{stem_index}"

                        if stem_name not in processed_chunks_by_stem:
                            processed_chunks_by_stem[stem_name] = []

                        abs_path = stem_path if os.path.isabs(stem_path) else os.path.join(temp_dir, stem_path)
                        processed_chunks_by_stem[stem_name].append(abs_path)
                finally:
                    self.chunk_duration = original_chunk_duration
                    self.output_dir = original_output_dir
                    if self.model_instance:
                        self.model_instance.output_dir = original_model_output_dir

            base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
            output_files = []

            for stem_name in sorted(processed_chunks_by_stem.keys()):
                chunk_paths_for_stem = processed_chunks_by_stem[stem_name]

                if not chunk_paths_for_stem:
                    continue

                if custom_output_names and stem_name in custom_output_names:
                    output_filename = custom_output_names[stem_name]
                else:
                    output_filename = f"{base_name}_({stem_name})"

                output_path = os.path.join(self.output_dir, f"{output_filename}.{self.output_format.lower()}")

                self.logger.info(f"Merging {len(chunk_paths_for_stem)} chunks for stem: {stem_name}")
                chunker.merge_chunks(chunk_paths_for_stem, output_path)
                output_files.append(output_path)

            return output_files
        finally:
            self._skip_auto_tune = original_skip_auto_tune
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def download_model_and_data(self, model_filename):
        self.logger.info(f"Downloading model {model_filename}...")
        model_filename, model_type, model_friendly_name, model_path, yaml_config_filename = self.download_model_files(model_filename)

        if model_path.lower().endswith(".yaml"):
            yaml_config_filename = model_path

        if yaml_config_filename is not None:
            model_data = self.load_model_data_from_yaml(yaml_config_filename)
        else:
            model_data = self.load_model_data_using_hash(model_path)

        self.logger.info(
            f"Model downloaded, type: {model_type}, friendly name: {model_friendly_name}, "
            f"model_path: {model_path}, model_data: {len(model_data)} items"
        )
