"""Tests for the Separator class."""

import logging

import pytest

from mlx_audio_separator.core import Separator


class TestSeparatorInit:
    def test_info_only(self, tmp_path):
        sep = Separator(info_only=True, model_file_dir=str(tmp_path / "models"))
        assert sep is not None

    def test_default_output_format(self, tmp_path):
        sep = Separator(
            info_only=True,
            model_file_dir=str(tmp_path / "models"),
            output_dir=str(tmp_path / "out"),
        )
        assert sep.output_format == "WAV"

    def test_normalization_out_of_range(self, tmp_path):
        with pytest.raises(ValueError, match="normalization_threshold"):
            Separator(
                info_only=True,
                model_file_dir=str(tmp_path / "models"),
                normalization_threshold=0,
            )

    def test_normalization_too_high(self, tmp_path):
        with pytest.raises(ValueError, match="normalization_threshold"):
            Separator(
                info_only=True,
                model_file_dir=str(tmp_path / "models"),
                normalization_threshold=1.5,
            )

    def test_amplification_out_of_range(self, tmp_path):
        with pytest.raises(ValueError, match="amplification_threshold"):
            Separator(
                info_only=True,
                model_file_dir=str(tmp_path / "models"),
                amplification_threshold=-1,
            )

    def test_sample_rate_zero(self, tmp_path):
        with pytest.raises(ValueError):
            Separator(
                info_only=True,
                model_file_dir=str(tmp_path / "models"),
                sample_rate=0,
            )

    def test_chunk_duration_negative(self, tmp_path):
        with pytest.raises(ValueError, match="chunk_duration"):
            Separator(
                info_only=True,
                model_file_dir=str(tmp_path / "models"),
                chunk_duration=-1,
            )

    def test_arch_specific_params_defaults(self, tmp_path):
        sep = Separator(info_only=True, model_file_dir=str(tmp_path / "models"))
        params = sep.arch_specific_params
        assert "Demucs" in params
        assert "MDXC" in params
        assert "MDX" in params
        assert "VR" in params
        assert params["VR"]["batch_size"] == 1

    def test_custom_vr_params(self, tmp_path):
        custom = {"batch_size": 4, "window_size": 1024, "aggression": 10,
                  "enable_tta": True, "enable_post_process": False,
                  "post_process_threshold": 0.5, "high_end_process": False}
        sep = Separator(
            info_only=True,
            model_file_dir=str(tmp_path / "models"),
            vr_params=custom,
        )
        assert sep.arch_specific_params["VR"]["batch_size"] == 4

    def test_log_level(self, tmp_path):
        sep = Separator(
            info_only=True,
            model_file_dir=str(tmp_path / "models"),
            log_level=logging.DEBUG,
        )
        assert sep.log_level == logging.DEBUG
