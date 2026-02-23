"""Tests for Demucs spectral wrapper runtime controls."""

from __future__ import annotations

import os

from mlx_audio_separator.demucs_mlx.spec_mlx import _resolve_demucs_istft_allow_fused


def test_resolve_demucs_istft_allow_fused_uses_argument_by_default():
    key = "MLX_AUDIO_SEPARATOR_DEMUCS_ISTFT_ALLOW_FUSED"
    previous = os.environ.pop(key, None)
    try:
        assert _resolve_demucs_istft_allow_fused(True) is True
        assert _resolve_demucs_istft_allow_fused(False) is False
    finally:
        if previous is not None:
            os.environ[key] = previous


def test_resolve_demucs_istft_allow_fused_honors_env_override():
    key = "MLX_AUDIO_SEPARATOR_DEMUCS_ISTFT_ALLOW_FUSED"
    previous = os.environ.get(key)
    try:
        os.environ[key] = "0"
        assert _resolve_demucs_istft_allow_fused(True) is False
        os.environ[key] = "1"
        assert _resolve_demucs_istft_allow_fused(False) is True
        os.environ[key] = "invalid"
        assert _resolve_demucs_istft_allow_fused(False) is False
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
