"""Tests for Demucs Wiener runtime controls."""

from __future__ import annotations

import os

from mlx_audio_separator.demucs_mlx.mlx_hdemucs import _demucs_strict_eval_enabled as hdemucs_strict_eval
from mlx_audio_separator.demucs_mlx.mlx_hdemucs import _demucs_wiener_use_vmap as hdemucs_use_vmap
from mlx_audio_separator.demucs_mlx.mlx_htdemucs import _demucs_strict_eval_enabled as htdemucs_strict_eval
from mlx_audio_separator.demucs_mlx.mlx_htdemucs import _demucs_wiener_use_vmap as htdemucs_use_vmap


def test_demucs_wiener_use_vmap_defaults_true():
    key = "MLX_AUDIO_SEPARATOR_DEMUCS_WIENER_USE_VMAP"
    previous = os.environ.pop(key, None)
    try:
        assert hdemucs_use_vmap() is True
        assert htdemucs_use_vmap() is True
    finally:
        if previous is not None:
            os.environ[key] = previous


def test_demucs_wiener_use_vmap_honors_false_values():
    key = "MLX_AUDIO_SEPARATOR_DEMUCS_WIENER_USE_VMAP"
    previous = os.environ.get(key)
    try:
        for value in ("0", "false", "no", "off"):
            os.environ[key] = value
            assert hdemucs_use_vmap() is False
            assert htdemucs_use_vmap() is False
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def test_demucs_strict_eval_defaults_false():
    key = "MLX_AUDIO_SEPARATOR_DEMUCS_STRICT_EVAL"
    previous = os.environ.pop(key, None)
    try:
        assert hdemucs_strict_eval() is False
        assert htdemucs_strict_eval() is False
    finally:
        if previous is not None:
            os.environ[key] = previous


def test_demucs_strict_eval_honors_true_values():
    key = "MLX_AUDIO_SEPARATOR_DEMUCS_STRICT_EVAL"
    previous = os.environ.get(key)
    try:
        for value in ("1", "true", "yes", "on"):
            os.environ[key] = value
            assert hdemucs_strict_eval() is True
            assert htdemucs_strict_eval() is True
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
