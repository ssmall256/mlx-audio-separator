"""Tests for fused Metal kernel runtime controls."""

from __future__ import annotations

import os

from mlx_audio_separator.demucs_mlx import metal_kernels as mk


def test_stable_threadgroup_size_default_allows_large_groups():
    key = "MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED"
    previous = os.environ.pop(key, None)
    try:
        size = mk._stable_threadgroup_size(33072)
        assert size == 1024
    finally:
        if previous is not None:
            os.environ[key] = previous


def test_stable_threadgroup_size_deterministic_caps_to_256():
    key = "MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED"
    previous = os.environ.get(key)
    os.environ[key] = "1"
    try:
        size = mk._stable_threadgroup_size(33072)
        assert size == 256
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
