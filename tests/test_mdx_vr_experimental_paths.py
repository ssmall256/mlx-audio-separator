"""Tests for experimental VR performance path."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_audio_separator.separator.architectures.vr_separator import VRSeparator


def test_vr_device_residency_matches_cpu_mask_path():
    class _FakeVRModel:
        offset = 2

        def predict_mask(self, x):
            # x: (B, F, window, 2) -> output (B, F, roi, 2)
            roi = x.shape[2] - 2 * self.offset
            return mx.ones((x.shape[0], x.shape[1], roi, x.shape[3]), dtype=mx.float32)

    # Build lightweight separator instances without full model loading.
    sep_cpu = VRSeparator.__new__(VRSeparator)
    sep_dev = VRSeparator.__new__(VRSeparator)
    for sep in (sep_cpu, sep_dev):
        sep.batch_size = 2
        sep.window_size = 8
        sep.model_run = _FakeVRModel()
        sep.primary_stem_name = "Vocals"
        sep.enable_post_process = False
        sep.enable_tta = False
        sep.post_process_threshold = 0.2
    sep_cpu.experimental_vr_device_residency = False
    sep_dev.experimental_vr_device_residency = True

    rng = np.random.default_rng(1234)
    X_spec = (rng.standard_normal((2, 12, 32)) + 1j * rng.standard_normal((2, 12, 32))).astype(np.complex64)
    aggressiveness = {"value": 0.05, "split_bin": 6, "aggr_correction": None}

    y_cpu, v_cpu = sep_cpu._inference_vr(X_spec, aggressiveness)
    y_dev, v_dev = sep_dev._inference_vr(X_spec, aggressiveness)

    assert y_cpu.shape == y_dev.shape
    assert v_cpu.shape == v_dev.shape
    assert np.allclose(y_cpu, y_dev, atol=1e-6, rtol=1e-6)
    assert np.allclose(v_cpu, v_dev, atol=1e-6, rtol=1e-6)
