"""Tests for Roformer fused scaled dot-product attention default and parity."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio_separator.separator.models.roformer.bs_roformer import Attention


def test_roformer_attention_fast_sdp_parity():
    """Verify that Fast SDP attention produces near-exact parity with manual attention."""
    b, h, n, d = 4, 8, 256, 64
    dim = h * d

    attn = Attention(dim=dim, heads=h, dim_head=d, rotary_embed=True)
    x = mx.random.normal((b, n, dim), dtype=mx.float32)

    # 1. Manual attention output
    import os
    old_env = os.environ.get("MLX_USE_FAST_SDP")
    try:
        os.environ["MLX_USE_FAST_SDP"] = "0"
        out_manual = attn(x)
        mx.eval(out_manual)

        # 2. Fast SDP attention output (default)
        os.environ["MLX_USE_FAST_SDP"] = "1"
        out_fast = attn(x)
        mx.eval(out_fast)

        diff = mx.max(mx.abs(out_manual - out_fast)).item()
        noise = np.array(out_manual) - np.array(out_fast)
        sp = np.sum(np.array(out_manual)**2)
        np_noise = np.sum(noise**2)
        snr = 10 * np.log10(sp / max(np_noise, 1e-12))

        assert diff < 1e-5, f"Max diff {diff} exceeded threshold"
        assert snr > 70.0, f"SNR {snr:.2f} dB below 70 dB threshold"
    finally:
        if old_env is None:
            os.environ.pop("MLX_USE_FAST_SDP", None)
        else:
            os.environ["MLX_USE_FAST_SDP"] = old_env
