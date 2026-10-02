"""Unit tests verifying bit-exact parity and behavior of single-stem Roformer extraction."""

import mlx.core as mx
import numpy as np

from mlx_audio_separator.separator.models.roformer.bs_roformer import BSRoformerMLX
from mlx_audio_separator.separator.models.roformer.mel_band_roformer import MelBandRoformerMLX


def test_bs_roformer_single_stem_bit_exact_parity():
    """Verify single-stem extraction matches full multi-stem inference bit-for-bit."""
    mx.random.seed(42)
    # Small 4-stem BS-Roformer
    model = BSRoformerMLX(
        dim=64,
        depth=1,
        stereo=True,
        num_stems=4,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        linear_transformer_depth=0,
        freqs_per_bands=tuple([25] * 41),  # 1025 freq bins
        dim_head=32,
        heads=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        mlp_expansion_factor=2,
        mask_estimator_depth=1,
    )

    # Input: (batch=1, channels=2, samples=2048)
    audio = mx.random.normal((1, 2, 2048))

    # 1. Full multi-stem forward pass
    model.set_target_stem(None)
    full_stems = model(audio)  # (1, 4, 2, 2048)
    mx.eval(full_stems)

    assert full_stems.shape == (1, 4, 2, 2048)

    # 2. Test each individual stem via set_target_stem
    for stem_idx in range(4):
        model.set_target_stem(stem_idx)
        single_stem = model(audio)  # (1, 2, 2048)
        mx.eval(single_stem)

        assert single_stem.shape == (1, 2, 2048)
        expected = full_stems[0, stem_idx]
        diff = float(mx.max(mx.abs(single_stem[0] - expected)))
        assert diff == 0.0, f"Stem {stem_idx} differs from full multi-stem by {diff}"

    # 3. Test explicit stem_idx argument to __call__
    model.set_target_stem(None)
    for stem_idx in range(4):
        single_stem = model(audio, stem_idx=stem_idx)
        mx.eval(single_stem)
        expected = full_stems[0, stem_idx]
        diff = float(mx.max(mx.abs(single_stem[0] - expected)))
        assert diff == 0.0, f"Stem {stem_idx} via arg differs from full by {diff}"

    # 4. Reset target stem
    model.set_target_stem(None)
    full_again = model(audio)
    mx.eval(full_again)
    assert full_again.shape == (1, 4, 2, 2048)


def test_mel_band_roformer_single_stem_bit_exact_parity():
    """Verify MelBand-Roformer single-stem extraction matches full inference bit-for-bit."""
    mx.random.seed(42)
    # Small 4-stem MelBand-Roformer
    model = MelBandRoformerMLX(
        dim=64,
        depth=1,
        stereo=True,
        num_stems=4,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        linear_transformer_depth=0,
        num_bands=60,
        dim_head=32,
        heads=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        mlp_expansion_factor=2,
        mask_estimator_depth=1,
    )

    audio = mx.random.normal((1, 2, 2048))

    # 1. Full multi-stem forward pass
    model.set_target_stem(None)
    full_stems = model(audio)  # (1, 4, 2, 2048)
    mx.eval(full_stems)

    assert full_stems.shape == (1, 4, 2, 2048)

    # 2. Test each individual stem
    for stem_idx in range(4):
        model.set_target_stem(stem_idx)
        single_stem = model(audio)  # (1, 2, 2048)
        mx.eval(single_stem)

        assert single_stem.shape == (1, 2, 2048)
        expected = full_stems[0, stem_idx]
        diff = float(mx.max(mx.abs(single_stem[0] - expected)))
        assert diff == 0.0, f"MelBand stem {stem_idx} differs from full by {diff}"

    model.set_target_stem(None)


class _NoopLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


def test_mdxc_separator_single_stem_demix(monkeypatch):
    """Verify MDXCSeparator._demix_mlx routes single stem extraction and returns only target stem."""
    from mlx_audio_separator.separator.architectures import mdxc_separator as mdxc_mod

    monkeypatch.setattr(mdxc_mod, "tqdm", lambda it, **kwargs: it)

    # 4-stem model
    model = BSRoformerMLX(
        dim=64,
        depth=1,
        stereo=True,
        num_stems=4,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        linear_transformer_depth=0,
        freqs_per_bands=tuple([25] * 41),
        dim_head=32,
        heads=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        mlp_expansion_factor=2,
        mask_estimator_depth=1,
    )

    sep = mdxc_mod.MDXCSeparator.__new__(mdxc_mod.MDXCSeparator)
    sep.logger = _NoopLogger()
    sep.batch_size = 2
    sep.model_run = model
    sep.model = model
    sep.model_data = {
        "training": {
            "instruments": ["drums", "bass", "other", "vocals"]
        },
        "inference": {
            "dim_t": 5,
        },
        "audio": {
            "hop_length": 512,
            "sample_rate": 44100,
        },
        "model": {},
    }
    sep.override_model_segment_size = False
    sep.segment_size = 5
    sep.sample_rate = 44100
    sep.overlap = 0.5
    sep.normalization_threshold = 0.9
    sep.amplification_threshold = 0.2
    sep._np_window_cache = {}
    sep._mlx_window_cache = {}
    sep.experimental_compile_model_forward = False
    sep.experimental_roformer_static_compiled_demix = False
    sep.experimental_vectorized_chunking = True
    sep.experimental_mdxc_precompute_gather_idx = False
    sep.experimental_roformer_fused_overlap_add = False
    sep.primary_stem_name = "vocals"
    sep.secondary_stem_name = "instrumental"

    t = np.linspace(0, 1.0, 44100, dtype=np.float32)
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    mix = np.stack([tone, tone], axis=0)

    # 1. Full 4-stem separation
    sep.output_single_stem = None
    sources_all = sep._demix_mlx(mix)
    assert set(sources_all.keys()) == {"drums", "bass", "other", "vocals"}
    assert model.target_stem_idx is None

    # 2. Single stem extraction for 'vocals'
    sep.output_single_stem = "vocals"
    sources_single = sep._demix_mlx(mix)
    assert set(sources_single.keys()) == {"vocals"}
    assert model.target_stem_idx is None  # cleanly reset after inference
    np.testing.assert_allclose(sources_single["vocals"], sources_all["vocals"], rtol=1e-5, atol=1e-5)
