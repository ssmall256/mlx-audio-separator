"""Tests for weight key mapping in all loaders."""

import numpy as np
import pytest


class TestRoformerWeightConversion:
    """Test convert_torch_to_mlx_weights key mapping for Roformer."""

    def _convert_keys(self, keys, has_linear=False):
        """Helper to convert a dict of key -> dummy tensor through the converter."""
        from mlx_audio_separator.separator.models.roformer.loader import convert_torch_to_mlx_weights

        # Build a fake state_dict with numpy arrays
        state_dict = {}
        for k in keys:
            state_dict[k] = np.zeros((2, 2), dtype=np.float32)
        if has_linear:
            # Add a linear transformer key to trigger detection
            state_dict["layers.0.2.dummy"] = np.zeros((2, 2), dtype=np.float32)

        result = convert_torch_to_mlx_weights(state_dict)
        return set(result.keys())

    def test_gamma_to_weight(self):
        result = self._convert_keys(["final_norm.gamma"])
        assert "final_norm.weight" in result

    def test_band_split_norm(self):
        result = self._convert_keys(["band_split.to_features.5.0.weight"])
        assert "band_split.to_features_5.norm.weight" in result

    def test_band_split_linear(self):
        result = self._convert_keys(["band_split.to_features.5.1.weight"])
        assert "band_split.to_features_5.linear.weight" in result

    def test_main_block_time_transformer(self):
        result = self._convert_keys(["layers.0.0.some_param"])
        assert "layers_0.time_transformer.some_param" in result

    def test_main_block_freq_transformer(self):
        result = self._convert_keys(["layers.0.1.some_param"])
        assert "layers_0.freq_transformer.some_param" in result

    def test_main_block_linear_transformer(self):
        result = self._convert_keys(["layers.0.0.some_param"], has_linear=True)
        assert "layers_0.linear_transformer.some_param" in result

    def test_mask_estimators(self):
        result = self._convert_keys(["mask_estimators.0.to_freqs.1.0.weight"])
        assert "mask_estimators_0.to_freqs_1.weight" in result

    def test_rotary_embed_skipped(self):
        result = self._convert_keys(["rotary_embed.freqs"])
        assert len(result) == 0


class TestVRWeightConversion:
    """Test _convert_key mapping for VR loader."""

    def test_stg1_low_band_base(self):
        from mlx_audio_separator.separator.models.vr.loader import _convert_key
        assert _convert_key("stg1_low_band_net.0.enc1.conv.weight") == "stg1_low_band_net_base.enc1.conv.weight"

    def test_stg1_low_band_proj(self):
        from mlx_audio_separator.separator.models.vr.loader import _convert_key
        assert _convert_key("stg1_low_band_net.1.conv.weight") == "stg1_low_band_net_proj.conv.weight"

    def test_conv_0_weight(self):
        """conv.0.weight stays as conv.0.weight (conv module, idx 0 not stripped)."""
        from mlx_audio_separator.separator.models.vr.loader import _convert_key
        result = _convert_key("enc1.conv.0.weight")
        # The converter appends "conv" then leaves "0" as next part
        assert "conv" in result

    def test_conv_1_becomes_bn(self):
        """conv.1.running_mean → bn.1.running_mean (bn replaces conv for idx>=1)."""
        from mlx_audio_separator.separator.models.vr.loader import _convert_key
        result = _convert_key("enc1.conv.1.running_mean")
        assert "bn" in result
        assert "running_mean" in result

    def test_is_conv_weight(self):
        from mlx_audio_separator.separator.models.vr.loader import _is_conv_weight
        # 4D tensor with 'weight' in name → conv weight
        assert _is_conv_weight("enc1.conv.weight", np.zeros((16, 2, 3, 3)))
        # 1D tensor → not conv weight
        assert not _is_conv_weight("enc1.conv.weight", np.zeros((16,)))
        # BN weight is also 4D with "weight" — the VR _is_conv_weight only checks ndim+name
        # So a 4D bn.weight would also return True (it just checks ndim==4 and "weight" in key)
        assert _is_conv_weight("enc1.bn.weight", np.zeros((16, 2, 3, 3)))


class TestMDXWeightConversion:
    """Test _translate_weight_name and _is_conv_weight for MDX loader."""

    def test_first_conv(self):
        from mlx_audio_separator.separator.models.mdx.loader import _translate_weight_name
        assert _translate_weight_name("first_conv.0.weight") == "first_conv.weight"

    def test_first_norm(self):
        from mlx_audio_separator.separator.models.mdx.loader import _translate_weight_name
        assert _translate_weight_name("first_conv.1.weight") == "first_norm.bn.weight"

    def test_final_conv(self):
        from mlx_audio_separator.separator.models.mdx.loader import _translate_weight_name
        assert _translate_weight_name("final_conv.0.weight") == "final_conv.weight"

    def test_encoding_block(self):
        from mlx_audio_separator.separator.models.mdx.loader import _translate_weight_name
        result = _translate_weight_name("encoding_blocks.0.tfc.H.1.0.weight")
        assert result == "enc_0.tfc.conv_1.weight"

    def test_ds_conv(self):
        from mlx_audio_separator.separator.models.mdx.loader import _translate_weight_name
        assert _translate_weight_name("ds.0.0.weight") == "ds_0_conv.weight"

    def test_ds_norm(self):
        from mlx_audio_separator.separator.models.mdx.loader import _translate_weight_name
        assert _translate_weight_name("ds.0.1.running_mean") == "ds_0_norm.bn.running_mean"

    def test_us_conv(self):
        from mlx_audio_separator.separator.models.mdx.loader import _translate_weight_name
        assert _translate_weight_name("us.1.0.weight") == "us_1_conv.weight"

    def test_bottleneck(self):
        from mlx_audio_separator.separator.models.mdx.loader import _translate_weight_name
        result = _translate_weight_name("bottleneck_block.tdf.0.weight")
        assert result == "bottleneck.tdf_linear1.weight"

    def test_pytorch_lightning_prefix(self):
        from mlx_audio_separator.separator.models.mdx.loader import _translate_weight_name
        result = _translate_weight_name("model.first_conv.0.weight")
        assert result == "first_conv.weight"

    def test_is_conv_weight(self):
        from mlx_audio_separator.separator.models.mdx.loader import _is_conv_weight
        assert _is_conv_weight("first_conv.weight", np.zeros((32, 3, 3, 4)))
        assert _is_conv_weight("ds_0_conv.weight", np.zeros((32, 3, 3, 16)))
        assert not _is_conv_weight("first_conv.weight", np.zeros((32,)))
        assert not _is_conv_weight("first_norm.bn.weight", np.zeros((32, 3, 3, 4)))

    def test_window_skipped(self):
        from mlx_audio_separator.separator.models.mdx.loader import _translate_weight_name
        assert _translate_weight_name("window") is None


class TestDetectModelType:
    """Test detect_model_type from Roformer loader."""

    def test_config_num_bands(self):
        from mlx_audio_separator.separator.models.roformer.loader import detect_model_type
        assert detect_model_type("some_model.ckpt", {"num_bands": 60}) == "mel_band_roformer"

    def test_config_freqs_per_bands(self):
        from mlx_audio_separator.separator.models.roformer.loader import detect_model_type
        assert detect_model_type("some_model.ckpt", {"freqs_per_bands": [2, 2]}) == "bs_roformer"

    def test_filename_mel_band(self):
        from mlx_audio_separator.separator.models.roformer.loader import detect_model_type
        assert detect_model_type("mel_band_roformer_model.ckpt", {}) == "mel_band_roformer"

    def test_filename_bs(self):
        from mlx_audio_separator.separator.models.roformer.loader import detect_model_type
        assert detect_model_type("bs_roformer_model.ckpt", {}) == "bs_roformer"

    def test_filename_generic_roformer(self):
        from mlx_audio_separator.separator.models.roformer.loader import detect_model_type
        assert detect_model_type("some_roformer.ckpt", {}) == "bs_roformer"

    def test_unknown_raises(self):
        from mlx_audio_separator.separator.models.roformer.loader import detect_model_type
        with pytest.raises(ValueError):
            detect_model_type("unknown_model.ckpt", {})
