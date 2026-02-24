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
        from mlx_audio_separator.separator.models.vr.loader import _convert_key
        assert _convert_key("enc1.conv.0.weight") == "enc1.conv.weight"

    def test_conv_1_becomes_bn(self):
        from mlx_audio_separator.separator.models.vr.loader import _convert_key
        assert _convert_key("enc1.conv.1.running_mean") == "enc1.bn.running_mean"

    def test_decoder_conv_mapping(self):
        from mlx_audio_separator.separator.models.vr.loader import _convert_key
        assert _convert_key("stg1_low_band_net.dec1.conv.conv.0.weight") == "stg1_low_band_net.dec1.conv1.conv.weight"
        assert _convert_key("stg1_low_band_net.dec1.conv.conv.1.running_var") == "stg1_low_band_net.dec1.conv1.bn.running_var"

    def test_aspp_bottleneck_mapping(self):
        from mlx_audio_separator.separator.models.vr.loader import _convert_key
        assert (
            _convert_key("stg1_low_band_net.aspp.bottleneck.0.conv.0.weight")
            == "stg1_low_band_net.aspp.bottleneck_conv.conv.weight"
        )
        assert (
            _convert_key("stg1_low_band_net.aspp.bottleneck.0.conv.1.bias")
            == "stg1_low_band_net.aspp.bottleneck_conv.bn.bias"
        )

    def test_aspp_separable_mapping(self):
        from mlx_audio_separator.separator.models.vr.loader import _convert_key
        assert _convert_key("stg1_low_band_net.aspp.conv3.conv.0.weight") == "stg1_low_band_net.aspp.conv3.dw_conv.weight"
        assert _convert_key("stg1_low_band_net.aspp.conv3.conv.1.weight") == "stg1_low_band_net.aspp.conv3.pw_conv.weight"
        assert _convert_key("stg1_low_band_net.aspp.conv3.conv.2.running_mean") == "stg1_low_band_net.aspp.conv3.bn.running_mean"

    def test_aspp_separable_mapping_conv6_conv7(self):
        from mlx_audio_separator.separator.models.vr.loader import _convert_key
        assert _convert_key("stg1_low_band_net.aspp.conv6.conv.0.weight") == "stg1_low_band_net.aspp.conv6.dw_conv.weight"
        assert _convert_key("stg1_low_band_net.aspp.conv7.conv.1.weight") == "stg1_low_band_net.aspp.conv7.pw_conv.weight"

    def test_aspp_plain_mapping_profile(self):
        from mlx_audio_separator.separator.models.vr.loader import NEW_CASCADED_ASPP_PLAIN, _convert_key
        assert (
            _convert_key(
                "stg1_low_band_net_base.aspp.conv3.conv.0.weight",
                mapping_profile=NEW_CASCADED_ASPP_PLAIN,
            )
            == "stg1_low_band_net_base.aspp.conv3.conv.weight"
        )
        assert (
            _convert_key(
                "stg1_low_band_net_base.aspp.conv3.conv.1.running_var",
                mapping_profile=NEW_CASCADED_ASPP_PLAIN,
            )
            == "stg1_low_band_net_base.aspp.conv3.bn.running_var"
        )

    def test_detect_mapping_profile_new_cascaded(self):
        from mlx_audio_separator.separator.models.vr.loader import NEW_CASCADED_ASPP_PLAIN, _detect_mapping_profile
        state_dict = {"stg1_low_band_net.0.enc1.conv.weight": np.zeros((1, 1, 1, 1))}
        assert _detect_mapping_profile(state_dict) == NEW_CASCADED_ASPP_PLAIN

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

    def test_numeric_mapping_disabled_for_voc_ft(self):
        from mlx_audio_separator.separator.models.mdx.loader import _should_include_numeric_mapping
        assert not _should_include_numeric_mapping("/tmp/UVR-MDX-NET-Voc_FT.onnx")
        assert _should_include_numeric_mapping("/tmp/UVR-MDX-NET-Inst_HQ_3.onnx")

    def test_numeric_mapping_can_be_disabled(self):
        from mlx_audio_separator.separator.models.mdx.loader import convert_onnx_to_mlx_weights
        weights = {"0": np.zeros((4, 4, 3, 3), dtype=np.float32)}
        result = convert_onnx_to_mlx_weights(
            weights,
            g=32,
            n=4,
            num_tdf_layers=3,
            dim_f=3072,
            bn=16,
            include_numeric=False,
        )
        assert result == {}

    def test_numeric_mapping_error_is_explicit(self):
        from mlx_audio_separator.separator.models.mdx.loader import convert_onnx_to_mlx_weights
        weights = {"0": np.zeros((4,), dtype=np.float32)}
        with pytest.raises(ValueError, match="expected 4D conv tensor"):
            convert_onnx_to_mlx_weights(
                weights,
                g=32,
                n=4,
                num_tdf_layers=3,
                dim_f=3072,
                bn=16,
                include_numeric=True,
            )

    def test_extract_conv_profile_sequence(self):
        from mlx_audio_separator.separator.models.mdx.loader import _extract_conv_profile_sequence

        named = {
            "Conv_684.weight": np.zeros((48, 4, 1, 1), dtype=np.float32),
            "Conv_684.bias": np.zeros((48,), dtype=np.float32),
            "Conv_687.weight": np.zeros((48, 48, 3, 3), dtype=np.float32),
        }
        seq = _extract_conv_profile_sequence(named)
        assert len(seq) == 3
        assert seq[0][1].shape == (48, 4, 1, 1)
        assert seq[1][1].shape == (48,)
        assert seq[2][1].shape == (48, 48, 3, 3)

    def test_infer_params_prefers_1x1_conv_for_g_dim_c(self):
        from mlx_audio_separator.separator.models.mdx.loader import _infer_params_from_onnx

        onnx_weights = {
            "Conv_684.weight": np.zeros((48, 4, 1, 1), dtype=np.float32),
            "Conv_696.weight": np.zeros((96, 48, 2, 2), dtype=np.float32),
            "Conv_708.weight": np.zeros((144, 96, 2, 2), dtype=np.float32),
            "Conv_720.weight": np.zeros((192, 144, 2, 2), dtype=np.float32),
            "Conv_732.weight": np.zeros((240, 192, 2, 2), dtype=np.float32),
            "Conv_744.weight": np.zeros((288, 240, 2, 2), dtype=np.float32),
            "668": np.zeros((3072, 384), dtype=np.float32),
        }
        params = _infer_params_from_onnx(onnx_weights, dim_f=3072)
        assert params["g"] == 48
        assert params["dim_c"] == 4
        assert params["num_blocks"] == 11
        assert params["bn"] == 8

    def test_infer_params_ds_dense_profile(self):
        from mlx_audio_separator.separator.models.mdx.loader import _infer_params_from_onnx

        onnx_weights = {
            "100": np.zeros((32, 4, 1, 1), dtype=np.float32),
            "ds_dense.0.tdf.0.weight": np.zeros((512, 2048), dtype=np.float32),
            "ds_dense.4.tdf.0.weight": np.zeros((16, 64), dtype=np.float32),
        }
        params = _infer_params_from_onnx(onnx_weights, dim_f=2048)
        assert params["num_blocks"] == 11
        assert params["bn"] == 4


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

    def test_explicit_mdx23c_8kfft_filename(self):
        from mlx_audio_separator.separator.models.roformer.loader import detect_model_type
        assert detect_model_type("MDX23C-8KFFT-InstVoc_HQ.ckpt", {}) == "bs_roformer"

    def test_explicit_mdx23c_de_reverb_filename(self):
        from mlx_audio_separator.separator.models.roformer.loader import detect_model_type
        assert detect_model_type("MDX23C-De-Reverb-aufr33-jarredou.ckpt", {}) == "bs_roformer"

    def test_explicit_mdx23c_drumsep_filename(self):
        from mlx_audio_separator.separator.models.roformer.loader import detect_model_type
        assert detect_model_type("MDX23C-DrumSep-aufr33-jarredou.ckpt", {}) == "bs_roformer"

    def test_unknown_raises(self):
        from mlx_audio_separator.separator.models.roformer.loader import detect_model_type
        with pytest.raises(ValueError):
            detect_model_type("unknown_model.ckpt", {})


class TestMDXCSchemaRouting:
    def _mdx23c_config(self):
        return {
            "audio": {
                "dim_f": 1024,
                "n_fft": 2048,
                "hop_length": 512,
                "num_channels": 2,
            },
            "model": {
                "num_subbands": 4,
                "num_scales": 5,
                "num_blocks_per_scale": 2,
                "num_channels": 128,
                "growth": 128,
                "bottleneck_factor": 4,
                "scale": [2, 2],
                "act": "gelu",
                "norm": "InstanceNorm",
            },
            "training": {
                "target_instrument": None,
                "instruments": ["Vocals", "Instrumental"],
            },
        }

    def test_classify_mdx23c_schema(self):
        from mlx_audio_separator.separator.models.mdxc.loader import SCHEMA_MDX23C_TFC_TDF_V3, classify_mdxc_schema

        schema = classify_mdxc_schema("MDX23C-8KFFT-InstVoc_HQ.ckpt", self._mdx23c_config())
        assert schema == SCHEMA_MDX23C_TFC_TDF_V3

    def test_classify_roformer_schema(self):
        from mlx_audio_separator.separator.models.mdxc.loader import SCHEMA_ROFORMER, classify_mdxc_schema

        schema = classify_mdxc_schema("model_bs_roformer.ckpt", {"model": {"freqs_per_bands": [1, 2, 3]}})
        assert schema == SCHEMA_ROFORMER

    def test_unsupported_schema_error_is_explicit(self):
        from mlx_audio_separator.separator.models.mdxc.loader import classify_mdxc_schema

        with pytest.raises(ValueError, match="Unsupported MDXC schema"):
            classify_mdxc_schema("broken.ckpt", {"model": {"foo": 1}, "audio": {"bar": 2}})

    def test_translate_torch_key_mappings(self):
        from mlx_audio_separator.separator.models.mdxc.loader import _translate_torch_key

        assert (
            _translate_torch_key("encoder_blocks.3.tfc_tdf.blocks.1.tfc1.0.weight")
            == "encoder_blocks_3.tfc_tdf.blocks_1.tfc1_norm.weight"
        )
        assert (
            _translate_torch_key("decoder_blocks.2.upscale.conv.2.weight")
            == "decoder_blocks_2.upscale.conv.weight"
        )
        assert _translate_torch_key("final_conv.2.weight") == "final_conv2.weight"

    def test_mdx23c_weight_conversion_transposes_convs(self):
        from mlx_audio_separator.separator.models.mdxc.loader import convert_mdx23c_torch_to_mlx_weights

        state_dict = {
            "encoder_blocks.0.downscale.conv.2.weight": np.zeros((5, 7, 2, 3), dtype=np.float32),
            "decoder_blocks.0.upscale.conv.2.weight": np.zeros((7, 5, 2, 3), dtype=np.float32),
            "encoder_blocks.0.tfc_tdf.blocks.0.tdf.2.weight": np.zeros((16, 32), dtype=np.float32),
        }
        converted = convert_mdx23c_torch_to_mlx_weights(state_dict)
        assert converted["encoder_blocks_0.downscale.conv.weight"].shape == (5, 2, 3, 7)
        assert converted["decoder_blocks_0.upscale.conv.weight"].shape == (5, 2, 3, 7)
        assert converted["encoder_blocks_0.tfc_tdf.blocks_0.tdf_linear1.weight"].shape == (16, 32)

    def test_load_mdxc_model_routes_to_roformer(self, monkeypatch):
        from mlx_audio_separator.separator.models.mdxc import loader as mdxc_loader

        def fake_roformer_loader(model_path, config):
            return "roformer-model", "bs_roformer"

        monkeypatch.setattr(mdxc_loader, "load_roformer_model", fake_roformer_loader)
        model, model_type = mdxc_loader.load_mdxc_model(
            model_path="model_bs_roformer.ckpt",
            config={"model": {"freqs_per_bands": [1, 2, 3]}},
        )
        assert model == "roformer-model"
        assert model_type == "bs_roformer"

    def test_load_mdxc_model_routes_to_mdx23c(self, monkeypatch):
        from mlx_audio_separator.separator.models.mdxc import loader as mdxc_loader

        def fake_mdx23c_loader(model_path, config):
            return "mdx23c-model"

        monkeypatch.setattr(mdxc_loader, "load_mdx23c_model", fake_mdx23c_loader)
        model, model_type = mdxc_loader.load_mdxc_model(
            model_path="MDX23C-8KFFT-InstVoc_HQ.ckpt",
            config=self._mdx23c_config(),
        )
        assert model == "mdx23c-model"
        assert model_type == mdxc_loader.SCHEMA_MDX23C_TFC_TDF_V3
