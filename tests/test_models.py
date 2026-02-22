"""Forward-pass shape tests for all model architectures."""

import mlx.core as mx
from mlx.utils import tree_flatten


class TestBSRoformer:
    def test_forward_shape(self, bs_roformer_config):
        from mlx_audio_separator.separator.models.roformer.loader import create_bs_roformer_mlx

        model = create_bs_roformer_mlx(bs_roformer_config)
        x = mx.zeros((1, 2, 44100))
        out = model(x)
        mx.eval(out)
        assert out.shape[0] == 1
        assert out.shape[1] == 2
        # Output length may differ slightly due to STFT framing; should be close to input
        assert abs(out.shape[2] - 44100) < 2048


class TestMelBandRoformer:
    def test_forward_shape(self, mel_band_roformer_config):
        from mlx_audio_separator.separator.models.roformer.loader import create_mel_band_roformer_mlx

        model = create_mel_band_roformer_mlx(mel_band_roformer_config)
        x = mx.zeros((1, 2, 44100))
        out = model(x)
        mx.eval(out)
        assert out.shape[0] == 1
        assert out.shape[1] == 2
        assert abs(out.shape[2] - 44100) < 2048


class TestConvTDFNet:
    def test_forward_shape(self, convtdfnet_config):
        from mlx_audio_separator.separator.models.mdx.convtdfnet import ConvTDFNet

        model = ConvTDFNet(**convtdfnet_config)
        # Input: (N, C, F, T) in NCHW convention
        x = mx.zeros((1, 4, 2048, 256))
        out = model(x)
        mx.eval(out)
        assert out.shape == (1, 4, 2048, 256)


class TestCascadedASPPNet:
    def test_forward_shape_31191(self):
        from mlx_audio_separator.separator.models.vr.nets import determine_model_capacity

        model = determine_model_capacity(1024, 31191)
        # Input: (N, H, W, C) NHWC
        x = mx.zeros((1, 512, 64, 2))
        out = model(x)
        mx.eval(out)
        assert out.shape[0] == 1
        assert out.shape[1] == 513  # output_bin = n_fft//2 + 1 = 512 + 1
        assert out.shape[2] == 64
        assert out.shape[3] == 2


class TestCascadedNet:
    def test_forward_shape_56817(self):
        from mlx_audio_separator.separator.models.vr.nets_new import CascadedNet

        model = CascadedNet(768, nn_arch_size=56817, nout=32, nout_lstm=128)
        # Input: (N, H, W, C) NHWC
        x = mx.zeros((1, 384, 64, 2))
        out = model(x)
        mx.eval(out)
        assert out.shape[0] == 1
        assert out.shape[1] == 385  # output_bin = 768//2 + 1 = 385
        assert out.shape[2] == 64
        assert out.shape[3] == 2


class TestVRLayers:
    def test_separable_conv_depthwise_shapes(self):
        from mlx_audio_separator.separator.models.vr.layers import SeperableConv2DBNActiv

        layer = SeperableConv2DBNActiv(8, 8, ksize=3, stride=1, pad=1)
        params = dict(tree_flatten(layer.parameters()))

        # Depthwise grouped conv: in/groups == 1 for each output channel.
        assert params["dw_conv.weight"].shape == (8, 3, 3, 1)
        assert params["pw_conv.weight"].shape == (8, 1, 1, 8)
