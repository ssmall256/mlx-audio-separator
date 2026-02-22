"""VR network architectures (standard models) ported to MLX.

CascadedASPPNet: Three-stage cascaded network for multi-band frequency processing.
All operations use NHWC layout.
"""

import mlx.core as mx
import mlx.nn as nn

from . import layers


class ASPPModuleOld(nn.Module):
    """ASPP module for standard VR models (nets.py style).

    Supports architecture-specific layer counts (5, 6, or 7 paths).
    """

    def __init__(self, nn_architecture, nin, nout, dilations=(4, 8, 16), activ="relu"):
        super().__init__()
        self.nn_architecture = nn_architecture
        self.six_layer = [129605]
        self.seven_layer = [537238, 537227, 33966]

        # Global context
        self.conv1_proj = layers.Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ)
        self.conv2 = layers.Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ)
        self.conv3 = layers.SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[0], dilations[0], activ=activ)
        self.conv4 = layers.SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[1], dilations[1], activ=activ)
        self.conv5 = layers.SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[2], dilations[2], activ=activ)

        if nn_architecture in self.six_layer:
            self.conv6 = layers.SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[2], dilations[2], activ=activ)
            nin_x = 6
        elif nn_architecture in self.seven_layer:
            self.conv6 = layers.SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[2], dilations[2], activ=activ)
            self.conv7 = layers.SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[2], dilations[2], activ=activ)
            nin_x = 7
        else:
            nin_x = 5

        self.bottleneck_conv = layers.Conv2DBNActiv(nin * nin_x, nout, 1, 1, 0, activ=activ)

    def __call__(self, x):
        N, H, W, C = x.shape
        # Global avg pool over H → (N, 1, W, C)
        feat1 = mx.mean(x, axis=1, keepdims=True)
        feat1 = self.conv1_proj(feat1)
        feat1 = mx.repeat(feat1, repeats=H, axis=1)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if self.nn_architecture in self.six_layer:
            feat6 = self.conv6(x)
            out = mx.concatenate([feat1, feat2, feat3, feat4, feat5, feat6], axis=-1)
        elif self.nn_architecture in self.seven_layer:
            feat6 = self.conv6(x)
            feat7 = self.conv7(x)
            out = mx.concatenate([feat1, feat2, feat3, feat4, feat5, feat6, feat7], axis=-1)
        else:
            out = mx.concatenate([feat1, feat2, feat3, feat4, feat5], axis=-1)

        return self.bottleneck_conv(out)


class BaseASPPNet(nn.Module):
    """Base ASPP network with encoder-decoder and skip connections."""

    def __init__(self, nn_architecture, nin, ch, dilations=(4, 8, 16)):
        super().__init__()
        self.nn_architecture = nn_architecture

        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)

        if nn_architecture == 129605:
            self.enc5 = layers.Encoder(ch * 8, ch * 16, 3, 2, 1)
            self.aspp = ASPPModuleOld(nn_architecture, ch * 16, ch * 32, dilations)
            self.dec5 = layers.Decoder(ch * (16 + 32), ch * 16, 3, 1, 1)
        else:
            self.aspp = ASPPModuleOld(nn_architecture, ch * 8, ch * 16, dilations)

        self.dec4 = layers.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def __call__(self, x):
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)

        if self.nn_architecture == 129605:
            h, e5 = self.enc5(h)
            h = self.aspp(h)
            h = self.dec5(h, e5)
        else:
            h = self.aspp(h)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)
        return h


def determine_model_capacity(n_fft_bins, nn_architecture):
    """Select model configuration based on architecture size."""
    sp_model_arch = [31191, 33966, 129605]
    hp_model_arch = [123821, 123812]
    hp2_model_arch = [537238, 537227]

    if nn_architecture in sp_model_arch:
        model_capacity_data = [(2, 16), (2, 16), (18, 8, 1, 1, 0), (8, 16), (34, 16, 1, 1, 0), (16, 32), (32, 2, 1), (16, 2, 1), (16, 2, 1)]
    elif nn_architecture in hp_model_arch:
        model_capacity_data = [
            (2, 32), (2, 32), (34, 16, 1, 1, 0), (16, 32),
            (66, 32, 1, 1, 0), (32, 64), (64, 2, 1), (32, 2, 1), (32, 2, 1),
        ]
    elif nn_architecture in hp2_model_arch:
        model_capacity_data = [
            (2, 64), (2, 64), (66, 32, 1, 1, 0), (32, 64),
            (130, 64, 1, 1, 0), (64, 128), (128, 2, 1), (64, 2, 1), (64, 2, 1),
        ]
    else:
        raise ValueError(f"Unknown VR architecture size: {nn_architecture}")

    return CascadedASPPNet(n_fft_bins, model_capacity_data, nn_architecture)


class CascadedASPPNet(nn.Module):
    """Three-stage cascaded ASPP network for VR source separation.

    Stage 1: Low/high band processing
    Stage 2: Full band refinement
    Stage 3: Final refinement
    Output: Sigmoid mask
    """

    def __init__(self, n_fft, model_capacity_data, nn_architecture):
        super().__init__()
        self.stg1_low_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[0])
        self.stg1_high_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[1])

        self.stg2_bridge = layers.Conv2DBNActiv(*model_capacity_data[2])
        self.stg2_full_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[3])

        self.stg3_bridge = layers.Conv2DBNActiv(*model_capacity_data[4])
        self.stg3_full_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[5])

        # Output convolutions (1x1, no bias)
        out_args = model_capacity_data[6]
        self.out = nn.Conv2d(out_args[0], out_args[1], kernel_size=out_args[2], bias=False)
        aux1_args = model_capacity_data[7]
        self.aux1_out = nn.Conv2d(aux1_args[0], aux1_args[1], kernel_size=aux1_args[2], bias=False)
        aux2_args = model_capacity_data[8]
        self.aux2_out = nn.Conv2d(aux2_args[0], aux2_args[1], kernel_size=aux2_args[2], bias=False)

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.offset = 128

    def __call__(self, x):
        """Forward pass. Input: (N, H, W, C) NHWC where H=freq, W=time, C=2 (stereo mag)."""
        # Truncate to max_bin frequency bins
        x = x[:, :self.max_bin, :, :]

        # Split into low/high bands along freq axis (axis 1 in NHWC)
        bandwidth = x.shape[1] // 2
        aux1 = mx.concatenate([
            self.stg1_low_band_net(x[:, :bandwidth, :, :]),
            self.stg1_high_band_net(x[:, bandwidth:, :, :]),
        ], axis=1)

        # Stage 2
        h = mx.concatenate([x, aux1], axis=-1)  # concat channels
        aux2 = self.stg2_full_band_net(self.stg2_bridge(h))

        # Stage 3
        h = mx.concatenate([x, aux1, aux2], axis=-1)
        h = self.stg3_full_band_net(self.stg3_bridge(h))

        # Output mask
        mask = mx.sigmoid(self.out(h))

        # Pad frequency to output_bin
        if mask.shape[1] < self.output_bin:
            pad_size = self.output_bin - mask.shape[1]
            # Replicate last frequency bin
            last_bin = mask[:, -1:, :, :]
            pad = mx.repeat(last_bin, repeats=pad_size, axis=1)
            mask = mx.concatenate([mask, pad], axis=1)

        return mask

    def predict_mask(self, x):
        mask = self.__call__(x)
        if self.offset > 0:
            mask = mask[:, :, self.offset:-self.offset, :]
        return mask
