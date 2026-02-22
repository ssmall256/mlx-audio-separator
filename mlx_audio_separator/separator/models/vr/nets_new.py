"""VR 5.1 network architectures ported to MLX.

CascadedNet: Multi-stage cascaded network with LSTM for temporal modeling.
All operations use NHWC layout.
"""

import mlx.core as mx
import mlx.nn as nn

from . import layers


class BaseNet(nn.Module):
    """Base network with encoder-decoder, ASPP, and LSTM temporal modeling."""

    def __init__(self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))):
        super().__init__()
        self.enc1 = layers.Conv2DBNActiv(nin, nout, 3, 1, 1)
        self.enc2 = layers.EncoderNew(nout, nout * 2, 3, 2, 1)
        self.enc3 = layers.EncoderNew(nout * 2, nout * 4, 3, 2, 1)
        self.enc4 = layers.EncoderNew(nout * 4, nout * 6, 3, 2, 1)
        self.enc5 = layers.EncoderNew(nout * 6, nout * 8, 3, 2, 1)

        self.aspp = layers.ASPPModule(nout * 8, nout * 8, dilations, dropout=True)

        self.dec4 = layers.Decoder(nout * (6 + 8), nout * 6, 3, 1, 1)
        self.dec3 = layers.Decoder(nout * (4 + 6), nout * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(nout * (2 + 4), nout * 2, 3, 1, 1)
        self.lstm_dec2 = layers.LSTMModule(nout * 2, nin_lstm, nout_lstm)
        self.dec1 = layers.Decoder(nout * (1 + 2) + 1, nout * 1, 3, 1, 1)

    def __call__(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        h = self.aspp(e5)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        # LSTM output is (N, H, W, 1) — concat on channel dim
        h = mx.concatenate([h, self.lstm_dec2(h)], axis=-1)
        h = self.dec1(h, e1)

        return h


class CascadedNet(nn.Module):
    """Three-stage cascaded network for VR 5.1 models.

    Stage 1: Low/high frequency band processing
    Stage 2: Refinement with concatenated Stage 1 outputs
    Stage 3: Full-band final processing
    """

    def __init__(self, n_fft, nn_arch_size=51000, nout=32, nout_lstm=128):
        super().__init__()
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64
        nout = 64 if nn_arch_size == 218409 else nout

        # Stage 1
        self.stg1_low_band_net_base = BaseNet(2, nout // 2, self.nin_lstm // 2, nout_lstm)
        self.stg1_low_band_net_proj = layers.Conv2DBNActiv(nout // 2, nout // 4, 1, 1, 0)
        self.stg1_high_band_net = BaseNet(2, nout // 4, self.nin_lstm // 2, nout_lstm // 2)

        # Stage 2
        self.stg2_low_band_net_base = BaseNet(nout // 4 + 2, nout, self.nin_lstm // 2, nout_lstm)
        self.stg2_low_band_net_proj = layers.Conv2DBNActiv(nout, nout // 2, 1, 1, 0)
        self.stg2_high_band_net = BaseNet(nout // 4 + 2, nout // 2, self.nin_lstm // 2, nout_lstm // 2)

        # Stage 3
        self.stg3_full_band_net = BaseNet(3 * nout // 4 + 2, nout, self.nin_lstm, nout_lstm)

        # Output
        self.out = nn.Conv2d(nout, 2, kernel_size=1, bias=False)
        self.aux_out = nn.Conv2d(3 * nout // 4, 2, kernel_size=1, bias=False)

    def __call__(self, x):
        """Forward pass. Input: (N, H, W, C) NHWC."""
        x = x[:, :self.max_bin, :, :]

        bandw = x.shape[1] // 2
        l1_in = x[:, :bandw, :, :]
        h1_in = x[:, bandw:, :, :]

        # Stage 1
        l1 = self.stg1_low_band_net_proj(self.stg1_low_band_net_base(l1_in))
        h1 = self.stg1_high_band_net(h1_in)
        aux1 = mx.concatenate([l1, h1], axis=1)

        # Stage 2
        l2_in = mx.concatenate([l1_in, l1], axis=-1)
        h2_in = mx.concatenate([h1_in, h1], axis=-1)
        l2 = self.stg2_low_band_net_proj(self.stg2_low_band_net_base(l2_in))
        h2 = self.stg2_high_band_net(h2_in)
        aux2 = mx.concatenate([l2, h2], axis=1)

        # Stage 3
        f3_in = mx.concatenate([x, aux1, aux2], axis=-1)
        f3 = self.stg3_full_band_net(f3_in)

        # Output mask
        mask = mx.sigmoid(self.out(f3))

        # Pad frequency to output_bin
        if mask.shape[1] < self.output_bin:
            pad_size = self.output_bin - mask.shape[1]
            last_bin = mask[:, -1:, :, :]
            pad = mx.repeat(last_bin, repeats=pad_size, axis=1)
            mask = mx.concatenate([mask, pad], axis=1)

        return mask

    def predict_mask(self, x):
        mask = self.__call__(x)
        if self.offset > 0:
            mask = mask[:, :, self.offset:-self.offset, :]
        return mask
