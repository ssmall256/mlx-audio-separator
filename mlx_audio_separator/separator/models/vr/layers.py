"""VR network layers ported to MLX.

All operations use NHWC layout (MLX convention).
PyTorch weight conversion must transpose Conv2d from OIHW → OHWI.
"""

import mlx.core as mx
import mlx.nn as nn


def crop_center(h1, h2):
    """Crop h1 to match h2's time dimension (axis 2 in NHWC)."""
    h1_t = h1.shape[2]
    h2_t = h2.shape[2]
    if h1_t == h2_t:
        return h1
    if h1_t < h2_t:
        raise ValueError("h1 time dim must be >= h2 time dim")
    s = (h1_t - h2_t) // 2
    return h1[:, :, s:s + h2_t, :]


def _upsample_2x(x):
    """2x bilinear upsampling for NHWC tensors.

    x: (N, H, W, C) → (N, 2*H, 2*W, C)
    Uses nearest-neighbor + averaging for simplicity and correctness.
    """
    N, H, W, C = x.shape
    # Repeat along H and W dimensions
    x = mx.repeat(x, repeats=2, axis=1)  # (N, 2H, W, C)
    x = mx.repeat(x, repeats=2, axis=2)  # (N, 2H, 2W, C)
    return x


class Conv2DBNActiv(nn.Module):
    """Conv2d + BatchNorm + activation in NHWC layout."""

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ="relu"):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm(nout)
        self.activ = activ

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activ == "relu":
            x = nn.relu(x)
        elif self.activ == "leaky_relu":
            x = nn.leaky_relu(x)
        return x


class SeperableConv2DBNActiv(nn.Module):
    """Depthwise separable Conv2d + BatchNorm + activation.

    Uses grouped Conv2d for depthwise stage (groups=nin), then 1x1 pointwise conv.
    """

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ="relu"):
        super().__init__()
        # Depthwise: one spatial kernel per input channel.
        self.dw_conv = nn.Conv2d(
            nin, nin, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation, groups=nin, bias=False
        )
        self.pw_conv = nn.Conv2d(nin, nout, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm(nout)
        self.activ = activ

    def __call__(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.bn(x)
        if self.activ == "relu":
            x = nn.relu(x)
        elif self.activ == "leaky_relu":
            x = nn.leaky_relu(x)
        return x


class Encoder(nn.Module):
    """Two-layer encoder with skip connection output."""

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ="leaky_relu"):
        super().__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, stride, pad, activ=activ)

    def __call__(self, x):
        skip = self.conv1(x)
        hidden = self.conv2(skip)
        return hidden, skip


class EncoderNew(nn.Module):
    """Encoder for VR 5.1 models (no skip connection, stride on conv1)."""

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ="leaky_relu"):
        super().__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, stride, pad, activ=activ)
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    """Decoder with 2x upsampling and skip connection concatenation."""

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ="relu", dropout=False):
        super().__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.use_dropout = dropout

    def __call__(self, x, skip=None):
        x = _upsample_2x(x)
        if skip is not None:
            skip = crop_center(skip, x)
            x = mx.concatenate([x, skip], axis=-1)  # NHWC: concat on C
        x = self.conv1(x)
        return x


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling for layers_new.py style (no nn_architecture branching)."""

    def __init__(self, nin, nout, dilations=((4, 2), (8, 4), (12, 6)), activ="relu", dropout=False):
        super().__init__()
        # Global context: adaptive avg pool to (1, W) then 1x1 conv
        self.conv1_proj = Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ)
        self.conv2 = Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ)
        # Dilated convolutions using first element of each dilation tuple
        d0, d1, d2 = dilations[0][0], dilations[1][0], dilations[2][0]
        self.conv3 = Conv2DBNActiv(nin, nout, 3, 1, d0, dilation=d0, activ=activ)
        self.conv4 = Conv2DBNActiv(nin, nout, 3, 1, d1, dilation=d1, activ=activ)
        self.conv5 = Conv2DBNActiv(nin, nout, 3, 1, d2, dilation=d2, activ=activ)
        self.bottleneck = Conv2DBNActiv(nout * 5, nout, 1, 1, 0, activ=activ)

    def __call__(self, x):
        # x: (N, H, W, C) in NHWC
        N, H, W, C = x.shape
        # Global average pool over H dimension → (N, 1, W, C)
        feat1 = mx.mean(x, axis=1, keepdims=True)
        feat1 = self.conv1_proj(feat1)
        # Upsample back to (N, H, W, C) via repeat
        feat1 = mx.repeat(feat1, repeats=H, axis=1)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        out = mx.concatenate([feat1, feat2, feat3, feat4, feat5], axis=-1)
        return self.bottleneck(out)


class LSTMModule(nn.Module):
    """LSTM temporal modeling for VR 5.1 models.

    Squeezes to 1 channel, runs bidirectional LSTM, then projects back.
    Input is NHWC: (N, H, W, C) where H=freq bins, W=time frames.
    """

    def __init__(self, nin_conv, nin_lstm, nout_lstm):
        super().__init__()
        self.squeeze_conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0, activ="relu")
        # MLX LSTM: input_size, hidden_size
        # Bidirectional = two LSTMs (forward + backward)
        self.lstm_fwd = nn.LSTM(input_size=nin_lstm, hidden_size=nout_lstm // 2)
        self.lstm_bwd = nn.LSTM(input_size=nin_lstm, hidden_size=nout_lstm // 2)
        self.dense_linear = nn.Linear(nout_lstm, nin_lstm)
        self.dense_bn = nn.BatchNorm(nin_lstm)

    def __call__(self, x):
        # x: (N, H, W, C) NHWC
        N, H, W, C = x.shape
        # Squeeze to 1 channel: (N, H, W, 1) → take channel 0 → (N, H, W)
        h = self.squeeze_conv(x)[:, :, :, 0]  # (N, H, W)

        # NHWC convention: H=freq, W=time
        # For LSTM we need (N, W, H) = (N, time, freq_features)
        h = mx.transpose(h, (0, 2, 1))  # (N, W, H) = (N, nframes, nbins)

        # Forward LSTM — nn.LSTM returns (output, (h_n, c_n))
        fwd_out, _ = self.lstm_fwd(h)  # (N, W, nout_lstm//2)
        # Backward LSTM: reverse time, run, reverse back
        h_rev = h[:, ::-1, :]
        bwd_out, _ = self.lstm_bwd(h_rev)  # (N, W, nout_lstm//2)
        bwd_out = bwd_out[:, ::-1, :]

        # Concatenate forward and backward
        lstm_out = mx.concatenate([fwd_out, bwd_out], axis=-1)  # (N, W, nout_lstm)

        # Dense projection: (N*W, nout_lstm) → (N*W, nin_lstm)
        lstm_flat = mx.reshape(lstm_out, (N * W, -1))
        h = self.dense_linear(lstm_flat)
        h = self.dense_bn(h)
        h = nn.relu(h)

        # Reshape back: (N, W, nin_lstm) → (N, 1, H, W) in NCHW equiv → (N, H, W, 1) NHWC
        h = mx.reshape(h, (N, W, -1))  # (N, W, nbins)
        h = mx.transpose(h, (0, 2, 1))  # (N, nbins, W) = (N, H, W)
        h = mx.expand_dims(h, axis=-1)  # (N, H, W, 1)

        return h
