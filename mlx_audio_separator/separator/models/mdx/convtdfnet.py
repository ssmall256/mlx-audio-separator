"""ConvTDFNet (MDX-Net) architecture ported to MLX.

This is a U-Net encoder-decoder with TFC_TDF (Time-Frequency Convolution +
Time-Distributed Frequency) blocks for music source separation.

MLX uses channels-last (NHWC) layout. All Conv2d operations are adapted accordingly.
PyTorch weight conversion must transpose from OIHW → OHWI.

Reference: mdxnet.py + modules.py from python-audio-separator
"""

import mlx.core as mx
import mlx.nn as nn


class BatchNorm2d(nn.Module):
    """Batch normalization for 2D inputs in NHWC layout."""

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm(num_features)

    def __call__(self, x):
        # x: (N, H, W, C) — BatchNorm operates on the last dim
        return self.bn(x)


class GroupNorm2d(nn.Module):
    """Group normalization for 2D inputs in NHWC layout."""

    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)

    def __call__(self, x):
        return self.gn(x)


class TFC(nn.Module):
    """Time-Frequency Convolution block.

    Stack of Conv2d + Norm + ReLU layers. In MLX, inputs are NHWC.
    """

    def __init__(self, c, num_layers, k, norm_fn):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            conv = nn.Conv2d(
                in_channels=c, out_channels=c,
                kernel_size=k, stride=1, padding=k // 2,
            )
            setattr(self, f"conv_{i}", conv)
            setattr(self, f"norm_{i}", norm_fn(c))

    def __call__(self, x):
        for i in range(self.num_layers):
            conv = getattr(self, f"conv_{i}")
            norm = getattr(self, f"norm_{i}")
            x = nn.relu(norm(conv(x)))
        return x


class TFC_TDF(nn.Module):
    """Combined TFC and TDF (Time-Distributed Frequency) block.

    The TDF applies linear projections along the frequency dimension,
    optionally with a bottleneck.
    """

    def __init__(self, c, num_layers, f, k, bn, bias=True, norm_fn=None):
        super().__init__()
        self.use_tdf = bn is not None
        self.tfc = TFC(c, num_layers, k, norm_fn)

        if self.use_tdf:
            if bn == 0:
                self.tdf_linear1 = nn.Linear(f, f, bias=bias)
                self.tdf_norm1 = norm_fn(c)
                self.tdf_bottleneck = False
            else:
                self.tdf_linear1 = nn.Linear(f, f // bn, bias=bias)
                self.tdf_norm1 = norm_fn(c)
                self.tdf_linear2 = nn.Linear(f // bn, f, bias=bias)
                self.tdf_norm2 = norm_fn(c)
                self.tdf_bottleneck = True

    def __call__(self, x):
        x = self.tfc(x)
        if not self.use_tdf:
            return x

        # TDF: apply linear along frequency dim
        # x shape: (N, T, F, C) — need F in last position for Linear
        # Transpose to (N, T, C, F), apply linear, transpose back
        tdf_out = mx.transpose(x, (0, 1, 3, 2))  # (N, T, C, F)
        tdf_out = self.tdf_linear1(tdf_out)
        tdf_out = mx.transpose(tdf_out, (0, 1, 3, 2))  # (N, T, F, C)
        tdf_out = nn.relu(self.tdf_norm1(tdf_out))
        if self.tdf_bottleneck:
            tdf_out = mx.transpose(tdf_out, (0, 1, 3, 2))  # (N, T, C, F)
            tdf_out = self.tdf_linear2(tdf_out)
            tdf_out = mx.transpose(tdf_out, (0, 1, 3, 2))  # (N, T, F, C)
            tdf_out = nn.relu(self.tdf_norm2(tdf_out))
        return x + tdf_out


class ConvTDFNet(nn.Module):
    """ConvTDFNet U-Net model for MDX source separation.

    Architecture:
    1. first_conv: Project input channels to g features
    2. Encoder: TFC_TDF blocks + Conv2d downsampling (stride 2)
    3. Bottleneck: TFC_TDF block
    4. Decoder: ConvTranspose2d upsampling + multiplicative skip connections + TFC_TDF blocks
    5. final_conv: Project back to input channels

    All operations use NHWC layout (MLX convention).
    """

    def __init__(
        self,
        dim_c=4,
        dim_f=2048,
        dim_t=256,
        n_fft=6144,
        hop_length=1024,
        num_blocks=9,
        num_tdf_layers=3,
        g=32,
        k=3,
        bn=16,
        bias=True,
        optimizer="rmsprop",
    ):
        super().__init__()
        self.dim_c = dim_c
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length
        self.num_blocks = num_blocks

        # Choose normalization type based on optimizer flag
        if optimizer == "rmsprop":
            norm_fn = BatchNorm2d
        else:
            def norm_fn(c):
                return GroupNorm2d(2, c)

        n = num_blocks // 2
        self.n = n

        # First conv: (dim_c) -> (g)
        self.first_conv = nn.Conv2d(dim_c, g, kernel_size=1)
        self.first_norm = norm_fn(g)

        # Encoder
        f = dim_f
        c = g
        for i in range(n):
            setattr(self, f"enc_{i}", TFC_TDF(c, num_tdf_layers, f, k, bn, bias=bias, norm_fn=norm_fn))
            setattr(self, f"ds_{i}_conv", nn.Conv2d(c, c + g, kernel_size=2, stride=2))
            setattr(self, f"ds_{i}_norm", norm_fn(c + g))
            f = f // 2
            c += g

        # Bottleneck
        self.bottleneck = TFC_TDF(c, num_tdf_layers, f, k, bn, bias=bias, norm_fn=norm_fn)

        # Decoder
        for i in range(n):
            setattr(self, f"us_{i}_conv", nn.ConvTranspose2d(c, c - g, kernel_size=2, stride=2))
            setattr(self, f"us_{i}_norm", norm_fn(c - g))
            f = f * 2
            c -= g
            setattr(self, f"dec_{i}", TFC_TDF(c, num_tdf_layers, f, k, bn, bias=bias, norm_fn=norm_fn))

        # Final conv: (g) -> (dim_c)
        self.final_conv = nn.Conv2d(c, dim_c, kernel_size=1)

    def __call__(self, x):
        """Forward pass.

        Args:
            x: Spectrogram input, shape (N, C, F, T) in PyTorch convention.
                Internally converted to NHWC for MLX processing.

        Returns:
            Mask output, same shape as input (N, C, F, T).
        """
        # Convert from NCHW (PyTorch) to NHWC (MLX)
        # Input: (N, C, F, T) → (N, F, T, C)
        x = mx.transpose(x, (0, 2, 3, 1))

        # First conv
        x = nn.relu(self.first_norm(self.first_conv(x)))

        # Transpose freq/time: (N, F, T, C) → (N, T, F, C)
        x = mx.transpose(x, (0, 2, 1, 3))

        # Encoder with skip connections
        ds_outputs = []
        for i in range(self.n):
            enc = getattr(self, f"enc_{i}")
            ds_conv = getattr(self, f"ds_{i}_conv")
            ds_norm = getattr(self, f"ds_{i}_norm")
            x = enc(x)
            ds_outputs.append(x)
            x = nn.relu(ds_norm(ds_conv(x)))

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with multiplicative skip connections
        for i in range(self.n):
            us_conv = getattr(self, f"us_{i}_conv")
            us_norm = getattr(self, f"us_{i}_norm")
            dec = getattr(self, f"dec_{i}")
            x = nn.relu(us_norm(us_conv(x)))
            x = x * ds_outputs[-i - 1]
            x = dec(x)

        # Transpose back: (N, T, F, C) → (N, F, T, C)
        x = mx.transpose(x, (0, 2, 1, 3))

        # Final conv
        x = self.final_conv(x)

        # Convert back from NHWC to NCHW: (N, F, T, C) → (N, C, F, T)
        x = mx.transpose(x, (0, 3, 1, 2))

        return x
