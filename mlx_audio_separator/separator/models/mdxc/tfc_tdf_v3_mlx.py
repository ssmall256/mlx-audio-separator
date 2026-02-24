"""MLX-native MDX23C TFC_TDF_v3 model."""

from __future__ import annotations

from typing import Dict, Sequence

import mlx.core as mx
import mlx.nn as nn

from mlx_audio_separator.separator.models.mdx.stft import STFT


def _make_norm(norm_type: str | None, channels: int) -> nn.Module:
    """Construct a norm layer that operates on NHWC tensors."""
    if norm_type is None:
        return nn.Identity()
    if norm_type == "BatchNorm":
        return nn.BatchNorm(channels)
    if norm_type == "InstanceNorm":
        # InstanceNorm2d(affine=True) is equivalent to GroupNorm(C, C).
        return nn.GroupNorm(channels, channels)
    if norm_type.startswith("GroupNorm"):
        groups = int(norm_type.replace("GroupNorm", ""))
        return nn.GroupNorm(groups, channels)
    return nn.Identity()


def _make_act(act_type: str) -> nn.Module:
    if act_type == "gelu":
        return nn.GELU()
    if act_type == "relu":
        return nn.ReLU()
    if act_type.startswith("elu"):
        alpha = float(act_type.replace("elu", ""))
        return nn.ELU(alpha)
    raise ValueError(f"Unsupported activation: {act_type}")


def _apply_act(act_type: str, x: mx.array) -> mx.array:
    if act_type == "gelu":
        return nn.gelu(x)
    if act_type == "relu":
        return nn.relu(x)
    if act_type.startswith("elu"):
        alpha = float(act_type.replace("elu", ""))
        return nn.elu(x, alpha=alpha)
    raise ValueError(f"Unsupported activation: {act_type}")


def _crop_spatial_to_smallest(a: mx.array, b: mx.array) -> tuple[mx.array, mx.array]:
    """Crop NHWC tensors to shared H/W shape (handles odd edge effects)."""
    h = min(a.shape[1], b.shape[1])
    w = min(a.shape[2], b.shape[2])
    return a[:, :h, :w, :], b[:, :h, :w, :]


class DownscaleMLX(nn.Module):
    """Norm + activation + strided Conv2d downsample block."""

    def __init__(self, in_c: int, out_c: int, scale: Sequence[int], norm_type: str | None, act_type: str):
        super().__init__()
        self.act_type = act_type
        self.norm = _make_norm(norm_type, in_c)
        self.conv = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=tuple(scale),
            stride=tuple(scale),
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        x = _apply_act(self.act_type, x)
        return self.conv(x)


class UpscaleMLX(nn.Module):
    """Norm + activation + ConvTranspose2d upsample block."""

    def __init__(self, in_c: int, out_c: int, scale: Sequence[int], norm_type: str | None, act_type: str):
        super().__init__()
        self.act_type = act_type
        self.norm = _make_norm(norm_type, in_c)
        self.conv = nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=tuple(scale),
            stride=tuple(scale),
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        x = _apply_act(self.act_type, x)
        return self.conv(x)


class TfcTdfInnerBlockMLX(nn.Module):
    """One residual TFC+TDF block from MDX23C (NHWC variant)."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        freq_bins: int,
        bottleneck_factor: int,
        norm_type: str | None,
        act_type: str,
    ):
        super().__init__()
        self.act_type = act_type

        self.tfc1_norm = _make_norm(norm_type, in_c)
        self.tfc1_conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)

        self.tdf_norm1 = _make_norm(norm_type, out_c)
        self.tdf_linear1 = nn.Linear(freq_bins, freq_bins // bottleneck_factor, bias=False)
        self.tdf_norm2 = _make_norm(norm_type, out_c)
        self.tdf_linear2 = nn.Linear(freq_bins // bottleneck_factor, freq_bins, bias=False)

        self.tfc2_norm = _make_norm(norm_type, out_c)
        self.tfc2_conv = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = self.shortcut(x)

        y = self.tfc1_norm(x)
        y = _apply_act(self.act_type, y)
        y = self.tfc1_conv(y)

        # Apply TDF along the frequency axis (NHWC: axis=2).
        tdf = self.tdf_norm1(y)
        tdf = _apply_act(self.act_type, tdf)
        tdf = mx.transpose(tdf, (0, 1, 3, 2))  # N,T,C,F
        tdf = self.tdf_linear1(tdf)
        tdf = mx.transpose(tdf, (0, 1, 3, 2))  # N,T,F,C
        tdf = self.tdf_norm2(tdf)
        tdf = _apply_act(self.act_type, tdf)
        tdf = mx.transpose(tdf, (0, 1, 3, 2))  # N,T,C,F'
        tdf = self.tdf_linear2(tdf)
        tdf = mx.transpose(tdf, (0, 1, 3, 2))  # N,T,F,C
        y = y + tdf

        y = self.tfc2_norm(y)
        y = _apply_act(self.act_type, y)
        y = self.tfc2_conv(y)

        return y + shortcut


class TfcTdfStackMLX(nn.Module):
    """Stack of residual TFC+TDF inner blocks."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        num_blocks: int,
        freq_bins: int,
        bottleneck_factor: int,
        norm_type: str | None,
        act_type: str,
    ):
        super().__init__()
        self.num_blocks = int(num_blocks)
        current_in = int(in_c)
        for idx in range(self.num_blocks):
            block = TfcTdfInnerBlockMLX(
                in_c=current_in,
                out_c=out_c,
                freq_bins=freq_bins,
                bottleneck_factor=bottleneck_factor,
                norm_type=norm_type,
                act_type=act_type,
            )
            setattr(self, f"blocks_{idx}", block)
            current_in = out_c

    def __call__(self, x: mx.array) -> mx.array:
        for idx in range(self.num_blocks):
            block = getattr(self, f"blocks_{idx}")
            x = block(x)
        return x


class TfcTdfV3MLX(nn.Module):
    """MLX implementation of MDX23C TFC_TDF_v3."""

    def __init__(self, config: Dict):
        super().__init__()

        model_cfg = config["model"]
        audio_cfg = config["audio"]
        training_cfg = config["training"]

        norm_type = model_cfg.get("norm")
        act_type = model_cfg.get("act", "gelu")

        target_instrument = training_cfg.get("target_instrument")
        instruments = training_cfg.get("instruments", [])

        self.num_target_instruments = 1 if target_instrument else len(instruments)
        self.num_subbands = int(model_cfg["num_subbands"])
        self.num_scales = int(model_cfg["num_scales"])
        self.act = _make_act(act_type)

        dim_c = self.num_subbands * int(audio_cfg["num_channels"]) * 2
        scale = tuple(int(v) for v in model_cfg["scale"])
        blocks_per_scale = int(model_cfg["num_blocks_per_scale"])
        channels = int(model_cfg["num_channels"])
        growth = int(model_cfg["growth"])
        bottleneck_factor = int(model_cfg["bottleneck_factor"])
        freq_bins = int(audio_cfg["dim_f"]) // self.num_subbands

        self.first_conv = nn.Conv2d(dim_c, channels, kernel_size=1, stride=1, padding=0, bias=False)

        current_channels = channels
        current_freq_bins = freq_bins

        for idx in range(self.num_scales):
            tfc_tdf = TfcTdfStackMLX(
                in_c=current_channels,
                out_c=current_channels,
                num_blocks=blocks_per_scale,
                freq_bins=current_freq_bins,
                bottleneck_factor=bottleneck_factor,
                norm_type=norm_type,
                act_type=act_type,
            )
            down = DownscaleMLX(
                in_c=current_channels,
                out_c=current_channels + growth,
                scale=scale,
                norm_type=norm_type,
                act_type=act_type,
            )
            setattr(self, f"encoder_blocks_{idx}", nn.Module())
            block = getattr(self, f"encoder_blocks_{idx}")
            block.tfc_tdf = tfc_tdf
            block.downscale = down
            current_freq_bins = current_freq_bins // scale[1]
            current_channels += growth

        self.bottleneck_block = TfcTdfStackMLX(
            in_c=current_channels,
            out_c=current_channels,
            num_blocks=blocks_per_scale,
            freq_bins=current_freq_bins,
            bottleneck_factor=bottleneck_factor,
            norm_type=norm_type,
            act_type=act_type,
        )

        for idx in range(self.num_scales):
            up = UpscaleMLX(
                in_c=current_channels,
                out_c=current_channels - growth,
                scale=scale,
                norm_type=norm_type,
                act_type=act_type,
            )
            current_freq_bins = current_freq_bins * scale[1]
            current_channels -= growth
            tfc_tdf = TfcTdfStackMLX(
                in_c=2 * current_channels,
                out_c=current_channels,
                num_blocks=blocks_per_scale,
                freq_bins=current_freq_bins,
                bottleneck_factor=bottleneck_factor,
                norm_type=norm_type,
                act_type=act_type,
            )
            setattr(self, f"decoder_blocks_{idx}", nn.Module())
            block = getattr(self, f"decoder_blocks_{idx}")
            block.upscale = up
            block.tfc_tdf = tfc_tdf

        self.final_conv1 = nn.Conv2d(current_channels + dim_c, current_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.final_conv2 = nn.Conv2d(
            current_channels,
            self.num_target_instruments * dim_c,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.stft = STFT(
            n_fft=int(audio_cfg["n_fft"]),
            hop_length=int(audio_cfg["hop_length"]),
            dim_f=int(audio_cfg["dim_f"]),
        )

    def cac2cws(self, x: mx.array) -> mx.array:
        """Reshape channel-as-complex to channel-with-subbands space."""
        k = self.num_subbands
        b, c, f, t = x.shape
        x = mx.reshape(x, (b, c, k, f // k, t))
        return mx.reshape(x, (b, c * k, f // k, t))

    def cws2cac(self, x: mx.array) -> mx.array:
        """Reverse subband channel reshaping."""
        k = self.num_subbands
        b, c, f, t = x.shape
        x = mx.reshape(x, (b, c // k, k, f, t))
        return mx.reshape(x, (b, c // k, f * k, t))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, 2, T)
        x = self.stft(x)  # (B, C, F, T)

        mix = x = self.cac2cws(x)  # (B, C', F', T)
        mix_nhwc = mx.transpose(mix, (0, 2, 3, 1))  # (B, F', T, C')

        x = self.first_conv(mix_nhwc)  # (B, F', T, C)
        first_conv_out = x

        # Match PyTorch path that swaps the two spatial dimensions before the UNet.
        x = mx.transpose(x, (0, 2, 1, 3))  # (B, T, F', C)

        encoder_outputs = []
        for idx in range(self.num_scales):
            block = getattr(self, f"encoder_blocks_{idx}")
            x = block.tfc_tdf(x)
            encoder_outputs.append(x)
            x = block.downscale(x)

        x = self.bottleneck_block(x)

        for dec_idx in range(self.num_scales):
            block = getattr(self, f"decoder_blocks_{dec_idx}")
            x = block.upscale(x)
            skip = encoder_outputs.pop()
            x, skip = _crop_spatial_to_smallest(x, skip)
            x = mx.concatenate([x, skip], axis=-1)
            x = block.tfc_tdf(x)

        x = mx.transpose(x, (0, 2, 1, 3))  # (B, F', T, C)

        x, first_conv_out = _crop_spatial_to_smallest(x, first_conv_out)
        x = x * first_conv_out  # artifact reduction from original model

        x, mix_nhwc = _crop_spatial_to_smallest(x, mix_nhwc)
        x = mx.concatenate([mix_nhwc, x], axis=-1)
        x = self.final_conv1(x)
        x = self.act(x)
        x = self.final_conv2(x)

        x = mx.transpose(x, (0, 3, 1, 2))  # (B, C', F', T)
        x = self.cws2cac(x)  # (B, C, F, T)

        if self.num_target_instruments > 1:
            b, c, f, t = x.shape
            x = mx.reshape(x, (b, self.num_target_instruments, -1, f, t))

        return self.stft.inverse(x)
