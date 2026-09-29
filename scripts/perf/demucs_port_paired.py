"""Compare the pre-port and optimized embedded Demucs paths in one process.

Run via ``metalq submit -w``. Two independently loaded models preserve the
old fused-module construction and the new fast GroupNorm construction. Each
path gets its own whole-model compiled graph, then warmed AB/BA pairs run on
the same input with identical settings.
"""

import argparse
import os
import statistics
import time

import mlx.core as mx
import numpy as np

from mlx_audio_separator.demucs_mlx import mlx_demucs
from mlx_audio_separator.demucs_mlx.api import Separator
from mlx_audio_separator.demucs_mlx.mlx_demucs import GroupNorm
from mlx_audio_separator.demucs_mlx.mlx_layers import (
    ConvTranspose1dNCL,
    ConvTranspose2dNCHW,
    GroupNormNCHW,
    GroupNormNCL,
)


def signal(seconds):
    rng = np.random.default_rng(481 + seconds)
    n = 44_100 * seconds
    t = np.arange(n, dtype=np.float32) / 44_100
    tones = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = rng.standard_normal((2, n), dtype=np.float32) * 0.01
    return (0.05 * tones[None, :] + noise).astype(np.float32)


def old_group_norm(self, x):
    batch, channels = x.shape[:2]
    grouped = x.reshape(batch, self.num_groups, channels // self.num_groups, *x.shape[2:])
    axes = tuple(range(2, grouped.ndim))
    mean = mx.mean(grouped, axis=axes, keepdims=True)
    variance = mx.mean((grouped - mean) ** 2, axis=axes, keepdims=True)
    normalized = ((grouped - mean) * mx.rsqrt(variance + self.eps)).reshape(x.shape)
    if not self.affine:
        return normalized
    affine_shape = (1, channels) + (1,) * (x.ndim - 2)
    return normalized * self.weight.reshape(affine_shape) + self.bias.reshape(affine_shape)


def old_1d(self, x):
    return self.conv(x.transpose(0, 2, 1)).transpose(0, 2, 1)


def old_2d(self, x):
    return self.conv(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)


group_norm_classes = (GroupNorm, GroupNormNCL, GroupNormNCHW)
group_norm_calls = {cls: cls.__call__ for cls in group_norm_classes}
current_1d = ConvTranspose1dNCL.__call__
current_2d = ConvTranspose2dNCHW.__call__
current_fused = mlx_demucs._use_fused_gn_glu
old_env = {
    key: os.environ.get(key)
    for key in (
        "DEMUCS_MLX_COMPILE_DCONV",
        "MLX_AUDIO_SEPARATOR_DEMUCS_DCONV_COMPILE",
        "MLX_AUDIO_SEPARATOR_DEMUCS_COMPILE",
        "MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE",
    )
}
parser = argparse.ArgumentParser()
parser.add_argument("--comparison", choices=("full", "dconv"), default="full")
parser.add_argument("--outer-compile", choices=("on", "off"), default="on")
parser.add_argument("--shifts", type=int, default=1)
args = parser.parse_args()


def select(path):
    new = path == "after" or args.comparison == "dconv"
    for cls in group_norm_classes:
        cls.__call__ = group_norm_calls[cls] if new else old_group_norm
    ConvTranspose1dNCL.__call__ = current_1d if new else old_1d
    ConvTranspose2dNCHW.__call__ = current_2d if new else old_2d
    os.environ["MLX_AUDIO_SEPARATOR_DEMUCS_DCONV_COMPILE"] = (
        "1" if args.comparison == "dconv" and path == "after" else "0"
    )


def run(separator, path, audio):
    select(path)
    start = time.perf_counter()
    _, stems = separator.separate_tensor(audio)
    return time.perf_counter() - start, stems


os.environ["MLX_AUDIO_SEPARATOR_DEMUCS_COMPILE"] = "1" if args.outer_compile == "on" else "0"
os.environ["MLX_AUDIO_SEPARATOR_FUSED_GROUPNORM_MODE"] = "off"
print(f"## Embedded Demucs paired {args.comparison} comparison", flush=True)
print(
    f"**Settings:** htdemucs, {args.shifts} shift(s), 25% overlap, batch two, "
    f"whole-model compile {args.outer_compile}",
    flush=True,
)
if args.comparison == "dconv":
    print("**Before:** eager DConv; **after:** compiled DConv; all other optimizations on", flush=True)
else:
    print("**Before:** pre-port path; **after:** all adopted optimizations", flush=True)
print("| Input | Pair | Before | After | Less wall time | More audio/s | Minimum SNR | Peak error |", flush=True)
print("|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)
try:
    # The old path built fused wrapper modules even when its Metal kernel mode
    # was off. They used the older pure-MLX normalization fallback.
    mlx_demucs._use_fused_gn_glu = current_fused if args.comparison == "dconv" else lambda: True
    select("before")
    before_separator = Separator(seed=481, shifts=args.shifts)
    mlx_demucs._use_fused_gn_glu = current_fused
    select("after")
    after_separator = Separator(seed=481, shifts=args.shifts)

    for seconds in (30, 60):
        audio = signal(seconds)
        for path, separator in (("before", before_separator), ("after", after_separator)):
            run(separator, path, audio)  # Eager first call tunes the STFT.
            run(separator, path, audio)  # Second call builds that model's compiled graph.
        pairs = []
        for index, order in enumerate((("before", "after"), ("after", "before")), start=1):
            times = {}
            results = {}
            for path in order:
                separator = before_separator if path == "before" else after_separator
                times[path], results[path] = run(separator, path, audio)
            snrs = []
            peaks = []
            for stem, reference in results["before"].items():
                want = reference.astype(np.float64)
                error = want - results["after"][stem].astype(np.float64)
                noise = np.sum(error * error)
                snrs.append(float("inf") if noise == 0 else 10 * np.log10(np.sum(want * want) / noise))
                peaks.append(float(np.max(np.abs(error))))
            before = times["before"]
            after = times["after"]
            pairs.append((before, after))
            print(
                f"| {seconds}s | {index} | {before:.3f}s | **{after:.3f}s** | "
                f"{100 * (before - after) / before:.1f}% | {100 * (before / after - 1):.1f}% | "
                f"{min(snrs):.2f} dB | {max(peaks):.3g} |",
                flush=True,
            )
        before_mean = statistics.mean(pair[0] for pair in pairs)
        after_mean = statistics.mean(pair[1] for pair in pairs)
        print(
            f"> **{seconds}s mean:** {before_mean:.3f}s → {after_mean:.3f}s; "
            f"**{100 * (before_mean - after_mean) / before_mean:.1f}% less wall time**, "
            f"**{100 * (before_mean / after_mean - 1):.1f}% more audio/s**.",
            flush=True,
        )
finally:
    select("after")
    mlx_demucs._use_fused_gn_glu = current_fused
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
