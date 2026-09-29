"""Compare actual per-update and per-batch Demucs overlap-add evaluation.

Run through ``metalq submit -w``. The internal evaluation preference is
temporarily overridden for each arm of the comparison.
"""

import os
import statistics
import time

import numpy as np

from mlx_audio_separator.demucs_mlx import apply_mlx
from mlx_audio_separator.demucs_mlx.api import Separator


def signal(seconds):
    rng = np.random.default_rng(917 + seconds)
    n = seconds * 44_100
    t = np.arange(n, dtype=np.float32) / 44_100
    return (0.04 * np.sin(2 * np.pi * 220 * t)[None, :] +
            0.01 * rng.standard_normal((2, n), dtype=np.float32)).astype(np.float32)


separator = Separator(shifts=1, seed=481)
original_preference = apply_mlx._prefer_per_update_eval


def run(path, wav):
    apply_mlx._prefer_per_update_eval = lambda _count, _batch: path == "per-update"
    os.environ["MLX_AUDIO_SEPARATOR_DETERMINISTIC_ACCUMULATION"] = "0"
    start = time.perf_counter()
    _, stems = separator.separate_tensor(wav)
    return time.perf_counter() - start, stems


print("## Demucs overlap-add evaluation boundary", flush=True)
print("**Settings:** htdemucs, one shift, 25% overlap, batch two, whole-model compile", flush=True)
print("| Input | Pair | Per batch | Per update | Less wall time with per update | Minimum SNR | Peak error |", flush=True)
print("|---:|---:|---:|---:|---:|---:|---:|", flush=True)
try:
    for seconds in (40, 45, 50):
        wav = signal(seconds)
        run("per-batch", wav)  # Spectral tuning and first eager shape call.
        run("per-update", wav)  # Compile the repeated shape.
        pairs = []
        for index, order in enumerate((
            ("per-batch", "per-update"),
            ("per-update", "per-batch"),
            ("per-batch", "per-update"),
        ), start=1):
            measurements = {path: run(path, wav) for path in order}
            batch_time, batch_out = measurements["per-batch"]
            update_time, update_out = measurements["per-update"]
            snrs, peaks = [], []
            for stem in batch_out:
                want = batch_out[stem].astype(np.float64)
                error = want - update_out[stem].astype(np.float64)
                noise = np.sum(error * error)
                snrs.append(float("inf") if noise == 0 else 10 * np.log10(np.sum(want * want) / noise))
                peaks.append(np.max(np.abs(error)))
            pairs.append((batch_time, update_time))
            print(f"| {seconds}s | {index} | {batch_time:.3f}s | {update_time:.3f}s | "
                  f"{100 * (batch_time - update_time) / batch_time:.1f}% | "
                  f"{min(snrs):.1f} dB | {max(peaks):.3g} |", flush=True)
        before = statistics.mean(t[0] for t in pairs)
        after = statistics.mean(t[1] for t in pairs)
        print(f"> **{seconds}s mean:** per batch {before:.3f}s, per update {after:.3f}s; "
              f"{100 * (before - after) / before:.1f}% less wall time with per update.", flush=True)
finally:
    apply_mlx._prefer_per_update_eval = original_preference
    os.environ.pop("MLX_AUDIO_SEPARATOR_DETERMINISTIC_ACCUMULATION", None)
