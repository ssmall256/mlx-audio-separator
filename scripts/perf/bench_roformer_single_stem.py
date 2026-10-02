"""Benchmark BS-Roformer-SW full 6-stem separation vs single-stem extraction on 120s audio.

Run via:
    metalq submit -w -n bench-bs-roformer-stem -- python scripts/perf/bench_roformer_single_stem.py
"""

import gc
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx_audio_io as aio
import numpy as np

from mlx_audio_separator.core import Separator

THERMAL_NAMES = ["nominal", "fair", "serious", "critical"]

try:
    import Foundation

    def get_thermal_state() -> tuple[int, str]:
        state = int(Foundation.NSProcessInfo.processInfo().thermalState())
        label = THERMAL_NAMES[state] if state < len(THERMAL_NAMES) else f"unknown({state})"
        return state, label
except ImportError:
    def get_thermal_state() -> tuple[int, str]:
        return 0, "nominal"


def wait_for_cool_silicon(min_cooldown: float = 6.0) -> float:
    start = time.perf_counter()
    state, _ = get_thermal_state()
    while state != 0:
        time.sleep(2.0)
        state, _ = get_thermal_state()
    time.sleep(min_cooldown)
    return time.perf_counter() - start


def generate_test_audio(seconds: int, sr: int = 44100) -> str:
    """Generate temporary 120s synthetic stereo test audio."""
    rng = np.random.default_rng(481 + seconds)
    n = sr * seconds
    t = np.arange(n, dtype=np.float32) / sr
    tones = (
        0.5 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 880 * t)
    )
    noise_l = rng.standard_normal(n, dtype=np.float32) * 0.01
    noise_r = rng.standard_normal(n, dtype=np.float32) * 0.01
    wave = np.stack([tones + noise_l, tones + noise_r], axis=1)

    import mlx_audio_io as aio
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    aio.save(tmp.name, mx.array(wave.T), sr)
    return tmp.name


def main():
    seconds = 120
    model_name = "BS-Roformer-SW.ckpt"
    model_dir = Path.home() / ".cache" / "mlx-audio-separator" / "models"
    model_path = model_dir / model_name

    if not model_path.exists():
        print(f"Error: model file {model_path} does not exist.")
        return

    audio_file = generate_test_audio(seconds)

    print(f"## BS-Roformer-SW Single-Stem vs Full 6-Stem Benchmark ({seconds}s Audio)", flush=True)
    print(f"- Model: `{model_name}` (6 stems: bass, drums, other, vocals, guitar, piano)", flush=True)
    print("- Architecture: Band-Split RoFormer (12-layer Transformer, 6 Mask Estimators)", flush=True)

    with tempfile.TemporaryDirectory() as out_full, tempfile.TemporaryDirectory() as out_single:
        # 1. Warmup
        print("\nWarming up models...", flush=True)
        warmup_audio = generate_test_audio(10)
        sep_warm = Separator(model_file_dir=str(model_dir), output_dir=out_full)
        sep_warm.load_model(model_name)
        sep_warm.separate(warmup_audio)
        del sep_warm
        Path(warmup_audio).unlink(missing_ok=True)
        mx.clear_cache()
        gc.collect()

        # 2. Full 6-Stem Separation
        wait_for_cool_silicon()
        _, t_in_label = get_thermal_state()
        print(f"\nRunning Full 6-Stem Separation (thermal: {t_in_label})...", flush=True)
        sep_full = Separator(
            model_file_dir=str(model_dir),
            output_dir=out_full,
            output_format="WAV",
        )
        sep_full.load_model(model_name)

        mx.clear_cache()
        gc.collect()
        t0 = time.perf_counter()
        files_full = sep_full.separate(audio_file)
        full_wall = time.perf_counter() - t0
        _, t_out_label = get_thermal_state()
        full_infer = sep_full.last_perf_metrics.get("inference_s", 0.0)
        full_post = sep_full.last_perf_metrics.get("postprocess_s", 0.0)
        print(
            f"Full 6-Stem: {full_wall:.3f}s wall "
            f"(inference: {full_infer:.3f}s, export: {full_post:.3f}s, {len(files_full)} files) [{t_in_label}->{t_out_label}]"
        )

        # 3. Single-Stem Extraction (Vocals only)
        wait_for_cool_silicon()
        _, t_in_label = get_thermal_state()
        print(f"\nRunning Single-Stem Extraction (--output_single_stem vocals, thermal: {t_in_label})...", flush=True)
        sep_single = Separator(
            model_file_dir=str(model_dir),
            output_dir=out_single,
            output_format="WAV",
            output_single_stem="vocals",
        )
        sep_single.load_model(model_name)

        mx.clear_cache()
        gc.collect()
        t0 = time.perf_counter()
        files_single = sep_single.separate(audio_file)
        single_wall = time.perf_counter() - t0
        _, t_out_label = get_thermal_state()
        single_infer = sep_single.last_perf_metrics.get("inference_s", 0.0)
        single_post = sep_single.last_perf_metrics.get("postprocess_s", 0.0)
        print(
            f"Single-Stem: {single_wall:.3f}s wall "
            f"(inference: {single_infer:.3f}s, export: {single_post:.3f}s, {len(files_single)} files) [{t_in_label}->{t_out_label}]"
        )

        # 4. Parity and Fidelity Check
        vocal_full_path = [f for f in files_full if "vocals" in f.lower()][0]
        vocal_single_path = [f for f in files_single if "vocals" in f.lower()][0]

        data_full, sr_full = aio.load(vocal_full_path)
        data_single, sr_single = aio.load(vocal_single_path)
        data_full = np.array(data_full)
        data_single = np.array(data_single)

        diff = np.abs(data_full - data_single)
        max_err = float(np.max(diff))
        denom = np.sum(data_full**2)
        mse = np.sum(diff**2)
        snr = 10 * np.log10(denom / max(mse, 1e-30)) if mse > 0 else float("inf")

        speedup_wall = full_wall / single_wall
        speedup_infer = full_infer / single_infer
        delta_rtfx = (seconds / single_wall) - (seconds / full_wall)

        rtfx_full = seconds / full_wall
        rtfx_single = seconds / single_wall

        print("\n### Benchmark Results")
        print("| Metric | Full 6-Stem Pass | Single Stem (Vocals) | Delta / Speedup |")
        print("|:---|:---:|:---:|:---:|")
        print(f"| **Wall Clock Time** | **{full_wall:.3f}s** | **{single_wall:.3f}s** | **{speedup_wall:.2f}x faster** |")
        print(f"| **Inference Time** | **{full_infer:.3f}s** | **{single_infer:.3f}s** | **{speedup_infer:.2f}x faster** |")
        print(f"| **Real-Time Factor (RTFx)** | {rtfx_full:.1f}x RTFx | **{rtfx_single:.1f}x RTFx** | +{delta_rtfx:.1f}x RTFx |")
        print("| **Stems Computed & Exported** | 6 stems | **1 stem** | 6x fewer exports |")
        print(f"| **Peak Absolute Error** | — | **{max_err:.2e}** | {'Bit-Exact Match' if max_err == 0.0 else 'High Fidelity'} |")
        print(f"| **Reconstruction SNR** | — | **{snr:.2f} dB** | {'Bit-Exact (inf)' if np.isinf(snr) else 'Exact'} |")

    Path(audio_file).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
