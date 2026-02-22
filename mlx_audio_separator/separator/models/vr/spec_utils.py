"""Spectral utility functions for VR models.

Uses mlx-spectro for GPU-accelerated STFT/iSTFT operations on Apple Silicon.
"""

import math
import traceback

import mlx.core as mx
import numpy as np
from mlx_spectro import get_transform_mlx


def _get_transform(n_fft, hop_length):
    """Get cached SpectralTransform for given parameters."""
    return get_transform_mlx(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window_fn="hann",
        window=None,
        periodic=False,   # symmetric window to match scipy
        center=True,       # only mode with good roundtrip accuracy
        normalized=False,
    )


def preprocess(X_spec):
    """Separate spectrogram into magnitude and phase."""
    X_mag = np.abs(X_spec)
    X_phase = np.angle(X_spec)
    return X_mag, X_phase


def make_padding(width, cropsize, offset):
    """Calculate padding for windowed processing."""
    left = offset
    roi_size = cropsize - offset * 2
    if roi_size == 0:
        roi_size = cropsize
    right = roi_size - (width % roi_size) + left
    return left, right, roi_size


def crop_center(h1, h2):
    """Crop h1 to match h2's time dimension (axis 3 in NCHW)."""
    if h1.shape[3] == h2.shape[3]:
        return h1
    if h1.shape[3] < h2.shape[3]:
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")
    s = (h1.shape[3] - h2.shape[3]) // 2
    return h1[:, :, :, s:s + h2.shape[3]]


def wave_to_spectrogram(wave, hop_length, n_fft, mp, band, is_v51_model=False):
    """Convert waveform to spectrogram using mlx-spectro STFT.

    Accepts either numpy or mlx arrays. Channel prep and STFT run on GPU,
    result is returned as numpy for downstream numpy operations.
    """
    # Ensure mlx array
    if isinstance(wave, np.ndarray):
        wave = mx.array(wave, dtype=mx.float32)
    if wave.ndim == 1:
        wave = mx.stack([wave, wave])

    if not is_v51_model:
        if mp.param["reverse"]:
            wave_left = wave[0][::-1]
            wave_right = wave[1][::-1]
        elif mp.param["mid_side"]:
            wave_left = (wave[0] + wave[1]) / 2
            wave_right = wave[0] - wave[1]
        elif mp.param["mid_side_b2"]:
            wave_left = wave[1] + wave[0] * 0.5
            wave_right = wave[0] - wave[1] * 0.5
        else:
            wave_left = wave[0]
            wave_right = wave[1]
    else:
        wave_left = wave[0]
        wave_right = wave[1]

    transform = _get_transform(n_fft, hop_length)
    wave_mx = mx.stack([wave_left, wave_right])  # (2, T)
    spec_mx = transform.stft(wave_mx)  # (2, F, frames) complex
    spec = np.array(spec_mx)  # back to numpy for combine_spectrograms

    if is_v51_model:
        spec = convert_channels(spec, mp, band)

    return spec


def spectrogram_to_wave(spec, hop_length=1024, mp=None, band=0, is_v51_model=True):
    """Convert spectrogram to waveform using mlx-spectro iSTFT."""
    spec_left = np.ascontiguousarray(spec[0])
    spec_right = np.ascontiguousarray(spec[1])

    n_fft = (spec_left.shape[0] - 1) * 2

    transform = _get_transform(n_fft, hop_length)
    spec_mx = mx.array(np.stack([spec_left, spec_right]))  # (2, F, T) complex
    wave_mx = transform.istft(spec_mx)  # (2, T)
    wave_left = np.array(wave_mx[0])
    wave_right = np.array(wave_mx[1])

    if mp is not None and is_v51_model:
        cc = mp.param["band"][band].get("convert_channels")
        if "mid_side_c" == cc:
            return np.asarray([np.subtract(wave_left / 1.0625, wave_right / 4.25), np.add(wave_right / 1.0625, wave_left / 4.25)])
        elif "mid_side" == cc:
            return np.asarray([np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)])
        elif "stereo_n" == cc:
            return np.asarray([np.subtract(wave_left, wave_right * 0.25), np.subtract(wave_right, wave_left * 0.25)])
    elif mp is not None:
        if mp.param["reverse"]:
            return np.asarray([np.flip(wave_left), np.flip(wave_right)])
        elif mp.param["mid_side"]:
            return np.asarray([np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)])
        elif mp.param["mid_side_b2"]:
            return np.asarray([np.add(wave_right / 1.25, 0.4 * wave_left), np.subtract(wave_left / 1.25, 0.4 * wave_right)])

    return np.asarray([wave_left, wave_right])


def convert_channels(spec, mp, band):
    """Apply channel conversion for VR 5.1 models."""
    cc = mp.param["band"][band].get("convert_channels")

    if "mid_side_c" == cc:
        spec_left = np.add(spec[0], spec[1] * 0.25)
        spec_right = np.subtract(spec[1], spec[0] * 0.25)
    elif "mid_side" == cc:
        spec_left = np.add(spec[0], spec[1]) / 2
        spec_right = np.subtract(spec[0], spec[1])
    elif "stereo_n" == cc:
        spec_left = np.add(spec[0], spec[1] * 0.25) / 0.9375
        spec_right = np.add(spec[1], spec[0] * 0.25) / 0.9375
    else:
        return spec

    return np.asarray([spec_left, spec_right])


def combine_spectrograms(specs, mp, is_v51_model=False):
    """Combine multi-band spectrograms into a single spectrogram."""
    min_len = min([specs[i].shape[2] for i in specs])
    spec_c = np.zeros(shape=(2, mp.param["bins"] + 1, min_len), dtype=np.complex64)
    offset = 0
    bands_n = len(mp.param["band"])

    for d in range(1, bands_n + 1):
        h = mp.param["band"][d]["crop_stop"] - mp.param["band"][d]["crop_start"]
        crop_start = mp.param["band"][d]["crop_start"]
        crop_stop = mp.param["band"][d]["crop_stop"]
        spec_c[:, offset:offset + h, :min_len] = specs[d][:, crop_start:crop_stop, :min_len]
        offset += h

    if offset > mp.param["bins"]:
        raise ValueError("Too many bins")

    # Low-pass filter
    if mp.param["pre_filter_start"] > 0:
        if is_v51_model:
            spec_c *= get_lp_filter_mask(spec_c.shape[1], mp.param["pre_filter_start"], mp.param["pre_filter_stop"])
        else:
            if bands_n == 1:
                spec_c = fft_lp_filter(spec_c, mp.param["pre_filter_start"], mp.param["pre_filter_stop"])
            else:
                gp = 1
                for b in range(mp.param["pre_filter_start"] + 1, mp.param["pre_filter_stop"]):
                    g = math.pow(10, -(b - mp.param["pre_filter_start"]) * (3.5 - gp) / 20.0)
                    gp = g
                    spec_c[:, b, :] *= g

    return np.ascontiguousarray(spec_c)


def cmb_spectrogram_to_wave(spec_m, mp, extra_bins_h=None, extra_bins=None, is_v51_model=False):
    """Convert combined spectrogram back to waveform via multi-band iSTFT."""
    import mlx_audio_io as mac

    bands_n = len(mp.param["band"])
    offset = 0
    wave = None

    for d in range(1, bands_n + 1):
        bp = mp.param["band"][d]
        spec_s = np.zeros(shape=(2, bp["n_fft"] // 2 + 1, spec_m.shape[2]), dtype=complex)
        h = bp["crop_stop"] - bp["crop_start"]
        spec_s[:, bp["crop_start"]:bp["crop_stop"], :] = spec_m[:, offset:offset + h, :]

        offset += h
        if d == bands_n:  # highest band
            if extra_bins_h:
                max_bin = bp["n_fft"] // 2
                spec_s[:, max_bin - extra_bins_h:max_bin, :] = extra_bins[:, :extra_bins_h, :]
            if bp.get("hpf_start", 0) > 0:
                if is_v51_model:
                    spec_s *= get_hp_filter_mask(spec_s.shape[1], bp["hpf_start"], bp["hpf_stop"] - 1)
                else:
                    spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
            if bands_n == 1:
                wave = spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model)
            else:
                new_wave = spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model)
                min_len = min(wave.shape[-1], new_wave.shape[-1])
                wave = np.add(wave[..., :min_len], new_wave[..., :min_len])
        else:
            sr = mp.param["band"][d + 1]["sr"]
            if d == 1:  # lowest band
                if is_v51_model:
                    spec_s *= get_lp_filter_mask(spec_s.shape[1], bp["lpf_start"], bp["lpf_stop"])
                else:
                    spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])

                band_wave = spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model)
                audio_mx = mx.array(band_wave.T, dtype=mx.float32)  # (frames, 2)
                resampled = mac.resample(audio_mx, int(bp["sr"]), int(sr))
                wave = np.array(resampled).T.astype(np.float32)  # (2, frames)
            else:  # mid band
                if is_v51_model:
                    spec_s *= get_hp_filter_mask(spec_s.shape[1], bp["hpf_start"], bp["hpf_stop"] - 1)
                    spec_s *= get_lp_filter_mask(spec_s.shape[1], bp["lpf_start"], bp["lpf_stop"])
                else:
                    spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
                    spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])

                new_wave = spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model)
                min_len = min(wave.shape[-1], new_wave.shape[-1])
                wave2 = np.add(wave[..., :min_len], new_wave[..., :min_len])
                audio_mx = mx.array(wave2.T, dtype=mx.float32)  # (frames, 2)
                resampled = mac.resample(audio_mx, int(bp["sr"]), int(sr))
                wave = np.array(resampled).T.astype(np.float32)  # (2, frames)

    return wave


def get_lp_filter_mask(n_bins, bin_start, bin_stop):
    """Create low-pass filter mask."""
    mask = np.concatenate([
        np.ones((bin_start - 1, 1)),
        np.linspace(1, 0, bin_stop - bin_start + 1)[:, None],
        np.zeros((n_bins - bin_stop, 1)),
    ], axis=0)
    return mask


def get_hp_filter_mask(n_bins, bin_start, bin_stop):
    """Create high-pass filter mask."""
    mask = np.concatenate([
        np.zeros((bin_stop + 1, 1)),
        np.linspace(0, 1, 1 + bin_start - bin_stop)[:, None],
        np.ones((n_bins - bin_start - 2, 1)),
    ], axis=0)
    return mask


def fft_lp_filter(spec, bin_start, bin_stop):
    """Apply FFT-based low-pass filter."""
    g = 1.0
    for b in range(bin_start, bin_stop):
        g -= 1 / (bin_stop - bin_start)
        spec[:, b, :] = g * spec[:, b, :]
    spec[:, bin_stop:, :] *= 0
    return spec


def fft_hp_filter(spec, bin_start, bin_stop):
    """Apply FFT-based high-pass filter."""
    g = 1.0
    for b in range(bin_start, bin_stop, -1):
        g -= 1 / (bin_start - bin_stop)
        spec[:, b, :] = g * spec[:, b, :]
    spec[:, 0:bin_stop + 1, :] *= 0
    return spec


def adjust_aggr(mask, is_non_accom_stem, aggressiveness):
    """Adjust mask aggressiveness."""
    aggr = aggressiveness["value"] * 2

    if aggr != 0:
        if is_non_accom_stem:
            aggr = 1 - aggr

        aggr = [aggr, aggr]

        if aggressiveness["aggr_correction"] is not None:
            aggr[0] += aggressiveness["aggr_correction"]["left"]
            aggr[1] += aggressiveness["aggr_correction"]["right"]

        for ch in range(2):
            mask[ch, :aggressiveness["split_bin"]] = np.power(mask[ch, :aggressiveness["split_bin"]], 1 + aggr[ch] / 3)
            mask[ch, aggressiveness["split_bin"]:] = np.power(mask[ch, aggressiveness["split_bin"]:], 1 + aggr[ch])

    return mask


def merge_artifacts(y_mask, thres=0.01, min_range=64, fade_size=32):
    """Merge detected artifacts in the mask."""
    mask = y_mask

    try:
        if min_range < fade_size * 2:
            raise ValueError("min_range must be >= fade_size * 2")

        idx = np.where(y_mask.min(axis=(0, 1)) > thres)[0]
        if len(idx) == 0:
            return mask

        start_idx = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
        end_idx = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
        artifact_idx = np.where(end_idx - start_idx > min_range)[0]
        weight = np.zeros_like(y_mask)

        if len(artifact_idx) > 0:
            start_idx = start_idx[artifact_idx]
            end_idx = end_idx[artifact_idx]
            old_e = None
            for s, e in zip(start_idx, end_idx):
                if old_e is not None and s - old_e < fade_size:
                    s = old_e - fade_size * 2

                if s != 0:
                    weight[:, :, s:s + fade_size] = np.linspace(0, 1, fade_size)
                else:
                    s -= fade_size

                if e != y_mask.shape[2]:
                    weight[:, :, e - fade_size:e] = np.linspace(1, 0, fade_size)
                else:
                    e += fade_size

                weight[:, :, s + fade_size:e - fade_size] = 1
                old_e = e

        v_mask = 1 - y_mask
        y_mask += weight * v_mask
        mask = y_mask
    except Exception as e:
        error_name = f"{type(e).__name__}"
        traceback_text = "".join(traceback.format_tb(e.__traceback__))
        print(f"Post Process Failed: {error_name}: \"{e}\"\n{traceback_text}")

    return mask


def mirroring(a, spec_m, input_high_end, mp):
    """Mirror high-end frequencies."""
    if "mirroring" == a:
        mirror = np.flip(
            np.abs(spec_m[:, mp.param["pre_filter_start"] - 10 - input_high_end.shape[1]:mp.param["pre_filter_start"] - 10, :]),
            1,
        )
        mirror = mirror * np.exp(1.0j * np.angle(input_high_end))
        return np.where(np.abs(input_high_end) <= np.abs(mirror), input_high_end, mirror)

    if "mirroring2" == a:
        mirror = np.flip(
            np.abs(spec_m[:, mp.param["pre_filter_start"] - 10 - input_high_end.shape[1]:mp.param["pre_filter_start"] - 10, :]),
            1,
        )
        mi = np.multiply(mirror, input_high_end * 1.7)
        return np.where(np.abs(input_high_end) <= np.abs(mi), input_high_end, mi)
