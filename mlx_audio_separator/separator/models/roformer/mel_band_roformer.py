"""
MelBand-Roformer implementation for Apple MLX.

Shares transformer, attention, and mask estimation components with BS-Roformer.
The key difference is mel-scale frequency bands computed via mel filter bank,
with index-based gathering and scatter-add for overlapping band masks.

Based on: https://arxiv.org/abs/2309.02612
"""

import math
import os
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_spectro import get_transform_mlx

from .bs_roformer import (
    BandSplit,
    BSRoformerBlock,
    L2Norm,
    MaskEstimator,
    Transformer,
    pack,
    rearrange,
    unpack,
)


def _hz_to_mel(freq: np.ndarray, htk: bool = False) -> np.ndarray:
    """Convert Hz to mel scale."""
    if htk:
        return 2595.0 * np.log10(1.0 + freq / 700.0)
    # Slaney formula
    f_min = 0.0
    f_sp = 200.0 / 3
    mel = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_mask = freq >= min_log_hz
    mel[log_mask] = min_log_mel + np.log(freq[log_mask] / min_log_hz) / logstep
    return mel


def _mel_to_hz(mel: np.ndarray, htk: bool = False) -> np.ndarray:
    """Convert mel to Hz."""
    if htk:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    freq = f_min + f_sp * mel
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_mask = mel >= min_log_mel
    freq[log_mask] = min_log_hz * np.exp(logstep * (mel[log_mask] - min_log_mel))
    return freq


def create_mel_filter_bank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    norm: Optional[str] = None,
) -> np.ndarray:
    """Create a mel filter bank matching librosa.filters.mel output.

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency (default: sample_rate / 2)
        htk: Use HTK formula (default: Slaney)
        norm: Normalization type (None or "slaney")

    Returns:
        (n_mels, n_fft // 2 + 1) filter bank matrix
    """
    if fmax is None:
        fmax = float(sample_rate) / 2

    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, float(sample_rate) / 2, n_freqs)

    min_mel = _hz_to_mel(np.array([fmin]), htk=htk)[0]
    max_mel = _hz_to_mel(np.array([fmax]), htk=htk)[0]
    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    mel_freqs = _mel_to_hz(mels, htk=htk)

    fdiff = np.diff(mel_freqs)
    ramps = np.subtract.outer(mel_freqs, fft_freqs)

    weights = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == "slaney":
        enorm = 2.0 / (mel_freqs[2 : n_mels + 2] - mel_freqs[:n_mels])
        weights *= enorm[:, np.newaxis]

    return weights


class MelBandRoformerMLX(nn.Module):
    """
    MelBand-Roformer for music source separation.

    Uses mel-scale frequency bands instead of explicit frequency splits,
    providing perceptually-motivated frequency analysis.

    Reuses transformer, attention, and mask estimation from BS-Roformer.
    """

    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        linear_transformer_depth=0,
        num_bands=60,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        mlp_expansion_factor=4,
        mask_estimator_depth=2,
        sample_rate=44100,
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
        stft_normalized=False,
        chunk_seconds: float = 8.0,
        overlap_seconds: float = 1.0,
        match_input_audio_length: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.mlp_expansion_factor = mlp_expansion_factor
        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.match_input_audio_length = match_input_audio_length

        # STFT
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.stft_win_length = stft_win_length
        self.stft_normalized = stft_normalized
        self._stft_transform = get_transform_mlx(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            window_fn="hann",
            window=None,
            periodic=True,
            center=True,
            normalized=stft_normalized,
        )

        # Chunked inference
        chunk_env = os.environ.get("MLX_CHUNK_SECONDS")
        if chunk_env is not None:
            chunk_seconds = float(chunk_env)
        self.chunk_seconds = float(chunk_seconds)
        self.overlap_seconds = float(overlap_seconds)

        # Build mel filter bank
        mel_fb = create_mel_filter_bank(
            sample_rate=sample_rate,
            n_fft=stft_n_fft,
            n_mels=num_bands,
        )

        # Match PyTorch: ensure DC and Nyquist are covered
        mel_fb[0, 0] = 1.0
        mel_fb[-1, -1] = 1.0

        # Binary mask: which freqs belong to which bands
        freqs_per_band_mask = mel_fb > 0  # (num_bands, n_freqs)

        # Verify all freqs are covered
        assert freqs_per_band_mask.any(axis=0).all(), (
            "Not all frequencies covered by mel bands"
        )

        # Num freqs per band and bands per freq
        num_freqs_per_band = freqs_per_band_mask.sum(axis=1).astype(np.int32)  # (num_bands,)
        num_bands_per_freq = freqs_per_band_mask.sum(axis=0).astype(np.int32)  # (n_freqs,)

        # Build flat frequency index array (which freqs go to which bands)
        freq_indices_list = []
        for band_idx in range(num_bands):
            freq_idx = np.where(freqs_per_band_mask[band_idx])[0]
            freq_indices_list.append(freq_idx)
        freq_indices = np.concatenate(freq_indices_list)

        # Handle stereo interleaving
        if stereo:
            freq_indices_expanded = np.stack(
                [freq_indices * 2, freq_indices * 2 + 1], axis=-1
            )
            freq_indices = freq_indices_expanded.reshape(-1)

        # Store as non-trainable arrays
        self.freq_indices = mx.array(freq_indices)
        self.num_freqs_per_band = mx.array(num_freqs_per_band)
        self.num_bands_per_freq = mx.array(num_bands_per_freq)

        # Band dimensions for BandSplit / MaskEstimator
        freqs_per_bands_with_complex = tuple(
            int(2 * f * self.audio_channels)
            for f in num_freqs_per_band.tolist()
        )

        # Transformer kwargs
        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            ff_mult=mlp_expansion_factor,
            norm_output=True,
        )

        rotary_embed = True

        # Transformer layers
        for i in range(depth):
            linear_tran = None
            if linear_transformer_depth > 0:
                linear_tran = Transformer(
                    depth=linear_transformer_depth,
                    rotary_embed=rotary_embed,
                    linear_attn=False,
                    **transformer_kwargs,
                )
            time_tran = Transformer(
                depth=time_transformer_depth,
                rotary_embed=rotary_embed,
                **transformer_kwargs,
            )
            freq_tran = Transformer(
                depth=freq_transformer_depth,
                rotary_embed=rotary_embed,
                **transformer_kwargs,
            )
            setattr(self, f"layers_{i}", BSRoformerBlock(linear_tran, time_tran, freq_tran))

        self.final_norm = L2Norm(dim)

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        for i in range(num_stems):
            setattr(
                self,
                f"mask_estimators_{i}",
                MaskEstimator(
                    dim=dim,
                    dim_inputs=freqs_per_bands_with_complex,
                    depth=mask_estimator_depth,
                    mlp_expansion_factor=mlp_expansion_factor,
                ),
            )

        if os.environ.get("MLX_ENABLE_COMPILE") == "1":
            self._forward_transformers = mx.compile(self._forward_transformers)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, raw_audio):
        """Forward pass: raw audio -> STFT -> mel gather -> transform -> scatter -> iSTFT.

        Args:
            raw_audio: (batch, channels, time) or (batch, time)

        Returns:
            Separated audio: (batch, num_stems, channels, time) or (batch, channels, time)
        """
        if raw_audio.ndim == 2:
            raw_audio = mx.expand_dims(raw_audio, axis=1)

        batch_size, channels, time_samples = raw_audio.shape

        # Optional fixed-length padding
        fixed_len_env = os.environ.get("MLX_FIXED_CHUNK_SAMPLES")
        if fixed_len_env:
            fixed_len = int(fixed_len_env)
            if time_samples > fixed_len:
                raise ValueError(
                    f"Input length {time_samples} exceeds MLX_FIXED_CHUNK_SAMPLES={fixed_len}"
                )
            if time_samples < fixed_len:
                raw_audio = mx.pad(raw_audio, [(0, 0), (0, 0), (0, fixed_len - time_samples)])

        if (self.stereo and channels != 2) or (not self.stereo and channels != 1):
            raise ValueError(
                f"Config mismatch: stereo={self.stereo} but input has {channels} channel(s)"
            )

        # STFT
        audio_flat = rearrange(raw_audio, "b c t -> (b c) t")
        stft_complex = self._stft_transform.stft(audio_flat)  # (b*c, F, T) complex
        stft_real = mx.stack([stft_complex.real, stft_complex.imag], axis=-1)  # (b*c, F, T, 2)

        # Reshape to (b, c, F, T, 2) then interleave to (b, F*c, T, 2)
        stft_repr = mx.reshape(
            stft_real,
            (batch_size, channels, stft_real.shape[1], stft_real.shape[2], 2),
        )
        stft_repr = mx.transpose(stft_repr, (0, 2, 1, 3, 4))  # (b, F, c, T, 2)
        stft_repr = mx.reshape(
            stft_repr,
            (batch_size, stft_repr.shape[1] * channels, stft_repr.shape[3], 2),
        )

        # Gather mel-band frequencies and get masks
        masks = self._forward_model(stft_repr)
        # masks: (b, n_stems, num_gathered_freqs, T, 2)

        # Scatter-add masks back to full spectrum
        n_freqs_total = self.stft_n_fft // 2 + 1
        if self.stereo:
            n_freqs_total *= 2

        _, num_stems, _, time_steps, _ = masks.shape
        masks_summed = mx.zeros(
            (batch_size, num_stems, n_freqs_total, time_steps, 2),
            dtype=masks.dtype,
        )
        masks_summed = masks_summed.at[:, :, self.freq_indices, :, :].add(masks)

        # Average overlapping bands
        denom = self.num_bands_per_freq
        if self.stereo:
            denom = mx.repeat(denom, 2)
        denom = denom.astype(masks_summed.dtype).reshape(1, 1, -1, 1, 1)
        masks_averaged = masks_summed / mx.maximum(denom, mx.array(1e-8, dtype=masks_summed.dtype))

        # Apply masks (complex multiplication)
        stft_repr_expanded = mx.expand_dims(stft_repr, axis=1)  # (b, 1, F*c, T, 2)
        stft_complex = stft_repr_expanded[..., 0] + 1j * stft_repr_expanded[..., 1]
        mask_complex = masks_averaged[..., 0] + 1j * masks_averaged[..., 1]
        stft_masked = stft_complex * mask_complex  # (b, n, F*c, T)

        # Reshape for iSTFT
        stft_masked = rearrange(
            stft_masked, "b n (f c) t -> (b n c) f t", c=self.audio_channels
        )

        original_length = raw_audio.shape[-1]
        istft_length = original_length if self.match_input_audio_length else None
        recon_audio = self._stft_transform.istft(stft_masked, length=istft_length)
        recon_audio = rearrange(
            recon_audio,
            "(b n c) t -> b n c t",
            b=batch_size,
            n=self.num_stems,
            c=self.audio_channels,
        )

        if self.num_stems == 1:
            recon_audio = rearrange(recon_audio, "b 1 c t -> b c t")

        return recon_audio

    # ------------------------------------------------------------------
    # Chunked inference
    # ------------------------------------------------------------------

    def separate_audio_chunked(
        self,
        raw_audio: mx.array,
        *,
        sr: int = 44100,
        chunk_seconds: Optional[float] = None,
        overlap_seconds: Optional[float] = None,
        use_hann_window: bool = True,
        batch_hops: int = 1,
    ) -> mx.array:
        """Chunked overlap-add inference for long audio."""
        if raw_audio.ndim == 1:
            raw_audio = raw_audio[None, None, :]
        elif raw_audio.ndim == 2:
            if raw_audio.shape[0] in (1, 2):
                raw_audio = raw_audio[None, ...]
            else:
                raw_audio = raw_audio[:, None, :]
        elif raw_audio.ndim != 3:
            raise ValueError(f"Expected audio with 1-3 dims, got shape {tuple(raw_audio.shape)}")

        B, C, T = raw_audio.shape

        chunk_s = float(self.chunk_seconds if chunk_seconds is None else chunk_seconds)
        overlap_s = float(self.overlap_seconds if overlap_seconds is None else overlap_seconds)

        chunk_len = int(round(chunk_s * sr))
        overlap_len = int(round(overlap_s * sr))
        if chunk_len <= 0:
            raise ValueError(f"chunk_seconds too small -> chunk_len={chunk_len}")
        if overlap_len >= chunk_len:
            raise ValueError(f"overlap ({overlap_len}) must be < chunk_len ({chunk_len})")

        hop_len = chunk_len - overlap_len

        n_hops = int(math.ceil(max(T - overlap_len, 1) / hop_len))
        total_len = (n_hops - 1) * hop_len + chunk_len
        pad_len = total_len - T
        padded = mx.pad(raw_audio, [(0, 0), (0, 0), (0, pad_len)]) if pad_len > 0 else raw_audio

        if use_hann_window and chunk_len > 1:
            w = np.hanning(chunk_len).astype(np.float32)
        else:
            w = np.ones((chunk_len,), dtype=np.float32)

        if self.num_stems == 1:
            out_acc = np.zeros((B, C, total_len), dtype=np.float32)
            w_acc = np.zeros((1, 1, total_len), dtype=np.float32)
        else:
            out_acc = np.zeros((B, self.num_stems, C, total_len), dtype=np.float32)
            w_acc = np.zeros((1, 1, 1, total_len), dtype=np.float32)

        starts = [hop * hop_len for hop in range(n_hops)]

        for i in range(0, n_hops, batch_hops):
            hops = list(range(i, min(i + batch_hops, n_hops)))
            chunk_list = [padded[..., starts[h] : starts[h] + chunk_len] for h in hops]
            chunk_batch = mx.concatenate(chunk_list, axis=0)

            batch_out = self(chunk_batch)
            mx.eval(batch_out)
            batch_np = np.array(batch_out, dtype=np.float32)

            del chunk_batch, batch_out

            H = len(hops)
            if self.num_stems == 1:
                batch_np = batch_np.reshape(H, B, C, chunk_len)
                for j, hop in enumerate(hops):
                    start = starts[hop]
                    end = start + chunk_len
                    out_acc[..., start:end] += batch_np[j] * w[None, None, :]
                    w_acc[..., start:end] += w[None, None, :]
            else:
                batch_np = batch_np.reshape(H, B, self.num_stems, C, chunk_len)
                for j, hop in enumerate(hops):
                    start = starts[hop]
                    end = start + chunk_len
                    out_acc[..., start:end] += batch_np[j] * w[None, None, None, :]
                    w_acc[..., start:end] += w[None, None, None, :]

        out_acc = out_acc / np.maximum(w_acc, 1e-8)
        out_acc = out_acc[..., :T]
        return mx.array(out_acc)

    def separate(self, wav: mx.array, *, sr: int = 44100) -> mx.array:
        """Convenience wrapper: handles shape normalization and chunking."""
        if wav.ndim == 1:
            wav_bct = wav[None, None, :]
            squeeze_b = True
        elif wav.ndim == 2:
            if wav.shape[0] in (1, 2):
                wav_bct = wav[None, ...]
                squeeze_b = True
            else:
                wav_bct = wav[:, None, :]
                squeeze_b = False
        elif wav.ndim == 3:
            wav_bct = wav
            squeeze_b = wav.shape[0] == 1
        else:
            raise ValueError(f"Expected wav with 1-3 dims, got shape {tuple(wav.shape)}")

        chunk_len = int(round(self.chunk_seconds * sr))
        if wav_bct.shape[-1] > chunk_len:
            out = self.separate_audio_chunked(wav_bct, sr=sr)
        else:
            out = self(wav_bct)

        if squeeze_b:
            out = out[0]
        return out

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _forward_transformers(self, x):
        """Process through transformer stack: (batch, time, bands, dim) -> same."""
        for i in range(self.depth):
            block = getattr(self, f"layers_{i}")

            if block.has_linear:
                x, ft_ps = pack([x], "b * d")
                x = block.linear_transformer(x)
                (x,) = unpack(x, ft_ps, "b * d")

            # Time transformer
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = block.time_transformer(x)
            (x,) = unpack(x, ps, "* t d")

            # Frequency transformer
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = block.freq_transformer(x)
            (x,) = unpack(x, ps, "* f d")

        x = self.final_norm(x)
        return x

    def _estimate_masks(self, x):
        """Generate masks from transformer output."""
        masks = []
        for i in range(self.num_stems):
            estimator = getattr(self, f"mask_estimators_{i}")
            masks.append(estimator(x))
        masks = mx.stack(masks, axis=1)
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)
        return masks

    def _forward_model(self, stft_repr):
        """Gather mel freqs -> band split -> transform -> estimate masks."""
        x_gathered = mx.take(stft_repr, self.freq_indices, axis=1)
        x = rearrange(x_gathered, "b f t c -> b t (f c)")
        x = self.band_split(x)
        x = self._forward_transformers(x)
        return self._estimate_masks(x)
