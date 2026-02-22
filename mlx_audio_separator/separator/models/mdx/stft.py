"""MDX-specific STFT wrapper using mlx-spectro.

Adapts the mlx-spectro SpectralTransform to the MDX model's expected format:
- Forward STFT: audio → 4-channel spectrogram (real/imag interleaved for stereo)
- Inverse STFT: 4-channel spectrogram → audio
- Frequency dimension truncation to dim_f
"""

import mlx.core as mx
from mlx_spectro import get_transform_mlx


class STFT:
    """STFT processor for MDX models.

    Converts stereo audio to/from the 4-channel real/imag spectrogram format
    expected by ConvTDFNet.

    Forward:  (N, 2, T) → STFT → interleave real/imag → (N, 4, dim_f, frames)
    Inverse:  (N, 4, dim_f, frames) → pad freq → deinterleave → iSTFT → (N, 2, T)
    """

    def __init__(self, n_fft, hop_length, dim_f):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dim_f = dim_f
        self.n_bins = n_fft // 2 + 1
        self._transform = get_transform_mlx(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window_fn="hann",
            window=None,
            periodic=True,
            center=True,
            normalized=False,
        )

    def __call__(self, x):
        """Forward STFT.

        Args:
            x: Audio tensor (N, 2, T) — stereo

        Returns:
            Spectrogram (N, 4, dim_f, frames) — stereo real/imag interleaved
        """
        N, C, T = x.shape

        # Flatten to (N*C, T) for batch STFT
        x_flat = mx.reshape(x, (N * C, T))

        # STFT → (N*C, F, frames) complex
        spec_complex = self._transform.stft(x_flat)

        # Convert to real representation (N*C, F, frames, 2)
        spec = mx.stack([spec_complex.real, spec_complex.imag], axis=-1)

        # Reshape to (N, C, F, frames, 2)
        F = spec.shape[1]
        frames = spec.shape[2]
        spec = mx.reshape(spec, (N, C, F, frames, 2))

        # Permute to (N, C, 2, F, frames) then reshape to (N, C*2, F, frames)
        # This interleaves channels with real/imag: [ch0_real, ch0_imag, ch1_real, ch1_imag]
        spec = mx.transpose(spec, (0, 1, 4, 2, 3))  # (N, C, 2, F, frames)
        spec = mx.reshape(spec, (N, C * 2, F, frames))  # (N, 4, F, frames)

        # Truncate frequency dimension to dim_f
        spec = spec[:, :, :self.dim_f, :]

        return spec

    def inverse(self, spec):
        """Inverse STFT.

        Args:
            spec: Spectrogram (N, 4, dim_f, frames) — stereo real/imag interleaved

        Returns:
            Audio tensor (N, 2, T)
        """
        N = spec.shape[0]
        frames = spec.shape[3]

        # Pad frequency dimension back to n_bins
        if self.dim_f < self.n_bins:
            pad_size = self.n_bins - self.dim_f
            freq_pad = mx.zeros((N, 4, pad_size, frames), dtype=spec.dtype)
            spec = mx.concatenate([spec, freq_pad], axis=2)

        # spec: (N, 4, n_bins, frames)
        # Reshape to (N, 2, 2, n_bins, frames) — (N, channels, real_imag, F, frames)
        spec = mx.reshape(spec, (N, 2, 2, self.n_bins, frames))

        # Permute to (N, 2, n_bins, frames, 2) — channels, F, frames, real/imag
        spec = mx.transpose(spec, (0, 1, 3, 4, 2))

        # Create complex: (N, 2, n_bins, frames)
        spec_complex = spec[..., 0] + 1j * spec[..., 1]

        # Flatten to (N*2, n_bins, frames) for batch iSTFT
        spec_flat = mx.reshape(spec_complex, (N * 2, self.n_bins, frames))

        # Inverse STFT
        audio = self._transform.istft(spec_flat)

        # Reshape to (N, 2, T)
        T = audio.shape[-1]
        audio = mx.reshape(audio, (N, 2, T))

        return audio
