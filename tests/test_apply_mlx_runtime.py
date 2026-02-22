"""Runtime toggle tests for Demucs apply_mlx helpers."""

from mlx_audio_separator.demucs_mlx import apply_mlx


def test_deterministic_accumulation_default_off(monkeypatch):
    monkeypatch.delenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_ACCUMULATION", raising=False)
    monkeypatch.delenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED", raising=False)
    assert apply_mlx._deterministic_accumulation_enabled() is False


def test_deterministic_accumulation_inherits_deterministic_fused(monkeypatch):
    monkeypatch.delenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_ACCUMULATION", raising=False)
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED", "1")
    assert apply_mlx._deterministic_accumulation_enabled() is True


def test_deterministic_accumulation_explicit_override_off(monkeypatch):
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_ACCUMULATION", "0")
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED", "1")
    assert apply_mlx._deterministic_accumulation_enabled() is False


def test_deterministic_accumulation_explicit_override_on(monkeypatch):
    monkeypatch.setenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_ACCUMULATION", "true")
    monkeypatch.delenv("MLX_AUDIO_SEPARATOR_DETERMINISTIC_FUSED", raising=False)
    assert apply_mlx._deterministic_accumulation_enabled() is True
