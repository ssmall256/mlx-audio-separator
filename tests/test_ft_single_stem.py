"""Fine-tuned single-stem inference preserves the full bag's shift and output."""

import logging
import random
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio_separator.demucs_mlx.api import Separator
from mlx_audio_separator.demucs_mlx.apply_mlx import apply_model
from mlx_audio_separator.demucs_mlx.mlx_convert import BagOfModelsMLX
from mlx_audio_separator.separator.architectures.demucs_separator import DemucsSeparator
from mlx_audio_separator.separator.common_separator import CommonSeparator

SOURCES = ("drums", "bass", "other", "vocals")


class _SourceModel:
    samplerate = 100
    audio_channels = 2
    segment = 0.3
    sources = SOURCES

    def __init__(self, number):
        self.number = number
        self.calls = 0

    def valid_length(self, length):
        return length

    def __call__(self, x):
        self.calls += 1
        return mx.stack([x * (self.number + source + 1) for source in range(4)], axis=1)


def _bag():
    models = [_SourceModel(index) for index in range(4)]
    weights = [[float(model == source) for source in range(4)] for model in range(4)]
    return BagOfModelsMLX(models, weights)


@pytest.mark.parametrize("shifts,split", [(0, False), (2, False), (1, True)])
def test_selected_source_matches_full_bag(shifts, split):
    bag = _bag()
    mix = mx.arange(240, dtype=mx.float32).reshape(1, 2, 120) / 1000
    full = apply_model(bag, mix, shifts=shifts, split=split, seed=481)
    mx.eval(full)
    for index in range(4):
        for model in bag.models:
            model.calls = 0
        selected = apply_model(bag, mix, shifts=shifts, split=split, seed=481, source_index=index)
        mx.eval(selected)
        np.testing.assert_array_equal(np.asarray(selected), np.asarray(full)[:, index : index + 1])
        assert [model.calls > 0 for model in bag.models] == [j == index for j in range(4)]


def test_unseeded_selection_preserves_global_random_state():
    bag = _bag()
    mix = mx.arange(240, dtype=mx.float32).reshape(1, 2, 120) / 1000
    random.seed(481)
    full = apply_model(bag, mix, shifts=3, split=False)
    full_state = random.getstate()
    for index in range(4):
        random.seed(481)
        selected = apply_model(bag, mix, shifts=3, split=False, source_index=index)
        assert random.getstate() == full_state
        np.testing.assert_array_equal(np.asarray(selected), np.asarray(full)[:, index : index + 1])


def test_separator_returns_only_requested_stem():
    bag = _bag()
    audio = np.arange(240, dtype=np.float32).reshape(2, 120) / 1000
    with patch("mlx_audio_separator.demucs_mlx.model_converter.get_mlx_model", return_value=bag):
        full = Separator(model="htdemucs_ft", seed=481, shifts=2, split=False)
        chosen = Separator(model="htdemucs_ft", stem="vocals", seed=481, shifts=2, split=False)
    _, full_stems = full.separate_tensor(audio)
    _, chosen_stems = chosen.separate_tensor(audio)
    assert list(chosen_stems) == ["vocals"]
    np.testing.assert_array_equal(chosen_stems["vocals"], full_stems["vocals"])


def test_single_stem_rejects_unsupported_models_and_sources():
    with pytest.raises(ValueError, match="only supports htdemucs_ft"):
        Separator(stem="vocals")
    with patch("mlx_audio_separator.demucs_mlx.model_converter.get_mlx_model", return_value=_bag()):
        with pytest.raises(ValueError, match="Unknown stem"):
            Separator(model="htdemucs_ft", stem="piano")
    with pytest.raises(ValueError, match="exactly one contributing model"):
        apply_model(
            BagOfModelsMLX([_SourceModel(0), _SourceModel(1)]), mx.zeros((1, 2, 5)), source_index=0
        )


@pytest.mark.parametrize("model,expected", [("htdemucs_ft", "bass"), ("htdemucs", None)])
def test_adapter_routes_single_stem_only_for_fine_tuned_bag(monkeypatch, model, expected):
    captured = {}

    def fake_common_init(self, config):
        self.performance_params = {}
        self.output_single_stem = "Bass"
        self.log_level = 20
        self.logger = logging.getLogger("test-demucs-stem")

    def fake_separator(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(samplerate=44_100, audio_channels=2)

    monkeypatch.setattr(CommonSeparator, "__init__", fake_common_init)
    monkeypatch.setattr(DemucsSeparator, "_resolve_demucs_model_name", lambda self: model)
    monkeypatch.setattr("mlx_audio_separator.demucs_mlx.api.Separator", fake_separator)
    DemucsSeparator({}, {})
    assert captured["stem"] == expected
