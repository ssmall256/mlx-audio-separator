"""Regression tests for the Demucs safetensors cache key contract.

0.1.7 wrote the cache from ``tree_flatten(model.state_dict())`` (MLX attribute
names) but read it back through ``_load_weights_into_model``, which expects
PyTorch-style names with the ``conv``/``layers`` wrappers collapsed. Only 213
of 573 tensors matched; the rest stayed at random initialization and every
Demucs model emitted noise-floor output with no error. These tests pin the
contract that the writer and the cache reader use the same key convention and
that any mismatch fails loudly.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from mlx_audio_separator.demucs_mlx import mlx_convert
from mlx_audio_separator.demucs_mlx.mlx_layers import Conv1dNCL
from mlx_audio_separator.demucs_mlx.mlx_utils import MLXStateDictMixin


class _WrappedModel(MLXStateDictMixin, nn.Module):
    """Exercises both wrapper shapes the old loader used to collapse.

    ``Conv1dNCL`` nests the real parameters under ``.conv``, and a list of
    submodules produces ``.<idx>.`` path segments -- exactly the naming that
    diverged between the cache writer and the cache reader.
    """

    def __init__(self):
        super().__init__()
        self.encoder = [Conv1dNCL(2, 4, 3, 1, 1), Conv1dNCL(4, 4, 3, 1, 1)]
        self.out = Conv1dNCL(4, 2, 1, 1, 0)


def _flat_state(model):
    return dict(tree_flatten(model.state_dict()))


def _randomize(model):
    """Give every parameter a distinct nonzero value.

    Freshly constructed models zero-initialize their biases, so two of them
    compare equal on those entries and a failed load can masquerade as a
    successful one. Randomizing makes "was this actually loaded?" answerable.
    """
    state = model.state_dict()
    flat = tree_flatten(state)
    model.update(
        tree_unflatten(
            [(key, mx.random.uniform(shape=value.shape) + 1.0) for key, value in flat]
        )
    )
    return model


def test_cache_keys_match_model_state_keys():
    """What _save_safe_cache writes is exactly what the loader looks up."""
    model = _randomize(_WrappedModel())
    written = _flat_state(model)

    # The wrapper segments must be present -- if this ever stops being true the
    # test is no longer covering the case that broke.
    assert any(".conv.weight" in key for key in written)

    fresh = _WrappedModel()
    mlx_convert._load_exact_model_state(
        fresh, written, context="test", regeneration="<cmd>"
    )

    reloaded = _flat_state(fresh)
    assert set(reloaded) == set(written)
    for key, value in written.items():
        assert mx.array_equal(reloaded[key], value), key


def test_every_parameter_is_actually_loaded():
    """A silent partial load is the exact failure mode from issue #4."""
    written = _flat_state(_randomize(_WrappedModel()))
    target = _randomize(_WrappedModel())
    before = _flat_state(target)

    # Every parameter must start out different, or "loaded" is unfalsifiable.
    assert all(
        not mx.array_equal(before[key], written[key]) for key in written
    ), "fixture is degenerate; this test cannot detect a failed load"

    mlx_convert._load_exact_model_state(
        target, written, context="test", regeneration="<cmd>"
    )

    after = _flat_state(target)
    assert set(after) == set(written)
    missed = [key for key in written if not mx.array_equal(after[key], written[key])]
    assert not missed, f"{len(missed)}/{len(written)} parameters were not loaded: {missed}"


def test_renamed_key_raises_instead_of_loading_silently():
    model = _WrappedModel()
    weights = _flat_state(model)
    victim = next(key for key in weights if key.endswith(".conv.weight"))
    corrupted = dict(weights)
    corrupted["bogus." + victim] = corrupted.pop(victim)

    with pytest.raises(mlx_convert.SafeCacheError) as excinfo:
        mlx_convert._load_exact_model_state(
            _WrappedModel(), corrupted, context="Demucs cache", regeneration="<cmd>"
        )

    message = str(excinfo.value)
    assert victim in message
    assert "bogus." + victim in message
    assert "<cmd>" in message


def test_pytorch_style_keys_are_rejected():
    """The 0.1.7 reader accepted these by collapsing wrappers; we must not."""
    model = _WrappedModel()
    weights = _flat_state(model)
    collapsed = {
        key.replace(".conv.weight", ".weight").replace(".conv.bias", ".bias"): value
        for key, value in weights.items()
    }
    assert set(collapsed) != set(weights)

    with pytest.raises(mlx_convert.SafeCacheError):
        mlx_convert._load_exact_model_state(
            _WrappedModel(), collapsed, context="Demucs cache", regeneration="<cmd>"
        )


def test_shape_mismatch_raises():
    model = _WrappedModel()
    weights = _flat_state(model)
    victim = next(iter(weights))
    corrupted = dict(weights)
    corrupted[victim] = mx.zeros((1, 1, 1))

    with pytest.raises(mlx_convert.SafeCacheError) as excinfo:
        mlx_convert._load_exact_model_state(
            _WrappedModel(), corrupted, context="Demucs cache", regeneration="<cmd>"
        )
    assert victim in str(excinfo.value)
