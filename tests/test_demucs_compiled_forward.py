"""The Demucs forward is compiled once per input shape.

The path otherwise runs entirely uncompiled -- `mx.compile` appears only in
`wiener_mlx.py`, which htdemucs never reaches -- and compiling it is worth
+15.9% end to end. These pin the parts that could silently go wrong: the
default, the opt-out, and that the cache is keyed on shape rather than
recompiling per call.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_audio_separator.demucs_mlx import apply_mlx


@pytest.fixture(autouse=True)
def _clear_cache():
    apply_mlx._COMPILED_FORWARDS.clear()
    yield
    apply_mlx._COMPILED_FORWARDS.clear()


class _Model:
    """Weak-referenceable stand-in; counts how often it is actually traced."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return x * 2.0 + 1.0


def test_compiled_by_default(monkeypatch):
    monkeypatch.delenv(apply_mlx._DEMUCS_COMPILE_ENV, raising=False)
    assert apply_mlx._demucs_compile_enabled()


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_the_opt_out_is_honoured(monkeypatch, value):
    monkeypatch.setenv(apply_mlx._DEMUCS_COMPILE_ENV, value)
    assert not apply_mlx._demucs_compile_enabled()
    model = _Model()
    x = mx.ones((1, 2, 64))
    apply_mlx._forward(model, x)
    assert apply_mlx._COMPILED_FORWARDS == {}, "must not cache when disabled"


def test_output_matches_the_eager_path(monkeypatch):
    x = mx.random.normal((2, 2, 128))
    monkeypatch.setenv(apply_mlx._DEMUCS_COMPILE_ENV, "0")
    eager = apply_mlx._forward(_Model(), x)
    monkeypatch.setenv(apply_mlx._DEMUCS_COMPILE_ENV, "1")
    compiled = apply_mlx._forward(_Model(), x)
    mx.eval(eager, compiled)
    assert float(mx.max(mx.abs(eager - compiled))) == 0.0


def test_one_graph_per_shape_not_per_call(monkeypatch):
    """A cache keyed per call would make compilation a per-chunk cost."""
    monkeypatch.setenv(apply_mlx._DEMUCS_COMPILE_ENV, "1")
    model = _Model()
    a, b = mx.ones((1, 2, 64)), mx.ones((2, 2, 64))
    for _ in range(4):
        mx.eval(apply_mlx._forward(model, a))
    # One eager call, then one trace: four calls, two invocations of the model.
    assert model.calls == 2, "same shape should trace once after the eager call"
    for _ in range(4):
        mx.eval(apply_mlx._forward(model, b))
    assert model.calls == 4, "a second shape adds one eager call and one graph"

    (_ref, per_shape), = apply_mlx._COMPILED_FORWARDS.values()
    assert len(per_shape) == 2


def test_the_first_call_at_a_shape_runs_eagerly(monkeypatch):
    """mlx-spectro tunes its STFT threadgroup sizes with timing runs, and MLX
    forbids mx.eval inside a compile trace. Compiling the very first call would
    leave a machine with no tuning cache no way to build one -- and before this,
    that surfaced as `no usable threadgroup size ... Attempting to eval an array
    during function transformations`, with no output produced at all.
    """
    monkeypatch.setenv(apply_mlx._DEMUCS_COMPILE_ENV, "1")
    model = _Model()
    x = mx.ones((1, 2, 64))

    traced = []
    real_compile = apply_mlx.mx.compile
    monkeypatch.setattr(
        apply_mlx.mx, "compile",
        lambda fn, *a, **k: (traced.append(fn), real_compile(fn, *a, **k))[1],
    )

    mx.eval(apply_mlx._forward(model, x))
    assert traced == [], "the first call must not be compiled"
    assert model.calls == 1, "the first call must still produce its output"

    mx.eval(apply_mlx._forward(model, x))
    assert len(traced) == 1, "the second call compiles"


def test_the_eager_first_call_returns_the_right_answer(monkeypatch):
    """The warm-up call is not a throwaway -- its output is used."""
    monkeypatch.setenv(apply_mlx._DEMUCS_COMPILE_ENV, "1")
    x = mx.random.normal((2, 2, 128))
    first = apply_mlx._forward(_Model(), x)
    expected = x * 2.0 + 1.0
    mx.eval(first, expected)
    assert float(mx.max(mx.abs(first - expected))) == 0.0


def test_a_second_model_gets_its_own_entry(monkeypatch):
    monkeypatch.setenv(apply_mlx._DEMUCS_COMPILE_ENV, "1")
    x = mx.ones((1, 2, 64))
    first, second = _Model(), _Model()
    mx.eval(apply_mlx._forward(first, x))
    mx.eval(apply_mlx._forward(second, x))
    assert len(apply_mlx._COMPILED_FORWARDS) == 2


def test_a_recycled_id_does_not_return_a_stale_graph(monkeypatch):
    """id() alone is unsafe: CPython reuses addresses of collected objects."""
    monkeypatch.setenv(apply_mlx._DEMUCS_COMPILE_ENV, "1")
    x = mx.ones((1, 2, 64))
    model = _Model()
    mx.eval(apply_mlx._forward(model, x))
    key = id(model)
    # Forge the collision the weakref check exists to catch.
    stale_ref, per_shape = apply_mlx._COMPILED_FORWARDS[key]
    replacement = _Model()
    apply_mlx._COMPILED_FORWARDS[id(replacement)] = (stale_ref, per_shape)
    mx.eval(apply_mlx._forward(replacement, x))
    assert replacement.calls == 1, "stale entry must be discarded, not reused"
