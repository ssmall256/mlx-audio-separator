"""Every mlx-spectro STFT call must state its layout.

mlx-spectro's `stft()`/`istft()` default to the `bfn` layout (batch, freq,
frames) while its compiled API defaults to `bnf`. That split is scheduled to be
resolved -- mlx-spectro 0.9.x deprecates relying on either default and 1.0 will
converge them -- so any call here that relies on the default is a latent
silent-corruption bug waiting for that release.

It would be silent, not loud. Every one of these sites reshapes or slices on the
frequency axis (the `[..., :-1, :]` Nyquist drop in the Demucs models, the
`[:, :, :dim_f, :]` crop in MDX, the 5-D band rearrangements in the Roformers),
so a transposed result stays a valid array of the wrong thing. `istft` has a
bin-count guard that catches most mismatches, but it cannot catch the case where
the frame count happens to equal the bin count.

This is a source-level check on purpose: it fails when a new call site is added,
rather than waiting for an upstream default to change under us.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "mlx_audio_separator"

# Attribute names that belong to this repo's own STFT wrapper classes rather
# than to an mlx_spectro SpectralTransform.
_LOCAL_WRAPPERS = {"self.stft", "self.istft"}


def _layout_kwarg(method: str) -> str:
    return "output_layout" if method == "stft" else "input_layout"


def _unqualified(node: ast.Call) -> str:
    """Render the call target, e.g. 'transform.stft' or 'self.stft'."""
    func = node.func
    if not isinstance(func, ast.Attribute):
        return ""
    value = func.value
    if isinstance(value, ast.Name):
        return f"{value.id}.{func.attr}"
    if isinstance(value, ast.Attribute):
        return f"{value.attr}.{func.attr}"
    return func.attr


def _collect_calls():
    """Yield (path, lineno, target, kwargs, has_kwargs_unpack) per stft/istft call."""
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = _unqualified(node)
            if not target:
                continue
            method = target.rsplit(".", 1)[-1]
            if method not in {"stft", "istft"}:
                continue
            if target in _LOCAL_WRAPPERS:
                continue  # this repo's own STFT class, not mlx_spectro
            names = {kw.arg for kw in node.keywords if kw.arg is not None}
            unpack = any(kw.arg is None for kw in node.keywords)
            yield path, node.lineno, target, names, unpack


def test_there_are_calls_to_check():
    """Guard against the collector silently matching nothing."""
    assert list(_collect_calls()), "found no stft/istft calls; the AST walk is broken"


def test_every_stft_call_states_its_layout():
    offenders = []
    for path, lineno, target, names, unpack in _collect_calls():
        method = target.rsplit(".", 1)[-1]
        if _layout_kwarg(method) in names:
            continue
        if unpack:
            # Passed via **kwargs; the dict is checked separately below.
            continue
        offenders.append(f"{path.relative_to(PACKAGE)}:{lineno} {target}()")

    assert not offenders, (
        "these calls rely on mlx-spectro's default STFT layout, which is "
        "deprecated and changes in 1.0:\n  " + "\n  ".join(offenders)
    )


def test_kwargs_dicts_that_feed_istft_state_their_layout():
    """`transform.istft(z, **istft_kw)` is only safe if the dict sets the layout."""
    unpack_sites = [
        (path, lineno) for path, lineno, _t, _n, unpack in _collect_calls() if unpack
    ]
    if not unpack_sites:
        pytest.skip("no **kwargs call sites")

    for path, lineno in unpack_sites:
        source = path.read_text(encoding="utf-8")
        assert "input_layout=" in source, (
            f"{path.relative_to(PACKAGE)}:{lineno} forwards **kwargs to istft but "
            "the file never sets input_layout"
        )


def test_the_layouts_requested_are_all_bfn():
    """Mixed layouts within the package would be a latent bug of its own.

    Every model here treats axis 1 as frequency, so bfn is the only correct
    answer; a stray bnf would be a real defect rather than a style choice.
    """
    wrong = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "output_layout=" in line or "input_layout=" in line:
                if '"bfn"' not in line and "'bfn'" not in line:
                    wrong.append(f"{path.relative_to(PACKAGE)}:{lineno}: {line.strip()}")
    assert not wrong, "non-bfn layout requested:\n  " + "\n  ".join(wrong)
