"""Tests for reference manifest scaffold utility."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "perf" / "generate_reference_manifest.py"
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("generate_reference_manifest", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load generate_reference_manifest module.")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_render_template_uses_mix_stem_and_stem(tmp_path: Path):
    mod = _load_module()
    mix = tmp_path / "track.wav"
    out = mod._render_template("{mix_stem}_{stem}.wav", mix, "vocals")
    assert out == "track_vocals.wav"


def test_resolve_stem_path_prefers_existing(tmp_path: Path):
    mod = _load_module()
    mix = tmp_path / "song.wav"
    mix.write_bytes(b"x")
    stem = tmp_path / "song_vocals.wav"
    stem.write_bytes(b"y")

    resolution = mod._resolve_stem_path(
        mix_path=mix,
        stem="vocals",
        templates=["{mix_stem}_{stem}.wav"],
        search_dirs=[tmp_path],
    )
    assert resolution.exists is True
    assert resolution.resolved_path == str(stem.resolve())


def test_load_corpus_ignores_comments_and_blanks(tmp_path: Path):
    mod = _load_module()
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(f"# comment\n\n{a}\n{b}\n", encoding="utf-8")
    out = mod._load_corpus(corpus)
    assert out == [os.path.abspath(str(a)), os.path.abspath(str(b))]
