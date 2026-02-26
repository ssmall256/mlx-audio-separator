"""Lightweight documentation sanity checks."""

from pathlib import Path


def test_readme_links_release_docs():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert "docs/release-validation.md" in readme
    assert "docs/release-first.md" in readme
    assert "docs/reproducibility.md" in readme
    assert "docs/wave4-opt-in.md" in readme
