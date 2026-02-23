#!/usr/bin/env python3
"""Generate a scaffold JSON manifest for reference stem quality evaluation."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_STEMS = ["vocals", "drums", "bass", "other"]
DEFAULT_TEMPLATES = [
    "{mix_stem}_{stem}.wav",
    "{mix_stem}-({stem}).wav",
    "{mix_stem}-[{stem}].wav",
    "{mix_stem}-{stem}.wav",
    "{mix_stem} ({stem}).wav",
    "{mix_stem}/{stem}.wav",
]


@dataclass
class StemResolution:
    stem: str
    resolved_path: str
    exists: bool


def _load_corpus(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        audio_path = os.path.abspath(os.path.expanduser(line))
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Corpus entry does not exist: {audio_path}")
        out.append(audio_path)
    if not out:
        raise ValueError(f"Corpus file {path} contains no valid entries.")
    return out


def _resolve_search_dirs(mix_path: Path, extra_dirs: list[str]) -> list[Path]:
    dirs: list[Path] = [mix_path.parent.resolve()]
    for raw in extra_dirs:
        p = Path(os.path.abspath(os.path.expanduser(raw))).resolve()
        if p not in dirs:
            dirs.append(p)
    return dirs


def _render_template(template: str, mix_path: Path, stem: str) -> str:
    return template.format(
        stem=stem,
        mix_stem=mix_path.stem,
        mix_name=mix_path.name,
        mix_path=str(mix_path),
        mix_dir=str(mix_path.parent.resolve()),
    )


def _resolve_stem_path(
    *,
    mix_path: Path,
    stem: str,
    templates: list[str],
    search_dirs: list[Path],
) -> StemResolution:
    first_guess: Path | None = None
    for search_dir in search_dirs:
        for template in templates:
            rendered = _render_template(template, mix_path, stem)
            candidate = Path(rendered)
            if not candidate.is_absolute():
                candidate = (search_dir / candidate).resolve()
            if first_guess is None:
                first_guess = candidate
            if candidate.is_file():
                return StemResolution(stem=stem, resolved_path=str(candidate), exists=True)

    if first_guess is None:
        first_guess = (mix_path.parent / f"{mix_path.stem}_{stem}.wav").resolve()
    return StemResolution(stem=stem, resolved_path=str(first_guess), exists=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scaffold manifest for quality reference stems.")
    parser.add_argument("--corpus-file", required=True, help="Text file with one input mix path per line.")
    parser.add_argument("--output-json", required=True, help="Output manifest JSON path.")
    parser.add_argument(
        "--stems",
        default=",".join(DEFAULT_STEMS),
        help=f"Comma-separated stem names (default: {','.join(DEFAULT_STEMS)}).",
    )
    parser.add_argument(
        "--template",
        action="append",
        default=[],
        help=(
            "Filename template (repeatable). Variables: {mix_stem}, {stem}, {mix_name}, {mix_path}, {mix_dir}. "
            "If omitted, built-in templates are used."
        ),
    )
    parser.add_argument(
        "--search-dir",
        action="append",
        default=[],
        help="Extra directory to search for stem files (repeatable). Mix directory is always searched first.",
    )
    parser.add_argument(
        "--require-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if any stem file cannot be resolved to an existing file.",
    )
    parser.add_argument(
        "--output-summary-json",
        default=None,
        help="Optional path to write resolution summary JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    corpus = _load_corpus(Path(args.corpus_file))
    templates = args.template if args.template else list(DEFAULT_TEMPLATES)
    stems = [x.strip().lower() for x in str(args.stems).split(",") if x.strip()]
    if not stems:
        raise ValueError("No stems provided.")

    manifest: dict[str, dict[str, str]] = {}
    summary_rows: list[dict[str, object]] = []
    missing_count = 0

    for mix in corpus:
        mix_path = Path(mix).resolve()
        search_dirs = _resolve_search_dirs(mix_path, args.search_dir)
        stem_map: dict[str, str] = {}
        for stem in stems:
            resolution = _resolve_stem_path(
                mix_path=mix_path,
                stem=stem,
                templates=templates,
                search_dirs=search_dirs,
            )
            stem_map[stem] = resolution.resolved_path
            if not resolution.exists:
                missing_count += 1
            summary_rows.append(
                {
                    "mix": str(mix_path),
                    "stem": stem,
                    "resolved_path": resolution.resolved_path,
                    "exists": resolution.exists,
                }
            )
        manifest[str(mix_path)] = stem_map

    if args.require_existing and missing_count > 0:
        raise FileNotFoundError(
            f"Could not resolve {missing_count} stem files. "
            "Re-run without --require-existing to scaffold anyway."
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "corpus_files": len(corpus),
        "stems_per_file": len(stems),
        "total_entries": len(summary_rows),
        "missing_entries": int(missing_count),
        "templates": templates,
        "search_dirs": [str(x) for x in (_resolve_search_dirs(Path(corpus[0]).resolve(), args.search_dir) if corpus else [])],
        "rows": summary_rows,
    }
    if args.output_summary_json:
        summary_path = Path(args.output_summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote manifest: {output_path}")
    if args.output_summary_json:
        print(f"Wrote summary: {args.output_summary_json}")
    print(
        f"Scaffolded {len(corpus)} mixes x {len(stems)} stems "
        f"({len(summary_rows)} entries), missing={missing_count}"
    )


if __name__ == "__main__":
    main()
