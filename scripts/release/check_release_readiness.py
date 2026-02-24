#!/usr/bin/env python3
"""Evaluate benchmark JSON against release-readiness gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_results(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Benchmark JSON missing 'results' list.")
    return results


def evaluate_release_readiness(
    results: list[dict[str, Any]],
    *,
    require_total_models: int | None = None,
    max_failures: int = 0,
    min_success_rate: float = 1.0,
    require_models_ok: list[str] | None = None,
    allow_skipped: bool = False,
) -> dict[str, Any]:
    total = len(results)
    ok_rows = [r for r in results if r.get("status") == "ok"]
    skipped_rows = [r for r in results if str(r.get("status", "")).startswith("skipped:")]
    fail_rows = [r for r in results if r not in ok_rows and r not in skipped_rows]

    if allow_skipped:
        denominator = total
        success_count = len(ok_rows) + len(skipped_rows)
        failure_count = len(fail_rows)
    else:
        denominator = total
        success_count = len(ok_rows)
        failure_count = total - len(ok_rows)

    success_rate = (success_count / denominator) if denominator > 0 else 0.0

    reasons: list[str] = []
    if require_total_models is not None and total != int(require_total_models):
        reasons.append(f"total models mismatch: got {total}, expected {require_total_models}")
    if failure_count > int(max_failures):
        reasons.append(f"failure count {failure_count} exceeds max_failures {max_failures}")
    if success_rate < float(min_success_rate):
        reasons.append(
            f"success rate {success_rate:.4f} below min_success_rate {float(min_success_rate):.4f}"
        )

    missing_required_ok: list[str] = []
    if require_models_ok:
        by_name = {str(r.get("filename")): r for r in results}
        for name in require_models_ok:
            row = by_name.get(name)
            if row is None or row.get("status") != "ok":
                missing_required_ok.append(name)
        if missing_required_ok:
            reasons.append(f"required models not ok: {', '.join(missing_required_ok)}")

    summary = {
        "pass": len(reasons) == 0,
        "total": total,
        "ok": len(ok_rows),
        "skipped": len(skipped_rows),
        "failures": failure_count,
        "success_rate": success_rate,
        "reasons": reasons,
        "failed_models": [
            {
                "filename": r.get("filename"),
                "arch": r.get("arch"),
                "status": r.get("status"),
            }
            for r in (fail_rows if allow_skipped else [x for x in results if x.get("status") != "ok"])
        ],
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check benchmark JSON against release-readiness gates.")
    p.add_argument("--benchmark-json", required=True, help="Path to benchmark_results.json")
    p.add_argument("--require-total-models", type=int, default=None, help="Expected model count.")
    p.add_argument("--max-failures", type=int, default=0, help="Maximum allowed failures.")
    p.add_argument(
        "--min-success-rate",
        type=float,
        default=1.0,
        help="Minimum success ratio in [0,1]. Default 1.0 for release gates.",
    )
    p.add_argument(
        "--require-model-ok",
        action="append",
        default=[],
        help="Require this model filename to have status=ok. Repeatable.",
    )
    p.add_argument(
        "--allow-skipped",
        action="store_true",
        help="Treat skipped models as non-failures for gate evaluation.",
    )
    p.add_argument("--output-json", default=None, help="Optional path to write summary JSON.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    benchmark_path = Path(args.benchmark_json).expanduser().resolve()
    try:
        results = _load_results(benchmark_path)
        summary = evaluate_release_readiness(
            results,
            require_total_models=args.require_total_models,
            max_failures=args.max_failures,
            min_success_rate=args.min_success_rate,
            require_models_ok=list(args.require_model_ok or []),
            allow_skipped=bool(args.allow_skipped),
        )
    except Exception as exc:
        print(f"release-readiness check error: {exc}", file=sys.stderr)
        return 2

    output = json.dumps(summary, indent=2)
    print(output)
    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")

    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
