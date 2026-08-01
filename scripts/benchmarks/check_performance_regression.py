"""Fail when a schema-v2 native benchmark regresses on the same runner.

This command intentionally does not compare records across different machines.
Use it after capturing a fixed-runner baseline with at least nine steady or
cold repeats that share the exact same measurement contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:  # Supports both ``python script.py`` and package-style test imports.
    from .performance_policy import NATIVE_ENGINES, compare_records
except ImportError:  # pragma: no cover - exercised by direct CLI invocation.
    from performance_policy import NATIVE_ENGINES, compare_records


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _report(checks: list[Any]) -> dict[str, Any]:
    return {
        "passed": bool(checks) and all(check.passed for check in checks),
        "checked_cases": len(checks),
        "checks": [
            {
                **asdict(check),
                "key": asdict(check.key),
                "reasons": list(check.reasons),
            }
            for check in checks
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--engine",
        action="append",
        choices=tuple(sorted(NATIVE_ENGINES)),
        help="Engine to gate; defaults to every native engine present in the baseline.",
    )
    parser.add_argument("--cpu-max-slowdown", type=float, default=1.10)
    parser.add_argument("--gpu-max-slowdown", type=float, default=1.15)
    parser.add_argument("--min-repeats", type=int, default=9)
    parser.add_argument("--cpu-max-relative-mad", type=float, default=0.05)
    parser.add_argument("--gpu-max-relative-mad", type=float, default=0.10)
    parser.add_argument("--max-iteration-delta", type=int, default=1)
    parser.add_argument(
        "--max-competitor-slowdown",
        type=float,
        default=1.0,
        help="Native/reference median ratio allowed for the matched NumPy or CuPy competitor.",
    )
    parser.add_argument(
        "--no-require-competitor-parity",
        action="store_true",
        help="Diagnostic-only: do not require native to match its paired NumPy/CuPy case.",
    )
    parser.add_argument(
        "--allow-different-hardware",
        action="store_true",
        help=(
            "Report timing differences across runners without treating the fingerprint "
            "as a failure."
        ),
    )
    parser.add_argument("--output", type=Path, help="Optional machine-readable gate report.")
    args = parser.parse_args()

    try:
        checks = compare_records(
            _load(args.baseline),
            _load(args.candidate),
            engines=args.engine or NATIVE_ENGINES,
            cpu_max_slowdown=args.cpu_max_slowdown,
            gpu_max_slowdown=args.gpu_max_slowdown,
            min_repeats=args.min_repeats,
            cpu_max_relative_mad=args.cpu_max_relative_mad,
            gpu_max_relative_mad=args.gpu_max_relative_mad,
            max_iteration_delta=args.max_iteration_delta,
            require_competitor_parity=not args.no_require_competitor_parity,
            max_competitor_slowdown=args.max_competitor_slowdown,
            require_same_hardware=not args.allow_different_hardware,
        )
    except ValueError as error:
        parser.error(str(error))
    if not checks:
        parser.error("no requested native benchmark cases were found in the baseline")
    report = _report(checks)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        regression_ratio = "n/a" if check.slowdown_ratio is None else f"{check.slowdown_ratio:.3f}x"
        competitor_ratio = (
            "n/a" if check.competitor_ratio is None else f"{check.competitor_ratio:.3f}x"
        )
        competitor = check.competitor_engine or "none"
        print(
            f"{status} {check.key.engine} {check.key.shape_name}: "
            f"baseline={regression_ratio}, {competitor}={competitor_ratio}"
        )
        for reason in check.reasons:
            print(f"  - {reason}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
