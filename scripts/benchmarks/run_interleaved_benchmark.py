"""Capture baseline/candidate shape sweeps in alternating A/B order and gate them."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

try:
    from .interleaved_regression import (
        compare_interleaved_records,
        merge_round_records,
        report,
    )
    from .performance_policy import NATIVE_ENGINES
except ImportError:  # pragma: no cover - direct CLI invocation.
    from interleaved_regression import compare_interleaved_records, merge_round_records, report
    from performance_policy import NATIVE_ENGINES


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _run_round(
    *,
    python: Path,
    repo: Path,
    output: Path,
    benchmark_args: list[str],
) -> dict[str, Any]:
    script = repo / "scripts" / "benchmarks" / "benchmark_shape_sweep.py"
    if not python.is_file():
        raise ValueError(f"Python executable does not exist: {python}")
    if not script.is_file():
        raise ValueError(f"benchmark script does not exist: {script}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(python), str(script), *benchmark_args, "--repeats", "1", "--output", str(output)]
    subprocess.run(command, cwd=repo, check=True)
    return _load(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", required=True, type=Path)
    parser.add_argument("--baseline-repo", required=True, type=Path)
    parser.add_argument("--candidate-python", required=True, type=Path)
    parser.add_argument("--candidate-repo", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pair-id", default=f"interleaved-{uuid.uuid4().hex}")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--case", action="append")
    parser.add_argument(
        "--backend",
        choices=("native-cpu", "native_cuda", "cpu", "gpu", "all"),
        default="all",
    )
    parser.add_argument("--penalty", choices=("none", "l1", "both"), default="none")
    parser.add_argument("--dtype", choices=("float32", "float64", "both"), default="both")
    parser.add_argument("--lifecycle", choices=("cold", "steady"), default="cold")
    parser.add_argument("--operation", choices=("fit", "partial-fit"), default="partial-fit")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--minimum-sample-seconds", type=float, default=0.25)
    parser.add_argument("--max-sample-repetitions", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-max-slowdown", type=float, default=1.10)
    parser.add_argument("--gpu-max-slowdown", type=float, default=1.15)
    parser.add_argument("--cpu-max-relative-mad", type=float, default=0.05)
    parser.add_argument("--gpu-max-relative-mad", type=float, default=0.10)
    parser.add_argument("--max-competitor-slowdown", type=float, default=1.0)
    args = parser.parse_args()
    if args.rounds < 3 or args.warmup < 0 or args.minimum_sample_seconds < 0:
        parser.error("rounds must be at least 3; warmup and sample duration must be non-negative")

    benchmark_args = [
        "--profile",
        args.profile,
        "--backend",
        args.backend,
        "--penalty",
        args.penalty,
        "--dtype",
        args.dtype,
        "--lifecycle",
        args.lifecycle,
        "--operation",
        args.operation,
        "--warmup",
        str(args.warmup),
        "--minimum-sample-seconds",
        str(args.minimum_sample_seconds),
        "--max-sample-repetitions",
        str(args.max_sample_repetitions),
        "--max-iter",
        str(args.max_iter),
        "--tol",
        str(args.tol),
        "--seed",
        str(args.seed),
    ]
    for case in args.case or ():
        benchmark_args.extend(("--case", case))

    # Do not resolve Python symlinks: resolving ``.venv/bin/python`` to the
    # system interpreter silently drops the virtual environment on Linux.
    variants = {
        "baseline": (args.baseline_python.absolute(), args.baseline_repo.resolve()),
        "candidate": (args.candidate_python.absolute(), args.candidate_repo.resolve()),
    }
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in variants}
    execution_order: dict[str, list[int]] = {name: [] for name in variants}
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "rounds"
    for round_index in range(args.rounds):
        order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
        print(f"round {round_index + 1}/{args.rounds}: {' -> '.join(order)}", flush=True)
        for position, variant in enumerate(order):
            python, repo = variants[variant]
            records[variant].append(
                _run_round(
                    python=python,
                    repo=repo,
                    output=raw_dir / f"{round_index:02d}-{variant}.json",
                    benchmark_args=benchmark_args,
                )
            )
            execution_order[variant].append(position)

    output_dir.mkdir(parents=True, exist_ok=True)
    merged = {
        variant: merge_round_records(
            variant_records,
            variant=variant,
            pair_id=args.pair_id,
            execution_order=execution_order[variant],
        )
        for variant, variant_records in records.items()
    }
    for variant, record in merged.items():
        (output_dir / f"{variant}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    checks = compare_interleaved_records(
        merged["baseline"],
        merged["candidate"],
        engines=NATIVE_ENGINES,
        cpu_max_slowdown=args.cpu_max_slowdown,
        gpu_max_slowdown=args.gpu_max_slowdown,
        min_pairs=args.rounds,
        cpu_max_relative_mad=args.cpu_max_relative_mad,
        gpu_max_relative_mad=args.gpu_max_relative_mad,
        max_competitor_slowdown=args.max_competitor_slowdown,
    )
    gate = report(checks)
    (output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        paired_value = (
            "n/a"
            if check.paired_median_slowdown is None
            else f"{check.paired_median_slowdown:.3f}x"
        )
        print(f"{status} {check.key.engine} {check.key.shape_name}: paired={paired_value}")
        for reason in check.reasons:
            print(f"  - {reason}")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
