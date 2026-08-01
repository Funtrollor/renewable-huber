"""Benchmark Native CPU thread scaling under the shape-sweep timing contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from renewable_huber import RenewableHuberRegressor  # noqa: E402
from renewable_huber.state import RenewableHuberState  # noqa: E402
from scripts.benchmarks.benchmark_shape_sweep import (  # noqa: E402
    PROFILES,
    _calibration_run,
    _dataset_checksum,
    _fit_batch,
    _lifecycle_metadata,
    _measure,
    _restore_empty_state,
    _run_operation,
    environment_metadata,
    make_batches,
)

SCALING_SCHEMA = "renewable-huber-native-cpu-thread-scaling"
SCALING_SCHEMA_VERSION = 1
DEFAULT_N_JOBS = (1, 2, 4, 8, -1)


def parse_n_jobs(value: str) -> tuple[int, ...]:
    """Parse a unique list containing positive counts or the all-core value -1."""

    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError("n-jobs must be a comma-separated list of integers") from error
    if not values or any(item == 0 or item < -1 for item in values):
        raise ValueError("n-jobs values must be positive integers or -1")
    if len(set(values)) != len(values):
        raise ValueError("n-jobs values must not contain duplicates")
    if 1 not in values:
        raise ValueError("n-jobs must include 1 as the scaling baseline")
    return values


def add_speedups(cases: list[dict[str, Any]]) -> None:
    """Add deterministic speedup and efficiency fields relative to n_jobs=1."""

    baseline = next((case for case in cases if case["requested_n_jobs"] == 1), None)
    if baseline is None:
        raise ValueError("thread scaling cases require an n_jobs=1 baseline")
    baseline_seconds = float(baseline["result"]["median_seconds"])
    if baseline_seconds <= 0:
        raise ValueError("baseline median_seconds must be positive")
    for case in cases:
        seconds = float(case["result"]["median_seconds"])
        if seconds <= 0:
            raise ValueError("case median_seconds must be positive")
        speedup = baseline_seconds / seconds
        case["speedup_vs_n_jobs_1"] = speedup
        threads = case.get("effective_threads")
        case["parallel_efficiency"] = (
            speedup / int(threads) if isinstance(threads, int) and threads > 0 else None
        )


def _effective_threads(model: RenewableHuberRegressor, requested: int) -> tuple[int, str]:
    """Read the native resolved count when exposed, with an explicit fallback."""

    backend = getattr(model, "_backend", None)
    candidates = (
        (model, "n_jobs_"),
        (backend, "effective_n_jobs"),
        (backend, "resolved_n_jobs"),
        (backend, "n_jobs"),
        (backend, "thread_count"),
    )
    for owner, name in candidates:
        value = getattr(owner, name, None)
        if isinstance(value, int) and value > 0:
            return value, f"{type(owner).__name__}.{name}"
    if requested > 0:
        return requested, "requested_n_jobs"
    return max(1, os.cpu_count() or 1), "os.cpu_count fallback for n_jobs=-1"


def _new_model(
    *, n_jobs: int, dtype: str, penalty: str, max_iter: int, tol: float
) -> RenewableHuberRegressor:
    return RenewableHuberRegressor(
        backend="native_cpu",
        device="cpu",
        dtype=dtype,
        penalty=penalty,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs,
    )


def benchmark_one(
    batches: list[tuple[Any, Any]],
    *,
    n_jobs: int,
    dtype: str,
    penalty: str,
    lifecycle: str,
    operation: str,
    warmup: int,
    repeats: int,
    max_iter: int,
    tol: float,
    minimum_sample_seconds: float,
    max_sample_repetitions: int,
) -> tuple[dict[str, Any], int, str]:
    """Measure one requested thread count without changing process globals."""

    fit_batch = _fit_batch(batches, xp=np)
    batch_count = 1 if operation == "fit" else len(batches)
    effective: tuple[int, str] | None = None

    def create_model() -> RenewableHuberRegressor:
        return _new_model(
            n_jobs=n_jobs,
            dtype=dtype,
            penalty=penalty,
            max_iter=max_iter,
            tol=tol,
        )

    if lifecycle == "cold":
        cold_model: RenewableHuberRegressor | None = None

        def cold_operation() -> tuple[int, bool]:
            nonlocal cold_model, effective
            cold_model = create_model()
            outcome = _run_operation(cold_model, batches, fit_batch, operation=operation)
            effective = _effective_threads(cold_model, n_jobs)
            return outcome

        def finalize() -> None:
            nonlocal cold_model
            cold_model = None

        for _ in range(warmup):
            cold_operation()
            finalize()
        estimate = _calibration_run(cold_operation, synchronize=None, finalize=finalize)
        result = _measure(
            cold_operation,
            repeats=repeats,
            finalize=finalize,
            estimated_operation_seconds=estimate,
            minimum_sample_seconds=minimum_sample_seconds,
            max_sample_repetitions=max_sample_repetitions,
        )
    elif lifecycle == "steady":
        if operation != "partial_fit":
            raise ValueError("steady scaling is defined only for partial_fit")
        model = create_model()

        def steady_operation() -> tuple[int, bool]:
            nonlocal effective
            outcome = _run_operation(model, batches, fit_batch, operation=operation)
            effective = _effective_threads(model, n_jobs)
            return outcome

        steady_operation()
        empty_state = RenewableHuberState.empty(
            batches[0][0].shape[1],
            fit_intercept=model.fit_intercept,
            xp=model._backend.xp,
            dtype=model._backend.dtype,
        )

        def prepare() -> None:
            _restore_empty_state(model, empty_state)

        for _ in range(warmup):
            prepare()
            steady_operation()
        estimate = _calibration_run(steady_operation, synchronize=None, prepare=prepare)
        result = _measure(
            steady_operation,
            repeats=repeats,
            prepare=prepare,
            estimated_operation_seconds=estimate,
            minimum_sample_seconds=minimum_sample_seconds,
            max_sample_repetitions=max_sample_repetitions,
        )
    else:
        raise ValueError("lifecycle must be 'cold' or 'steady'")

    result.update(
        _lifecycle_metadata(
            lifecycle=lifecycle,
            operation=operation,
            input_location="host",
            includes_input_transfer=False,
            batch_count=batch_count,
        )
    )
    if effective is None:
        raise RuntimeError("native CPU benchmark did not execute an operation")
    return result, effective[0], effective[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="standard")
    parser.add_argument("--case", default="reference")
    parser.add_argument("--n-jobs", default=",".join(map(str, DEFAULT_N_JOBS)))
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--penalty", choices=("none", "l1"), default="none")
    parser.add_argument("--lifecycle", choices=("cold", "steady"), default="steady")
    parser.add_argument("--operation", choices=("fit", "partial-fit"), default="partial-fit")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--minimum-sample-seconds", type=float, default=0.1)
    parser.add_argument("--max-sample-repetitions", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        n_jobs_values = parse_n_jobs(args.n_jobs)
    except ValueError as error:
        parser.error(str(error))
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be non-negative and repeats must be positive")
    if args.minimum_sample_seconds < 0 or args.max_sample_repetitions < 1:
        parser.error("sample duration must be non-negative and repetition cap positive")
    if args.lifecycle == "steady" and args.operation != "partial-fit":
        parser.error("steady scaling is defined only for partial-fit")

    shapes = {shape.name: shape for shape in PROFILES[args.profile]}
    if args.case not in shapes:
        parser.error(f"unknown case {args.case!r} for profile {args.profile!r}")
    shape = shapes[args.case]
    batches = make_batches(shape, seed=args.seed, dtype=args.dtype)
    cases: list[dict[str, Any]] = []
    operation = args.operation.replace("-", "_")
    for requested in n_jobs_values:
        result, effective_threads, source = benchmark_one(
            batches,
            n_jobs=requested,
            dtype=args.dtype,
            penalty=args.penalty,
            lifecycle=args.lifecycle,
            operation=operation,
            warmup=args.warmup,
            repeats=args.repeats,
            max_iter=args.max_iter,
            tol=args.tol,
            minimum_sample_seconds=args.minimum_sample_seconds,
            max_sample_repetitions=args.max_sample_repetitions,
        )
        cases.append(
            {
                "requested_n_jobs": requested,
                "effective_threads": effective_threads,
                "effective_threads_source": source,
                "result": result,
            }
        )
    add_speedups(cases)
    for case in cases:
        print(
            f"n_jobs={case['requested_n_jobs']:>2} "
            f"threads={case['effective_threads']:>2}: "
            f"{case['result']['median_seconds']:.6f}s, "
            f"speedup={case['speedup_vs_n_jobs_1']:.3f}x"
        )

    record = {
        "schema": SCALING_SCHEMA,
        "schema_version": SCALING_SCHEMA_VERSION,
        "environment": environment_metadata(),
        "shape": asdict(shape),
        "dataset_seed": args.seed,
        "dataset_sha256": _dataset_checksum(batches),
        "dtype": args.dtype,
        "penalty": args.penalty,
        "lifecycle": args.lifecycle,
        "operation": operation,
        "n_jobs": list(n_jobs_values),
        "baseline_n_jobs": 1,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} scaling measurements to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
