"""Benchmark strict, CUDA Graph, and opt-in TF32 native CUDA execution."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from renewable_huber import RenewableHuberRegressor  # noqa: E402
from renewable_huber.state import RenewableHuberState  # noqa: E402


def _relative_mad(values: list[float]) -> float:
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values) / median


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--features", type=int, default=90)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--minimum-sample-seconds", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        min(args.samples, args.features, args.repeats, args.max_iter) < 1
        or args.warmup < 0
        or args.minimum_sample_seconds < 0
    ):
        parser.error("shape/repeat/iteration values must be positive and warmup non-negative")

    dtype = np.dtype(args.dtype)
    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(args.samples, args.features)).astype(dtype)
    beta = rng.normal(size=args.features).astype(dtype)
    y = (X @ beta + rng.normal(scale=0.2, size=args.samples)).astype(dtype)
    y[::97] += dtype.type(8)
    modes = [("strict", False, False), ("graph", True, False)]
    if dtype == np.dtype("float32"):
        modes.append(("graph_tf32", True, True))

    results: dict[str, object] = {}
    strict_coefficients: np.ndarray | None = None
    empty = RenewableHuberState.empty(args.features, fit_intercept=True, xp=np, dtype=dtype)
    for label, graphs, fast_math in modes:
        model = RenewableHuberRegressor(
            backend="native_cuda",
            device="cuda",
            dtype=args.dtype,
            max_iter=args.max_iter,
            tol=args.tol,
            cuda_graphs=graphs,
            cuda_fast_math=fast_math,
        )
        model.fit(X, y)
        for _ in range(args.warmup):
            model._backend.restore_native_state(empty)
            model._state = empty.copy()
            model.partial_fit(X, y)

        model._backend.restore_native_state(empty)
        model._state = empty.copy()
        calibration_start = perf_counter()
        model.partial_fit(X, y)
        calibration_seconds = perf_counter() - calibration_start
        sample_repetitions = min(
            64,
            max(1, ceil(args.minimum_sample_seconds / calibration_seconds)),
        )

        timings: list[float] = []
        iteration_counts: list[int] = []
        for _ in range(args.repeats):
            gc.collect()
            sample_seconds = 0.0
            sample_iterations: list[int] = []
            for _sample_run in range(sample_repetitions):
                model._backend.restore_native_state(empty)
                model._state = empty.copy()
                start = perf_counter()
                model.partial_fit(X, y)
                sample_seconds += perf_counter() - start
                sample_iterations.append(model.n_iter_)
            timings.append(sample_seconds / sample_repetitions)
            iteration_counts.append(int(statistics.median(sample_iterations)))
        coefficients = np.asarray(model.coef_).copy()
        if strict_coefficients is None:
            strict_coefficients = coefficients
        difference = np.abs(coefficients - strict_coefficients)
        record = {
            "seconds": timings,
            "median_seconds": statistics.median(timings),
            "relative_mad": _relative_mad(timings),
            "iterations": iteration_counts,
            "sample_repetitions": sample_repetitions,
            "calibration_seconds": calibration_seconds,
            "cuda_features": model.cuda_features_,
            "max_abs_coefficient_error_vs_strict": float(difference.max(initial=0.0)),
            "relative_l2_coefficient_error_vs_strict": float(
                np.linalg.norm(difference) / max(np.linalg.norm(strict_coefficients), 1e-30)
            ),
        }
        results[label] = record
        print(
            f"{label}: {record['median_seconds'] * 1e3:.3f} ms; "
            f"relative MAD {record['relative_mad']:.2%}; {model.cuda_features_}"
        )

    strict_seconds = results["strict"]["median_seconds"]  # type: ignore[index]
    for record in results.values():
        record["speedup_vs_strict"] = strict_seconds / record["median_seconds"]  # type: ignore[index]
    payload = {
        "schema": "renewable-huber-native-cuda-tuning",
        "schema_version": 1,
        "configuration": vars(args) | {"output": str(args.output) if args.output else None},
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
