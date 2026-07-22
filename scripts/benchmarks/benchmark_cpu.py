"""Measure renewable-Huber NumPy CPU throughput with reproducible warm runs.

Set ``OPENBLAS_NUM_THREADS`` before starting Python to compare the BLAS thread
count used by the NumPy wheel.  For example:

``OPENBLAS_NUM_THREADS=8 python scripts/benchmarks/benchmark_cpu.py``

In PowerShell, use ``$env:OPENBLAS_NUM_THREADS = "8"`` before running Python.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from renewable_huber import RenewableHuberRegressor  # noqa: E402


def make_batches(
    n_samples: int, n_features: int, batch_size: int, seed: int, dtype: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic streamed batches outside the measured region."""

    rng = np.random.default_rng(seed)
    array_dtype = np.dtype(dtype)
    X = rng.normal(size=(n_samples, n_features)).astype(array_dtype)
    coefficients = rng.normal(size=n_features).astype(array_dtype)
    y = X @ coefficients + rng.normal(scale=0.2, size=n_samples).astype(array_dtype)
    return [
        (X[start : start + batch_size], y[start : start + batch_size])
        for start in range(0, n_samples, batch_size)
    ]


def run_once(
    batches: list[tuple[np.ndarray, np.ndarray]],
    *,
    penalty: str,
    dtype: str,
    max_iter: int,
) -> tuple[float, int]:
    """Fit one fresh estimator and return elapsed seconds plus total iterations."""

    model = RenewableHuberRegressor(
        penalty=penalty, dtype=dtype, max_iter=max_iter, backend="numpy", device="cpu"
    )
    iterations = 0
    start = perf_counter()
    for X_batch, y_batch in batches:
        model.partial_fit(X_batch, y_batch)
        iterations += model.diagnostics_.iterations
    return perf_counter() - start, iterations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=262_144)
    parser.add_argument("--features", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=32_768)
    parser.add_argument("--penalty", choices=["none", "l1"], default="none")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    batches = make_batches(args.samples, args.features, args.batch_size, args.seed, args.dtype)
    for _ in range(args.warmup):
        run_once(batches, penalty=args.penalty, dtype=args.dtype, max_iter=args.max_iter)

    measurements = [
        run_once(batches, penalty=args.penalty, dtype=args.dtype, max_iter=args.max_iter)
        for _ in range(args.repeats)
    ]
    seconds = [measurement[0] for measurement in measurements]
    iterations = [measurement[1] for measurement in measurements]
    median_seconds = statistics.median(seconds)
    print(f"NumPy {args.dtype}, penalty={args.penalty}")
    print(
        f"samples={args.samples:,} features={args.features} batch_size={args.batch_size:,} "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', 'default')}"
    )
    print(
        f"median={median_seconds:.3f}s throughput={args.samples / median_seconds:,.0f} samples/s "
        f"iterations={statistics.median(iterations):.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
