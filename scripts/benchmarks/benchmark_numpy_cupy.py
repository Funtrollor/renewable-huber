"""Compare streamed NumPy CPU and CuPy CUDA renewable-Huber throughput."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from renewable_huber import BackendUnavailableError, RenewableHuberRegressor  # noqa: E402


def make_batches(n_samples: int, n_features: int, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    coefficients = rng.normal(size=n_features)
    y = X @ coefficients + rng.normal(scale=0.2, size=n_samples)
    return [
        (X[start : start + batch_size], y[start : start + batch_size])
        for start in range(0, n_samples, batch_size)
    ]


def _fit_all(model: RenewableHuberRegressor, batches: list[tuple[np.ndarray, np.ndarray]]) -> None:
    for X_batch, y_batch in batches:
        model.partial_fit(X_batch, y_batch)


def run_cpu(batches: list[tuple[np.ndarray, np.ndarray]], dtype: str) -> float:
    model = RenewableHuberRegressor(dtype=dtype)
    start = perf_counter()
    _fit_all(model, batches)
    return perf_counter() - start


def run_gpu(batches: list[tuple[np.ndarray, np.ndarray]], dtype: str) -> float:
    import cupy as cp

    # Load cuBLAS and allocate its workspaces before measuring steady-state
    # batch throughput.  Cold-start time is relevant to services but obscures
    # the performance of a long-running streaming job.
    _fit_all(RenewableHuberRegressor(backend="cupy", device="cuda", dtype=dtype), batches[:1])
    cp.cuda.get_current_stream().synchronize()
    model = RenewableHuberRegressor(backend="cupy", device="cuda", dtype=dtype)
    cp.cuda.get_current_stream().synchronize()
    start = perf_counter()
    _fit_all(model, batches)
    cp.cuda.get_current_stream().synchronize()
    return perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=32_768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()

    batches = make_batches(args.samples, args.features, args.batch_size, args.seed)
    cpu_seconds = run_cpu(batches, args.dtype)
    cpu_throughput = args.samples / cpu_seconds
    print(f"NumPy CPU ({args.dtype}): {cpu_seconds:.3f}s ({cpu_throughput:,.0f} samples/s)")

    try:
        gpu_seconds = run_gpu(batches, args.dtype)
    except (BackendUnavailableError, ImportError) as error:
        print(f"CuPy CUDA benchmark unavailable: {error}")
        return 0
    gpu_throughput = args.samples / gpu_seconds
    print(f"CuPy CUDA ({args.dtype}): {gpu_seconds:.3f}s ({gpu_throughput:,.0f} samples/s)")
    print(f"Speed-up: {cpu_seconds / gpu_seconds:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
