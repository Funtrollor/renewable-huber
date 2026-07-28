"""CUDA workload with NVTX ranges for Nsight Systems and Nsight Compute."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from renewable_huber import RenewableHuberRegressor  # noqa: E402


def _git_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def make_batches(
    *,
    samples: int,
    features: int,
    batch_size: int,
    dtype: str,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    array_dtype = np.dtype(dtype)
    X = rng.normal(size=(samples, features)).astype(array_dtype)
    coefficients = rng.normal(size=features).astype(array_dtype)
    y = X @ coefficients + rng.normal(scale=0.2, size=samples).astype(array_dtype)
    outlier_rows = np.arange(0, samples, max(100, samples // 100))
    y[outlier_rows] += rng.normal(scale=8.0, size=outlier_rows.size).astype(array_dtype)
    return [
        (X[start : start + batch_size], y[start : start + batch_size])
        for start in range(0, samples, batch_size)
    ]


def _fit(
    batches: list[tuple[Any, Any]],
    *,
    dtype: str,
    penalty: str,
    max_iter: int,
    tol: float,
    annotate_batches: bool,
) -> tuple[int, bool]:
    from cupyx.profiler import time_range

    model = RenewableHuberRegressor(
        backend="cupy",
        device="cuda",
        dtype=dtype,
        penalty=penalty,
        max_iter=max_iter,
        tol=tol,
    )
    iterations = 0
    converged = True
    for batch_index, (X_batch, y_batch) in enumerate(batches):
        if annotate_batches:
            with time_range(f"profile/batch-{batch_index}", color_id=batch_index % 8):
                model.partial_fit(X_batch, y_batch)
        else:
            model.partial_fit(X_batch, y_batch)
        iterations += model.diagnostics_.iterations
        converged = converged and model.diagnostics_.converged
    return iterations, converged


def _metadata(
    cp: Any,
    args: argparse.Namespace,
    elapsed: list[float],
    iteration_counts: list[int],
    convergence: list[bool],
) -> dict[str, Any]:
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    return {
        "schema": "renewable-huber-nsight-profile",
        "schema_version": 1,
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "gpu": name,
        "gpu_compute_capability": f"{properties['major']}.{properties['minor']}",
        "configuration": {
            "samples": args.samples,
            "features": args.features,
            "batch_size": args.batch_size,
            "batches": (args.samples + args.batch_size - 1) // args.batch_size,
            "dtype": args.dtype,
            "penalty": args.penalty,
            "input_location": args.input_location,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "seed": args.seed,
        },
        "elapsed_seconds": elapsed,
        "iteration_counts": iteration_counts,
        "all_batches_converged": all(convergence),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=32_768)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--penalty", choices=("none", "l1"), default="none")
    parser.add_argument("--input-location", choices=("host", "device"), default="device")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()
    if (
        args.samples < 1
        or args.features < 1
        or args.batch_size < 1
        or args.warmup < 0
        or args.repeats < 1
    ):
        parser.error("sizes and repeats must be positive; warmup must be non-negative")

    try:
        import cupy as cp
        from cupyx.profiler import time_range
    except ImportError as error:
        parser.error(f"CuPy CUDA is required: {error}")

    host_batches = make_batches(
        samples=args.samples,
        features=args.features,
        batch_size=args.batch_size,
        dtype=args.dtype,
        seed=args.seed,
    )
    if args.input_location == "device":
        batches = [(cp.asarray(X), cp.asarray(y)) for X, y in host_batches]
    else:
        batches = host_batches
    cp.cuda.get_current_stream().synchronize()

    with time_range("warmup", color_id=7):
        for _ in range(args.warmup):
            _fit(
                batches,
                dtype=args.dtype,
                penalty=args.penalty,
                max_iter=args.max_iter,
                tol=args.tol,
                annotate_batches=False,
            )
        cp.cuda.get_current_stream().synchronize()

    elapsed = []
    iteration_counts = []
    convergence = []
    for repeat in range(args.repeats):
        start = perf_counter()
        with time_range(f"profile/repeat-{repeat}", color_id=repeat % 6):
            iterations, converged = _fit(
                batches,
                dtype=args.dtype,
                penalty=args.penalty,
                max_iter=args.max_iter,
                tol=args.tol,
                annotate_batches=True,
            )
            cp.cuda.get_current_stream().synchronize()
        seconds = perf_counter() - start
        elapsed.append(seconds)
        iteration_counts.append(iterations)
        convergence.append(converged)
        print(
            f"repeat={repeat} elapsed={seconds:.6f}s iterations={iterations} converged={converged}"
        )

    metadata = _metadata(cp, args, elapsed, iteration_counts, convergence)
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote profile metadata to {args.metadata_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
