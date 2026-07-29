"""Run reproducible native-core shape sweeps across NumPy and CUDA engines."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from renewable_huber import BackendUnavailableError, RenewableHuberRegressor  # noqa: E402
from renewable_huber.state import RenewableHuberState  # noqa: E402


@dataclass(frozen=True, slots=True)
class Shape:
    name: str
    samples: int
    features: int
    batch_size: int


PROFILES = {
    "smoke": (
        Shape("latency-smoke", 2_048, 8, 1_024),
        Shape("reference-smoke", 8_192, 32, 4_096),
    ),
    "standard": (
        Shape("latency", 4_096, 16, 4_096),
        Shape("reference", 100_000, 90, 32_768),
        Shape("wide", 16_384, 256, 4_096),
        Shape("streaming", 1_000_000, 32, 65_536),
    ),
}


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


def environment_metadata() -> dict[str, Any]:
    """Return comparison metadata without requiring a CUDA installation."""

    metadata: dict[str, Any] = {
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "numpy": np.__version__,
    }
    try:
        import cupy as cp

        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        metadata.update(
            {
                "cupy": cp.__version__,
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "gpu": name,
                "gpu_compute_capability": (f"{properties['major']}.{properties['minor']}"),
            }
        )
    except Exception as error:
        metadata["gpu_unavailable"] = str(error)
    try:
        from renewable_huber import _native_cuda

        metadata.update(
            {
                "native_cuda_abi": _native_cuda.version(),
                "native_cuda_available": bool(_native_cuda.is_available()),
            }
        )
    except (ImportError, OSError, RuntimeError) as error:
        metadata["native_cuda_unavailable"] = str(error)
    return metadata


def make_batches(
    shape: Shape,
    *,
    seed: int,
    dtype: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic host batches outside measured regions."""

    rng = np.random.default_rng(seed)
    array_dtype = np.dtype(dtype)
    X = rng.normal(size=(shape.samples, shape.features)).astype(array_dtype)
    coefficients = rng.normal(size=shape.features).astype(array_dtype)
    y = X @ coefficients + rng.normal(scale=0.2, size=shape.samples).astype(array_dtype)
    if shape.samples >= 100:
        outlier_rows = np.arange(0, shape.samples, max(100, shape.samples // 100))
        y[outlier_rows] += rng.normal(scale=8.0, size=outlier_rows.size).astype(array_dtype)
    return [
        (X[start : start + shape.batch_size], y[start : start + shape.batch_size])
        for start in range(0, shape.samples, shape.batch_size)
    ]


def _fit_batches(
    batches: list[tuple[Any, Any]],
    *,
    backend: str,
    device: str,
    dtype: str,
    penalty: str,
    max_iter: int,
    tol: float,
) -> tuple[int, bool]:
    model = RenewableHuberRegressor(
        backend=backend,
        device=device,
        dtype=dtype,
        penalty=penalty,
        max_iter=max_iter,
        tol=tol,
    )
    iterations = 0
    converged = True
    for X_batch, y_batch in batches:
        model.partial_fit(X_batch, y_batch)
        iterations += model.diagnostics_.iterations
        converged = converged and model.diagnostics_.converged
    return iterations, converged


def _measure(
    operation: Any,
    *,
    repeats: int,
    synchronize: Any | None = None,
    prepare: Any | None = None,
) -> dict[str, Any]:
    seconds = []
    iterations = []
    convergence = []
    for _ in range(repeats):
        if prepare is not None:
            prepare()
        if synchronize is not None:
            synchronize()
        start = perf_counter()
        iteration_count, converged = operation()
        if synchronize is not None:
            synchronize()
        seconds.append(perf_counter() - start)
        iterations.append(iteration_count)
        convergence.append(converged)
    median_seconds = statistics.median(seconds)
    return {
        "seconds": seconds,
        "median_seconds": median_seconds,
        "minimum_seconds": min(seconds),
        "maximum_seconds": max(seconds),
        "iterations": iterations,
        "median_iterations": statistics.median(iterations),
        "all_batches_converged": all(convergence),
    }


def benchmark_numpy(
    batches: list[tuple[np.ndarray, np.ndarray]],
    *,
    dtype: str,
    penalty: str,
    warmup: int,
    repeats: int,
    max_iter: int,
    tol: float,
) -> dict[str, Any]:
    def operation() -> tuple[int, bool]:
        return _fit_batches(
            batches,
            backend="numpy",
            device="cpu",
            dtype=dtype,
            penalty=penalty,
            max_iter=max_iter,
            tol=tol,
        )

    for _ in range(warmup):
        operation()
    result = _measure(operation, repeats=repeats)
    result.update(
        {
            "input_location": "host",
            "includes_input_transfer": False,
            "includes_engine_initialization": True,
            "resident_engine": False,
            "engine_prime_runs": 0,
            "repeat_state_reset": "new_estimator",
        }
    )
    return result


def benchmark_cupy(
    batches: list[tuple[np.ndarray, np.ndarray]],
    *,
    dtype: str,
    penalty: str,
    warmup: int,
    repeats: int,
    max_iter: int,
    tol: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import cupy as cp

    stream = cp.cuda.get_current_stream()
    device_batches = [(cp.asarray(X), cp.asarray(y)) for X, y in batches]
    stream.synchronize()

    def host_operation() -> tuple[int, bool]:
        return _fit_batches(
            batches,
            backend="cupy",
            device="cuda",
            dtype=dtype,
            penalty=penalty,
            max_iter=max_iter,
            tol=tol,
        )

    def device_operation() -> tuple[int, bool]:
        return _fit_batches(
            device_batches,
            backend="cupy",
            device="cuda",
            dtype=dtype,
            penalty=penalty,
            max_iter=max_iter,
            tol=tol,
        )

    for _ in range(warmup):
        host_operation()
        device_operation()
    stream.synchronize()

    host_result = _measure(host_operation, repeats=repeats, synchronize=stream.synchronize)
    host_result.update(
        {
            "input_location": "host",
            "includes_input_transfer": True,
            "includes_engine_initialization": True,
            "resident_engine": False,
            "engine_prime_runs": 0,
            "repeat_state_reset": "new_estimator",
        }
    )
    device_result = _measure(device_operation, repeats=repeats, synchronize=stream.synchronize)
    device_result.update(
        {
            "input_location": "device",
            "includes_input_transfer": False,
            "includes_engine_initialization": True,
            "resident_engine": False,
            "engine_prime_runs": 0,
            "repeat_state_reset": "new_estimator",
        }
    )
    host_result["transfer_and_conversion_overhead_seconds"] = max(
        0.0, host_result["median_seconds"] - device_result["median_seconds"]
    )
    return host_result, device_result


def benchmark_native_cuda(
    batches: list[tuple[np.ndarray, np.ndarray]],
    *,
    dtype: str,
    warmup: int,
    repeats: int,
    max_iter: int,
    tol: float,
) -> dict[str, Any]:
    """Measure the host-fed whole-batch solver with state resident on CUDA."""

    model = RenewableHuberRegressor(
        backend="native_cuda",
        device="cuda",
        dtype=dtype,
        penalty="none",
        max_iter=max_iter,
        tol=tol,
    )

    def operation() -> tuple[int, bool]:
        iterations = 0
        converged = True
        for X_batch, y_batch in batches:
            model.partial_fit(X_batch, y_batch)
            iterations += model.diagnostics_.iterations
            converged = converged and model.diagnostics_.converged
        return iterations, converged

    # Prime handles and maximum-batch workspaces once. Every timed repeat then
    # restores an empty portable state into the same opaque engine.
    operation()
    empty_state = RenewableHuberState.empty(
        batches[0][0].shape[1],
        fit_intercept=model.fit_intercept,
        xp=np,
        dtype=np.dtype(dtype),
    )

    def prepare() -> None:
        model._backend.restore_native_state(empty_state)
        model._state = empty_state.copy()
        model._diagnostics = None
        model.n_samples_seen_ = 0
        model.n_iter_ = 0
        model._sync_public_coefficients()

    for _ in range(warmup):
        prepare()
        operation()
    result = _measure(operation, repeats=repeats, prepare=prepare)
    result.update(
        {
            "input_location": "host",
            "includes_input_transfer": True,
            "includes_engine_initialization": False,
            "resident_engine": True,
            "engine_prime_runs": 1,
            "repeat_state_reset": "restore_empty_portable_state",
        }
    )
    return result


def _add_throughput(result: dict[str, Any], samples: int) -> None:
    result["median_samples_per_second"] = samples / result["median_seconds"]


def _print_result(case: dict[str, Any]) -> None:
    result = case["result"]
    print(
        f"{case['shape']['name']} {case['penalty']} {case['dtype']} "
        f"{case['engine']}: {result['median_seconds']:.4f}s, "
        f"{result['median_samples_per_second']:,.0f} samples/s"
    )


def _estimated_host_mib(shape: Shape, dtype: str) -> float:
    itemsize = np.dtype(dtype).itemsize
    values = shape.samples * shape.features + shape.samples + shape.features
    return values * itemsize / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--case", action="append", help="Run only a named shape; repeatable")
    parser.add_argument(
        "--backend",
        choices=("numpy", "cupy", "native_cuda", "both", "all"),
        default="both",
    )
    parser.add_argument("--penalty", choices=("none", "l1", "both"), default="both")
    parser.add_argument("--dtype", choices=("float32", "float64", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-host-memory-mib", type=float, default=2048.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.max_iter < 1:
        parser.error("warmup must be non-negative; repeats and max-iter must be positive")

    shapes = list(PROFILES[args.profile])
    if args.case:
        requested = set(args.case)
        known = {shape.name for shape in shapes}
        if unknown := requested - known:
            parser.error(f"unknown case(s) for {args.profile}: {', '.join(sorted(unknown))}")
        shapes = [shape for shape in shapes if shape.name in requested]
    penalties = ("none", "l1") if args.penalty == "both" else (args.penalty,)
    dtypes = ("float32", "float64") if args.dtype == "both" else (args.dtype,)
    run_numpy = args.backend in ("numpy", "both", "all")
    run_cupy = args.backend in ("cupy", "both", "all")
    run_native_cuda = args.backend in ("native_cuda", "all")

    record: dict[str, Any] = {
        "schema": "renewable-huber-shape-sweep",
        "schema_version": 1,
        "profile": args.profile,
        "environment": environment_metadata(),
        "arguments": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "seed": args.seed,
        },
        "cases": [],
    }

    profile_shape_indexes = {
        shape.name: index for index, shape in enumerate(PROFILES[args.profile])
    }
    for shape in shapes:
        for dtype in dtypes:
            estimated_mib = _estimated_host_mib(shape, dtype)
            if estimated_mib > args.max_host_memory_mib:
                parser.error(
                    f"{shape.name}/{dtype} needs about {estimated_mib:.1f} MiB host memory; "
                    "raise --max-host-memory-mib explicitly to proceed"
                )
            shape_seed = args.seed + profile_shape_indexes[shape.name]
            batches = make_batches(shape, seed=shape_seed, dtype=dtype)
            for penalty in penalties:
                base = {
                    "shape": asdict(shape),
                    "dataset_seed": shape_seed,
                    "dtype": dtype,
                    "penalty": penalty,
                    "estimated_host_data_mib": estimated_mib,
                }
                if run_numpy:
                    result = benchmark_numpy(
                        batches,
                        dtype=dtype,
                        penalty=penalty,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        max_iter=args.max_iter,
                        tol=args.tol,
                    )
                    _add_throughput(result, shape.samples)
                    case = {**base, "engine": "numpy_cpu", "result": result}
                    record["cases"].append(case)
                    _print_result(case)
                if run_cupy:
                    try:
                        host_result, device_result = benchmark_cupy(
                            batches,
                            dtype=dtype,
                            penalty=penalty,
                            warmup=args.warmup,
                            repeats=args.repeats,
                            max_iter=args.max_iter,
                            tol=args.tol,
                        )
                    except (BackendUnavailableError, ImportError) as error:
                        record.setdefault("unavailable", {})["cupy_cuda"] = str(error)
                        print(f"CuPy CUDA unavailable: {error}")
                        run_cupy = False
                    else:
                        for engine, result in (
                            ("cupy_cuda_host_input", host_result),
                            ("cupy_cuda_device_input", device_result),
                        ):
                            _add_throughput(result, shape.samples)
                            case = {**base, "engine": engine, "result": result}
                            record["cases"].append(case)
                            _print_result(case)
                if run_native_cuda and penalty == "none":
                    try:
                        result = benchmark_native_cuda(
                            batches,
                            dtype=dtype,
                            warmup=args.warmup,
                            repeats=args.repeats,
                            max_iter=args.max_iter,
                            tol=args.tol,
                        )
                    except (BackendUnavailableError, ImportError, OSError) as error:
                        record.setdefault("unavailable", {})["native_cuda"] = str(error)
                        print(f"Native CUDA unavailable: {error}")
                        run_native_cuda = False
                    else:
                        _add_throughput(result, shape.samples)
                        case = {**base, "engine": "native_cuda_host_input", "result": result}
                        record["cases"].append(case)
                        _print_result(case)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(record['cases'])} measurements to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
