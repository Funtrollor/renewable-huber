"""Reproducible NumPy/CuPy end-to-end renewable-Huber benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from renewable_huber import BackendUnavailableError, RenewableHuberRegressor  # noqa: E402

Batch = tuple[Any, Any]


def make_batches(
    n_samples: int,
    n_features: int,
    batch_size: int,
    seed: int,
    dtype: str,
) -> list[Batch]:
    """Generate deterministic, already-cast host batches."""

    rng = np.random.default_rng(seed)
    numpy_dtype = np.dtype(dtype)
    X = rng.normal(size=(n_samples, n_features)).astype(numpy_dtype)
    coefficients = rng.normal(size=n_features).astype(numpy_dtype)
    noise = rng.normal(scale=0.2, size=n_samples).astype(numpy_dtype)
    y = X @ coefficients + noise
    return [
        (X[start : start + batch_size], y[start : start + batch_size])
        for start in range(0, n_samples, batch_size)
    ]


def _fit_all(model: RenewableHuberRegressor, batches: list[Batch]) -> None:
    for X_batch, y_batch in batches:
        model.partial_fit(X_batch, y_batch)


def _measure(operation: Callable[[], None], repeats: int) -> dict[str, Any]:
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        operation()
        samples.append(perf_counter() - start)
    return {
        "seconds": samples,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
    }


def _add_throughput(result: dict[str, Any], n_samples: int) -> dict[str, Any]:
    result["median_samples_per_second"] = n_samples / result["median_seconds"]
    return result


def benchmark_cpu(batches: list[Batch], dtype: str, repeats: int, n_samples: int) -> dict[str, Any]:
    """Measure steady-state NumPy fits with data already at the requested dtype."""

    _fit_all(RenewableHuberRegressor(dtype=dtype), batches[:1])

    def operation() -> None:
        _fit_all(RenewableHuberRegressor(dtype=dtype), batches)

    return _add_throughput(_measure(operation, repeats), n_samples)


def benchmark_gpu(
    batches: list[Batch], dtype: str, repeats: int, n_samples: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure host-fed and device-resident CuPy paths independently."""

    import cupy as cp

    stream = cp.cuda.get_current_stream()
    warmup = RenewableHuberRegressor(backend="cupy", device="cuda", dtype=dtype)
    _fit_all(warmup, batches[:1])
    stream.synchronize()

    device_batches = [(cp.asarray(X), cp.asarray(y)) for X, y in batches]
    stream.synchronize()

    def host_operation() -> None:
        model = RenewableHuberRegressor(backend="cupy", device="cuda", dtype=dtype)
        _fit_all(model, batches)
        stream.synchronize()

    def device_operation() -> None:
        model = RenewableHuberRegressor(backend="cupy", device="cuda", dtype=dtype)
        _fit_all(model, device_batches)
        stream.synchronize()

    host_result = _add_throughput(_measure(host_operation, repeats), n_samples)
    device_result = _add_throughput(_measure(device_operation, repeats), n_samples)
    host_result["transfer_and_conversion_overhead_seconds"] = max(
        0.0, host_result["median_seconds"] - device_result["median_seconds"]
    )
    return host_result, device_result


def environment_metadata() -> dict[str, Any]:
    """Return enough runtime metadata to compare benchmark records responsibly."""

    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
    }
    try:
        import cupy as cp

        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        metadata["cupy"] = cp.__version__
        metadata["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
        metadata["gpu"] = name
    except Exception as error:
        metadata["gpu_unavailable"] = str(error)
    return metadata


def _print_result(name: str, result: dict[str, Any]) -> None:
    print(
        f"{name}: {result['median_seconds']:.4f}s median "
        f"({result['median_samples_per_second']:,.0f} samples/s, "
        f"best {result['minimum_seconds']:.4f}s)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=32_768)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    args = parser.parse_args()
    if args.samples < 1 or args.features < 1 or args.batch_size < 1 or args.repeats < 1:
        parser.error("samples, features, batch-size, and repeats must be positive")

    batches = make_batches(
        args.samples,
        args.features,
        args.batch_size,
        args.seed,
        args.dtype,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "configuration": {
            "samples": args.samples,
            "features": args.features,
            "batch_size": args.batch_size,
            "batches": len(batches),
            "repeats": args.repeats,
            "seed": args.seed,
            "dtype": args.dtype,
        },
        "environment": environment_metadata(),
        "numpy_cpu": benchmark_cpu(batches, args.dtype, args.repeats, args.samples),
    }
    _print_result("NumPy CPU", result["numpy_cpu"])

    try:
        gpu_host, gpu_device = benchmark_gpu(batches, args.dtype, args.repeats, args.samples)
    except (BackendUnavailableError, ImportError) as error:
        result["cupy_cuda_unavailable"] = str(error)
        print(f"CuPy CUDA benchmark unavailable: {error}")
    else:
        result["cupy_cuda_host_input"] = gpu_host
        result["cupy_cuda_device_input"] = gpu_device
        _print_result("CuPy CUDA (host input)", gpu_host)
        _print_result("CuPy CUDA (device input)", gpu_device)
        print(
            "Device-resident speed-up: "
            f"{result['numpy_cpu']['median_seconds'] / gpu_device['median_seconds']:.2f}x"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Wrote benchmark record to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
