"""One runner per engine, each fixing the transport contract it measures.

The engine-specific part of a sweep is small and consists almost entirely of
*which* comparison key a result carries: backend, device, input location and
whether host-to-device transfer is inside the timed interval. Keeping the four
runners side by side is what makes an accidental mismatch visible, because a
host record and a device record are never interchangeable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.benchmarks.shape_sweep.timing import _benchmark_engine


def benchmark_numpy(
    batches: list[tuple[np.ndarray, np.ndarray]],
    *,
    dtype: str,
    penalty: str,
    lifecycle: str,
    operation: str,
    warmup: int,
    repeats: int,
    max_iter: int,
    tol: float,
    minimum_sample_seconds: float = 0.0,
    max_sample_repetitions: int = 64,
) -> dict[str, Any]:
    return _benchmark_engine(
        batches,
        xp=np,
        backend="numpy",
        device="cpu",
        dtype=dtype,
        penalty=penalty,
        lifecycle=lifecycle,
        operation=operation,
        input_location="host",
        includes_input_transfer=False,
        warmup=warmup,
        repeats=repeats,
        max_iter=max_iter,
        tol=tol,
        minimum_sample_seconds=minimum_sample_seconds,
        max_sample_repetitions=max_sample_repetitions,
    )


def benchmark_native_cpu(
    batches: list[tuple[np.ndarray, np.ndarray]],
    *,
    dtype: str,
    penalty: str,
    lifecycle: str,
    operation: str,
    warmup: int,
    repeats: int,
    max_iter: int,
    tol: float,
    minimum_sample_seconds: float = 0.0,
    max_sample_repetitions: int = 64,
) -> dict[str, Any]:
    """Measure the opt-in whole-batch Rust CPU engine."""

    return _benchmark_engine(
        batches,
        xp=np,
        backend="native_cpu",
        device="cpu",
        dtype=dtype,
        penalty=penalty,
        lifecycle=lifecycle,
        operation=operation,
        input_location="host",
        includes_input_transfer=False,
        warmup=warmup,
        repeats=repeats,
        max_iter=max_iter,
        tol=tol,
        minimum_sample_seconds=minimum_sample_seconds,
        max_sample_repetitions=max_sample_repetitions,
    )


def benchmark_cupy(
    batches: list[tuple[np.ndarray, np.ndarray]],
    *,
    dtype: str,
    penalty: str,
    lifecycle: str,
    operation: str,
    warmup: int,
    repeats: int,
    max_iter: int,
    tol: float,
    minimum_sample_seconds: float = 0.0,
    max_sample_repetitions: int = 64,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import cupy as cp

    stream = cp.cuda.get_current_stream()
    device_batches = [(cp.asarray(X), cp.asarray(y)) for X, y in batches]
    stream.synchronize()
    host_result = _benchmark_engine(
        batches,
        xp=np,
        backend="cupy",
        device="cuda",
        dtype=dtype,
        penalty=penalty,
        lifecycle=lifecycle,
        operation=operation,
        input_location="host",
        includes_input_transfer=True,
        warmup=warmup,
        repeats=repeats,
        max_iter=max_iter,
        tol=tol,
        synchronize=stream.synchronize,
        minimum_sample_seconds=minimum_sample_seconds,
        max_sample_repetitions=max_sample_repetitions,
    )
    device_result = _benchmark_engine(
        device_batches,
        xp=cp,
        backend="cupy",
        device="cuda",
        dtype=dtype,
        penalty=penalty,
        lifecycle=lifecycle,
        operation=operation,
        input_location="device",
        includes_input_transfer=False,
        warmup=warmup,
        repeats=repeats,
        max_iter=max_iter,
        tol=tol,
        synchronize=stream.synchronize,
        minimum_sample_seconds=minimum_sample_seconds,
        max_sample_repetitions=max_sample_repetitions,
    )
    host_result["transfer_and_conversion_overhead_seconds"] = max(
        0.0, host_result["median_seconds"] - device_result["median_seconds"]
    )
    return host_result, device_result


def benchmark_native_cuda(
    batches: list[tuple[np.ndarray, np.ndarray]],
    *,
    dtype: str,
    lifecycle: str,
    operation: str,
    warmup: int,
    repeats: int,
    max_iter: int,
    tol: float,
    minimum_sample_seconds: float = 0.0,
    max_sample_repetitions: int = 64,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure native CUDA with equivalent host and DLPack device contracts."""

    import cupy as cp

    stream = cp.cuda.get_current_stream()
    device_batches = [(cp.asarray(X), cp.asarray(y)) for X, y in batches]
    stream.synchronize()
    host_result = _benchmark_engine(
        batches,
        xp=np,
        backend="native_cuda",
        device="cuda",
        dtype=dtype,
        penalty="none",
        lifecycle=lifecycle,
        operation=operation,
        input_location="host",
        includes_input_transfer=True,
        warmup=warmup,
        repeats=repeats,
        max_iter=max_iter,
        tol=tol,
        minimum_sample_seconds=minimum_sample_seconds,
        max_sample_repetitions=max_sample_repetitions,
    )
    device_result = _benchmark_engine(
        device_batches,
        xp=cp,
        backend="native_cuda",
        device="cuda",
        dtype=dtype,
        penalty="none",
        lifecycle=lifecycle,
        operation=operation,
        input_location="device",
        includes_input_transfer=False,
        warmup=warmup,
        repeats=repeats,
        max_iter=max_iter,
        tol=tol,
        synchronize=stream.synchronize,
        minimum_sample_seconds=minimum_sample_seconds,
        max_sample_repetitions=max_sample_repetitions,
    )
    host_result["transfer_and_conversion_overhead_seconds"] = max(
        0.0, host_result["median_seconds"] - device_result["median_seconds"]
    )
    return host_result, device_result
