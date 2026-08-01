"""Run reproducible shape sweeps across NumPy and native CPU/CUDA engines."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import get_clock_info, perf_counter
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

_THREAD_ENVIRONMENT_KEYS = (
    "MATMUL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


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

    timer = get_clock_info("perf_counter")
    metadata: dict[str, Any] = {
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "numpy": np.__version__,
        "threading": {key: os.environ.get(key, "unset") for key in _THREAD_ENVIRONMENT_KEYS},
        "numpy_blas_provider": _numpy_blas_provider(),
        "perf_counter": {
            "implementation": timer.implementation,
            "resolution_seconds": timer.resolution,
            "monotonic": timer.monotonic,
            "adjustable": timer.adjustable,
        },
    }
    try:
        from renewable_huber import _native_cpu

        metadata["native_cpu"] = dict(_native_cpu.version())
    except Exception as error:
        metadata["native_cpu_unavailable"] = str(error)
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


def _numpy_blas_provider() -> dict[str, str]:
    """Return the BLAS/LAPACK identity without embedding build-machine paths."""

    config = getattr(np.__config__, "CONFIG", {})
    dependencies = config.get("Build Dependencies", {}) if isinstance(config, dict) else {}
    provider: dict[str, str] = {}
    for library in ("blas", "lapack"):
        details = dependencies.get(library, {})
        if not isinstance(details, dict):
            provider[library] = "unknown"
            continue
        provider[library] = (
            " | ".join(
                str(details[key])
                for key in ("name", "version", "openblas configuration")
                if details.get(key)
            )
            or "unknown"
        )
    return provider


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


def _dataset_checksum(batches: list[tuple[np.ndarray, np.ndarray]]) -> str:
    """Fingerprint actual generated values, not only their seed and shape."""

    digest = hashlib.sha256()
    for X_batch, y_batch in batches:
        for array in (X_batch, y_batch):
            contiguous = np.ascontiguousarray(array)
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(repr(contiguous.shape).encode("ascii"))
            digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _new_model(
    *,
    backend: str,
    device: str,
    dtype: str,
    penalty: str,
    max_iter: int,
    tol: float,
) -> RenewableHuberRegressor:
    return RenewableHuberRegressor(
        backend=backend,
        device=device,
        dtype=dtype,
        penalty=penalty,
        max_iter=max_iter,
        tol=tol,
    )


def _fit_batch(batches: list[tuple[Any, Any]], *, xp: Any) -> tuple[Any, Any]:
    """Combine a stream outside a measured ``fit`` interval."""

    if len(batches) == 1:
        return batches[0]
    return (
        xp.concatenate([X_batch for X_batch, _ in batches], axis=0),
        xp.concatenate([y_batch for _, y_batch in batches], axis=0),
    )


def _run_operation(
    model: RenewableHuberRegressor,
    batches: list[tuple[Any, Any]],
    fit_batch: tuple[Any, Any],
    *,
    operation: str,
) -> tuple[int, bool]:
    """Run one public API workload and return its diagnostics summary."""

    if operation == "fit":
        model.fit(*fit_batch)
        diagnostics = model.diagnostics_
        return diagnostics.iterations, diagnostics.converged
    if operation != "partial_fit":
        raise ValueError(f"unknown benchmark operation {operation!r}")
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
    finalize: Any | None = None,
    estimated_operation_seconds: float | None = None,
    minimum_sample_seconds: float = 0.0,
    max_sample_repetitions: int = 64,
) -> dict[str, Any]:
    """Measure per-operation time using independent, fixed-size sample blocks.

    A single cold GPU call can be shorter than a Windows/WDDM scheduling
    quantum.  Treating nine such calls as nine statistical samples makes the
    scheduler, DPCs, and timer-call overhead dominate relative MAD.  Each
    reported duration is therefore the arithmetic mean of a fixed number of
    *independent* public operations.  Cold operations still construct a fresh
    estimator/engine every time, and no observation is trimmed or discarded.

    ``estimated_operation_seconds`` comes from the explicit warmup/calibration
    run outside the measured samples.  It only selects the fixed block size;
    it is never included in the result.
    """

    sample_repetitions = _sample_repetitions(
        estimated_operation_seconds,
        minimum_sample_seconds=minimum_sample_seconds,
        maximum=max_sample_repetitions,
    )
    seconds = []
    iterations = []
    convergence = []
    for _ in range(repeats):
        # Cyclic GC is unrelated to the public solver operation and can pause a
        # short sample unpredictably.  Collect before each statistical sample,
        # then disable cyclic GC while timing.  CPython reference-counted
        # destruction still happens normally in ``finalize``.
        gc.collect()
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        sample_seconds = 0.0
        sample_iterations: list[int] = []
        sample_convergence: list[bool] = []
        try:
            for _sample_run in range(sample_repetitions):
                if prepare is not None:
                    prepare()
                if synchronize is not None:
                    synchronize()
                start = perf_counter()
                iteration_count, converged = operation()
                if synchronize is not None:
                    synchronize()
                sample_seconds += perf_counter() - start
                # Releasing a fitted estimator is a different lifecycle
                # operation from fit/partial_fit. Keep destruction outside
                # every timed interval for all engines.
                if finalize is not None:
                    finalize()
                sample_iterations.append(iteration_count)
                sample_convergence.append(converged)
        finally:
            if gc_was_enabled:
                gc.enable()
        seconds.append(sample_seconds / sample_repetitions)
        iterations.append(statistics.median(sample_iterations))
        convergence.append(all(sample_convergence))
    median_seconds = statistics.median(seconds)
    return {
        "seconds": seconds,
        "median_seconds": median_seconds,
        "minimum_seconds": min(seconds),
        "maximum_seconds": max(seconds),
        "iterations": iterations,
        "median_iterations": statistics.median(iterations),
        "all_batches_converged": all(convergence),
        "timer": "time.perf_counter",
        "sample_aggregation": "arithmetic_mean",
        "sample_repetitions": sample_repetitions,
        "minimum_sample_seconds": minimum_sample_seconds,
        "gc_collected_before_sample": True,
        "gc_disabled_during_timing": True,
    }


def _sample_repetitions(
    estimated_operation_seconds: float | None,
    *,
    minimum_sample_seconds: float,
    maximum: int,
) -> int:
    """Choose a bounded fixed block size without looking at measured samples."""

    if minimum_sample_seconds <= 0 or estimated_operation_seconds is None:
        return 1
    if not math.isfinite(estimated_operation_seconds) or estimated_operation_seconds <= 0:
        return 1
    return min(maximum, max(1, math.ceil(minimum_sample_seconds / estimated_operation_seconds)))


def _calibration_run(
    operation: Any,
    *,
    synchronize: Any | None,
    prepare: Any | None = None,
    finalize: Any | None = None,
) -> float:
    """Time one unreported run used only to select the sampling block size."""

    if prepare is not None:
        prepare()
    if synchronize is not None:
        synchronize()
    start = perf_counter()
    operation()
    if synchronize is not None:
        synchronize()
    elapsed = perf_counter() - start
    if finalize is not None:
        finalize()
    return elapsed


def _restore_empty_state(model: RenewableHuberRegressor, empty_state: RenewableHuberState) -> None:
    """Reset a primed benchmark model without rebuilding its backend object."""

    # Use one fresh snapshot for both the estimator and an optional resident
    # native engine.  Restoring ``empty_state`` first and then assigning a copy
    # would give the two mirrors different process-local tokens, forcing the
    # backend to repeat the p^2 restore inside the measured operation.
    reset_state = empty_state.copy()
    backend = model._backend
    restore_native_state = getattr(backend, "restore_native_state", None)
    if callable(restore_native_state):
        restore_native_state(reset_state)
    model._state = reset_state
    model._diagnostics = None
    model.n_features_in_ = empty_state.n_features_in
    model.n_samples_seen_ = 0
    model.n_iter_ = 0
    model._sync_public_coefficients()


def _lifecycle_metadata(
    *,
    lifecycle: str,
    operation: str,
    input_location: str,
    includes_input_transfer: bool,
    batch_count: int,
) -> dict[str, Any]:
    if lifecycle == "cold":
        return {
            "lifecycle": lifecycle,
            "operation": operation,
            "input_location": input_location,
            "includes_input_transfer": includes_input_transfer,
            "includes_engine_initialization": True,
            "includes_engine_destruction": False,
            "resident_engine": False,
            "engine_prime_runs": 0,
            "repeat_state_reset": "new_estimator_inside_timing",
            "state_reset_timed": True,
            "model_reused": False,
            "device_preload_timed": False,
            "workload_batch_count": batch_count,
        }
    if lifecycle == "steady":
        return {
            "lifecycle": lifecycle,
            "operation": operation,
            "input_location": input_location,
            "includes_input_transfer": includes_input_transfer,
            "includes_engine_initialization": False,
            "includes_engine_destruction": False,
            "resident_engine": True,
            "engine_prime_runs": 1,
            "repeat_state_reset": "restore_empty_state_outside_timing",
            "state_reset_timed": False,
            "model_reused": True,
            "device_preload_timed": False,
            "workload_batch_count": batch_count,
        }
    raise ValueError(f"unknown benchmark lifecycle {lifecycle!r}")


def _benchmark_engine(
    batches: list[tuple[Any, Any]],
    *,
    xp: Any,
    backend: str,
    device: str,
    dtype: str,
    penalty: str,
    lifecycle: str,
    operation: str,
    input_location: str,
    includes_input_transfer: bool,
    warmup: int,
    repeats: int,
    max_iter: int,
    tol: float,
    synchronize: Any | None = None,
    minimum_sample_seconds: float = 0.0,
    max_sample_repetitions: int = 64,
) -> dict[str, Any]:
    """Measure one engine under an explicit cold or steady-state contract."""

    fit_batch = _fit_batch(batches, xp=xp)
    batch_count = 1 if operation == "fit" else len(batches)

    def new_model() -> RenewableHuberRegressor:
        return _new_model(
            backend=backend,
            device=device,
            dtype=dtype,
            penalty=penalty,
            max_iter=max_iter,
            tol=tol,
        )

    if lifecycle == "cold":
        cold_model: RenewableHuberRegressor | None = None

        def cold_operation() -> tuple[int, bool]:
            nonlocal cold_model
            cold_model = new_model()
            return _run_operation(cold_model, batches, fit_batch, operation=operation)

        def release_cold_model() -> None:
            nonlocal cold_model
            cold_model = None

        calibration_seconds: list[float] = []
        for _ in range(warmup):
            calibration_seconds.append(
                _calibration_run(
                    cold_operation,
                    synchronize=synchronize,
                    finalize=release_cold_model,
                )
            )
        if not calibration_seconds:
            calibration_seconds.append(
                _calibration_run(
                    cold_operation,
                    synchronize=synchronize,
                    finalize=release_cold_model,
                )
            )
        result = _measure(
            cold_operation,
            repeats=repeats,
            synchronize=synchronize,
            finalize=release_cold_model,
            estimated_operation_seconds=statistics.median(calibration_seconds),
            minimum_sample_seconds=minimum_sample_seconds,
            max_sample_repetitions=max_sample_repetitions,
        )
        result["sampling_calibration_runs"] = len(calibration_seconds)
    elif lifecycle == "steady":
        if operation != "partial_fit":
            raise ValueError("steady-state measurement is defined only for partial_fit")
        model = new_model()

        def steady_operation() -> tuple[int, bool]:
            return _run_operation(model, batches, fit_batch, operation=operation)

        # Prime global/library handles and maximum batch workspaces once.  The
        # portable empty state is restored before every warmup and repeat,
        # outside the measured region, for every engine (not just native CUDA).
        steady_operation()
        if synchronize is not None:
            synchronize()
        empty_state = RenewableHuberState.empty(
            batches[0][0].shape[1],
            fit_intercept=model.fit_intercept,
            xp=model._backend.xp,
            dtype=model._backend.dtype,
        )

        def prepare() -> None:
            _restore_empty_state(model, empty_state)

        calibration_seconds = []
        for _ in range(warmup):
            calibration_seconds.append(
                _calibration_run(
                    steady_operation,
                    synchronize=synchronize,
                    prepare=prepare,
                )
            )
        if not calibration_seconds:
            calibration_seconds.append(
                _calibration_run(
                    steady_operation,
                    synchronize=synchronize,
                    prepare=prepare,
                )
            )
        result = _measure(
            steady_operation,
            repeats=repeats,
            synchronize=synchronize,
            prepare=prepare,
            estimated_operation_seconds=statistics.median(calibration_seconds),
            minimum_sample_seconds=minimum_sample_seconds,
            max_sample_repetitions=max_sample_repetitions,
        )
        result["sampling_calibration_runs"] = len(calibration_seconds)
    else:
        raise ValueError(f"unknown benchmark lifecycle {lifecycle!r}")
    result.update(
        _lifecycle_metadata(
            lifecycle=lifecycle,
            operation=operation,
            input_location=input_location,
            includes_input_transfer=includes_input_transfer,
            batch_count=batch_count,
        )
    )
    return result


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


def _add_throughput(result: dict[str, Any], samples: int) -> None:
    result["median_samples_per_second"] = samples / result["median_seconds"]


def _print_result(case: dict[str, Any]) -> None:
    result = case["result"]
    print(
        f"{case['shape']['name']} {case['penalty']} {case['dtype']} "
        f"{result['lifecycle']}/{result['operation']} {case['engine']}: "
        f"{result['median_seconds']:.4f}s, "
        f"{result['median_samples_per_second']:,.0f} samples/s"
    )


def _record_skip(
    record: dict[str, Any],
    base: dict[str, Any],
    *,
    engine: str,
    lifecycle: str,
    operation: str,
    input_location: str,
    reason: str,
) -> None:
    record.setdefault("skipped", []).append(
        {
            **base,
            "engine": engine,
            "lifecycle": lifecycle,
            "operation": operation,
            "input_location": input_location,
            "reason": reason,
        }
    )
    print(
        f"Skipped {base['shape']['name']} {base['penalty']} {base['dtype']} "
        f"{lifecycle}/{operation} {engine}: {reason}"
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
        choices=(
            "numpy",
            "native-cpu",
            "cpu",
            "cupy",
            "gpu",
            "native_cuda",
            "both",
            "all",
        ),
        default="both",
    )
    parser.add_argument("--penalty", choices=("none", "l1", "both"), default="both")
    parser.add_argument("--dtype", choices=("float32", "float64", "both"), default="both")
    parser.add_argument(
        "--lifecycle",
        choices=("cold", "steady", "both"),
        default="cold",
        help=(
            "Cold includes a new estimator/native engine in every repeat; steady reuses a "
            "primed model and restores empty state outside timing."
        ),
    )
    parser.add_argument(
        "--operation",
        choices=("fit", "partial-fit", "both"),
        default="partial-fit",
        help="Measure public fit (one full batch) or streaming partial_fit calls.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--minimum-sample-seconds",
        type=float,
        default=0.1,
        help=(
            "Target aggregate timed work per statistical sample. Short operations are "
            "repeated independently and reported as a per-operation arithmetic mean."
        ),
    )
    parser.add_argument(
        "--max-sample-repetitions",
        type=int,
        default=64,
        help="Safety cap for independent operations aggregated into one timing sample.",
    )
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-host-memory-mib", type=float, default=2048.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.warmup < 0
        or args.repeats < 1
        or args.max_iter < 1
        or args.minimum_sample_seconds < 0
        or args.max_sample_repetitions < 1
    ):
        parser.error(
            "warmup and minimum-sample-seconds must be non-negative; repeats, "
            "max-iter, and max-sample-repetitions must be positive"
        )

    shapes = list(PROFILES[args.profile])
    if args.case:
        requested = set(args.case)
        known = {shape.name for shape in shapes}
        if unknown := requested - known:
            parser.error(f"unknown case(s) for {args.profile}: {', '.join(sorted(unknown))}")
        shapes = [shape for shape in shapes if shape.name in requested]
    penalties = ("none", "l1") if args.penalty == "both" else (args.penalty,)
    dtypes = ("float32", "float64") if args.dtype == "both" else (args.dtype,)
    lifecycles = ("cold", "steady") if args.lifecycle == "both" else (args.lifecycle,)
    operations = (
        ("fit", "partial_fit") if args.operation == "both" else (args.operation.replace("-", "_"),)
    )
    run_numpy = args.backend in ("numpy", "cpu", "both", "all")
    run_native_cpu = args.backend in ("native-cpu", "cpu", "all")
    run_cupy = args.backend in ("cupy", "gpu", "both", "all")
    run_native_cuda = args.backend in ("native_cuda", "gpu", "all")

    record: dict[str, Any] = {
        "schema": "renewable-huber-shape-sweep",
        "schema_version": 2,
        "profile": args.profile,
        "environment": environment_metadata(),
        "arguments": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "seed": args.seed,
            "lifecycle": list(lifecycles),
            "operation": list(operations),
            "minimum_sample_seconds": args.minimum_sample_seconds,
            "max_sample_repetitions": args.max_sample_repetitions,
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
            dataset_sha256 = _dataset_checksum(batches)
            for penalty in penalties:
                base = {
                    "shape": asdict(shape),
                    "dataset_seed": shape_seed,
                    "dataset_sha256": dataset_sha256,
                    "dtype": dtype,
                    "penalty": penalty,
                    "max_iter": args.max_iter,
                    "tol": args.tol,
                    "estimated_host_data_mib": estimated_mib,
                }
                for lifecycle in lifecycles:
                    for operation in operations:
                        # ``fit`` concatenates the generated stream before the
                        # timed call, so its real public-API batch contains all
                        # samples. Keep the comparison key honest instead of
                        # retaining the generation chunk size used by
                        # ``partial_fit``.
                        case_base = (
                            base
                            if operation == "partial_fit"
                            else {
                                **base,
                                "shape": {
                                    **base["shape"],
                                    "batch_size": shape.samples,
                                },
                            }
                        )
                        # ``fit`` deliberately calls reset(), so a resident-engine
                        # steady-state variant would not describe the public API.
                        if lifecycle == "steady" and operation == "fit":
                            if run_numpy:
                                _record_skip(
                                    record,
                                    case_base,
                                    engine="numpy_cpu",
                                    lifecycle=lifecycle,
                                    operation=operation,
                                    input_location="host",
                                    reason=(
                                        "fit resets the estimator; steady state is partial_fit only"
                                    ),
                                )
                            if run_native_cpu:
                                _record_skip(
                                    record,
                                    case_base,
                                    engine="rust_native_cpu",
                                    lifecycle=lifecycle,
                                    operation=operation,
                                    input_location="host",
                                    reason=(
                                        "fit resets the estimator; steady state is partial_fit only"
                                    ),
                                )
                            if run_cupy:
                                for engine, input_location in (
                                    ("cupy_cuda_host_input", "host"),
                                    ("cupy_cuda_device_input", "device"),
                                ):
                                    _record_skip(
                                        record,
                                        case_base,
                                        engine=engine,
                                        lifecycle=lifecycle,
                                        operation=operation,
                                        input_location=input_location,
                                        reason=(
                                            "fit resets the estimator; "
                                            "steady state is partial_fit only"
                                        ),
                                    )
                            if run_native_cuda:
                                _record_skip(
                                    record,
                                    case_base,
                                    engine="native_cuda_host_input",
                                    lifecycle=lifecycle,
                                    operation=operation,
                                    input_location="host",
                                    reason=(
                                        "fit resets the estimator; steady state is partial_fit only"
                                    ),
                                )
                            continue
                        if run_numpy:
                            result = benchmark_numpy(
                                batches,
                                dtype=dtype,
                                penalty=penalty,
                                lifecycle=lifecycle,
                                operation=operation,
                                warmup=args.warmup,
                                repeats=args.repeats,
                                max_iter=args.max_iter,
                                tol=args.tol,
                                minimum_sample_seconds=args.minimum_sample_seconds,
                                max_sample_repetitions=args.max_sample_repetitions,
                            )
                            _add_throughput(result, shape.samples)
                            case = {**case_base, "engine": "numpy_cpu", "result": result}
                            record["cases"].append(case)
                            _print_result(case)
                        if run_native_cpu:
                            try:
                                result = benchmark_native_cpu(
                                    batches,
                                    dtype=dtype,
                                    penalty=penalty,
                                    lifecycle=lifecycle,
                                    operation=operation,
                                    warmup=args.warmup,
                                    repeats=args.repeats,
                                    max_iter=args.max_iter,
                                    tol=args.tol,
                                    minimum_sample_seconds=args.minimum_sample_seconds,
                                    max_sample_repetitions=args.max_sample_repetitions,
                                )
                            except (BackendUnavailableError, ImportError) as error:
                                record.setdefault("unavailable", {})["native_cpu"] = str(error)
                                print(f"Native CPU unavailable: {error}")
                                run_native_cpu = False
                            else:
                                _add_throughput(result, shape.samples)
                                case = {
                                    **case_base,
                                    "engine": "rust_native_cpu",
                                    "result": result,
                                }
                                record["cases"].append(case)
                                _print_result(case)
                        if run_cupy:
                            try:
                                host_result, device_result = benchmark_cupy(
                                    batches,
                                    dtype=dtype,
                                    penalty=penalty,
                                    lifecycle=lifecycle,
                                    operation=operation,
                                    warmup=args.warmup,
                                    repeats=args.repeats,
                                    max_iter=args.max_iter,
                                    tol=args.tol,
                                    minimum_sample_seconds=args.minimum_sample_seconds,
                                    max_sample_repetitions=args.max_sample_repetitions,
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
                                    case = {**case_base, "engine": engine, "result": result}
                                    record["cases"].append(case)
                                    _print_result(case)
                        if run_native_cuda:
                            if penalty != "none":
                                _record_skip(
                                    record,
                                    case_base,
                                    engine="native_cuda_host_input",
                                    lifecycle=lifecycle,
                                    operation=operation,
                                    input_location="host",
                                    reason="P2 native CUDA supports penalty='none' only",
                                )
                            else:
                                try:
                                    host_result, device_result = benchmark_native_cuda(
                                        batches,
                                        dtype=dtype,
                                        lifecycle=lifecycle,
                                        operation=operation,
                                        warmup=args.warmup,
                                        repeats=args.repeats,
                                        max_iter=args.max_iter,
                                        tol=args.tol,
                                        minimum_sample_seconds=args.minimum_sample_seconds,
                                        max_sample_repetitions=args.max_sample_repetitions,
                                    )
                                except (BackendUnavailableError, ImportError, OSError) as error:
                                    record.setdefault("unavailable", {})["native_cuda"] = str(error)
                                    print(f"Native CUDA unavailable: {error}")
                                    run_native_cuda = False
                                else:
                                    for engine, result in (
                                        ("native_cuda_host_input", host_result),
                                        ("native_cuda_device_input", device_result),
                                    ):
                                        _add_throughput(result, shape.samples)
                                        case = {**case_base, "engine": engine, "result": result}
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
