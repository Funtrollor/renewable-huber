"""Lifecycle calibration and the measurement discipline itself.

The rules encoded here are the comparison contract described in
``docs/native-performance-policy.md``: what a sample is, what is inside the
timed interval, and what a cold or steady lifecycle promises. Changing
anything in this module invalidates existing schema-v2 captures even when the
numbers still look reasonable, so treat every constant and every ordering as
part of the record.
"""

from __future__ import annotations

import gc
import math
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

# The benchmarks run straight from a source checkout, so ``src`` has to be on
# the path before ``renewable_huber`` is imported. An import sorter would move
# a relative helper import below the third-party block, which is why the two
# lines are written out here instead of being shared.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from renewable_huber import RenewableHuberRegressor  # noqa: E402
from renewable_huber.state import RenewableHuberState  # noqa: E402


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
