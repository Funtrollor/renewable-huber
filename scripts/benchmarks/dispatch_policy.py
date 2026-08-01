"""Recommend a native backend only when a matching calibration proves it wins.

This is deliberately an offline advisor, not a change to
``RenewableHuberRegressor(backend=\"auto\")``.  It makes promotion decisions
auditable while the native engines remain opt-in.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

try:  # Supports both ``python script.py`` and package-style test imports.
    from .performance_policy import MEASUREMENT_SCHEMA_VERSION, measurement_key, validate_record
except ImportError:  # pragma: no cover - exercised by direct CLI invocation.
    from performance_policy import MEASUREMENT_SCHEMA_VERSION, measurement_key, validate_record


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    """Backends that the caller has already verified can be constructed."""

    cupy: bool = False
    native_cpu: bool = False
    native_cuda: bool = False


@dataclass(frozen=True, slots=True)
class Workload:
    """Runtime shape and measurement contract supplied to the advisor."""

    samples: int
    features: int
    batch_size: int
    dtype: str
    penalty: str
    input_location: str = "host"
    lifecycle: str = "steady"
    operation: str = "partial_fit"
    max_iter: int = 100
    tol: float = 1e-6


@dataclass(frozen=True, slots=True)
class DispatchDecision:
    """A deterministic recommendation with the evidence used to make it."""

    backend: str
    reason: str
    calibration_found: bool
    reference_backend: str | None
    reference_seconds: float | None
    native_seconds: float | None


def recommend_backend(
    record: Mapping[str, Any],
    workload: Workload,
    capabilities: RuntimeCapabilities,
    *,
    native_speedup_fraction: float = 0.10,
) -> DispatchDecision:
    """Return a conservative backend recommendation for one exact shape.

    Native CPU/CUDA is eligible only if a record with the same shape, dtype,
    penalty, operation, lifecycle, and input location shows it at least
    ``native_speedup_fraction`` faster than the fastest available reference
    engine.  Missing data, a changed contract, or an unsupported P2 transport
    falls back to a reference implementation rather than extrapolating.
    """

    _validate_workload(workload, native_speedup_fraction)
    validate_record(record)
    cases = _matching_cases(record, workload)
    _require_one_calibration_contract(cases)
    timings = {measurement_key(case).engine: _median_seconds(case) for case in cases}
    if workload.input_location == "device":
        return _recommend_device_input(timings, workload, capabilities, native_speedup_fraction)
    return _recommend_host_input(timings, workload, capabilities, native_speedup_fraction)


def _recommend_host_input(
    timings: Mapping[str, float],
    workload: Workload,
    capabilities: RuntimeCapabilities,
    native_speedup_fraction: float,
) -> DispatchDecision:
    references: list[tuple[str, float]] = []
    if "numpy_cpu" in timings:
        references.append(("numpy", timings["numpy_cpu"]))
    if capabilities.cupy and "cupy_cuda_host_input" in timings:
        references.append(("cupy", timings["cupy_cuda_host_input"]))
    if not references:
        return DispatchDecision(
            backend="numpy",
            reason="no exact host-input reference calibration; use the portable NumPy fallback",
            calibration_found=bool(timings),
            reference_backend=None,
            reference_seconds=None,
            native_seconds=None,
        )
    reference_backend, reference_seconds = min(references, key=lambda item: item[1])
    native_candidates: list[tuple[str, float]] = []
    if capabilities.native_cpu and "rust_native_cpu" in timings:
        native_candidates.append(("native_cpu", timings["rust_native_cpu"]))
    # P2 only accepts host NumPy input and does not implement L1.  Do not let
    # a stale or synthetic calibration bypass either product restriction.
    if (
        capabilities.native_cuda
        and workload.penalty == "none"
        and "native_cuda_host_input" in timings
    ):
        native_candidates.append(("native_cuda", timings["native_cuda_host_input"]))
    qualifying = [
        candidate
        for candidate in native_candidates
        if candidate[1] <= reference_seconds * (1.0 - native_speedup_fraction)
    ]
    if qualifying:
        backend, native_seconds = min(qualifying, key=lambda item: item[1])
        return DispatchDecision(
            backend=backend,
            reason=(
                f"exact calibration shows {backend} is at least "
                f"{native_speedup_fraction:.0%} faster than {reference_backend}"
            ),
            calibration_found=True,
            reference_backend=reference_backend,
            reference_seconds=reference_seconds,
            native_seconds=native_seconds,
        )
    if native_candidates:
        fastest_native, native_seconds = min(native_candidates, key=lambda item: item[1])
        return DispatchDecision(
            backend=reference_backend,
            reason=(
                f"{fastest_native} does not clear the {native_speedup_fraction:.0%} "
                "native speedup margin; use the faster reference engine"
            ),
            calibration_found=True,
            reference_backend=reference_backend,
            reference_seconds=reference_seconds,
            native_seconds=native_seconds,
        )
    return DispatchDecision(
        backend=reference_backend,
        reason="no eligible calibrated native engine for this workload",
        calibration_found=True,
        reference_backend=reference_backend,
        reference_seconds=reference_seconds,
        native_seconds=None,
    )


def _recommend_device_input(
    timings: Mapping[str, float],
    workload: Workload,
    capabilities: RuntimeCapabilities,
    native_speedup_fraction: float,
) -> DispatchDecision:
    reference_seconds = timings.get("cupy_cuda_device_input") if capabilities.cupy else None
    native_seconds = (
        timings.get("native_cuda_device_input")
        if capabilities.native_cuda and workload.penalty == "none"
        else None
    )
    if native_seconds is not None and (
        reference_seconds is None
        or native_seconds <= reference_seconds * (1.0 - native_speedup_fraction)
    ):
        return DispatchDecision(
            backend="native_cuda",
            reason=(
                "exact DLPack calibration selects native_cuda"
                if reference_seconds is None
                else f"exact DLPack calibration shows native_cuda is at least "
                f"{native_speedup_fraction:.0%} faster than cupy"
            ),
            calibration_found=True,
            reference_backend="cupy" if reference_seconds is not None else None,
            reference_seconds=reference_seconds,
            native_seconds=native_seconds,
        )
    if reference_seconds is not None:
        return DispatchDecision(
            backend="cupy",
            reason=(
                "native_cuda DLPack does not clear the native speedup margin"
                if native_seconds is not None
                else "no exact native CUDA DLPack calibration; retain the calibrated CuPy path"
            ),
            calibration_found=True,
            reference_backend="cupy",
            reference_seconds=reference_seconds,
            native_seconds=native_seconds,
        )
    return DispatchDecision(
        backend="numpy",
        reason=(
            "no calibrated device-input backend is available; explicit host conversion is required"
        ),
        calibration_found=bool(timings),
        reference_backend=None,
        reference_seconds=None,
        native_seconds=None,
    )


def _matching_cases(record: Mapping[str, Any], workload: Workload) -> list[Mapping[str, Any]]:
    matches: list[Mapping[str, Any]] = []
    for case in record["cases"]:
        key = measurement_key(case)
        if (
            key.samples == workload.samples
            and key.features == workload.features
            and key.batch_size == workload.batch_size
            and key.dtype == workload.dtype
            and key.penalty == workload.penalty
            and key.input_location == workload.input_location
            and key.lifecycle == workload.lifecycle
            and key.operation == workload.operation
            and key.max_iter == workload.max_iter
            and key.tol == workload.tol
            and _matches_requested_timing_contract(key, workload)
            and case["result"]["all_batches_converged"] is True
        ):
            matches.append(case)
    return matches


def _require_one_calibration_contract(cases: list[Mapping[str, Any]]) -> None:
    """Refuse to mix engines measured on different data or benchmark labels."""

    contracts = {
        (
            key.shape_name,
            key.dataset_seed,
            key.dataset_sha256,
            key.max_iter,
            key.tol,
        )
        for key in map(measurement_key, cases)
    }
    if len(contracts) > 1:
        raise ValueError(
            "matching calibration cases contain multiple dataset or solver contracts; "
            "select a record with one coherent workload"
        )


def _matches_requested_timing_contract(key: Any, workload: Workload) -> bool:
    """Make lifecycle/transfer parity explicit at the dispatch boundary."""

    expected_lifecycle = (
        (True, False, True) if workload.lifecycle == "cold" else (False, True, False)
    )
    observed_lifecycle = (
        key.includes_engine_initialization,
        key.resident_engine,
        key.state_reset_timed,
    )
    if observed_lifecycle != expected_lifecycle:
        return False
    expected_transfer = {
        "numpy_cpu": False,
        "rust_native_cpu": False,
        "cupy_cuda_host_input": True,
        "cupy_cuda_device_input": False,
        "native_cuda_host_input": True,
        "native_cuda_device_input": False,
    }.get(key.engine)
    return expected_transfer is None or key.includes_input_transfer == expected_transfer


def _median_seconds(case: Mapping[str, Any]) -> float:
    raw_seconds = case["result"].get("seconds")
    if not isinstance(raw_seconds, list) or not raw_seconds:
        raise ValueError("calibration case is missing timings")
    values = [float(value) for value in raw_seconds]
    if any(value <= 0 for value in values):
        raise ValueError("calibration timings must be positive")
    return float(median(values))


def _validate_workload(workload: Workload, native_speedup_fraction: float) -> None:
    if workload.samples <= 0 or workload.features <= 0 or workload.batch_size <= 0:
        raise ValueError("samples, features, and batch_size must be positive")
    if workload.input_location not in {"host", "device"}:
        raise ValueError("input_location must be 'host' or 'device'")
    if workload.lifecycle not in {"cold", "steady"}:
        raise ValueError("lifecycle must be 'cold' or 'steady'")
    if workload.operation not in {"fit", "partial_fit"}:
        raise ValueError("operation must be 'fit' or 'partial_fit'")
    if workload.dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")
    if workload.penalty not in {"none", "l1"}:
        raise ValueError("penalty must be 'none' or 'l1'")
    if workload.max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if not isinstance(workload.tol, (int, float)) or not 0 < workload.tol < float("inf"):
        raise ValueError("tol must be a positive finite number")
    if not 0 < native_speedup_fraction < 1:
        raise ValueError("native_speedup_fraction must be between 0 and 1")


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=int)
    parser.add_argument("--features", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--penalty", choices=("none", "l1"), required=True)
    parser.add_argument("--input-location", choices=("host", "device"), default="host")
    parser.add_argument("--lifecycle", choices=("cold", "steady"), default="steady")
    parser.add_argument("--operation", choices=("fit", "partial_fit"), default="partial_fit")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--cupy-available", action="store_true")
    parser.add_argument("--native-cpu-available", action="store_true")
    parser.add_argument("--native-cuda-available", action="store_true")
    parser.add_argument("--native-speedup-fraction", type=float, default=0.10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        decision = recommend_backend(
            _load(args.calibration),
            Workload(
                samples=args.samples,
                features=args.features,
                batch_size=args.batch_size,
                dtype=args.dtype,
                penalty=args.penalty,
                input_location=args.input_location,
                lifecycle=args.lifecycle,
                operation=args.operation,
                max_iter=args.max_iter,
                tol=args.tol,
            ),
            RuntimeCapabilities(
                cupy=args.cupy_available,
                native_cpu=args.native_cpu_available,
                native_cuda=args.native_cuda_available,
            ),
            native_speedup_fraction=args.native_speedup_fraction,
        )
    except ValueError as error:
        parser.error(str(error))
    output = (
        json.dumps(
            {
                "schema_version": MEASUREMENT_SCHEMA_VERSION,
                "decision": asdict(decision),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
