"""Shared validation and comparison rules for native-engine benchmark records.

The benchmark harness deliberately keeps this module dependency-free.  It is
used by the command-line regression gate and by the calibration-only dispatch
advisor, so neither tool has to import the package under test (or CUDA).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from math import ceil, isfinite
from statistics import median
from typing import Any

MEASUREMENT_SCHEMA_VERSION = 2
NATIVE_ENGINES = frozenset(
    {"rust_native_cpu", "native_cuda_host_input", "native_cuda_device_input"}
)
GPU_ENGINES = frozenset(
    {
        "cupy_cuda_host_input",
        "cupy_cuda_device_input",
        "native_cuda_host_input",
        "native_cuda_device_input",
    }
)
NATIVE_CUDA_ENGINES = frozenset({"native_cuda_host_input", "native_cuda_device_input"})
DIRECT_COMPETITORS = {
    "rust_native_cpu": "numpy_cpu",
    "native_cuda_host_input": "cupy_cuda_host_input",
    "native_cuda_device_input": "cupy_cuda_device_input",
}
#: The ``environment`` section in which each native engine's extension
#: reports its own ``version()`` metadata.
NATIVE_METADATA_SECTIONS = {
    "rust_native_cpu": "native_cpu",
    "native_cuda_host_input": "native_cuda_abi",
    "native_cuda_device_input": "native_cuda_abi",
}
#: Native extension interface versions. They are part of every native
#: fingerprint; only an interleaved A/B that explicitly opts in to comparing
#: across a native ABI/API change may leave them out, and nothing else.
NATIVE_VERSION_FIELDS = ("abi_version", "python_api_version")

#: Iteration policies. ``absolute`` holds every case to ``max_iteration_delta``.
#: ``relative`` applies only to float32 cases in an interleaved A/B: float32
#: sits at its rounding floor at the default ``tol``, so any change to a
#: summation order moves its iteration count by more than one without the
#: solver getting worse. Such a case passes only if the count moved by at most
#: ``FLOAT32_RELATIVE_ITERATION_FRACTION`` of the baseline (and never less than
#: ``max_iteration_delta``), each iteration is no slower than the slowdown
#: limit, and both builds reach the same final objective to within
#: ``FLOAT32_FINAL_OBJECTIVE_RTOL``. float64 and every stored-baseline gate
#: stay ``absolute``.
ITERATION_POLICIES = ("absolute", "relative")
FLOAT32_RELATIVE_ITERATION_FRACTION = 0.25
#: Measured, not guessed: across every standard shape the released 0.7.0 ->
#: 0.7.1 Rust CPU change moved float32 final objectives by at most 6.1e-5
#: (streaming L1, 56 -> 61 iterations), the summed float32 loss over 65,536
#: rows being only that accurate. float64 moved by under 1e-15.
FLOAT32_FINAL_OBJECTIVE_RTOL = 1e-4


@dataclass(frozen=True, slots=True)
class MeasurementKey:
    """Everything that must agree before two timings may be compared."""

    engine: str
    shape_name: str
    samples: int
    features: int
    batch_size: int
    dtype: str
    penalty: str
    dataset_seed: int
    dataset_sha256: str
    max_iter: int
    tol: float
    lifecycle: str
    operation: str
    input_location: str
    includes_input_transfer: bool
    includes_engine_initialization: bool
    includes_engine_destruction: bool
    resident_engine: bool
    state_reset_timed: bool
    timer: str
    sample_aggregation: str
    minimum_sample_seconds: float
    gc_collected_before_sample: bool
    gc_disabled_during_timing: bool


@dataclass(frozen=True, slots=True)
class RegressionCheck:
    """One baseline/candidate comparison and every condition behind its result."""

    key: MeasurementKey
    passed: bool
    baseline_seconds: float | None
    candidate_seconds: float | None
    slowdown_ratio: float | None
    max_slowdown: float
    competitor_engine: str | None
    competitor_seconds: float | None
    competitor_ratio: float | None
    max_competitor_slowdown: float | None
    baseline_relative_mad: float | None
    candidate_relative_mad: float | None
    reasons: tuple[str, ...]
    #: ``absolute`` or ``float32_relative``: the rule this case's iterations met.
    iteration_policy: str = "absolute"
    baseline_iterations: float | None = None
    candidate_iterations: float | None = None
    #: Candidate seconds per iteration over the baseline's.
    per_iteration_ratio: float | None = None
    #: |candidate - baseline| / |baseline| of the median final objective.
    final_objective_relative_difference: float | None = None
    #: The iteration count moved past ``max_iteration_delta`` and the float32
    #: relative policy accepted it. Visible in the report, never silent.
    iterations_changed: bool = False


def measurement_key(case: Mapping[str, Any]) -> MeasurementKey:
    """Extract a strict comparison key from a schema-v2 benchmark case."""

    shape = _mapping(case, "shape")
    result = _mapping(case, "result")
    try:
        return MeasurementKey(
            engine=_string(case, "engine"),
            shape_name=_string(shape, "name"),
            samples=_positive_int(shape, "samples"),
            features=_positive_int(shape, "features"),
            batch_size=_positive_int(shape, "batch_size"),
            dtype=_string(case, "dtype"),
            penalty=_string(case, "penalty"),
            dataset_seed=_non_negative_int(case, "dataset_seed"),
            dataset_sha256=_string(case, "dataset_sha256"),
            max_iter=_positive_int(case, "max_iter"),
            tol=_positive_float(case, "tol"),
            lifecycle=_string(result, "lifecycle"),
            operation=_string(result, "operation"),
            input_location=_string(result, "input_location"),
            includes_input_transfer=_bool(result, "includes_input_transfer"),
            includes_engine_initialization=_bool(result, "includes_engine_initialization"),
            # Early schema-v2 captures accidentally timed temporary-model
            # teardown and did not record it. Treat omission as the historical
            # True contract so those records never compare with corrected
            # operation-only captures that explicitly report False.
            includes_engine_destruction=(
                _bool(result, "includes_engine_destruction")
                if "includes_engine_destruction" in result
                else True
            ),
            resident_engine=_bool(result, "resident_engine"),
            state_reset_timed=_bool(result, "state_reset_timed"),
            # Historical schema-v2 records used one perf_counter interval per
            # sample and did not isolate cyclic GC. Keep those defaults
            # explicit so they cannot be compared with block-sampled records.
            timer=str(result.get("timer", "time.perf_counter")),
            sample_aggregation=str(result.get("sample_aggregation", "single_operation")),
            minimum_sample_seconds=_optional_non_negative_float(
                result, "minimum_sample_seconds", default=0.0
            ),
            gc_collected_before_sample=_optional_bool(
                result, "gc_collected_before_sample", default=False
            ),
            gc_disabled_during_timing=_optional_bool(
                result, "gc_disabled_during_timing", default=False
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid benchmark case: {error}") from error


def validate_record(record: Mapping[str, Any]) -> None:
    """Reject legacy or malformed records before they reach a performance gate.

    Schema v1 combines unlike lifecycles for the CUDA path.  Treating its
    values as a hard gate would turn an apples-to-oranges comparison into a
    pass/fail decision, so a new schema-v2 baseline must be captured first.
    """

    if record.get("schema") != "renewable-huber-shape-sweep":
        raise ValueError("record is not a renewable-huber shape sweep")
    if record.get("schema_version") != MEASUREMENT_SCHEMA_VERSION:
        raise ValueError(
            "performance gates require shape-sweep schema version "
            f"{MEASUREMENT_SCHEMA_VERSION}; recapture this baseline"
        )
    cases = record.get("cases")
    if not isinstance(cases, list):
        raise ValueError("record cases must be a list")
    # Only consulted for a native CUDA case that claims a penalty; a missing
    # or malformed environment then counts as no evidence of L1 support.
    environment = record.get("environment")
    keys: set[MeasurementKey] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("record contains a non-object benchmark case")
        key = measurement_key(case)
        if key in keys:
            raise ValueError(f"record contains duplicate benchmark case {key}")
        keys.add(key)
        _timings(case)
        result = _mapping(case, "result")
        _median_iterations(result)
        _all_batches_converged(result)
        _validate_sampling_result(result, key)
        _validate_timing_contract(key, environment)


def hardware_fingerprint(
    record: Mapping[str, Any],
    engine: str,
    *,
    include_native_versions: bool = True,
) -> tuple[tuple[str, str], ...]:
    """Return the runner attributes relevant to a CPU or GPU measurement.

    ``include_native_versions=False`` drops exactly the native extension's
    ``abi_version`` and ``python_api_version``; every hardware, driver,
    runtime, provider and transport field is still returned.
    """

    environment = _mapping(record, "environment")
    version_fields = NATIVE_VERSION_FIELDS if include_native_versions else ()
    fields = [
        "platform",
        "processor",
        "python",
        "numpy",
        "threading",
        "numpy_blas_provider",
        "perf_counter",
    ]
    if engine in GPU_ENGINES:
        fields.extend(["gpu", "gpu_compute_capability", "cuda_runtime", "cupy"])
    fingerprint = [(field, _normalise_metadata(environment.get(field))) for field in fields]
    if engine == "rust_native_cpu":
        fingerprint.extend(
            _nested_fingerprint(
                environment,
                "native_cpu",
                (
                    *version_fields,
                    "linear_algebra_provider",
                    "parallel_provider",
                    "parallel_threads",
                ),
            )
        )
    elif engine in NATIVE_CUDA_ENGINES:
        native_cuda_fields = [
            *version_fields,
            "driver_version",
            "runtime_version",
        ]
        if engine == "native_cuda_device_input":
            native_cuda_fields.append("device_input")
        fingerprint.extend(
            _nested_fingerprint(
                environment,
                "native_cuda_abi",
                tuple(native_cuda_fields),
            )
        )
    return tuple(fingerprint)


def native_versions(record: Mapping[str, Any], engine: str) -> dict[str, Any]:
    """Return the recorded native interface versions for a native engine.

    A field the record does not carry is returned as ``None`` rather than
    omitted, so evidence built from it cannot silently drop a side.
    """

    section = NATIVE_METADATA_SECTIONS.get(engine)
    if section is None:
        raise ValueError(f"{engine!r} is not a native engine")
    environment = record.get("environment")
    metadata = environment.get(section) if isinstance(environment, Mapping) else None
    mapping = metadata if isinstance(metadata, Mapping) else {}
    return {field: mapping.get(field) for field in NATIVE_VERSION_FIELDS}


def relative_mad(seconds: Iterable[float]) -> float:
    """Return median absolute deviation relative to the median runtime."""

    values = tuple(float(value) for value in seconds)
    if not values:
        raise ValueError("at least one timing is required")
    center = median(values)
    if center <= 0:
        raise ValueError("timings must be positive")
    return median(abs(value - center) for value in values) / center


def compare_records(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    engines: Iterable[str] = NATIVE_ENGINES,
    cpu_max_slowdown: float = 1.10,
    gpu_max_slowdown: float = 1.15,
    min_repeats: int = 9,
    cpu_max_relative_mad: float = 0.05,
    gpu_max_relative_mad: float = 0.10,
    max_iteration_delta: int = 1,
    require_competitor_parity: bool = True,
    max_competitor_slowdown: float = 1.0,
    require_same_hardware: bool = True,
    allow_native_version_change: bool = False,
    iteration_policy: str = "absolute",
) -> list[RegressionCheck]:
    """Compare strict-equivalent native cases from two v2 sweep records.

    A candidate must retain the same result contract: shape, solver settings
    carried by the record, input location, lifecycle, initialization policy,
    and state-reset policy.  The gate is intentionally conservative: it is
    useful only on a fixed runner and rejects short or noisy samples.

    ``allow_native_version_change`` leaves only the native extensions'
    ``abi_version`` and ``python_api_version`` out of the fingerprint
    comparison. It exists for the interleaved A/B runner, which measures both
    builds on the same host in alternating order; a gate against a stored
    baseline must keep the default and reject an interface change.

    ``iteration_policy="relative"`` is likewise for the interleaved runner
    only; see ``ITERATION_POLICIES``.
    """

    if iteration_policy not in ITERATION_POLICIES:
        raise ValueError(f"iteration_policy must be one of {ITERATION_POLICIES}")

    _validate_thresholds(
        cpu_max_slowdown=cpu_max_slowdown,
        gpu_max_slowdown=gpu_max_slowdown,
        min_repeats=min_repeats,
        cpu_max_relative_mad=cpu_max_relative_mad,
        gpu_max_relative_mad=gpu_max_relative_mad,
        max_iteration_delta=max_iteration_delta,
        max_competitor_slowdown=max_competitor_slowdown,
    )
    validate_record(baseline)
    validate_record(candidate)
    requested_engines = frozenset(engines)
    baseline_cases = _case_index(baseline)
    candidate_cases = _case_index(candidate)
    checks: list[RegressionCheck] = []

    for key, baseline_case in sorted(baseline_cases.items(), key=lambda item: repr(item[0])):
        if key.engine not in requested_engines:
            continue
        candidate_case = candidate_cases.get(key)
        max_slowdown = gpu_max_slowdown if key.engine in GPU_ENGINES else cpu_max_slowdown
        if candidate_case is None:
            checks.append(
                RegressionCheck(
                    key=key,
                    passed=False,
                    baseline_seconds=None,
                    candidate_seconds=None,
                    slowdown_ratio=None,
                    max_slowdown=max_slowdown,
                    competitor_engine=DIRECT_COMPETITORS.get(key.engine),
                    competitor_seconds=None,
                    competitor_ratio=None,
                    max_competitor_slowdown=(
                        max_competitor_slowdown if require_competitor_parity else None
                    ),
                    baseline_relative_mad=None,
                    candidate_relative_mad=None,
                    reasons=("candidate record is missing this benchmark case",),
                )
            )
            continue
        checks.append(
            _compare_case(
                baseline,
                candidate,
                key,
                baseline_case,
                candidate_case,
                candidate_cases=candidate_cases,
                max_slowdown=max_slowdown,
                min_repeats=min_repeats,
                max_relative_mad=(
                    gpu_max_relative_mad if key.engine in GPU_ENGINES else cpu_max_relative_mad
                ),
                max_iteration_delta=max_iteration_delta,
                require_competitor_parity=require_competitor_parity,
                max_competitor_slowdown=max_competitor_slowdown,
                require_same_hardware=require_same_hardware,
                allow_native_version_change=allow_native_version_change,
                iteration_policy=iteration_policy,
            )
        )
    return checks


def _compare_case(
    baseline_record: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    key: MeasurementKey,
    baseline_case: Mapping[str, Any],
    candidate_case: Mapping[str, Any],
    *,
    candidate_cases: Mapping[MeasurementKey, Mapping[str, Any]],
    max_slowdown: float,
    min_repeats: int,
    max_relative_mad: float,
    max_iteration_delta: int,
    require_competitor_parity: bool,
    max_competitor_slowdown: float,
    require_same_hardware: bool,
    allow_native_version_change: bool,
    iteration_policy: str = "absolute",
) -> RegressionCheck:
    baseline_seconds = _timings(baseline_case)
    candidate_seconds = _timings(candidate_case)
    baseline_median = median(baseline_seconds)
    candidate_median = median(candidate_seconds)
    baseline_mad = relative_mad(baseline_seconds)
    candidate_mad = relative_mad(candidate_seconds)
    reasons: list[str] = []
    include_native_versions = not allow_native_version_change
    if require_same_hardware and hardware_fingerprint(
        baseline_record, key.engine, include_native_versions=include_native_versions
    ) != hardware_fingerprint(
        candidate_record, key.engine, include_native_versions=include_native_versions
    ):
        reasons.append("hardware or runtime fingerprint differs")
    if len(baseline_seconds) < min_repeats:
        reasons.append(f"baseline has {len(baseline_seconds)} repeats; need at least {min_repeats}")
    if len(candidate_seconds) < min_repeats:
        reasons.append(
            f"candidate has {len(candidate_seconds)} repeats; need at least {min_repeats}"
        )
    if baseline_mad > max_relative_mad:
        reasons.append(f"baseline relative MAD {baseline_mad:.1%} exceeds {max_relative_mad:.1%}")
    if candidate_mad > max_relative_mad:
        reasons.append(f"candidate relative MAD {candidate_mad:.1%} exceeds {max_relative_mad:.1%}")
    baseline_result = _mapping(baseline_case, "result")
    candidate_result = _mapping(candidate_case, "result")
    if not _all_batches_converged(baseline_result):
        reasons.append("baseline did not converge for every batch")
    if not _all_batches_converged(candidate_result):
        reasons.append("candidate did not converge for every batch")
    baseline_iterations = _median_iterations(baseline_result)
    candidate_iterations = _median_iterations(candidate_result)
    slowdown = candidate_median / baseline_median
    if slowdown > max_slowdown:
        reasons.append(f"slowdown {slowdown:.3f} exceeds limit {max_slowdown:.3f}")
    per_iteration_ratio = (
        slowdown * baseline_iterations / candidate_iterations
        if baseline_iterations > 0 and candidate_iterations > 0
        else None
    )
    baseline_objective = _median_final_objective(baseline_result)
    candidate_objective = _median_final_objective(candidate_result)
    objective_difference = (
        abs(candidate_objective - baseline_objective) / max(abs(baseline_objective), 1e-300)
        if baseline_objective is not None and candidate_objective is not None
        else None
    )
    relative = iteration_policy == "relative" and key.dtype == "float32"
    iteration_delta = abs(candidate_iterations - baseline_iterations)
    iterations_changed = False
    if iteration_delta > max_iteration_delta:
        if not relative:
            reasons.append(
                "median solver iterations differ by "
                f"{iteration_delta:.0f}; allowed {max_iteration_delta}"
            )
        else:
            iterations_changed = True
            allowed = max(
                max_iteration_delta,
                ceil(FLOAT32_RELATIVE_ITERATION_FRACTION * baseline_iterations),
            )
            if iteration_delta > allowed:
                reasons.append(
                    f"median solver iterations differ by {iteration_delta:.0f}; the float32 "
                    f"relative policy allows {allowed}"
                )
            if per_iteration_ratio is None:
                reasons.append("per-iteration slowdown is undefined for zero iterations")
            elif per_iteration_ratio > max_slowdown:
                reasons.append(
                    f"per-iteration slowdown {per_iteration_ratio:.3f} exceeds limit "
                    f"{max_slowdown:.3f}"
                )
            if objective_difference is None:
                reasons.append(
                    "the float32 relative iteration policy needs median_final_objective "
                    "in both records"
                )
            elif objective_difference > FLOAT32_FINAL_OBJECTIVE_RTOL:
                reasons.append(
                    f"final objectives differ by {objective_difference:.2e} (relative); "
                    f"limit {FLOAT32_FINAL_OBJECTIVE_RTOL:.0e}"
                )
    competitor_engine = DIRECT_COMPETITORS.get(key.engine)
    competitor_seconds: float | None = None
    competitor_ratio: float | None = None
    if require_competitor_parity and competitor_engine is not None:
        competitor_case = candidate_cases.get(replace(key, engine=competitor_engine))
        if competitor_case is None:
            reasons.append(f"candidate lacks matched {competitor_engine} competitor case")
        else:
            competitor_seconds = median(_timings(competitor_case))
            competitor_ratio = candidate_median / competitor_seconds
            competitor_result = _mapping(competitor_case, "result")
            if not _all_batches_converged(competitor_result):
                reasons.append(f"matched {competitor_engine} competitor did not converge")
            if competitor_ratio > max_competitor_slowdown:
                reasons.append(
                    f"native/reference ratio {competitor_ratio:.3f} exceeds "
                    f"{max_competitor_slowdown:.3f} against {competitor_engine}"
                )
    return RegressionCheck(
        key=key,
        passed=not reasons,
        baseline_seconds=baseline_median,
        candidate_seconds=candidate_median,
        slowdown_ratio=slowdown,
        max_slowdown=max_slowdown,
        competitor_engine=competitor_engine,
        competitor_seconds=competitor_seconds,
        competitor_ratio=competitor_ratio,
        max_competitor_slowdown=(max_competitor_slowdown if require_competitor_parity else None),
        baseline_relative_mad=baseline_mad,
        candidate_relative_mad=candidate_mad,
        reasons=tuple(reasons),
        iteration_policy="float32_relative" if relative else "absolute",
        baseline_iterations=baseline_iterations,
        candidate_iterations=candidate_iterations,
        per_iteration_ratio=per_iteration_ratio,
        final_objective_relative_difference=objective_difference,
        iterations_changed=iterations_changed and not reasons,
    )


def _case_index(record: Mapping[str, Any]) -> dict[MeasurementKey, Mapping[str, Any]]:
    return {measurement_key(case): case for case in record["cases"]}


def _timings(case: Mapping[str, Any]) -> tuple[float, ...]:
    result = _mapping(case, "result")
    raw_seconds = result.get("seconds")
    if not isinstance(raw_seconds, list) or not raw_seconds:
        raise ValueError("benchmark case result.seconds must be a non-empty list")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in raw_seconds):
        raise ValueError("benchmark timings must be numeric")
    seconds = tuple(float(value) for value in raw_seconds)
    if any(not isfinite(value) or value <= 0 for value in seconds):
        raise ValueError("benchmark timings must be positive finite numbers")
    return seconds


def _median_final_objective(result: Mapping[str, Any]) -> float | None:
    value = result.get("median_final_objective")
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
        raise ValueError("benchmark median_final_objective must be a finite number")
    return float(value)


def _median_iterations(result: Mapping[str, Any]) -> float:
    value = result.get("median_iterations")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("benchmark result is missing numeric median_iterations")
    iterations = float(value)
    if not isfinite(iterations) or iterations < 0:
        raise ValueError("benchmark median_iterations must be a non-negative finite number")
    return iterations


def _all_batches_converged(result: Mapping[str, Any]) -> bool:
    value = result.get("all_batches_converged")
    if not isinstance(value, bool):
        raise ValueError("benchmark all_batches_converged must be a boolean")
    return value


def _mapping(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    nested = value.get(field)
    if not isinstance(nested, Mapping):
        raise ValueError(f"{field} must be an object")
    return nested


def _string(value: Mapping[str, Any], field: str) -> str:
    result = value[field]
    if not isinstance(result, str) or not result:
        raise ValueError(f"{field} must be a non-empty string")
    return result


def _positive_int(value: Mapping[str, Any], field: str) -> int:
    result = value[field]
    if not isinstance(result, int) or isinstance(result, bool) or result <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return result


def _non_negative_int(value: Mapping[str, Any], field: str) -> int:
    result = value[field]
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return result


def _positive_float(value: Mapping[str, Any], field: str) -> float:
    result = value[field]
    if (
        not isinstance(result, (int, float))
        or isinstance(result, bool)
        or not isfinite(float(result))
        or result <= 0
    ):
        raise ValueError(f"{field} must be a positive finite number")
    return float(result)


def _bool(value: Mapping[str, Any], field: str) -> bool:
    result = value[field]
    if not isinstance(result, bool):
        raise ValueError(f"{field} must be a boolean")
    return result


def _optional_bool(value: Mapping[str, Any], field: str, *, default: bool) -> bool:
    if field not in value:
        return default
    return _bool(value, field)


def _optional_non_negative_float(value: Mapping[str, Any], field: str, *, default: float) -> float:
    if field not in value:
        return default
    result = value[field]
    if (
        not isinstance(result, (int, float))
        or isinstance(result, bool)
        or not isfinite(float(result))
        or result < 0
    ):
        raise ValueError(f"{field} must be a non-negative finite number")
    return float(result)


def _validate_sampling_result(result: Mapping[str, Any], key: MeasurementKey) -> None:
    repetitions = result.get("sample_repetitions", 1)
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("sample_repetitions must be a positive integer")
    calibration_runs = result.get("sampling_calibration_runs", 0)
    if (
        not isinstance(calibration_runs, int)
        or isinstance(calibration_runs, bool)
        or calibration_runs < 0
    ):
        raise ValueError("sampling_calibration_runs must be a non-negative integer")
    if key.timer != "time.perf_counter":
        raise ValueError("shape-sweep timing must use time.perf_counter")
    if key.sample_aggregation not in {"single_operation", "arithmetic_mean"}:
        raise ValueError("unknown sample_aggregation")
    if key.sample_aggregation == "single_operation" and repetitions != 1:
        raise ValueError("single_operation samples must have one repetition")


def _normalise_metadata(value: Any) -> str:
    if value is None:
        return "<missing>"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return str(value)


def _nested_fingerprint(
    environment: Mapping[str, Any],
    section: str,
    fields: Iterable[str],
) -> list[tuple[str, str]]:
    metadata = environment.get(section)
    mapping = metadata if isinstance(metadata, Mapping) else {}
    return [(f"{section}.{field}", _normalise_metadata(mapping.get(field))) for field in fields]


def _records_native_cuda_l1_support(environment: Any) -> bool:
    """Return whether a record's own native CUDA metadata shows L1 support.

    ABI 2 added the L1 penalty to the native CUDA engine, and Python API 4
    advertises it in ``supported_penalties``. Either one recorded at capture
    time is evidence; anything else, including no metadata, is not.
    """

    metadata = environment.get("native_cuda_abi") if isinstance(environment, Mapping) else None
    if not isinstance(metadata, Mapping):
        return False
    abi_version = metadata.get("abi_version")
    if isinstance(abi_version, int) and not isinstance(abi_version, bool) and abi_version >= 2:
        return True
    penalties = metadata.get("supported_penalties")
    return isinstance(penalties, (list, tuple)) and "l1" in penalties


def _validate_timing_contract(key: MeasurementKey, environment: Any) -> None:
    """Reject records whose labels disagree with the measured lifecycle."""

    if key.lifecycle == "cold":
        expected = (True, False, True)
    elif key.lifecycle == "steady":
        expected = (False, True, False)
        if key.operation == "fit":
            raise ValueError("steady fit is not a valid public benchmark contract")
    else:
        raise ValueError("lifecycle must be 'cold' or 'steady'")
    observed = (
        key.includes_engine_initialization,
        key.resident_engine,
        key.state_reset_timed,
    )
    if observed != expected:
        raise ValueError(
            "lifecycle labels do not match initialization/residency/reset timing policy"
        )
    expected_transport = {
        "numpy_cpu": ("host", False),
        "rust_native_cpu": ("host", False),
        "cupy_cuda_host_input": ("host", True),
        "cupy_cuda_device_input": ("device", False),
        "native_cuda_host_input": ("host", True),
        "native_cuda_device_input": ("device", False),
    }.get(key.engine)
    if (
        expected_transport is not None
        and (
            key.input_location,
            key.includes_input_transfer,
        )
        != expected_transport
    ):
        raise ValueError("engine labels do not match its input-transfer policy")
    if key.engine in NATIVE_CUDA_ENGINES and key.penalty != "none":
        if key.penalty != "l1":
            raise ValueError(f"native CUDA benchmark records may not claim penalty {key.penalty!r}")
        if not _records_native_cuda_l1_support(environment):
            raise ValueError(
                "native CUDA benchmark records may claim L1 support only when the "
                "recorded native_cuda_abi shows abi_version >= 2 or lists 'l1' in "
                "supported_penalties; an ABI 1 engine implements only penalty 'none'"
            )


def _validate_thresholds(
    *,
    cpu_max_slowdown: float,
    gpu_max_slowdown: float,
    min_repeats: int,
    cpu_max_relative_mad: float,
    gpu_max_relative_mad: float,
    max_iteration_delta: int,
    max_competitor_slowdown: float,
) -> None:
    if (
        not isfinite(cpu_max_slowdown)
        or not isfinite(gpu_max_slowdown)
        or cpu_max_slowdown < 1
        or gpu_max_slowdown < 1
    ):
        raise ValueError("maximum slowdown must be at least 1.0")
    if min_repeats < 1:
        raise ValueError("min_repeats must be positive")
    if (
        not isfinite(cpu_max_relative_mad)
        or not isfinite(gpu_max_relative_mad)
        or not 0 <= cpu_max_relative_mad < 1
        or not 0 <= gpu_max_relative_mad < 1
    ):
        raise ValueError("maximum relative MAD must be in [0, 1)")
    if max_iteration_delta < 0:
        raise ValueError("max_iteration_delta must be non-negative")
    if not isfinite(max_competitor_slowdown) or max_competitor_slowdown <= 0:
        raise ValueError("max_competitor_slowdown must be positive")
