"""Merge and gate fixed-runner benchmark records captured in interleaved order."""

from __future__ import annotations

import copy
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any

try:
    from .performance_policy import GPU_ENGINES, MeasurementKey, compare_records, measurement_key
except ImportError:  # pragma: no cover - direct script imports use the sibling directory.
    from performance_policy import GPU_ENGINES, MeasurementKey, compare_records, measurement_key


@dataclass(frozen=True, slots=True)
class PairedRegressionCheck:
    """Existing fixed-runner checks plus the median of paired A/B ratios."""

    key: MeasurementKey
    passed: bool
    paired_median_slowdown: float | None
    max_paired_slowdown: float
    pairs: int
    candidate_faster_pairs: int
    candidate_slower_pairs: int
    reasons: tuple[str, ...]
    fixed_runner: dict[str, Any]


def _index(record: dict[str, Any]) -> dict[MeasurementKey, dict[str, Any]]:
    return {measurement_key(case): case for case in record["cases"]}


def merge_round_records(
    records: list[dict[str, Any]],
    *,
    variant: str,
    pair_id: str,
    execution_order: list[int],
) -> dict[str, Any]:
    """Combine one-sample round records without losing their round pairing."""

    if not records:
        raise ValueError("at least one benchmark round is required")
    if len(records) != len(execution_order):
        raise ValueError("execution_order must have one entry per benchmark round")
    merged = copy.deepcopy(records[0])
    reference_environment = records[0].get("environment")
    reference_arguments = records[0].get("arguments")
    reference_profile = records[0].get("profile")
    for record in records[1:]:
        if record.get("environment") != reference_environment:
            raise ValueError("hardware or runtime metadata changed between interleaved rounds")
        if (
            record.get("arguments") != reference_arguments
            or record.get("profile") != reference_profile
        ):
            raise ValueError("benchmark arguments changed between interleaved rounds")
    expected = set(_index(merged))
    if not expected:
        raise ValueError("benchmark rounds contain no measured cases")
    per_round = [_index(record) for record in records]
    for index in per_round[1:]:
        if set(index) != expected:
            raise ValueError("interleaved rounds do not contain identical benchmark cases")

    merged_index = _index(merged)
    for key in expected:
        results = [index[key]["result"] for index in per_round]
        sample_repetitions = {result.get("sample_repetitions") for result in results}
        if len(sample_repetitions) != 1:
            raise ValueError(
                "sample repetition calibration changed between interleaved rounds; "
                "lower --max-sample-repetitions or recapture"
            )
        seconds = [float(value) for result in results for value in result["seconds"]]
        iterations = [float(value) for result in results for value in result["iterations"]]
        if len(seconds) != len(records) or len(iterations) != len(records):
            raise ValueError("each interleaved round must contain exactly one statistical sample")
        result = merged_index[key]["result"]
        result["seconds"] = seconds
        result["iterations"] = iterations
        result["median_seconds"] = statistics.median(seconds)
        result["minimum_seconds"] = min(seconds)
        result["maximum_seconds"] = max(seconds)
        result["median_iterations"] = statistics.median(iterations)
        result["all_batches_converged"] = all(
            bool(round_result["all_batches_converged"]) for round_result in results
        )
        result["median_samples_per_second"] = key.samples / result["median_seconds"]

    merged["arguments"]["repeats"] = len(records)
    merged["interleaved_capture"] = {
        "pair_id": pair_id,
        "variant": variant,
        "rounds": len(records),
        "round_indexes": list(range(len(records))),
        "execution_order": execution_order,
        "sample_alignment": "round_index",
    }
    return merged


def compare_interleaved_records(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    engines: frozenset[str],
    cpu_max_slowdown: float = 1.10,
    gpu_max_slowdown: float = 1.15,
    min_pairs: int = 9,
    cpu_max_relative_mad: float = 0.05,
    gpu_max_relative_mad: float = 0.10,
    max_iteration_delta: int = 1,
    max_competitor_slowdown: float = 1.0,
) -> list[PairedRegressionCheck]:
    """Require both the existing policy and an aligned paired slowdown gate."""

    baseline_capture = baseline.get("interleaved_capture")
    candidate_capture = candidate.get("interleaved_capture")
    if not isinstance(baseline_capture, dict) or not isinstance(candidate_capture, dict):
        raise ValueError("records must be produced by the interleaved capture runner")
    if baseline_capture.get("pair_id") != candidate_capture.get("pair_id"):
        raise ValueError("baseline and candidate interleaved pair IDs differ")
    if baseline_capture.get("round_indexes") != candidate_capture.get("round_indexes"):
        raise ValueError("baseline and candidate round indexes differ")
    baseline_order = baseline_capture.get("execution_order")
    candidate_order = candidate_capture.get("execution_order")
    if not isinstance(baseline_order, list) or not isinstance(candidate_order, list):
        raise ValueError("interleaved records are missing execution order")
    if len(baseline_order) != len(candidate_order) or any(
        {baseline_position, candidate_position} != {0, 1}
        for baseline_position, candidate_position in zip(
            baseline_order, candidate_order, strict=True
        )
    ):
        raise ValueError("each round must contain one baseline and one candidate in opposite order")
    if any(current == previous for previous, current in zip(baseline_order, baseline_order[1:])):
        raise ValueError("baseline/candidate execution order must alternate between rounds")

    fixed_checks = compare_records(
        baseline,
        candidate,
        engines=engines,
        cpu_max_slowdown=cpu_max_slowdown,
        gpu_max_slowdown=gpu_max_slowdown,
        min_repeats=min_pairs,
        cpu_max_relative_mad=cpu_max_relative_mad,
        gpu_max_relative_mad=gpu_max_relative_mad,
        max_iteration_delta=max_iteration_delta,
        require_competitor_parity=True,
        max_competitor_slowdown=max_competitor_slowdown,
        require_same_hardware=True,
    )
    baseline_cases = _index(baseline)
    candidate_cases = _index(candidate)
    paired: list[PairedRegressionCheck] = []
    for fixed in fixed_checks:
        maximum = gpu_max_slowdown if fixed.key.engine in GPU_ENGINES else cpu_max_slowdown
        reasons = list(fixed.reasons)
        baseline_seconds = baseline_cases[fixed.key]["result"]["seconds"]
        candidate_seconds = candidate_cases[fixed.key]["result"]["seconds"]
        if len(baseline_seconds) != len(candidate_seconds):
            reasons.append("baseline and candidate have different paired sample counts")
            ratios: list[float] = []
        else:
            ratios = [
                float(candidate_value) / float(baseline_value)
                for baseline_value, candidate_value in zip(
                    baseline_seconds, candidate_seconds, strict=True
                )
            ]
        if len(ratios) < min_pairs:
            reasons.append(f"paired capture has {len(ratios)} pairs; need at least {min_pairs}")
        paired_median = statistics.median(ratios) if ratios else None
        if paired_median is not None and (
            not math.isfinite(paired_median) or paired_median > maximum
        ):
            reasons.append(
                f"paired median slowdown {paired_median:.3f} exceeds limit {maximum:.3f}"
            )
        paired.append(
            PairedRegressionCheck(
                key=fixed.key,
                passed=not reasons,
                paired_median_slowdown=paired_median,
                max_paired_slowdown=maximum,
                pairs=len(ratios),
                candidate_faster_pairs=sum(ratio < 1.0 for ratio in ratios),
                candidate_slower_pairs=sum(ratio > 1.0 for ratio in ratios),
                reasons=tuple(reasons),
                fixed_runner={**asdict(fixed), "key": asdict(fixed.key)},
            )
        )
    return paired


def report(checks: list[PairedRegressionCheck]) -> dict[str, Any]:
    return {
        "schema": "renewable-huber-interleaved-regression-gate",
        "schema_version": 1,
        "passed": bool(checks) and all(check.passed for check in checks),
        "checked_cases": len(checks),
        "checks": [{**asdict(check), "key": asdict(check.key)} for check in checks],
    }
