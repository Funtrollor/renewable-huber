from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

from scripts.benchmarks import run_interleaved_benchmark
from scripts.benchmarks.interleaved_regression import (
    GATE_SCHEMA_VERSION,
    compare_interleaved_records,
    merge_round_records,
    report,
)


def _case(engine: str, seconds: float) -> dict[str, object]:
    location = "device" if engine.endswith("device_input") else "host"
    return {
        "shape": {"name": "tiny", "samples": 128, "features": 4, "batch_size": 64},
        "dataset_seed": 42,
        "dataset_sha256": "abc",
        "dtype": "float64",
        "penalty": "none",
        "max_iter": 10,
        "tol": 1e-6,
        "engine": engine,
        "result": {
            "seconds": [seconds],
            "median_seconds": seconds,
            "minimum_seconds": seconds,
            "maximum_seconds": seconds,
            "iterations": [3.0],
            "median_iterations": 3.0,
            "all_batches_converged": True,
            "lifecycle": "cold",
            "operation": "partial_fit",
            "input_location": location,
            "includes_input_transfer": location == "host" and "cuda" in engine,
            "includes_engine_initialization": True,
            "includes_engine_destruction": False,
            "resident_engine": False,
            "state_reset_timed": True,
            "timer": "time.perf_counter",
            "sample_aggregation": "arithmetic_mean",
            "sample_repetitions": 1,
            "minimum_sample_seconds": 0.0,
            "gc_collected_before_sample": True,
            "gc_disabled_during_timing": True,
            "median_samples_per_second": 128 / seconds,
        },
    }


def _record(native_seconds: float, numpy_seconds: float = 2.0) -> dict[str, object]:
    return {
        "schema": "renewable-huber-shape-sweep",
        "schema_version": 2,
        "profile": "smoke",
        "arguments": {"repeats": 1},
        "environment": {
            "platform": "test",
            "processor": "cpu",
            "python": "3.12",
            "numpy": "2",
            "threading": {},
            "numpy_blas_provider": {},
            "perf_counter": "test",
            "native_cpu": {
                "abi_version": 1,
                "python_api_version": 2,
                "linear_algebra_provider": "test",
                "parallel_provider": "test",
                "parallel_threads": 1,
            },
        },
        "cases": [_case("numpy_cpu", numpy_seconds), _case("rust_native_cpu", native_seconds)],
    }


#: The N5 A/B: one RTX 5070 Ti host, identical driver and runtime, native CUDA
#: ABI 1 / Python API 3 at the baseline and ABI 2 / API 4 at the candidate.
_GPU_ENVIRONMENT = {
    "gpu": "NVIDIA GeForce RTX 5070 Ti",
    "gpu_compute_capability": "12.0",
    "cuda_runtime": 12090,
    "cupy": "14.1.1",
}
_NATIVE_CUDA_ABI_1 = {
    "abi_version": 1,
    "python_api_version": 3,
    "driver_version": 13040,
    "runtime_version": 12090,
    "device_input": "dlpack",
}
_NATIVE_CUDA_ABI_2 = {
    **_NATIVE_CUDA_ABI_1,
    "abi_version": 2,
    "python_api_version": 4,
    "supported_penalties": ["none", "l1"],
}
_CUDA_ENGINES = frozenset({"native_cuda_host_input"})
_CPU_ENGINES = frozenset({"rust_native_cpu"})


def _cuda_record(
    native_seconds: float,
    native_cuda_abi: dict[str, Any] = _NATIVE_CUDA_ABI_1,
    cupy_seconds: float = 2.0,
) -> dict[str, Any]:
    record: dict[str, Any] = _record(native_seconds)
    record["environment"].update(copy.deepcopy(_GPU_ENVIRONMENT))
    record["environment"]["native_cuda_abi"] = copy.deepcopy(native_cuda_abi)
    record["cases"] = [
        _case("cupy_cuda_host_input", cupy_seconds),
        _case("native_cuda_host_input", native_seconds),
    ]
    return record


def _merge(
    values: list[float],
    variant: str,
    factory: Callable[[float], dict[str, Any]] = _record,
) -> dict[str, Any]:
    records = [factory(value) for value in values]
    offset = 0 if variant == "baseline" else 1
    return merge_round_records(
        copy.deepcopy(records),
        variant=variant,
        pair_id="pair",
        execution_order=[(index + offset) % 2 for index in range(len(records))],
    )


def _abi_change() -> tuple[dict[str, Any], dict[str, Any]]:
    """Merged records whose only fingerprint difference is the native CUDA ABI/API."""

    baseline = _merge([1.0] * 9, "baseline", lambda value: _cuda_record(value))
    candidate = _merge(
        [0.95] * 9,
        "candidate",
        lambda value: _cuda_record(value, _NATIVE_CUDA_ABI_2),
    )
    return baseline, candidate


class InterleavedRegressionTests(unittest.TestCase):
    def _merged(self, values: list[float], variant: str) -> dict[str, object]:
        return _merge(values, variant)

    def test_merge_preserves_round_order_and_recomputes_summaries(self) -> None:
        merged = self._merged([1.0, 0.9, 1.1], "baseline")
        native = next(case for case in merged["cases"] if case["engine"] == "rust_native_cpu")
        self.assertEqual(native["result"]["seconds"], [1.0, 0.9, 1.1])
        self.assertEqual(native["result"]["median_seconds"], 1.0)
        self.assertEqual(merged["arguments"]["repeats"], 3)

    def test_paired_gate_accepts_stable_candidate(self) -> None:
        baseline = self._merged([1.0] * 9, "baseline")
        candidate = self._merged([0.9] * 9, "candidate")
        checks = compare_interleaved_records(
            baseline,
            candidate,
            engines=frozenset({"rust_native_cpu"}),
        )
        self.assertEqual(len(checks), 1)
        self.assertTrue(checks[0].passed, checks[0].reasons)
        self.assertAlmostEqual(checks[0].paired_median_slowdown or 0.0, 0.9)

    def test_paired_gate_rejects_slow_candidate(self) -> None:
        baseline = self._merged([1.0] * 9, "baseline")
        candidate = self._merged([1.2] * 9, "candidate")
        checks = compare_interleaved_records(
            baseline,
            candidate,
            engines=frozenset({"rust_native_cpu"}),
        )
        self.assertFalse(checks[0].passed)
        self.assertTrue(any("paired median slowdown" in reason for reason in checks[0].reasons))

    def test_pair_id_mismatch_is_rejected(self) -> None:
        baseline = self._merged([1.0] * 9, "baseline")
        candidate = self._merged([0.9] * 9, "candidate")
        candidate["interleaved_capture"]["pair_id"] = "different"
        with self.assertRaisesRegex(ValueError, "pair IDs differ"):
            compare_interleaved_records(
                baseline,
                candidate,
                engines=frozenset({"rust_native_cpu"}),
            )

    def test_non_alternating_execution_order_is_rejected(self) -> None:
        baseline = self._merged([1.0] * 9, "baseline")
        candidate = self._merged([0.9] * 9, "candidate")
        baseline["interleaved_capture"]["execution_order"] = [0] * 9
        candidate["interleaved_capture"]["execution_order"] = [1] * 9
        with self.assertRaisesRegex(ValueError, "must alternate"):
            compare_interleaved_records(
                baseline,
                candidate,
                engines=frozenset({"rust_native_cpu"}),
            )

    def test_merge_rejects_per_round_sampling_contract_drift(self) -> None:
        records = [_record(1.0), _record(1.0)]
        records[1]["cases"][0]["result"]["sample_repetitions"] = 2
        with self.assertRaisesRegex(ValueError, "sample repetition calibration changed"):
            merge_round_records(
                records,
                variant="baseline",
                pair_id="pair",
                execution_order=[0, 1],
            )


class NativeVersionChangeTests(unittest.TestCase):
    """``allow_native_version_change`` exempts the interface versions and nothing else."""

    def test_native_abi_change_is_rejected_without_the_option(self) -> None:
        baseline, candidate = _abi_change()
        for options in ({}, {"allow_native_version_change": False}):
            with self.subTest(options=options):
                checks = compare_interleaved_records(
                    baseline, candidate, engines=_CUDA_ENGINES, **options
                )

                self.assertEqual(len(checks), 1)
                self.assertFalse(checks[0].passed)
                self.assertEqual(checks[0].reasons, ("hardware or runtime fingerprint differs",))

    def test_native_abi_change_alone_passes_with_the_option(self) -> None:
        baseline, candidate = _abi_change()

        checks = compare_interleaved_records(
            baseline, candidate, engines=_CUDA_ENGINES, allow_native_version_change=True
        )

        self.assertEqual(len(checks), 1)
        self.assertTrue(checks[0].passed, checks[0].reasons)
        self.assertAlmostEqual(checks[0].paired_median_slowdown or 0.0, 0.95)

    def test_option_still_rejects_driver_runtime_and_gpu_changes(self) -> None:
        for section, field, value in (
            ("native_cuda_abi", "driver_version", 13020),
            ("native_cuda_abi", "runtime_version", 12080),
            (None, "cuda_runtime", 12080),
            (None, "gpu", "NVIDIA GeForce RTX 4090"),
            (None, "gpu_compute_capability", "8.9"),
            (None, "cupy", "13.6.0"),
        ):
            with self.subTest(field=field):
                baseline, candidate = _abi_change()
                environment = candidate["environment"]
                (environment if section is None else environment[section])[field] = value

                checks = compare_interleaved_records(
                    baseline, candidate, engines=_CUDA_ENGINES, allow_native_version_change=True
                )

                self.assertFalse(checks[0].passed)
                self.assertIn("hardware or runtime fingerprint differs", checks[0].reasons)

    def test_native_cpu_interface_versions_follow_the_same_option(self) -> None:
        baseline = _merge([1.0] * 9, "baseline")
        candidate = _merge([0.95] * 9, "candidate")
        candidate["environment"]["native_cpu"].update(abi_version=2, python_api_version=3)

        rejected = compare_interleaved_records(baseline, candidate, engines=_CPU_ENGINES)
        accepted = compare_interleaved_records(
            baseline, candidate, engines=_CPU_ENGINES, allow_native_version_change=True
        )
        candidate["environment"]["native_cpu"]["parallel_threads"] = 2
        provider_change = compare_interleaved_records(
            baseline, candidate, engines=_CPU_ENGINES, allow_native_version_change=True
        )

        self.assertFalse(rejected[0].passed)
        self.assertTrue(accepted[0].passed, accepted[0].reasons)
        self.assertFalse(provider_change[0].passed)

    def test_gate_records_the_option_and_both_sides_native_versions(self) -> None:
        baseline, candidate = _abi_change()
        for allowed in (False, True):
            with self.subTest(allow_native_version_change=allowed):
                checks = compare_interleaved_records(
                    baseline,
                    candidate,
                    engines=_CUDA_ENGINES,
                    allow_native_version_change=allowed,
                )

                gate = report(
                    checks,
                    baseline=baseline,
                    candidate=candidate,
                    allow_native_version_change=allowed,
                )

                self.assertEqual(GATE_SCHEMA_VERSION, 2)
                self.assertEqual(gate["schema_version"], GATE_SCHEMA_VERSION)
                self.assertIs(gate["allow_native_version_change"], allowed)
                self.assertIs(gate["passed"], allowed)
                self.assertEqual(
                    gate["native_versions"],
                    {
                        "native_cuda_abi": {
                            "baseline": {"abi_version": 1, "python_api_version": 3},
                            "candidate": {"abi_version": 2, "python_api_version": 4},
                            "changed": True,
                        }
                    },
                )
                json.dumps(gate, allow_nan=False)

    def test_gate_lists_only_the_gated_native_families(self) -> None:
        baseline = _merge([1.0] * 9, "baseline")
        candidate = _merge([0.95] * 9, "candidate")
        checks = compare_interleaved_records(baseline, candidate, engines=_CPU_ENGINES)

        gate = report(
            checks,
            baseline=baseline,
            candidate=candidate,
            allow_native_version_change=False,
        )

        self.assertIs(gate["allow_native_version_change"], False)
        self.assertEqual(
            gate["native_versions"],
            {
                "native_cpu": {
                    "baseline": {"abi_version": 1, "python_api_version": 2},
                    "candidate": {"abi_version": 1, "python_api_version": 2},
                    "changed": False,
                }
            },
        )

    def test_gate_records_missing_native_metadata_as_null(self) -> None:
        baseline, candidate = _abi_change()
        del candidate["environment"]["native_cuda_abi"]
        checks = compare_interleaved_records(
            baseline, candidate, engines=_CUDA_ENGINES, allow_native_version_change=True
        )

        gate = report(
            checks,
            baseline=baseline,
            candidate=candidate,
            allow_native_version_change=True,
        )

        self.assertFalse(gate["passed"])
        self.assertEqual(
            gate["native_versions"]["native_cuda_abi"]["candidate"],
            {"abi_version": None, "python_api_version": None},
        )


class InterleavedCliTests(unittest.TestCase):
    REQUIRED = (
        "--baseline-python",
        "python",
        "--baseline-repo",
        "baseline",
        "--candidate-python",
        "python",
        "--candidate-repo",
        "candidate",
        "--output-dir",
        "out",
    )

    def test_parser_exposes_the_option_defaulting_to_false(self) -> None:
        parser = run_interleaved_benchmark.build_parser()

        self.assertIs(parser.parse_args(self.REQUIRED).allow_native_version_change, False)
        self.assertIs(
            parser.parse_args(
                [*self.REQUIRED, "--allow-native-version-change"]
            ).allow_native_version_change,
            True,
        )

    def _run_main(self, root: Path, *extra: str) -> tuple[int, dict[str, Any]]:
        rounds = {
            "baseline": _cuda_record(1.0, _NATIVE_CUDA_ABI_1),
            "candidate": _cuda_record(0.95, _NATIVE_CUDA_ABI_2),
        }

        def fake_round(**kwargs: Any) -> dict[str, Any]:
            return copy.deepcopy(rounds[kwargs["repo"].name])

        output = root / "out"
        argv = [
            "run_interleaved_benchmark.py",
            "--baseline-python",
            str(root / "python"),
            "--baseline-repo",
            str(root / "baseline"),
            "--candidate-python",
            str(root / "python"),
            "--candidate-repo",
            str(root / "candidate"),
            "--output-dir",
            str(output),
            "--pair-id",
            "pair",
            "--backend",
            "native_cuda",
            *extra,
        ]
        with (
            mock.patch.object(run_interleaved_benchmark, "_run_round", side_effect=fake_round),
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = run_interleaved_benchmark.main()
        return status, json.loads((output / "gate.json").read_text(encoding="utf-8"))

    def test_main_writes_the_option_and_native_versions_to_gate_json(self) -> None:
        expected_versions = {
            "native_cuda_abi": {
                "baseline": {"abi_version": 1, "python_api_version": 3},
                "candidate": {"abi_version": 2, "python_api_version": 4},
                "changed": True,
            }
        }
        for extra, allowed in (((), False), (("--allow-native-version-change",), True)):
            with self.subTest(allowed=allowed), tempfile.TemporaryDirectory() as directory:
                status, gate = self._run_main(Path(directory), *extra)

                self.assertEqual(status, 0 if allowed else 1)
                self.assertEqual(gate["schema_version"], GATE_SCHEMA_VERSION)
                self.assertIs(gate["allow_native_version_change"], allowed)
                self.assertIs(gate["passed"], allowed)
                self.assertEqual(gate["native_versions"], expected_versions)


if __name__ == "__main__":
    unittest.main()
