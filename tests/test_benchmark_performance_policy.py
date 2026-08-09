from __future__ import annotations

import ast
import dataclasses
import importlib
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from renewable_huber.state import RenewableHuberState
from scripts.benchmarks import benchmark_auto_dispatch
from scripts.benchmarks.benchmark_native_cpu_scaling import (
    _effective_threads,
    add_speedups,
    parse_n_jobs,
)
from scripts.benchmarks.benchmark_shape_sweep import (
    _measure,
    _restore_empty_state,
    _sample_repetitions,
)
from scripts.benchmarks.dispatch_policy import (
    RuntimeCapabilities,
    Workload,
    recommend_backend,
)
from scripts.benchmarks.performance_policy import compare_records, validate_record
from scripts.benchmarks.shape_sweep.shapes import PROFILES, Shape, make_batches
from scripts.benchmarks.shape_sweep.timing import _fit_batch


def _case(
    engine: str,
    seconds: list[float],
    *,
    penalty: str = "none",
    input_location: str = "host",
    lifecycle: str = "steady",
    operation: str = "partial_fit",
    dataset_seed: int = 42,
    dataset_sha256: str = "a" * 64,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, Any]:
    gpu_engine = engine in {
        "cupy_cuda_host_input",
        "cupy_cuda_device_input",
        "native_cuda_host_input",
        "native_cuda_device_input",
    }
    return {
        "shape": {
            "name": "wide",
            "samples": 16_384,
            "features": 256,
            "batch_size": 4_096,
        },
        "dataset_seed": dataset_seed,
        "dataset_sha256": dataset_sha256,
        "dtype": "float32",
        "penalty": penalty,
        "max_iter": max_iter,
        "tol": tol,
        "engine": engine,
        "result": {
            "seconds": seconds,
            "median_seconds": sorted(seconds)[len(seconds) // 2],
            "median_iterations": 5,
            "all_batches_converged": True,
            "lifecycle": lifecycle,
            "operation": operation,
            "input_location": input_location,
            "includes_input_transfer": gpu_engine and input_location == "host",
            "includes_engine_initialization": lifecycle == "cold",
            "resident_engine": lifecycle == "steady",
            "state_reset_timed": lifecycle == "cold",
        },
    }


def _record(*cases: dict[str, Any], processor: str = "fixed-cpu") -> dict[str, Any]:
    return {
        "schema": "renewable-huber-shape-sweep",
        "schema_version": 2,
        "environment": {
            "platform": "Windows-fixed",
            "processor": processor,
            "python": "3.11.0",
            "numpy": "2.4.6",
            "threading": {
                "MATMUL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "unset",
                "OMP_NUM_THREADS": "unset",
                "RAYON_NUM_THREADS": "4",
            },
            "numpy_blas_provider": {
                "blas": "OpenBLAS 0.3",
                "lapack": "OpenBLAS 0.3",
            },
            "gpu": "fixed-gpu",
            "gpu_compute_capability": "12.0",
            "cuda_runtime": 12090,
            "cupy": "14.1.1",
        },
        "cases": list(cases),
    }


class DispatchPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workload = Workload(
            samples=16_384,
            features=256,
            batch_size=4_096,
            dtype="float32",
            penalty="none",
        )

    def test_recommends_native_cuda_only_after_ten_percent_exact_win(self) -> None:
        record = _record(
            _case("numpy_cpu", [10.0]),
            _case("cupy_cuda_host_input", [4.0]),
            _case("native_cuda_host_input", [3.5]),
        )

        decision = recommend_backend(
            record,
            self.workload,
            RuntimeCapabilities(cupy=True, native_cuda=True),
        )

        self.assertEqual(decision.backend, "native_cuda")
        self.assertEqual(decision.reference_backend, "cupy")
        self.assertTrue(decision.calibration_found)

    def test_falls_back_to_cupy_when_native_margin_is_not_met(self) -> None:
        record = _record(
            _case("numpy_cpu", [10.0]),
            _case("cupy_cuda_host_input", [4.0]),
            _case("native_cuda_host_input", [3.8]),
        )

        decision = recommend_backend(
            record,
            self.workload,
            RuntimeCapabilities(cupy=True, native_cuda=True),
        )

        self.assertEqual(decision.backend, "cupy")
        self.assertIn("does not clear", decision.reason)

    def test_l1_never_selects_p2_native_cuda(self) -> None:
        record = _record(
            _case("numpy_cpu", [10.0], penalty="l1"),
            _case("cupy_cuda_host_input", [4.0], penalty="l1"),
        )
        workload = Workload(
            samples=16_384,
            features=256,
            batch_size=4_096,
            dtype="float32",
            penalty="l1",
        )

        decision = recommend_backend(
            record,
            workload,
            RuntimeCapabilities(cupy=True, native_cuda=True),
        )

        self.assertEqual(decision.backend, "cupy")
        self.assertIsNone(decision.native_seconds)

    def test_device_input_retains_cupy_without_exact_native_dlpack_calibration(self) -> None:
        record = _record(
            _case("cupy_cuda_device_input", [3.0], input_location="device"),
            _case("native_cuda_host_input", [1.0], input_location="host"),
        )
        workload = Workload(
            samples=16_384,
            features=256,
            batch_size=4_096,
            dtype="float32",
            penalty="none",
            input_location="device",
        )

        decision = recommend_backend(
            record,
            workload,
            RuntimeCapabilities(cupy=True, native_cuda=True),
        )

        self.assertEqual(decision.backend, "cupy")
        self.assertIn("no exact native CUDA DLPack calibration", decision.reason)

    def test_device_input_selects_native_after_exact_dlpack_win(self) -> None:
        record = _record(
            _case("cupy_cuda_device_input", [3.0], input_location="device"),
            _case("native_cuda_device_input", [2.0], input_location="device"),
        )
        decision = recommend_backend(
            record,
            Workload(
                samples=16_384,
                features=256,
                batch_size=4_096,
                dtype="float32",
                penalty="none",
                input_location="device",
            ),
            RuntimeCapabilities(cupy=True, native_cuda=True),
        )

        self.assertEqual(decision.backend, "native_cuda")
        self.assertEqual(decision.native_seconds, 2.0)

    def test_host_input_without_an_exact_calibration_stays_on_numpy(self) -> None:
        record = _record(_case("numpy_cpu", [1.0]))
        workload = Workload(
            samples=8_192,
            features=32,
            batch_size=4_096,
            dtype="float32",
            penalty="none",
        )

        decision = recommend_backend(record, workload, RuntimeCapabilities(native_cpu=True))

        self.assertEqual(decision.backend, "numpy")
        self.assertFalse(decision.calibration_found)

    def test_dispatch_refuses_to_mix_different_dataset_contracts(self) -> None:
        record = _record(
            _case("numpy_cpu", [10.0], dataset_seed=42),
            _case("rust_native_cpu", [1.0], dataset_seed=43),
        )

        with self.assertRaisesRegex(ValueError, "multiple dataset or solver contracts"):
            recommend_backend(
                record,
                self.workload,
                RuntimeCapabilities(native_cpu=True),
            )

    def test_unconverged_native_calibration_is_not_eligible(self) -> None:
        native = _case("rust_native_cpu", [1.0])
        native["result"]["all_batches_converged"] = False
        record = _record(_case("numpy_cpu", [10.0]), native)

        decision = recommend_backend(
            record,
            self.workload,
            RuntimeCapabilities(native_cpu=True),
        )

        self.assertEqual(decision.backend, "numpy")
        self.assertIsNone(decision.native_seconds)

    def test_solver_settings_must_match_the_runtime_workload(self) -> None:
        record = _record(
            _case("numpy_cpu", [10.0], max_iter=50),
            _case("rust_native_cpu", [1.0], max_iter=50),
        )

        decision = recommend_backend(
            record,
            self.workload,
            RuntimeCapabilities(native_cpu=True),
        )

        self.assertEqual(decision.backend, "numpy")
        self.assertFalse(decision.calibration_found)


class PerformanceGateTests(unittest.TestCase):
    def test_gate_refuses_to_compare_different_sampling_contracts(self) -> None:
        baseline = _record(_case("numpy_cpu", [1.20] * 9), _case("rust_native_cpu", [1.0] * 9))
        candidate = _record(_case("numpy_cpu", [1.20] * 9), _case("rust_native_cpu", [1.0] * 9))
        for case in candidate["cases"]:
            case["result"].update(
                {
                    "sample_aggregation": "arithmetic_mean",
                    "sample_repetitions": 10,
                    "minimum_sample_seconds": 0.1,
                    "gc_collected_before_sample": True,
                    "gc_disabled_during_timing": True,
                }
            )

        checks = compare_records(baseline, candidate)

        self.assertFalse(checks[0].passed)
        self.assertIn("missing", " ".join(checks[0].reasons))

    def test_native_cpu_gate_accepts_a_small_regression_on_a_stable_fixed_runner(self) -> None:
        baseline = _record(_case("numpy_cpu", [1.20] * 9), _case("rust_native_cpu", [1.0] * 9))
        candidate = _record(_case("numpy_cpu", [1.20] * 9), _case("rust_native_cpu", [1.05] * 9))

        checks = compare_records(baseline, candidate)

        self.assertEqual(len(checks), 1)
        self.assertTrue(checks[0].passed)
        self.assertAlmostEqual(checks[0].slowdown_ratio or 0.0, 1.05)

    def test_native_cpu_gate_rejects_more_than_ten_percent_slowdown(self) -> None:
        baseline = _record(_case("numpy_cpu", [1.20] * 9), _case("rust_native_cpu", [1.0] * 9))
        candidate = _record(_case("numpy_cpu", [1.20] * 9), _case("rust_native_cpu", [1.11] * 9))

        checks = compare_records(baseline, candidate)

        self.assertFalse(checks[0].passed)
        self.assertIn("slowdown", " ".join(checks[0].reasons))

    def test_gate_rejects_hardware_changes_before_interpreting_timings(self) -> None:
        baseline = _record(
            _case("numpy_cpu", [1.20] * 9),
            _case("rust_native_cpu", [1.0] * 9),
            processor="fixed-cpu-a",
        )
        candidate = _record(
            _case("numpy_cpu", [1.20] * 9),
            _case("rust_native_cpu", [1.0] * 9),
            processor="fixed-cpu-b",
        )

        checks = compare_records(baseline, candidate)

        self.assertFalse(checks[0].passed)
        self.assertIn("fingerprint", " ".join(checks[0].reasons))

    def test_legacy_schema_cannot_be_promoted_to_a_hard_performance_gate(self) -> None:
        legacy = _record(_case("numpy_cpu", [1.20] * 9), _case("rust_native_cpu", [1.0] * 9))
        legacy["schema_version"] = 1

        with self.assertRaisesRegex(ValueError, "schema version 2"):
            validate_record(legacy)

    def test_gate_requires_native_to_match_its_direct_numpy_competitor(self) -> None:
        baseline = _record(_case("numpy_cpu", [1.20] * 9), _case("rust_native_cpu", [1.0] * 9))
        candidate = _record(_case("numpy_cpu", [0.90] * 9), _case("rust_native_cpu", [1.0] * 9))

        checks = compare_records(baseline, candidate)

        self.assertFalse(checks[0].passed)
        self.assertEqual(checks[0].competitor_engine, "numpy_cpu")
        self.assertIn("native/reference", " ".join(checks[0].reasons))

    def test_gate_requires_native_cuda_to_match_its_host_input_cupy_competitor(self) -> None:
        baseline = _record(
            _case("cupy_cuda_host_input", [1.20] * 9),
            _case("native_cuda_host_input", [1.0] * 9),
        )
        candidate = _record(
            _case("cupy_cuda_host_input", [0.90] * 9),
            _case("native_cuda_host_input", [1.0] * 9),
        )

        checks = compare_records(baseline, candidate)

        self.assertFalse(checks[0].passed)
        self.assertEqual(checks[0].competitor_engine, "cupy_cuda_host_input")
        self.assertIn("native/reference", " ".join(checks[0].reasons))

    def test_gate_refuses_a_solver_or_dataset_mismatch_as_a_missing_equivalent_case(self) -> None:
        baseline = _record(_case("numpy_cpu", [1.20] * 9), _case("rust_native_cpu", [1.0] * 9))
        candidate = _record(
            _case("numpy_cpu", [1.20] * 9, tol=1e-5),
            _case("rust_native_cpu", [1.0] * 9, tol=1e-5),
        )

        checks = compare_records(baseline, candidate)

        self.assertFalse(checks[0].passed)
        self.assertIn("missing", " ".join(checks[0].reasons))

    def test_gate_refuses_a_dataset_checksum_mismatch(self) -> None:
        baseline = _record(_case("numpy_cpu", [1.20] * 9), _case("rust_native_cpu", [1.0] * 9))
        candidate = _record(
            _case("numpy_cpu", [1.20] * 9, dataset_sha256="b" * 64),
            _case("rust_native_cpu", [1.0] * 9, dataset_sha256="b" * 64),
        )

        checks = compare_records(baseline, candidate)

        self.assertFalse(checks[0].passed)
        self.assertIn("missing", " ".join(checks[0].reasons))

    def test_record_rejects_mislabeled_lifecycle_timing_contract(self) -> None:
        malformed = _record(_case("numpy_cpu", [1.0]))
        malformed["cases"][0]["result"]["includes_engine_initialization"] = True
        malformed["cases"][0]["result"]["resident_engine"] = False
        malformed["cases"][0]["result"]["state_reset_timed"] = True

        with self.assertRaisesRegex(ValueError, "lifecycle labels"):
            validate_record(malformed)

    def test_record_rejects_non_finite_timings_and_iterations(self) -> None:
        for field, value in (
            ("seconds", float("nan")),
            ("seconds", float("inf")),
            ("median_iterations", float("nan")),
            ("median_iterations", float("inf")),
        ):
            with self.subTest(field=field, value=value):
                malformed = _record(_case("numpy_cpu", [1.0]))
                if field == "seconds":
                    malformed["cases"][0]["result"][field] = [value]
                else:
                    malformed["cases"][0]["result"][field] = value

                with self.assertRaisesRegex(ValueError, "finite"):
                    validate_record(malformed)

    def test_record_rejects_non_boolean_convergence(self) -> None:
        malformed = _record(_case("numpy_cpu", [1.0]))
        malformed["cases"][0]["result"]["all_batches_converged"] = "false"

        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            validate_record(malformed)

    def test_gate_rejects_non_finite_slowdown_threshold(self) -> None:
        baseline = _record(_case("numpy_cpu", [1.20] * 9), _case("rust_native_cpu", [1.0] * 9))
        candidate = _record(_case("numpy_cpu", [1.20] * 9), _case("rust_native_cpu", [1.0] * 9))

        with self.assertRaisesRegex(ValueError, "maximum slowdown"):
            compare_records(baseline, candidate, cpu_max_slowdown=float("nan"))

    def test_gate_rejects_thread_or_blas_provider_change(self) -> None:
        baseline = _record(_case("numpy_cpu", [1.20] * 9), _case("rust_native_cpu", [1.0] * 9))
        candidate = _record(_case("numpy_cpu", [1.20] * 9), _case("rust_native_cpu", [1.0] * 9))
        candidate["environment"]["threading"]["OPENBLAS_NUM_THREADS"] = "8"
        candidate["environment"]["numpy_blas_provider"]["blas"] = "MKL 2026"

        checks = compare_records(baseline, candidate)

        self.assertFalse(checks[0].passed)
        self.assertIn("fingerprint", " ".join(checks[0].reasons))

    def test_gate_rejects_native_cpu_provider_or_parallelism_change(self) -> None:
        native_cpu = {
            "abi_version": 1,
            "python_api_version": 1,
            "linear_algebra_provider": "nalgebra+matrixmultiply",
            "parallel_provider": "rayon",
            "parallel_threads": 24,
        }
        for field, value in (
            ("abi_version", 2),
            ("linear_algebra_provider", "different-provider"),
            ("parallel_provider", "different-parallel-runtime"),
            ("parallel_threads", 12),
        ):
            with self.subTest(field=field):
                baseline = _record(
                    _case("numpy_cpu", [1.20] * 9),
                    _case("rust_native_cpu", [1.0] * 9),
                )
                candidate = _record(
                    _case("numpy_cpu", [1.20] * 9),
                    _case("rust_native_cpu", [1.0] * 9),
                )
                baseline["environment"]["native_cpu"] = native_cpu
                candidate["environment"]["native_cpu"] = {**native_cpu, field: value}

                checks = compare_records(baseline, candidate)

                self.assertFalse(checks[0].passed)
                self.assertIn("fingerprint", " ".join(checks[0].reasons))

    def test_gate_rejects_native_cuda_driver_or_abi_change(self) -> None:
        native_cuda_abi = {
            "abi_version": 1,
            "python_api_version": 2,
            "driver_version": 13020,
            "runtime_version": 12090,
        }
        for field, value in (
            ("abi_version", 2),
            ("python_api_version", 3),
            ("driver_version", 14000),
            ("runtime_version", 13000),
        ):
            with self.subTest(field=field):
                baseline = _record(
                    _case("cupy_cuda_host_input", [1.20] * 9),
                    _case("native_cuda_host_input", [1.0] * 9),
                )
                candidate = _record(
                    _case("cupy_cuda_host_input", [1.20] * 9),
                    _case("native_cuda_host_input", [1.0] * 9),
                )
                baseline["environment"]["native_cuda_abi"] = native_cuda_abi
                candidate["environment"]["native_cuda_abi"] = {
                    **native_cuda_abi,
                    field: value,
                }

                checks = compare_records(baseline, candidate)

                self.assertFalse(checks[0].passed)
                self.assertIn("fingerprint", " ".join(checks[0].reasons))


class BenchmarkLifecycleTests(unittest.TestCase):
    def test_short_operations_are_aggregated_without_dropping_observations(self) -> None:
        calls = {"operation": 0, "prepare": 0, "finalize": 0}

        def operation() -> tuple[int, bool]:
            calls["operation"] += 1
            return 7, True

        def prepare() -> None:
            calls["prepare"] += 1

        def finalize() -> None:
            calls["finalize"] += 1

        result = _measure(
            operation,
            repeats=3,
            prepare=prepare,
            finalize=finalize,
            estimated_operation_seconds=0.02,
            minimum_sample_seconds=0.1,
            max_sample_repetitions=64,
        )

        self.assertEqual(result["sample_repetitions"], 5)
        self.assertEqual(calls, {"operation": 15, "prepare": 15, "finalize": 15})
        self.assertEqual(len(result["seconds"]), 3)
        self.assertEqual(result["iterations"], [7, 7, 7])
        self.assertTrue(result["all_batches_converged"])
        self.assertEqual(result["sample_aggregation"], "arithmetic_mean")

    def test_sample_repetition_calibration_is_bounded_and_fixed(self) -> None:
        self.assertEqual(
            _sample_repetitions(
                0.02,
                minimum_sample_seconds=0.1,
                maximum=64,
            ),
            5,
        )
        self.assertEqual(
            _sample_repetitions(
                0.0001,
                minimum_sample_seconds=0.1,
                maximum=64,
            ),
            64,
        )
        self.assertEqual(
            _sample_repetitions(None, minimum_sample_seconds=0.1, maximum=64),
            1,
        )

    def test_native_restore_and_estimator_share_the_same_reset_snapshot(self) -> None:
        class Backend:
            restored_token: int | None = None

            def restore_native_state(self, state: RenewableHuberState) -> None:
                self.restored_token = state.mirror_token

        class Model:
            fit_intercept = True

            def __init__(self) -> None:
                self._backend = Backend()
                self._state = None
                self._diagnostics = object()
                self.n_features_in_ = -1
                self.n_samples_seen_ = -1
                self.n_iter_ = -1

            def _sync_public_coefficients(self) -> None:
                pass

        empty_state = RenewableHuberState.empty(
            3,
            fit_intercept=True,
            xp=np,
            dtype=np.float64,
        )
        model = Model()

        _restore_empty_state(model, empty_state)

        self.assertIsNot(model._state, empty_state)
        self.assertEqual(model._backend.restored_token, model._state.mirror_token)
        self.assertEqual(model.n_features_in_, 3)
        self.assertEqual(model.n_samples_seen_, 0)
        self.assertEqual(model.n_iter_, 0)


class NativeCpuScalingPolicyTests(unittest.TestCase):
    def test_n_jobs_parser_requires_unique_one_thread_baseline(self) -> None:
        self.assertEqual(parse_n_jobs("1,2,4,8,-1"), (1, 2, 4, 8, -1))
        for value in ("", "2,4", "1,0", "1,-2", "1,2,2", "1,nope"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_n_jobs(value)

    def test_speedup_is_based_on_median_one_thread_result(self) -> None:
        cases = [
            {
                "requested_n_jobs": 4,
                "effective_threads": 4,
                "result": {"seconds": [0.3, 0.25, 0.2], "median_seconds": 0.25},
            },
            {
                "requested_n_jobs": 1,
                "effective_threads": 1,
                "result": {"seconds": [1.1, 1.0, 0.9], "median_seconds": 1.0},
            },
            {
                "requested_n_jobs": -1,
                "effective_threads": None,
                "result": {"seconds": [0.22, 0.2, 0.18], "median_seconds": 0.2},
            },
        ]

        add_speedups(cases)

        self.assertEqual(cases[0]["speedup_vs_n_jobs_1"], 4.0)
        self.assertEqual(cases[0]["parallel_efficiency"], 1.0)
        self.assertEqual(cases[1]["speedup_vs_n_jobs_1"], 1.0)
        self.assertEqual(cases[2]["speedup_vs_n_jobs_1"], 5.0)
        self.assertIsNone(cases[2]["parallel_efficiency"])

    def test_speedup_rejects_missing_or_nonpositive_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_jobs=1"):
            add_speedups([{"requested_n_jobs": 2, "result": {"median_seconds": 1.0}}])
        with self.assertRaisesRegex(ValueError, "positive"):
            add_speedups([{"requested_n_jobs": 1, "result": {"median_seconds": 0.0}}])

    def test_effective_thread_count_prefers_fitted_estimator_contract(self) -> None:
        class Backend:
            effective_n_jobs = 7

        class Model:
            n_jobs_ = 4
            _backend = Backend()

        self.assertEqual(
            _effective_threads(Model(), requested=-1),
            (4, "Model.n_jobs_"),
        )


class ShapeSweepReExportTests(unittest.TestCase):
    """``benchmark_shape_sweep`` is a stable entry point, so prove it stays one.

    The sweep implementation moved into ``scripts/benchmarks/shape_sweep/`` and
    the entry point now only re-exports. Nothing in an ordinary refactor fails
    if one of those re-exports disappears: the two consumers below break at
    import time, which shows up as a broken benchmark run rather than as a test
    failure. The required names are therefore read back out of the consumers'
    own import statements and checked against the entry point.
    """

    #: The names the split had to preserve. Kept literal as an anti-vacuity
    #: check on the parser below; the parser is what keeps this honest.
    DOCUMENTED = frozenset(
        {
            "PROFILES",
            "_calibration_run",
            "_dataset_checksum",
            "_fit_batch",
            "_lifecycle_metadata",
            "_measure",
            "_restore_empty_state",
            "_run_operation",
            "_sample_repetitions",
            "environment_metadata",
            "make_batches",
        }
    )
    CONSUMERS = (
        "scripts/benchmarks/benchmark_native_cpu_scaling.py",
        "tests/test_benchmark_performance_policy.py",
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = importlib.import_module("scripts.benchmarks.benchmark_shape_sweep")
        cls.root = Path(cls.module.__file__).resolve().parents[2]

    def _imported_names(self) -> set[str]:
        names: set[str] = set()
        for relative in self.CONSUMERS:
            tree = ast.parse((self.root / relative).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "scripts.benchmarks.benchmark_shape_sweep"
                ):
                    names.update(alias.name for alias in node.names)
        return names

    def test_the_consumers_still_import_the_documented_names(self) -> None:
        imported = self._imported_names()
        # A consumer dropping one is the signal to revisit the list above,
        # rather than to let it quietly go stale.
        self.assertEqual(self.DOCUMENTED - imported, set())

    def test_every_consumed_name_is_exported_and_reachable(self) -> None:
        imported = self._imported_names()
        self.assertGreaterEqual(len(imported), len(self.DOCUMENTED))
        exported = set(self.module.__all__)
        for name in sorted(imported):
            with self.subTest(name=name):
                # Extra exports are fine; a missing one is not.
                self.assertIn(name, exported)
                self.assertTrue(hasattr(self.module, name))

    def test_every_exported_name_resolves(self) -> None:
        # ``__all__`` must not promise something the module does not define.
        self.assertTrue(self.module.__all__)
        for name in self.module.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(self.module, name))

    def test_the_entry_point_still_exposes_the_cli(self) -> None:
        self.assertTrue(callable(self.module.main))
        self.assertIn("main", self.module.__all__)


class AutoDispatchBenchmarkContractTests(unittest.TestCase):
    """The auto-dispatch harness must run the work its record describes.

    Nothing here times anything: the estimator is replaced by a recorder, so
    these assertions are about *which calls are made with which arrays*, which
    is deterministic on any machine. A harness that fits one batch while its
    header, its ``shape`` record and its ``work_units`` describe the whole
    dataset reports a wrong number rather than failing, and only a check at
    this level notices.
    """

    #: Small enough to be free, and four batches deep so "first batch" and
    #: "whole dataset" cannot coincide.
    SHAPE = Shape("contract", 128, 4, 32)

    def _case(self, operation: str) -> Any:
        return benchmark_auto_dispatch.Case(
            shape=self.SHAPE,
            profile="contract",
            dtype="float64",
            penalty="none",
            operation=operation,
        )

    def _data(self) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
        batches = make_batches(self.SHAPE, seed=7, dtype="float64")
        return batches, _fit_batch(batches, xp=np)

    @staticmethod
    def _recorder() -> tuple[list[tuple[str, np.ndarray, np.ndarray]], type]:
        calls: list[tuple[str, np.ndarray, np.ndarray]] = []

        class Estimator:
            def __init__(self, **_: Any) -> None:
                self.backend_ = "numpy"

            def fit(self, features: np.ndarray, target: np.ndarray) -> Any:
                calls.append(("fit", features, target))
                return self

            def partial_fit(self, features: np.ndarray, target: np.ndarray) -> Any:
                calls.append(("partial_fit", features, target))
                return self

        return calls, Estimator

    def test_a_fit_case_consumes_the_whole_dataset_in_one_call(self) -> None:
        batches, fit_batch = self._data()
        calls, estimator = self._recorder()

        with mock.patch.object(benchmark_auto_dispatch, "RenewableHuberRegressor", estimator):
            benchmark_auto_dispatch._one_run("numpy", self._case("fit"), batches, fit_batch)

        self.assertEqual([name for name, _, _ in calls], ["fit"])
        _, features, target = calls[0]
        # The row count the header and the shape record both claim.
        self.assertEqual(features.shape[0], self.SHAPE.samples)
        self.assertEqual(target.shape[0], self.SHAPE.samples)
        self.assertGreater(len(batches), 1)

    def test_a_stream_case_stays_per_batch(self) -> None:
        batches, fit_batch = self._data()
        calls, estimator = self._recorder()

        with mock.patch.object(benchmark_auto_dispatch, "RenewableHuberRegressor", estimator):
            benchmark_auto_dispatch._one_run("numpy", self._case("stream"), batches, fit_batch)

        self.assertEqual([name for name, _, _ in calls], ["partial_fit"] * len(batches))
        for (_, features, target), (expected_X, expected_y) in zip(calls, batches, strict=True):
            self.assertIs(features, expected_X)
            self.assertIs(target, expected_y)

    def test_the_fit_batch_is_built_once_and_never_inside_a_timed_run(self) -> None:
        batches, fit_batch = self._data()
        calls, estimator = self._recorder()
        case = self._case("fit")

        with mock.patch.object(benchmark_auto_dispatch, "RenewableHuberRegressor", estimator):
            benchmark_auto_dispatch._measure(case, batches, fit_batch, warmup=1, repeats=2)

        # One untimed auto prime, then two steady rounds and a separate cold
        # phase; every measured engine sample has one warmup and one timed run.
        expected_runs = 1 + 2 * 2 * len(benchmark_auto_dispatch.STEADY_ENGINES) + 2 * 2
        self.assertEqual(len(calls), expected_runs)
        # Identity, not equality: rebuilding or copying the concatenation per
        # run would put that cost inside the timed region for every sample.
        for _, features, target in calls:
            self.assertIs(features, fit_batch[0])
            self.assertIs(target, fit_batch[1])

    def test_work_units_count_all_samples_for_fit_and_the_first_batch_for_a_stream(
        self,
    ) -> None:
        design_width = float(self.SHAPE.features + 1) ** 2
        self.assertEqual(
            benchmark_auto_dispatch._work_units(self._case("fit")),
            float(self.SHAPE.samples) * design_width,
        )
        self.assertEqual(
            benchmark_auto_dispatch._work_units(self._case("stream")),
            float(self.SHAPE.batch_size) * design_width,
        )

    def test_a_single_batch_stream_is_counted_as_the_whole_dataset(self) -> None:
        # ``make_batches`` never emits a batch longer than the dataset, so a
        # stream that fits in one batch decides on every sample there is.
        shape = Shape("one-batch", 64, 4, 4_096)
        case = dataclasses.replace(self._case("stream"), shape=shape)
        batches = make_batches(shape, seed=7, dtype="float64")

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0].shape[0], shape.samples)
        self.assertEqual(
            benchmark_auto_dispatch._work_units(case),
            float(shape.samples) * float(shape.features + 1) ** 2,
        )

    def test_reported_work_units_match_the_batch_the_policy_is_asked_about(self) -> None:
        batches, fit_batch = self._data()
        for operation, deciding in (("fit", fit_batch), ("stream", batches[0])):
            with self.subTest(operation=operation):
                case = self._case(operation)
                expected = float(deciding[0].shape[0]) * float(self.SHAPE.features + 1) ** 2
                # ``_summarise`` tolerates an empty result set; the field under
                # test is derived from the case, not from any timing.
                summary = benchmark_auto_dispatch._summarise({}, case)
                self.assertEqual(summary["work_units"], expected)

    def test_every_shipped_shape_reports_both_operations_consistently(self) -> None:
        for shapes in PROFILES.values():
            for shape in shapes:
                with self.subTest(shape=shape.name):
                    width = float(shape.features + 1) ** 2
                    fit_case = benchmark_auto_dispatch.Case(shape, "p", "float64", "none", "fit")
                    stream_case = dataclasses.replace(fit_case, operation="stream")
                    fit_units = benchmark_auto_dispatch._work_units(fit_case)
                    stream_units = benchmark_auto_dispatch._work_units(stream_case)
                    self.assertEqual(fit_units, float(shape.samples) * width)
                    self.assertEqual(
                        stream_units,
                        float(min(shape.batch_size, shape.samples)) * width,
                    )
                    self.assertGreaterEqual(fit_units, stream_units)

    def test_the_recorded_runtime_signature_makes_no_native_pool_claim(self) -> None:
        # ``parallel_threads`` is deliberately not an input to the signature:
        # reading it forces the extension to be imported, which is what
        # calibrating does, so the first calibration would be paid twice. The
        # harness must not describe the field as carrying it either.
        signature = benchmark_auto_dispatch.current_runtime_signature()
        fields = {field.name for field in dataclasses.fields(signature)}
        self.assertNotIn("parallel_threads", fields)
        source = Path(benchmark_auto_dispatch.__file__).read_text(encoding="utf-8")
        for stale in ("parallel_threads", "native pool"):
            self.assertNotIn(stale, source)

    def test_engine_order_alternates_forward_and_reverse(self) -> None:
        self.assertEqual(
            benchmark_auto_dispatch._engine_order(0), benchmark_auto_dispatch.STEADY_ENGINES
        )
        self.assertEqual(
            benchmark_auto_dispatch._engine_order(1),
            tuple(reversed(benchmark_auto_dispatch.STEADY_ENGINES)),
        )

    def test_an_odd_repeat_count_is_rejected_as_unbalanced(self) -> None:
        batches, fit_batch = self._data()
        with self.assertRaisesRegex(ValueError, "positive even"):
            benchmark_auto_dispatch._measure(
                self._case("fit"), batches, fit_batch, warmup=0, repeats=3
            )

    def test_regret_is_the_median_of_aligned_ratios(self) -> None:
        def result(seconds: list[float]) -> dict[str, Any]:
            return {
                "seconds": seconds,
                "median_seconds": float(np.median(seconds)),
                "auto_dispatch": None,
            }

        results = {
            "numpy": result([1.0, 1.0, 100.0]),
            "native_cpu": result([2.0, 2.0, 200.0]),
            "auto_warm": result([1.0, 100.0, 100.0]),
            "auto_cold": {
                **result([2.0, 101.0, 102.0]),
                "auto_dispatch": {"calibrated": True, "calibration_seconds": 1.0},
            },
        }
        summary = benchmark_auto_dispatch._summarise(results, self._case("fit"))
        self.assertEqual(summary["regret"], 1.0)
        self.assertEqual(summary["calibration_overhead_seconds"], 1.0)
        self.assertIn("aligned", summary["regret_statistic"])

    def test_no_calibration_reports_no_overhead_instead_of_timer_noise(self) -> None:
        result = {
            "seconds": [1.0, 2.0],
            "median_seconds": 1.5,
            "auto_dispatch": {"calibrated": False, "calibration_seconds": 0.0},
        }
        summary = benchmark_auto_dispatch._summarise(
            {"auto_cold": result, "auto_warm": result}, self._case("fit")
        )
        self.assertIsNone(summary["calibration_overhead_seconds"])


if __name__ == "__main__":
    unittest.main()
