from __future__ import annotations

import copy
import unittest

from scripts.benchmarks.interleaved_regression import (
    compare_interleaved_records,
    merge_round_records,
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


class InterleavedRegressionTests(unittest.TestCase):
    def _merged(self, values: list[float], variant: str) -> dict[str, object]:
        records = [_record(value) for value in values]
        offset = 0 if variant == "baseline" else 1
        return merge_round_records(
            copy.deepcopy(records),
            variant=variant,
            pair_id="pair",
            execution_order=[(index + offset) % 2 for index in range(len(records))],
        )

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


if __name__ == "__main__":
    unittest.main()
