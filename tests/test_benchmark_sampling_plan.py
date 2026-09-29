"""Frozen sample repetitions and one shared harness for an interleaved A/B.

Everything here runs on NumPy alone, so it lives in the ``performance``
profile: the plan decides how noisy every GPU A/B sample is, and a regression
would only show up as a gate that fails on a desktop in normal use.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from scripts.benchmarks import run_interleaved_benchmark
from scripts.benchmarks.interleaved_regression import (
    compare_interleaved_records,
    merge_round_records,
)
from scripts.benchmarks.performance_policy import NATIVE_ENGINES
from scripts.benchmarks.shape_sweep import cli, source_root, timing
from scripts.benchmarks.shape_sweep.sampling_plan import (
    PLAN_SCHEMA,
    PLAN_SCHEMA_VERSION,
    case_plan_key,
    load_plan,
    plan_from_records,
    plan_key,
    planned_repetitions,
    write_plan,
)
from tests.test_benchmark_interleaved_regression import _record


def _case(engine: str, repetitions: int, *, shape: str = "tiny") -> dict[str, Any]:
    return {
        "shape": {"name": shape},
        "dtype": "float64",
        "penalty": "none",
        "engine": engine,
        "result": {
            "lifecycle": "cold",
            "operation": "partial_fit",
            "sample_repetitions": repetitions,
        },
    }


def _sweep(*cases: dict[str, Any]) -> dict[str, Any]:
    return {"cases": list(cases)}


class SamplingPlanTests(unittest.TestCase):
    def test_plan_key_matches_the_key_of_a_measured_case(self) -> None:
        self.assertEqual(
            case_plan_key(_case("numpy_cpu", 3)),
            plan_key(
                shape="tiny",
                dtype="float64",
                penalty="none",
                lifecycle="cold",
                operation="partial_fit",
                engine="numpy_cpu",
            ),
        )

    def test_each_case_takes_the_largest_calibrated_block(self) -> None:
        plan = plan_from_records(
            [
                _sweep(_case("numpy_cpu", 42), _case("rust_native_cpu", 64)),
                _sweep(_case("numpy_cpu", 60), _case("rust_native_cpu", 36)),
            ]
        )

        self.assertEqual(list(plan.values()), [60, 64])
        self.assertEqual(list(plan), sorted(plan))

    def test_calibrations_must_cover_identical_cases(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical benchmark cases"):
            plan_from_records([_sweep(_case("numpy_cpu", 1)), _sweep(_case("rust_native_cpu", 1))])
        with self.assertRaisesRegex(ValueError, "duplicate case"):
            plan_from_records([_sweep(_case("numpy_cpu", 1), _case("numpy_cpu", 2))])
        with self.assertRaisesRegex(ValueError, "no measured cases"):
            plan_from_records([_sweep()])
        for bad in (0, True, 1.5):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "positive integer"):
                plan_from_records([_sweep(_case("numpy_cpu", bad))])  # type: ignore[arg-type]

    def test_written_plan_round_trips_with_the_hash_of_its_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            written = write_plan(path, {"b": 2, "a": 5})
            loaded, digest = load_plan(path)

            self.assertEqual(loaded, {"a": 5, "b": 2})
            self.assertEqual(written, digest)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_load_rejects_anything_but_a_valid_plan(self) -> None:
        documents: list[Any] = [
            [],
            {"schema": "other", "schema_version": PLAN_SCHEMA_VERSION, "sample_repetitions": {}},
            {"schema": PLAN_SCHEMA, "schema_version": 99, "sample_repetitions": {"a": 1}},
            {"schema": PLAN_SCHEMA, "schema_version": PLAN_SCHEMA_VERSION},
            {
                "schema": PLAN_SCHEMA,
                "schema_version": PLAN_SCHEMA_VERSION,
                "sample_repetitions": {},
            },
            {
                "schema": PLAN_SCHEMA,
                "schema_version": PLAN_SCHEMA_VERSION,
                "sample_repetitions": {"a": 0},
            },
            {
                "schema": PLAN_SCHEMA,
                "schema_version": PLAN_SCHEMA_VERSION,
                "sample_repetitions": {"a": True},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            for document in documents:
                with self.subTest(document=document):
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_plan(path)

    def test_a_plan_must_cover_every_case_it_is_asked_about(self) -> None:
        self.assertIsNone(planned_repetitions(None, "anything"))
        self.assertEqual(planned_repetitions({"a": 4}, "a"), 4)
        with self.assertRaisesRegex(ValueError, "no entry for b"):
            planned_repetitions({"a": 4}, "b")


class PlannedMeasureTests(unittest.TestCase):
    @staticmethod
    def _counting_operation() -> tuple[Any, list[int]]:
        calls: list[int] = []

        def operation() -> tuple[int, bool]:
            calls.append(1)
            return 3, True

        return operation, calls

    def test_a_planned_block_overrides_the_warmup_estimate(self) -> None:
        operation, calls = self._counting_operation()
        # The estimate alone would choose min(64, ceil(0.1 / 1e-6)) = 64.
        result = timing._measure(
            operation,
            repeats=2,
            estimated_operation_seconds=1e-6,
            minimum_sample_seconds=0.1,
            sample_repetitions=3,
        )

        self.assertEqual(len(calls), 6)
        self.assertEqual(result["sample_repetitions"], 3)
        self.assertEqual(result["sample_repetitions_source"], "plan")

    def test_without_a_plan_the_record_is_unchanged(self) -> None:
        operation, calls = self._counting_operation()
        result = timing._measure(
            operation,
            repeats=1,
            estimated_operation_seconds=0.025,
            minimum_sample_seconds=0.1,
        )

        self.assertEqual(result["sample_repetitions"], 4)
        self.assertEqual(len(calls), 4)
        self.assertNotIn("sample_repetitions_source", result)

    def test_a_planned_block_must_be_a_positive_integer(self) -> None:
        operation, _ = self._counting_operation()
        for bad in (0, -1, True, 2.0):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "positive integer"):
                timing._measure(operation, repeats=1, sample_repetitions=bad)  # type: ignore[arg-type]


class ShapeSweepPlanOptionTests(unittest.TestCase):
    ARGS = (
        "--profile",
        "smoke",
        "--case",
        "latency-smoke",
        "--backend",
        "numpy",
        "--penalty",
        "none",
        "--dtype",
        "float64",
        "--lifecycle",
        "cold",
        "--operation",
        "partial-fit",
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--max-iter",
        "5",
    )
    KEY = plan_key(
        shape="latency-smoke",
        dtype="float64",
        penalty="none",
        lifecycle="cold",
        operation="partial_fit",
        engine="numpy_cpu",
    )

    def _sweep(self, directory: Path, *extra: str) -> dict[str, Any]:
        output = directory / "sweep.json"
        argv = ["benchmark_shape_sweep.py", *self.ARGS, *extra, "--output", str(output)]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(), 0)
        return json.loads(output.read_text(encoding="utf-8"))

    def test_the_sweep_uses_and_records_the_plan(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            plan_path = directory / "plan.json"
            digest = write_plan(plan_path, {self.KEY: 2})

            record = self._sweep(directory, "--sample-repetitions-plan", str(plan_path))

        (case,) = record["cases"]
        self.assertEqual(case["result"]["sample_repetitions"], 2)
        self.assertEqual(case["result"]["sample_repetitions_source"], "plan")
        self.assertEqual(record["arguments"]["sample_repetitions_plan_sha256"], digest)

    def test_without_a_plan_the_record_carries_no_plan_fields(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            record = self._sweep(Path(name), "--minimum-sample-seconds", "0")

        (case,) = record["cases"]
        self.assertNotIn("sample_repetitions_plan_sha256", record["arguments"])
        self.assertNotIn("sample_repetitions_source", case["result"])
        self.assertNotIn("benchmark_harness_git_revision", record["environment"])

    def test_a_plan_missing_a_measured_case_stops_the_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            plan_path = directory / "plan.json"
            write_plan(plan_path, {"some/other/case/cold/partial_fit/numpy_cpu": 2})

            with self.assertRaisesRegex(ValueError, "no entry for latency-smoke"):
                self._sweep(directory, "--sample-repetitions-plan", str(plan_path))

    def test_an_unreadable_plan_is_a_usage_error(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            plan_path = directory / "plan.json"
            plan_path.write_text("{}", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self._sweep(directory, "--sample-repetitions-plan", str(plan_path))


class SourceRootTests(unittest.TestCase):
    def test_default_is_the_harness_checkout(self) -> None:
        with mock.patch.dict(os.environ, {source_root.ENVIRONMENT_VARIABLE: ""}):
            self.assertEqual(source_root.source_root(), source_root.HARNESS_ROOT)
            self.assertFalse(source_root.is_overridden())

    def test_an_override_must_contain_the_package(self) -> None:
        with (
            tempfile.TemporaryDirectory() as name,
            mock.patch.dict(os.environ, {source_root.ENVIRONMENT_VARIABLE: name}),
            self.assertRaisesRegex(RuntimeError, "does not contain src/renewable_huber"),
        ):
            source_root.source_root()

    def test_an_override_moves_its_src_first_and_drops_the_harness_src(self) -> None:
        harness_src = str(source_root.HARNESS_ROOT / "src")
        with tempfile.TemporaryDirectory() as name:
            package = Path(name) / "src" / "renewable_huber"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            override_src = str(Path(name).resolve() / "src")
            path = ["/elsewhere", harness_src, override_src]
            with (
                mock.patch.dict(os.environ, {source_root.ENVIRONMENT_VARIABLE: name}),
                mock.patch.object(sys, "path", path),
            ):
                self.assertTrue(source_root.is_overridden())
                source_root.put_source_on_path()
                self.assertEqual(sys.path, [override_src, "/elsewhere"])

                # The harness's own package is exactly what an override must refuse.
                with self.assertRaisesRegex(RuntimeError, "not the measured source tree"):
                    source_root.assert_measuring(
                        str(source_root.HARNESS_ROOT / "src" / "renewable_huber" / "__init__.py")
                    )
                source_root.assert_measuring(str(package / "__init__.py"))

    def test_without_an_override_an_importable_src_is_left_in_place(self) -> None:
        harness_src = str(source_root.HARNESS_ROOT / "src")
        path = ["/elsewhere", harness_src]
        with (
            mock.patch.dict(os.environ, {source_root.ENVIRONMENT_VARIABLE: ""}),
            mock.patch.object(sys, "path", path),
        ):
            source_root.put_source_on_path()
            self.assertEqual(sys.path, ["/elsewhere", harness_src])


class FrozenPlanRunnerTests(unittest.TestCase):
    def _run(self, root: Path, *extra: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        calibration = {"baseline": 42, "candidate": 60}

        def fake_round(**kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            record = _record(1.0 if kwargs["repo"].name == "baseline" else 0.95)
            if kwargs["output"].parent.name == "calibration":
                for case in record["cases"]:
                    case["result"]["sample_repetitions"] = calibration[kwargs["repo"].name]
            return copy.deepcopy(record)

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
            str(root / "out"),
            "--pair-id",
            "pair",
            "--backend",
            "cpu",
            "--rounds",
            "3",
            *extra,
        ]
        with (
            mock.patch.object(run_interleaved_benchmark, "_run_round", side_effect=fake_round),
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run_interleaved_benchmark.main()
        gate = json.loads((root / "out" / "gate.json").read_text(encoding="utf-8"))
        return calls, gate

    def test_frozen_plan_is_calibrated_once_and_applied_to_every_round(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            calls, gate = self._run(root, "--freeze-sample-repetitions")
            plan_path = root / "out" / "sample-repetitions-plan.json"
            plan, digest = load_plan(plan_path)

        calibrations, rounds = calls[:2], calls[2:]
        self.assertEqual([call["repo"].name for call in calibrations], ["baseline", "candidate"])
        self.assertTrue(all(call["output"].parent.name == "calibration" for call in calibrations))
        self.assertTrue(
            all("--sample-repetitions-plan" not in c["benchmark_args"] for c in calibrations)
        )
        self.assertEqual(len(rounds), 6)
        for call in [*calibrations, *rounds]:
            # One harness, the candidate's, measures both source trees.
            self.assertEqual(call["harness_repo"], root / "candidate")
        for call in rounds:
            arguments = call["benchmark_args"]
            self.assertEqual(
                arguments[arguments.index("--sample-repetitions-plan") + 1], str(plan_path)
            )
        self.assertEqual(set(plan.values()), {60})
        self.assertEqual(
            gate["sample_repetitions"],
            {"policy": "frozen_plan", "plan_sha256": digest, "harness": "candidate"},
        )

    def test_default_keeps_each_variant_on_its_own_harness(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            calls, gate = self._run(root)

        self.assertEqual(len(calls), 6)
        self.assertTrue(all(call.get("harness_repo") is None for call in calls))
        self.assertTrue(all("--sample-repetitions-plan" not in c["benchmark_args"] for c in calls))
        self.assertEqual(gate["sample_repetitions"], {"policy": "per_round_calibration"})

    def test_run_round_points_the_shared_harness_at_the_variant_source(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            python = root / "python"
            python.write_text("", encoding="utf-8")
            for checkout in ("harness", "variant"):
                script = root / checkout / "scripts" / "benchmarks" / "benchmark_shape_sweep.py"
                script.parent.mkdir(parents=True)
                script.write_text("", encoding="utf-8")
            output = root / "out.json"
            output.write_text("{}", encoding="utf-8")

            with mock.patch.object(run_interleaved_benchmark.subprocess, "run") as run:
                run_interleaved_benchmark._run_round(
                    python=python,
                    repo=root / "variant",
                    output=output,
                    benchmark_args=[],
                    harness_repo=root / "harness",
                )
                run_interleaved_benchmark._run_round(
                    python=python, repo=root / "variant", output=output, benchmark_args=[]
                )

        shared, own = run.call_args_list
        self.assertIn(str(root / "harness"), shared.args[0][1])
        self.assertEqual(
            shared.kwargs["env"][source_root.ENVIRONMENT_VARIABLE], str(root / "variant")
        )
        self.assertIn(str(root / "variant"), own.args[0][1])
        self.assertIsNone(own.kwargs["env"])

    def test_gate_refuses_variants_with_different_sampling_policies(self) -> None:
        records = [_record(1.0) for _ in range(3)]
        baseline = merge_round_records(
            copy.deepcopy(records), variant="baseline", pair_id="pair", execution_order=[0, 1, 0]
        )
        candidate = merge_round_records(
            copy.deepcopy(records), variant="candidate", pair_id="pair", execution_order=[1, 0, 1]
        )
        baseline["interleaved_capture"]["sample_repetitions"] = {"policy": "frozen_plan"}
        candidate["interleaved_capture"]["sample_repetitions"] = {"policy": "per_round_calibration"}

        with self.assertRaisesRegex(ValueError, "different sample repetition policies"):
            compare_interleaved_records(baseline, candidate, engines=NATIVE_ENGINES, min_pairs=3)


if __name__ == "__main__":
    unittest.main()
