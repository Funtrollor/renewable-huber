"""Contract for the named unittest profiles.

These tests exist because the failure mode being prevented is silent: a profile
that skips every one of its tests, or that quietly stops covering a module,
still reports success. Each check below therefore asserts the runner *fails*
under exactly those conditions.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import io
import os
import unittest
from unittest import mock

from scripts import run_test_profile as runner


def _class_names(module: str) -> set[str]:
    """Return the classes a test module defines, without importing it."""

    path = runner.TESTS_DIRECTORY / f"{module.rpartition('.')[2]}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def _unittest_skip_controls(module: str) -> set[str]:
    """Return semantic unittest skip controls referenced by one module."""

    path = runner.TESTS_DIRECTORY / f"{module.rpartition('.')[2]}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {"SkipTest", "expectedFailure", "skip", "skipIf", "skipTest", "skipUnless"}
    referenced = {
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Name | ast.Attribute)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "unittest":
            referenced.update(alias.name for alias in node.names)
    return referenced & forbidden


def _flatten(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    cases: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            cases.extend(_flatten(item))
        else:
            cases.append(item)
    return cases


def _module_names(suite: unittest.TestSuite) -> tuple[str, ...]:
    return tuple(sorted({type(case).__module__ for case in _flatten(suite)}))


def _quiet_run(profile: runner.Profile) -> tuple[int, str]:
    """Run a profile with all reporting captured."""

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
        code = runner.run_profile(profile, stream=io.StringIO())
    return code, stderr.getvalue()


def _quiet_main(argv: list[str]) -> tuple[int, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            code = runner.main(argv)
        except SystemExit as exit_error:  # argparse usage errors
            code = int(exit_error.code or 0)
    return code, stdout.getvalue() + stderr.getvalue()


class ProfileMembershipTests(unittest.TestCase):
    def test_the_declared_table_is_consistent(self) -> None:
        self.assertEqual(runner.validate_profiles(), [])

    def test_the_expected_profiles_all_exist(self) -> None:
        self.assertEqual(
            sorted(runner.PROFILES),
            ["all", "core", "cuda", "native-cpu", "optional-cpu", "performance"],
        )

    def test_every_test_module_on_disk_belongs_to_a_profile(self) -> None:
        discovered = runner.discovered_test_modules()
        # Without this the coverage check below would pass on an empty glob.
        self.assertIn("tests.test_profile_runner", discovered)
        self.assertGreater(len(discovered), 10)
        self.assertEqual(set(runner.PROFILES["all"].modules), set(discovered))

    def test_only_the_required_profiles_are_marked_required(self) -> None:
        required = {name for name, profile in runner.PROFILES.items() if profile.required}
        self.assertEqual(required, {"core", "cuda", "native-cpu", "optional-cpu", "performance"})
        self.assertFalse(runner.PROFILES["all"].required)

    def test_plain_discovery_still_imports_every_module(self) -> None:
        # Profiles are additive: ``python -m unittest discover -s tests`` stays
        # the documented entry point and must keep loading cleanly. A module
        # that fails to import turns into a synthetic ``_FailedTest`` case,
        # which a plain discovery run would otherwise report only at run time.
        suite = unittest.defaultTestLoader.discover(str(runner.TESTS_DIRECTORY))
        loaded = {type(case).__name__ for case in _flatten(suite)}
        self.assertNotIn("_FailedTest", loaded)
        # ``discover -s tests`` defaults its top-level directory to the start
        # directory, so it imports ``test_estimator`` where a profile imports
        # ``tests.test_estimator``. Compare the stems the two agree on.
        discovered = tuple(name.rpartition(".")[2] for name in runner.discovered_test_modules())
        self.assertEqual(discovered, _module_names(suite))

    def test_the_cuda_profile_demands_a_real_device(self) -> None:
        # The whole point of the required profiles: no device, no success.
        self.assertEqual(
            runner.PROFILES["cuda"].requirements, ("cupy-device", "native-cuda-device")
        )

    def test_portable_native_modules_are_declared_and_owned_by_core(self) -> None:
        self.assertIn("tests.test_native_cuda_selection", runner.PORTABLE_NATIVE_MODULES)
        for module in runner.PORTABLE_NATIVE_MODULES:
            with self.subTest(module=module):
                self.assertIn(module, runner.PROFILES["core"].modules)

    def test_the_portable_native_modules_really_are_portable(self) -> None:
        # A module claiming to need no device must not gate itself on one:
        # a skipUnless at module scope would make its presence in core
        # meaningless on a CI runner.
        for module in runner.PORTABLE_NATIVE_MODULES:
            with self.subTest(module=module):
                self.assertEqual(_unittest_skip_controls(module), set())

    def test_the_cuda_selection_tests_are_in_core_not_only_in_cuda(self) -> None:
        classes = _class_names("tests.test_native_cuda_selection")
        self.assertIn("NativeCudaSelectionTests", classes)
        self.assertNotIn("NativeCudaSelectionTests", _class_names("tests.test_native_cuda_backend"))
        module = importlib.import_module("tests.test_native_cuda_selection")
        self.assertTrue(issubclass(module.NativeCudaSelectionTests, unittest.TestCase))

    def test_the_cuda_selection_contract_executes_without_a_device_or_skips(self) -> None:
        expected_methods = {
            "test_cuda_tuning_reaches_engine_and_reports_capabilities",
            "test_engine_initialization_error_is_unavailable_and_not_fitted",
            "test_estimator_routes_a_complete_batch_through_one_native_call",
            "test_explicit_native_request_never_falls_back",
            "test_hard_native_error_discards_engine_before_retry",
            "test_incompatible_native_protocol_fails_before_engine_creation",
            "test_native_cuda_rejects_cpu_device",
            "test_requested_tuning_requires_advertised_capability",
            "test_resident_engine_restores_distinct_state_with_same_batch_count",
        }
        profile = runner.Profile(
            name="portable-native-cuda-selection",
            summary="fixture",
            modules=("tests.test_native_cuda_selection",),
            requirements=("numpy",),
            environment=(("CUDA_VISIBLE_DEVICES", ""),),
        )
        with runner.forced_environment(profile):
            suite = runner.build_suite(profile)
            cases = _flatten(suite)
            result = unittest.TestResult()
            suite.run(result)

        loaded_methods = {
            case._testMethodName
            for case in cases
            if type(case).__name__ == "NativeCudaSelectionTests"
        }
        self.assertLessEqual(expected_methods, loaded_methods)
        self.assertEqual(result.testsRun, len(cases))
        self.assertEqual(result.skipped, [])
        self.assertEqual(result.failures, [])
        self.assertEqual(result.errors, [])

    def test_native_capability_tests_sit_in_the_required_native_profiles(self) -> None:
        # They used to live in the portable capability module, where the only
        # thing they could do without an extension was skip. A required profile
        # exists to rule that outcome out, so assert where they live.
        for module, class_name, profile in (
            ("tests.test_native_cpu_backend", "NativeCpuCapabilityTests", "native-cpu"),
            ("tests.test_native_cuda_backend", "NativeCudaCapabilityTests", "cuda"),
        ):
            with self.subTest(profile=profile):
                self.assertIn(module, runner.PROFILES[profile].modules)
                self.assertIn(class_name, _class_names(module))
        portable = _class_names("tests.test_backend_capabilities")
        self.assertNotIn("NativeBackendCapabilityTests", portable)
        # Anti-vacuity: the portable module must still be the one being read.
        self.assertIn("LiveAccessorTests", portable)


class CpuIsolationTests(unittest.TestCase):
    """``optional-cpu`` means CPU-only on a GPU host as much as on CI."""

    def test_only_the_optional_cpu_profile_forces_an_environment(self) -> None:
        self.assertEqual(
            runner.PROFILES["optional-cpu"].environment, (("CUDA_VISIBLE_DEVICES", ""),)
        )
        for name in ("core", "cuda", "native-cpu", "performance", "all"):
            with self.subTest(profile=name):
                self.assertEqual(runner.PROFILES[name].environment, ())

    def test_the_forced_environment_is_restored_when_it_was_unset(self) -> None:
        profile = runner.PROFILES["optional-cpu"]
        with mock.patch.dict(os.environ):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            with runner.forced_environment(profile):
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)

    def test_the_forced_environment_is_restored_when_it_was_set(self) -> None:
        profile = runner.PROFILES["optional-cpu"]
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            with runner.forced_environment(profile):
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")

    def test_the_forced_environment_is_restored_after_a_failure(self) -> None:
        profile = runner.PROFILES["optional-cpu"]
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}):
            with self.assertRaises(RuntimeError):
                with runner.forced_environment(profile):
                    raise RuntimeError("suite exploded")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")

    def test_the_suite_is_built_inside_the_forced_environment(self) -> None:
        # A framework that reads the variable while being imported must see it,
        # so masking has to wrap loading and not only running. The fixture
        # borrows the real profile's `environment` but requires only NumPy, so
        # this asserts the masking rule rather than whether PyTorch happens to
        # be installed in the environment running the test.
        observed: list[str | None] = []
        profile = runner.Profile(
            name="masked",
            summary="fixture",
            modules=("tests.test_loss",),
            requirements=("numpy",),
            environment=runner.PROFILES["optional-cpu"].environment,
        )

        def record(_: runner.Profile) -> unittest.TestSuite:
            observed.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
            return RequiredProfileEnforcementTests._suite("pass")

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with mock.patch.object(runner, "build_suite", record):
                code, message = _quiet_run(profile)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(code, 0, message)
        self.assertEqual(observed, [""])


class ProfileValidationTests(unittest.TestCase):
    """``validate_profiles`` must name each kind of wiring mistake."""

    def _validate(self, *profiles: runner.Profile, shared: frozenset[str] | None = None) -> str:
        table = {profile.name: profile for profile in profiles}
        with (
            mock.patch.object(runner, "_LEAF_PROFILES", profiles),
            mock.patch.object(runner, "PROFILES", table),
            mock.patch.object(
                runner, "SHARED_MODULES", runner.SHARED_MODULES if shared is None else shared
            ),
        ):
            return "\n".join(runner.validate_profiles())

    @staticmethod
    def _profile(name: str, modules: tuple[str, ...], **kwargs: object) -> runner.Profile:
        return runner.Profile(
            name=name,
            summary="fixture",
            modules=modules,
            requirements=kwargs.pop("requirements", ("numpy",)),  # type: ignore[arg-type]
            **kwargs,  # type: ignore[arg-type]
        )

    def test_a_module_in_two_profiles_is_reported(self) -> None:
        problems = self._validate(
            self._profile("first", ("tests.test_loss",)),
            self._profile("second", ("tests.test_loss",)),
            shared=frozenset(),
        )
        self.assertIn("undeclared duplicate profiles", problems)
        self.assertIn("tests.test_loss", problems)

    def test_a_declared_shared_module_is_allowed(self) -> None:
        problems = self._validate(
            self._profile("first", ("tests.test_loss",)),
            self._profile("second", ("tests.test_loss",)),
            shared=frozenset({"tests.test_loss"}),
        )
        self.assertNotIn("duplicate", problems)

    def test_a_shared_module_with_a_single_owner_is_reported(self) -> None:
        problems = self._validate(
            self._profile("only", ("tests.test_loss",)),
            shared=frozenset({"tests.test_loss"}),
        )
        self.assertIn("declared shared but has one owner", problems)

    def test_a_misspelled_module_is_reported(self) -> None:
        problems = self._validate(self._profile("typo", ("tests.test_looss",)))
        self.assertIn("names missing module 'tests.test_looss'", problems)

    def test_a_module_outside_the_tests_package_is_reported(self) -> None:
        problems = self._validate(self._profile("stray", ("scripts.run_test_profile",)))
        self.assertIn("names non-test module", problems)

    def test_an_unassigned_module_on_disk_is_reported(self) -> None:
        problems = self._validate(self._profile("partial", ("tests.test_loss",)))
        self.assertIn("assigned to no profile", problems)
        self.assertIn("tests.test_estimator", problems)

    def test_an_unknown_requirement_is_reported(self) -> None:
        problems = self._validate(
            self._profile("bad-requirement", ("tests.test_loss",), requirements=("quantum",))
        )
        self.assertIn("unknown requirement 'quantum'", problems)

    def test_an_empty_profile_is_reported(self) -> None:
        problems = self._validate(self._profile("empty", ()))
        self.assertIn("declares no modules", problems)

    def test_a_repeated_module_inside_one_profile_is_reported(self) -> None:
        problems = self._validate(self._profile("doubled", ("tests.test_loss", "tests.test_loss")))
        self.assertIn("repeats a module", problems)

    def test_moving_a_portable_native_module_out_of_core_is_reported(self) -> None:
        # The regression this exists for: the CUDA selection tests need no
        # device, but lived in a module the `cuda` profile owned, so CPU CI
        # silently stopped running them and nothing failed.
        problems = self._validate(
            self._profile("core", ("tests.test_loss",)),
            self._profile("cuda", ("tests.test_native_cuda_selection",)),
        )
        self.assertIn("must stay in the 'core' profile", problems)
        self.assertIn("tests.test_native_cuda_selection", problems)
        self.assertIn("CPU CI would stop running it", problems)

    def test_a_portable_native_module_kept_in_core_is_accepted(self) -> None:
        problems = self._validate(
            self._profile("core", ("tests.test_native_cuda_selection",)),
            self._profile("cuda", ("tests.test_loss",)),
        )
        self.assertNotIn("must stay in the 'core' profile", problems)


class RequiredProfileEnforcementTests(unittest.TestCase):
    @staticmethod
    def _suite(*outcomes: str) -> unittest.TestSuite:
        """Build a suite of locally defined cases so discovery never sees them."""

        class _Fixture(unittest.TestCase):
            def runTest(self) -> None:  # noqa: N802 - unittest's own spelling
                outcome = self._outcome_name  # type: ignore[attr-defined]
                if outcome == "skip":
                    self.skipTest("no device")
                if outcome == "fail":
                    self.fail("deliberate")

        suite = unittest.TestSuite()
        for outcome in outcomes:
            case = _Fixture()
            case._outcome_name = outcome  # type: ignore[attr-defined]
            suite.addTest(case)
        return suite

    def _run_with(self, profile: runner.Profile, suite: unittest.TestSuite) -> tuple[int, str]:
        with mock.patch.object(runner, "build_suite", return_value=suite):
            return _quiet_run(profile)

    def test_an_all_skipped_required_profile_fails(self) -> None:
        code, message = self._run_with(runner.PROFILES["core"], self._suite("skip", "skip", "skip"))
        self.assertEqual(code, 2)
        self.assertIn("skipped all 3 tests", message)

    def test_an_all_skipped_optional_profile_still_succeeds(self) -> None:
        code, _ = self._run_with(runner.PROFILES["all"], self._suite("skip", "skip"))
        self.assertEqual(code, 0)

    def test_a_partly_skipped_required_profile_succeeds(self) -> None:
        code, _ = self._run_with(runner.PROFILES["core"], self._suite("skip", "pass"))
        self.assertEqual(code, 0)

    def test_an_empty_required_run_fails(self) -> None:
        code, message = self._run_with(runner.PROFILES["core"], self._suite())
        self.assertEqual(code, 2)
        self.assertIn("ran no tests", message)

    def test_a_test_failure_is_reported_as_exit_one(self) -> None:
        code, _ = self._run_with(runner.PROFILES["core"], self._suite("pass", "fail"))
        self.assertEqual(code, 1)

    def test_an_unmet_requirement_fails_a_required_profile_before_running(self) -> None:
        unavailable = runner.Requirement("absent-device", lambda: "no such device attached")
        profile = runner.Profile(
            name="probe",
            summary="fixture",
            modules=("tests.test_loss",),
            requirements=("absent-device",),
        )
        with (
            mock.patch.dict(runner.REQUIREMENTS, {"absent-device": unavailable}),
            mock.patch.object(runner, "build_suite", side_effect=AssertionError("must not run")),
        ):
            code, message = _quiet_run(profile)
        self.assertEqual(code, 2)
        self.assertIn("no such device attached", message)
        self.assertIn("must not report success as a suite of skips", message)

    def test_an_unmet_requirement_only_annotates_an_optional_profile(self) -> None:
        unavailable = runner.Requirement("absent-device", lambda: "no such device attached")
        profile = runner.Profile(
            name="probe",
            summary="fixture",
            modules=("tests.test_loss",),
            requirements=("absent-device",),
            required=False,
        )
        with (
            mock.patch.dict(runner.REQUIREMENTS, {"absent-device": unavailable}),
            mock.patch.object(runner, "build_suite", return_value=self._suite("pass")),
        ):
            code, _ = _quiet_run(profile)
        self.assertEqual(code, 0)

    def test_a_probe_that_raises_is_treated_as_unmet(self) -> None:
        def explode() -> str | None:
            raise RuntimeError("driver fell over")

        requirement = runner.Requirement("explosive", explode)
        self.assertEqual(requirement.unmet_reason(), "probe raised RuntimeError: driver fell over")

    def test_an_inconsistent_table_stops_the_run(self) -> None:
        broken = runner.Profile(
            name="broken", summary="fixture", modules=("tests.test_nope",), requirements=("numpy",)
        )
        with (
            mock.patch.object(runner, "_LEAF_PROFILES", (broken,)),
            mock.patch.object(runner, "PROFILES", {"broken": broken}),
            mock.patch.object(runner, "build_suite", side_effect=AssertionError("must not run")),
        ):
            code, message = _quiet_run(broken)
        self.assertEqual(code, 2)
        self.assertIn("profile table error", message)


class CommandLineTests(unittest.TestCase):
    def test_check_reports_success_on_the_real_table(self) -> None:
        code, output = _quiet_main(["--check"])
        self.assertEqual(code, 0)
        self.assertIn("profiles cover", output)

    def test_list_prints_every_profile_and_its_requirements(self) -> None:
        code, output = _quiet_main(["--list"])
        self.assertEqual(code, 0)
        for name in runner.PROFILES:
            self.assertIn(name, output)
        self.assertIn("cupy-device", output)
        self.assertIn("(optional)", output)

    def test_check_fails_on_an_inconsistent_table(self) -> None:
        broken = runner.Profile(
            name="broken", summary="fixture", modules=("tests.test_nope",), requirements=("numpy",)
        )
        with (
            mock.patch.object(runner, "_LEAF_PROFILES", (broken,)),
            mock.patch.object(runner, "PROFILES", {"broken": broken}),
        ):
            code, output = _quiet_main(["--check"])
        self.assertEqual(code, 2)
        self.assertIn("names missing module", output)

    def test_a_missing_profile_name_is_a_usage_error(self) -> None:
        code, _ = _quiet_main([])
        self.assertEqual(code, 2)

    def test_an_unknown_profile_name_is_a_usage_error(self) -> None:
        code, _ = _quiet_main(["gpu-ish"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
