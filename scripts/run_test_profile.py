"""Run one named, auditable slice of the unittest suite.

``unittest`` stays the authoritative runner: every profile below is an explicit
list of modules that ``python -m unittest`` can also be given by hand, and
``python -m unittest discover -s tests`` keeps working untouched.

The problem this solves is that a suite made entirely of skips reports success.
``tests/test_native_cuda_backend.py`` skips itself when no GPU is present, so a
"passing" CI job proves nothing about CUDA. A *required* profile therefore
probes its declared dependencies first and fails when one is missing, using the
same readiness conditions the test modules use to skip. The ``all`` profile is
the developer's everything-runnable view and keeps its documented optional
skips.

Exit codes: ``0`` success, ``1`` test failures or errors, ``2`` a profile
whose requirements are unmet or whose membership table is inconsistent.
"""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIRECTORY = PROJECT_ROOT / "tests"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True, slots=True)
class Requirement:
    """One dependency or device a profile needs before it can mean anything."""

    name: str
    #: Returns ``None`` when satisfied, otherwise why it is not.
    probe: Callable[[], str | None]

    def unmet_reason(self) -> str | None:
        try:
            return self.probe()
        except Exception as error:  # pragma: no cover - a probe must never crash a run
            return f"probe raised {type(error).__name__}: {error}"


def _module_probe(module: str, description: str) -> Callable[[], str | None]:
    """Mirror the ``find_spec`` gate the corresponding test modules use."""

    def probe() -> str | None:
        return None if find_spec(module) is not None else f"{description} is not installed"

    return probe


def _native_cpu_probe() -> str | None:
    try:
        from renewable_huber import _native_cpu
    except (ImportError, OSError, RuntimeError) as error:
        return f"the Rust native CPU extension is not importable: {error}"
    version = _native_cpu.version()
    if version.get("abi_version") != 1 or version.get("python_api_version") != 2:
        return f"the native CPU extension reports an unsupported contract: {dict(version)}"
    return None


def _cupy_device_probe() -> str | None:
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            return None
    except Exception as error:
        return f"CuPy is unusable: {error}"
    return "CuPy reports no CUDA device"


def _native_cuda_device_probe() -> str | None:
    try:
        from renewable_huber import _native_cuda
    except (ImportError, OSError, RuntimeError) as error:
        return f"the native CUDA extension is not importable: {error}"
    if not _native_cuda.is_available():
        return "the native CUDA extension reports no usable runtime"
    if not _native_cuda.device_count():
        return "the native CUDA extension reports no CUDA device"
    return None


REQUIREMENTS: dict[str, Requirement] = {
    requirement.name: requirement
    for requirement in (
        Requirement("numpy", _module_probe("numpy", "NumPy")),
        Requirement("pandas", _module_probe("pandas", "pandas")),
        Requirement("scipy", _module_probe("scipy", "SciPy")),
        Requirement("sklearn", _module_probe("sklearn", "scikit-learn")),
        Requirement("torch", _module_probe("torch", "PyTorch")),
        Requirement("tensorflow", _module_probe("tensorflow", "TensorFlow")),
        Requirement("native-cpu", _native_cpu_probe),
        Requirement("cupy-device", _cupy_device_probe),
        Requirement("native-cuda-device", _native_cuda_device_probe),
    )
}


@dataclass(frozen=True, slots=True)
class Profile:
    """An explicit slice of the suite plus what it needs in order to mean anything."""

    name: str
    summary: str
    modules: tuple[str, ...]
    requirements: tuple[str, ...]
    #: Required profiles fail on an unmet requirement or an all-skipped run.
    required: bool = True
    #: Environment forced for the duration of the run, then restored. Empty for
    #: every profile that must observe the machine exactly as it is.
    environment: tuple[tuple[str, str], ...] = ()


# Membership is deliberately written out rather than derived from a naming
# convention: which optional dependency a module really needs is not visible in
# its filename, and a wrong guess would silently weaken a required profile.
_LEAF_PROFILES: tuple[Profile, ...] = (
    Profile(
        name="core",
        summary="portable NumPy behaviour, contracts and persistence",
        modules=(
            "tests.test_backend_capabilities",
            "tests.test_checkpoint_payload",
            "tests.test_correctness_contract",
            "tests.test_cpu_optimizations",
            "tests.test_dlpack_adapters",
            "tests.test_estimator",
            "tests.test_loss",
            "tests.test_native_cuda_contract",
            # Selection and routing driven by fakes: no device, no extension.
            # It is a separate module from test_native_cuda_backend precisely so
            # that CPU CI keeps running it; see PORTABLE_NATIVE_MODULES below.
            "tests.test_native_cuda_selection",
            "tests.test_native_golden",
            "tests.test_native_release_metadata",
            "tests.test_profile_runner",
            "tests.test_wsl_environment_verifier",
        ),
        requirements=("numpy",),
    ),
    Profile(
        name="optional-cpu",
        summary="pandas, SciPy, scikit-learn, PyTorch and TensorFlow CPU integrations",
        modules=(
            "tests.test_data_integrations",
            "tests.test_sklearn_integration",
            "tests.test_tensorflow_backend",
            "tests.test_torch_backend",
        ),
        requirements=("pandas", "scipy", "sklearn", "torch", "tensorflow"),
        # This profile is CPU-only by definition, and hiding the GPU is what
        # makes that definition true rather than aspirational. TensorFlow and
        # PyTorch each initialise their own CUDA runtime in-process; on a host
        # with a device, importing both leaves torch's `cusolverDnCreate`
        # returning CUSOLVER_STATUS_INTERNAL_ERROR, so the profile's result
        # would depend on which framework touched CUDA first. CI runners have
        # no GPU and are unaffected; the `cuda` profile keeps an empty
        # environment and still sees the device.
        environment=(("CUDA_VISIBLE_DEVICES", ""),),
    ),
    Profile(
        name="native-cpu",
        summary="the opt-in Rust CPU engine",
        modules=("tests.test_native_cpu_backend",),
        requirements=("native-cpu",),
    ),
    Profile(
        name="cuda",
        summary="CuPy and native CUDA on a real device; local GPU host only",
        modules=(
            "tests.test_cuda_kernels",
            "tests.test_cupy_backend",
            "tests.test_dlpack_adapters",
            "tests.test_native_cuda_backend",
        ),
        requirements=("cupy-device", "native-cuda-device"),
    ),
    Profile(
        name="performance",
        summary="benchmark schema, sampling and regression-gate tooling",
        modules=(
            "tests.test_benchmark_interleaved_regression",
            "tests.test_benchmark_performance_policy",
        ),
        requirements=("numpy",),
    ),
)

# ``tests.test_dlpack_adapters`` is the one module that legitimately belongs to
# two profiles. Its adapter tests drive fakes and run on any CPU, while three
# integration classes need a real device. Listing it only under ``cuda`` would
# drop the CPU tests from CPU CI; listing it only under ``core`` would leave
# nothing that guarantees the integration classes ever run. Every other
# duplicate is a wiring mistake and is reported by ``--check``.
SHARED_MODULES = frozenset({"tests.test_dlpack_adapters"})

# Modules that carry native-backend tests but need no device, no driver and no
# built extension, because they substitute fakes. They must stay in ``core``:
# the profiles that would otherwise own them run only on the fixed GPU host or
# behind a built wheel, so moving one out silently removes it from CPU CI
# without any test failing. ``validate_profiles`` enforces the membership and
# ``tests/test_profile_runner.py`` checks the modules really are portable.
PORTABLE_NATIVE_MODULES = frozenset(
    {
        "tests.test_native_cuda_selection",
        "tests.test_native_cuda_contract",
        "tests.test_native_golden",
    }
)


def _all_modules() -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for profile in _LEAF_PROFILES:
        for module in profile.modules:
            seen.setdefault(module, None)
    return tuple(sorted(seen))


PROFILES: dict[str, Profile] = {profile.name: profile for profile in _LEAF_PROFILES}
PROFILES["all"] = Profile(
    name="all",
    summary="every module, with documented optional skips (developer default)",
    # Derived from the leaf profiles so the two can never drift apart.
    modules=_all_modules(),
    requirements=("numpy",),
    required=False,
)


def discovered_test_modules() -> tuple[str, ...]:
    """Return every ``tests/test_*.py`` module name found on disk."""

    return tuple(sorted(f"tests.{path.stem}" for path in TESTS_DIRECTORY.glob("test_*.py")))


def validate_profiles() -> list[str]:
    """Return every membership inconsistency, empty when the table is sound."""

    problems: list[str] = []
    on_disk = set(discovered_test_modules())

    for profile in PROFILES.values():
        if not profile.modules:
            problems.append(f"profile {profile.name!r} declares no modules")
        for requirement in profile.requirements:
            if requirement not in REQUIREMENTS:
                problems.append(
                    f"profile {profile.name!r} names unknown requirement {requirement!r}"
                )
        for module in profile.modules:
            if not module.startswith("tests."):
                problems.append(f"profile {profile.name!r} names non-test module {module!r}")
            elif module not in on_disk:
                problems.append(f"profile {profile.name!r} names missing module {module!r}")
        if len(set(profile.modules)) != len(profile.modules):
            problems.append(f"profile {profile.name!r} repeats a module")

    owners: dict[str, list[str]] = {}
    for profile in _LEAF_PROFILES:
        for module in profile.modules:
            owners.setdefault(module, []).append(profile.name)
    for module, profile_names in sorted(owners.items()):
        if len(profile_names) > 1 and module not in SHARED_MODULES:
            problems.append(
                f"module {module!r} appears in undeclared duplicate profiles: "
                f"{', '.join(profile_names)}"
            )
    for module in sorted(SHARED_MODULES):
        if len(owners.get(module, [])) < 2:
            problems.append(f"module {module!r} is declared shared but has one owner or none")

    core = PROFILES.get("core")
    core_modules = set(core.modules) if core is not None else set()
    for module in sorted(PORTABLE_NATIVE_MODULES):
        if module not in on_disk:
            problems.append(f"module {module!r} is declared portable but does not exist")
        elif module in owners and module not in core_modules:
            # Only checked once a table actually claims the module, so the
            # validator's own fixture tables do not trip on it.
            problems.append(
                f"module {module!r} is portable native coverage and must stay in the 'core' "
                f"profile; it currently belongs to {', '.join(owners[module])} and CPU CI "
                "would stop running it"
            )

    if unassigned := on_disk - set(owners):
        problems.append(f"test modules assigned to no profile: {', '.join(sorted(unassigned))}")
    return problems


def unmet_requirements(profile: Profile) -> list[tuple[str, str]]:
    """Return ``(requirement, reason)`` for everything the profile is missing."""

    unmet = []
    for name in profile.requirements:
        requirement = REQUIREMENTS[name]
        if reason := requirement.unmet_reason():
            unmet.append((name, reason))
    return unmet


def build_suite(profile: Profile) -> unittest.TestSuite:
    return unittest.TestLoader().loadTestsFromNames(profile.modules)


def _describe(profiles: Iterable[Profile]) -> str:
    lines = []
    for profile in profiles:
        kind = "required" if profile.required else "optional"
        lines.append(f"{profile.name} ({kind}) - {profile.summary}")
        lines.append(f"  requires: {', '.join(profile.requirements) or 'nothing'}")
        for name, value in profile.environment:
            lines.append(f"  forces: {name}={value!r}")
        for module in profile.modules:
            lines.append(f"  {module}")
    return "\n".join(lines)


@contextmanager
def forced_environment(profile: Profile) -> Iterator[None]:
    """Apply a profile's declared environment, then restore what was there.

    Restoration matters because ``run_profile`` is also called in-process by
    its own tests: leaking ``CUDA_VISIBLE_DEVICES=""`` would silently disable
    every GPU test that ran afterwards in the same interpreter.
    """

    previous: list[tuple[str, str | None]] = []
    try:
        for name, value in profile.environment:
            previous.append((name, os.environ.get(name)))
            os.environ[name] = value
        yield
    finally:
        for name, original in reversed(previous):
            if original is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original


def run_profile(
    profile: Profile,
    *,
    verbosity: int = 1,
    failfast: bool = False,
    stream: object | None = None,
) -> int:
    """Run one profile and return its process exit code."""

    if problems := validate_profiles():
        for problem in problems:
            print(f"profile table error: {problem}", file=sys.stderr)
        return 2

    unmet = unmet_requirements(profile)
    if unmet and profile.required:
        for name, reason in unmet:
            print(
                f"required profile {profile.name!r} cannot run: {name}: {reason}", file=sys.stderr
            )
        print(
            "A required profile must not report success as a suite of skips. "
            "Install the dependency, attach the device, or run the 'all' profile.",
            file=sys.stderr,
        )
        return 2
    for name, reason in unmet:
        print(f"note: {name} unavailable ({reason}); its tests will skip")
    for name, value in profile.environment:
        print(f"note: {profile.name!r} forces {name}={value!r} for this run")

    runner = unittest.TextTestRunner(
        stream=stream,  # type: ignore[arg-type]
        verbosity=verbosity,
        failfast=failfast,
    )
    # The suite is built inside the forced environment as well as run in it:
    # a framework that reads the variable at import time must see it too.
    with forced_environment(profile):
        result = runner.run(build_suite(profile))

    if profile.required:
        if result.testsRun == 0:
            print(f"required profile {profile.name!r} ran no tests", file=sys.stderr)
            return 2
        if len(result.skipped) == result.testsRun:
            print(
                f"required profile {profile.name!r} skipped all {result.testsRun} tests; "
                "its declared requirements were met, so this is a defect, not an absence",
                file=sys.stderr,
            )
            return 2
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("profile", nargs="?", choices=tuple(PROFILES), help="profile to run")
    parser.add_argument("--list", action="store_true", help="print every profile and exit")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the membership table without running tests",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--failfast", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        print(_describe(PROFILES.values()))
        return 0
    if args.check:
        problems = validate_profiles()
        for problem in problems:
            print(f"profile table error: {problem}", file=sys.stderr)
        if problems:
            return 2
        print(f"{len(PROFILES)} profiles cover {len(discovered_test_modules())} test modules")
        return 0
    if args.profile is None:
        parser.error("a profile name is required unless --list or --check is given")

    return run_profile(
        PROFILES[args.profile],
        verbosity=2 if args.verbose else 1,
        failfast=args.failfast,
    )


if __name__ == "__main__":
    raise SystemExit(main())
