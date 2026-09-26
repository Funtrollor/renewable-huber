"""Contract tests for the backend capability layer.

These abilities used to be discovered by ``getattr`` at a dozen call sites, so
losing one degraded silently to the portable path -- correct, but potentially
orders of magnitude slower, and nothing failed. Now there is one place that
probes and one place to assert against.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from renewable_huber.backends.capabilities import BackendCapabilities, capabilities_of
from renewable_huber.backends.numpy_backend import NumPyBackend


class CapabilityProbeTests(unittest.TestCase):
    def test_plain_numpy_backend_offers_no_optional_behaviour(self) -> None:
        capabilities = capabilities_of(NumPyBackend("float64"))
        self.assertIsNone(capabilities.native_update)
        self.assertIsNone(capabilities.native_predict)
        self.assertIsNone(capabilities.native_design_matrix)
        self.assertIsNone(capabilities.minimum_scalar)
        self.assertIsNone(capabilities.sum_scalar)
        self.assertIsNone(capabilities.huber_loss)
        self.assertIsNone(capabilities.read_n_jobs)
        self.assertIsNone(capabilities.read_cuda_features)
        # No restriction: the portable solver implements every penalty.
        self.assertIsNone(capabilities.native_update_penalties)

    def test_native_backends_declare_their_penalties_at_class_or_instance_level(self) -> None:
        from renewable_huber.backends.native_cpu_backend import NativeCpuBackend
        from renewable_huber.backends.native_cuda_backend import _DEFAULT_PENALTIES

        self.assertEqual(NativeCpuBackend.native_update_penalties, frozenset({"none", "l1"}))
        # An extension that does not advertise its penalties is assumed to
        # implement only the unpenalized solver, never L1 by default.
        self.assertEqual(_DEFAULT_PENALTIES, frozenset({"none"}))

    def test_core_refuses_a_penalty_the_native_engine_does_not_advertise(self) -> None:
        from renewable_huber.config import EstimatorConfig
        from renewable_huber.core import renewable_update
        from renewable_huber.exceptions import ValidationError
        from renewable_huber.state import RenewableHuberState

        calls: list[str] = []

        class NoneOnly(NumPyBackend):
            name = "none_only"
            native_update_penalties = frozenset({"none"})

            def renewable_update(self, *args: Any, **kwargs: Any) -> Any:
                calls.append("native")
                raise AssertionError("the engine must not be reached")

        backend = NoneOnly("float64")
        state = RenewableHuberState.empty(2, fit_intercept=True, xp=np, dtype=np.float64)
        X = np.column_stack((np.eye(3, 2), np.ones(3)))
        y = np.ones(3)
        config = EstimatorConfig(penalty="l1")
        with self.assertRaisesRegex(ValidationError, "does not support penalty='l1'"):
            renewable_update(X, y, state, config, backend)
        self.assertEqual(calls, [])
        self.assertEqual(capabilities_of(backend).native_update_penalties, frozenset({"none"}))

    def test_elementwise_workspace_matches_the_documented_backend_set(self) -> None:
        # Before the capability object this was `backend.name in {"numpy",
        # "cupy"}`. The native backends inherit from NumPyBackend, so they must
        # opt out explicitly or they would silently join the set.
        self.assertTrue(capabilities_of(NumPyBackend("float64")).elementwise_workspace)
        for module, attribute in (
            ("renewable_huber.backends.cupy_backend", "CuPyBackend"),
            ("renewable_huber.backends.native_cpu_backend", "NativeCpuBackend"),
            ("renewable_huber.backends.native_cuda_backend", "NativeCudaBackend"),
        ):
            backend_class = getattr(__import__(module, fromlist=[attribute]), attribute)
            expected = attribute == "CuPyBackend"
            with self.subTest(backend=attribute):
                self.assertEqual(
                    backend_class.supports_elementwise_workspace,
                    expected,
                    f"{attribute} changed its portable-workspace eligibility",
                )

    def test_a_declared_capability_object_is_used_verbatim(self) -> None:
        declared = BackendCapabilities(elementwise_workspace=True)

        class Declaring(NumPyBackend):
            capabilities = declared

        # Structural probing would report native_update from the method below;
        # the declaration must win instead.
        backend = Declaring("float64")
        backend.renewable_update = lambda *a, **k: None  # type: ignore[method-assign]
        self.assertIs(capabilities_of(backend), declared)
        self.assertIsNone(capabilities_of(backend).native_update)

    def test_result_is_cached_per_instance(self) -> None:
        backend = NumPyBackend("float64")
        self.assertIs(capabilities_of(backend), capabilities_of(backend))
        self.assertIsNot(capabilities_of(backend), capabilities_of(NumPyBackend("float64")))

    def test_probing_tolerates_a_backend_that_cannot_be_cached(self) -> None:
        class Slotted:
            __slots__ = ("dtype",)
            name = "slotted"
            device = "cpu"
            xp = np

        backend = Slotted()
        first = capabilities_of(backend)
        second = capabilities_of(backend)
        self.assertIsInstance(first, BackendCapabilities)
        self.assertEqual(first, second)


class LiveAccessorTests(unittest.TestCase):
    """The two mutable reports must not be snapshotted at probe time."""

    def test_n_jobs_is_re_read_on_every_call(self) -> None:
        class Reporting(NumPyBackend):
            def __init__(self) -> None:
                super().__init__("float64")
                self.threads: int | None = None

            @property
            def effective_n_jobs(self) -> int | None:
                return self.threads

        backend = Reporting()
        read = capabilities_of(backend).read_n_jobs
        assert read is not None
        self.assertIsNone(read())
        backend.threads = 24
        self.assertEqual(read(), 24, "capabilities snapshotted a value that changes over time")

    def test_cuda_features_are_re_read_on_every_call(self) -> None:
        class Counting(NumPyBackend):
            def __init__(self) -> None:
                super().__init__("float64")
                self.replays = 0

            @property
            def cuda_features(self) -> dict[str, Any]:
                return {"graph_replays": self.replays}

        backend = Counting()
        read = capabilities_of(backend).read_cuda_features
        assert read is not None
        self.assertEqual(read()["graph_replays"], 0)
        backend.replays = 7
        self.assertEqual(
            read()["graph_replays"], 7, "CUDA graph counters were frozen at probe time"
        )


if __name__ == "__main__":
    unittest.main()
