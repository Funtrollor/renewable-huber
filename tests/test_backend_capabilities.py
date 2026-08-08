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

from renewable_huber.backends import resolve_backend
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


class NativeBackendCapabilityTests(unittest.TestCase):
    """Whatever a native backend advertises, the core must be able to find."""

    def _capabilities(self, name: str) -> BackendCapabilities | None:
        from renewable_huber.exceptions import BackendUnavailableError

        try:
            backend = resolve_backend(name, device="cpu" if name == "native_cpu" else "cuda")
        except BackendUnavailableError:
            return None
        return capabilities_of(backend)

    def test_native_cpu_advertises_update_predict_and_threads(self) -> None:
        capabilities = self._capabilities("native_cpu")
        if capabilities is None:
            self.skipTest("the native CPU extension is not installed")
        self.assertIsNotNone(capabilities.native_update)
        self.assertIsNotNone(capabilities.native_predict)
        self.assertIsNotNone(capabilities.read_n_jobs)
        self.assertIsNone(capabilities.native_design_matrix)
        self.assertFalse(capabilities.elementwise_workspace)

    def test_native_cuda_advertises_design_ownership_and_device_reductions(self) -> None:
        capabilities = self._capabilities("native_cuda")
        if capabilities is None:
            self.skipTest("the native CUDA extension or a CUDA device is unavailable")
        self.assertIsNotNone(capabilities.native_update)
        self.assertIsNotNone(capabilities.native_predict)
        # The CUDA engine appends the intercept on device, so it owns the
        # design matrix; losing this silently reintroduces a full host copy of
        # every batch before each transfer.
        self.assertIsNotNone(capabilities.native_design_matrix)
        self.assertIsNotNone(capabilities.minimum_scalar)
        self.assertIsNotNone(capabilities.sum_scalar)
        self.assertIsNotNone(capabilities.read_cuda_features)
        self.assertFalse(capabilities.elementwise_workspace)


if __name__ == "__main__":
    unittest.main()
