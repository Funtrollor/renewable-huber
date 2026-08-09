"""Native CUDA backend selection, fallback and routing, verified with fakes.

Every test here substitutes a fake extension module, so none of them needs a
CUDA device, a driver, or the built extension. They live apart from
``test_native_cuda_backend.py`` for exactly that reason: that module belongs to
the ``cuda`` profile, which only ever runs on the fixed local GPU host, and
these portable tests would otherwise never run in CPU CI again.

Anything added here must keep that property. A test that needs a real device
belongs in ``test_native_cuda_backend.py``.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import numpy as np

import renewable_huber
from renewable_huber import RenewableHuberRegressor
from renewable_huber.backends import resolve_backend
from renewable_huber.exceptions import BackendUnavailableError, NotFittedError
from renewable_huber.state import RenewableHuberState


class NativeCudaSelectionTests(unittest.TestCase):
    def test_explicit_native_request_never_falls_back(self) -> None:
        unavailable = types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            version=lambda: {"abi_version": 1, "python_api_version": 3},
        )
        with (
            mock.patch.dict(sys.modules, {"renewable_huber._native_cuda": unavailable}),
            mock.patch.object(renewable_huber, "_native_cuda", unavailable, create=True),
        ):
            with self.assertRaisesRegex(BackendUnavailableError, "without CUDA"):
                resolve_backend("native_cuda", device="cuda")

    def test_incompatible_native_protocol_fails_before_engine_creation(self) -> None:
        incompatible = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            version=lambda: {"abi_version": 1, "python_api_version": 999},
        )
        with (
            mock.patch.dict(sys.modules, {"renewable_huber._native_cuda": incompatible}),
            mock.patch.object(renewable_huber, "_native_cuda", incompatible, create=True),
        ):
            with self.assertRaisesRegex(BackendUnavailableError, "incompatible"):
                resolve_backend("native_cuda", device="cuda")

    def test_requested_tuning_requires_advertised_capability(self) -> None:
        extension = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            version=lambda: {
                "abi_version": 1,
                "python_api_version": 3,
                "supports_cuda_graphs": False,
                "supports_fast_math": False,
            },
        )
        with (
            mock.patch.dict(sys.modules, {"renewable_huber._native_cuda": extension}),
            mock.patch.object(renewable_huber, "_native_cuda", extension, create=True),
        ):
            with self.assertRaisesRegex(BackendUnavailableError, "Graph"):
                resolve_backend("native_cuda", device="cuda", cuda_graphs=True)
            with self.assertRaisesRegex(BackendUnavailableError, "fast-math"):
                resolve_backend(
                    "native_cuda",
                    device="cuda",
                    dtype="float32",
                    cuda_fast_math=True,
                )

    def test_native_cuda_rejects_cpu_device(self) -> None:
        with self.assertRaisesRegex(BackendUnavailableError, "requires a CUDA device"):
            resolve_backend("native_cuda", device="cpu")

    def test_cuda_tuning_reaches_engine_and_reports_capabilities(self) -> None:
        received: dict[str, object] = {}

        class FakeEngine:
            def __init__(
                self,
                dtype: str,
                n_parameters: int,
                device_id: int,
                **tuning: object,
            ) -> None:
                received.update(
                    dtype=dtype,
                    n_parameters=n_parameters,
                    device_id=device_id,
                    **tuning,
                )

            def features(self) -> dict[str, object]:
                return {
                    "cuda_graphs_requested": True,
                    "cuda_graphs_enabled": True,
                    "fast_math_requested": True,
                    "fast_math_enabled": True,
                    "graph_captures": 0,
                    "graph_replays": 0,
                    "graph_fallbacks": 0,
                }

        extension = types.SimpleNamespace(
            NativeCudaEngine=FakeEngine,
            is_available=lambda: True,
            device_count=lambda: 1,
            version=lambda: {
                "abi_version": 1,
                "python_api_version": 3,
                "initial_state": "canonical_empty",
                "supports_cuda_graphs": True,
                "supports_fast_math": True,
            },
        )
        with (
            mock.patch.dict(sys.modules, {"renewable_huber._native_cuda": extension}),
            mock.patch.object(renewable_huber, "_native_cuda", extension, create=True),
        ):
            backend = resolve_backend(
                "native_cuda",
                device="cuda",
                dtype="float32",
                cuda_graphs=True,
                cuda_fast_math=True,
            )
            self.assertFalse(backend.cuda_features["cuda_graphs_enabled"])
            self.assertFalse(backend.cuda_features["fast_math_enabled"])
            state = RenewableHuberState.empty(2, fit_intercept=False, xp=np, dtype=np.float32)
            backend.restore_native_state(state)

        self.assertTrue(received["cuda_graphs"])
        self.assertTrue(received["fast_math"])
        self.assertTrue(backend.cuda_features["cuda_graphs_enabled"])

    def test_resident_engine_restores_distinct_state_with_same_batch_count(self) -> None:
        restore_calls = 0

        class FakeEngine:
            def __init__(self, dtype: str, n_parameters: int, device_id: int) -> None:
                del device_id
                self.dtype = np.dtype(dtype)
                self.coefficients = np.zeros(n_parameters, dtype=self.dtype)

            def restore(
                self,
                coefficients: np.ndarray,
                information: np.ndarray,
                n_samples_seen: int,
                batch_count: int,
                previous_lambda: float,
                weight_sum: float,
            ) -> None:
                nonlocal restore_calls
                del information, n_samples_seen, batch_count, previous_lambda, weight_sum
                restore_calls += 1
                self.coefficients = coefficients.copy()

            def predict(self, X: np.ndarray) -> np.ndarray:
                return X @ self.coefficients

        extension = types.SimpleNamespace(
            NativeCudaEngine=FakeEngine,
            is_available=lambda: True,
            device_count=lambda: 1,
            version=lambda: {"abi_version": 1, "python_api_version": 3},
        )
        with (
            mock.patch.dict(sys.modules, {"renewable_huber._native_cuda": extension}),
            mock.patch.object(renewable_huber, "_native_cuda", extension, create=True),
        ):
            backend = resolve_backend("native_cuda", device="cuda")
            first = RenewableHuberState.empty(2, fit_intercept=False, xp=np, dtype=np.float64)
            second = first.copy()
            second.coefficients[:] = [2.0, -1.0]
            X = np.asarray([[3.0, 4.0]], dtype=np.float64)

            np.testing.assert_array_equal(backend.native_predict(X, first), [0.0])
            np.testing.assert_array_equal(backend.native_predict(X, second), [2.0])
            np.testing.assert_array_equal(backend.native_predict(X, second), [2.0])
            self.assertEqual(first.batch_count, second.batch_count)
            self.assertNotEqual(first.mirror_token, second.mirror_token)
            self.assertEqual(restore_calls, 2)

    def test_engine_initialization_error_is_unavailable_and_not_fitted(self) -> None:
        class BrokenEngine:
            def __init__(self, dtype: str, n_parameters: int, device_id: int) -> None:
                del dtype, n_parameters, device_id
                raise RuntimeError("simulated initialization failure")

        available = types.SimpleNamespace(
            NativeCudaEngine=BrokenEngine,
            is_available=lambda: True,
            device_count=lambda: 1,
            version=lambda: {"abi_version": 1, "python_api_version": 3},
        )
        with (
            mock.patch.dict(sys.modules, {"renewable_huber._native_cuda": available}),
            mock.patch.object(renewable_huber, "_native_cuda", available, create=True),
        ):
            model = RenewableHuberRegressor(backend="native_cuda", device="cuda")
            X = np.arange(12, dtype=np.float64).reshape(4, 3)
            y = np.arange(4, dtype=np.float64)
            with self.assertRaisesRegex(BackendUnavailableError, "could not initialize"):
                model.partial_fit(X, y)
            with self.assertRaises(NotFittedError):
                model.predict(X)

    def test_estimator_routes_a_complete_batch_through_one_native_call(self) -> None:
        engines: list[object] = []

        class FakeEngine:
            def __init__(self, dtype: str, n_parameters: int, device_id: int) -> None:
                self.dtype = np.dtype(dtype)
                self.n_parameters = n_parameters
                self.device_id = device_id
                self.calls = 0
                engines.append(self)

            def restore(
                self,
                coefficients: np.ndarray,
                information: np.ndarray,
                n_samples_seen: int,
                batch_count: int,
                previous_lambda: float,
                weight_sum: float,
            ) -> None:
                self.coefficients = coefficients.copy()
                self.information = information.copy()
                self.n_samples_seen = n_samples_seen
                self.batch_count = batch_count
                self.previous_lambda = previous_lambda
                self.weight_sum = weight_sum

            def update(
                self,
                X: np.ndarray,
                y: np.ndarray,
                sample_weight: np.ndarray | None,
                batch_weight: float,
                n_features_in: int,
                fit_intercept: bool,
                tau: float,
                bandwidth_scale: float,
                max_iter: int,
                tol: float,
                ridge: float,
            ) -> dict[str, object]:
                del (
                    y,
                    sample_weight,
                    n_features_in,
                    fit_intercept,
                    tau,
                    bandwidth_scale,
                    max_iter,
                    ridge,
                )
                self.calls += 1
                self.coefficients = np.arange(self.n_parameters, dtype=self.dtype)
                self.information = self.information + np.eye(self.n_parameters, dtype=self.dtype)
                self.n_samples_seen += X.shape[0]
                self.batch_count += 1
                self.weight_sum += batch_weight
                return {
                    "coefficients": self.coefficients,
                    "information": self.information,
                    "n_samples_seen": self.n_samples_seen,
                    "batch_count": self.batch_count,
                    "previous_lambda": 0.0,
                    "weight_sum": self.weight_sum,
                    "iterations": 1,
                    "converged": True,
                    "used_regularized_fallback": False,
                    "objective": tol,
                    "lambda_value": 0.0,
                    "bandwidth": 0.5,
                }

            def predict(self, X: np.ndarray) -> np.ndarray:
                return X @ self.coefficients

        available = types.SimpleNamespace(
            NativeCudaEngine=FakeEngine,
            is_available=lambda: True,
            device_count=lambda: 1,
            version=lambda: {"abi_version": 1, "python_api_version": 3},
        )
        with (
            mock.patch.dict(sys.modules, {"renewable_huber._native_cuda": available}),
            mock.patch.object(renewable_huber, "_native_cuda", available, create=True),
        ):
            model = RenewableHuberRegressor(
                backend="native_cuda", device="cuda", fit_intercept=False
            )
            X = np.arange(12, dtype=np.float64).reshape(4, 3)
            y = np.arange(4, dtype=np.float64)
            model.partial_fit(X, y, sample_weight=np.array([1.0, 0.0, 2.0, 0.0]))
            model.partial_fit(X, y)

            self.assertEqual(len(engines), 1)
            self.assertEqual(engines[0].calls, 2)
            self.assertEqual(model.n_samples_seen_, 8)
            self.assertEqual(model.state_.effective_weight, 7.0)
            np.testing.assert_array_equal(model.predict(X), X @ np.arange(3))

    def test_hard_native_error_discards_engine_before_retry(self) -> None:
        engines: list[object] = []
        predict_failures: list[bool] = []

        class RecoveringEngine:
            def __init__(self, dtype: str, n_parameters: int, device_id: int) -> None:
                del device_id
                self.dtype = np.dtype(dtype)
                self.n_parameters = n_parameters
                engines.append(self)

            def restore(
                self,
                coefficients: np.ndarray,
                information: np.ndarray,
                n_samples_seen: int,
                batch_count: int,
                previous_lambda: float,
                weight_sum: float,
            ) -> None:
                self.coefficients = coefficients.copy()
                self.information = information.copy()
                self.n_samples_seen = n_samples_seen
                self.batch_count = batch_count
                self.previous_lambda = previous_lambda
                self.weight_sum = weight_sum

            def update(
                self,
                X: np.ndarray,
                y: np.ndarray,
                sample_weight: np.ndarray | None,
                **config: object,
            ) -> dict[str, object]:
                del y, sample_weight
                if len(engines) == 1:
                    self.coefficients[:] = np.nan
                    raise RuntimeError("simulated CUDA failure")
                batch_weight = float(config["batch_weight"])
                return {
                    "coefficients": self.coefficients,
                    "information": self.information,
                    "n_samples_seen": self.n_samples_seen + X.shape[0],
                    "batch_count": self.batch_count + 1,
                    "previous_lambda": 0.0,
                    "weight_sum": self.weight_sum + batch_weight,
                    "iterations": 1,
                    "converged": True,
                    "used_regularized_fallback": False,
                    "objective": 0.0,
                    "lambda_value": 0.0,
                    "bandwidth": 0.5,
                }

            def predict(self, X: np.ndarray) -> np.ndarray:
                if not predict_failures:
                    predict_failures.append(True)
                    self.coefficients[:] = np.nan
                    raise RuntimeError("simulated predict failure")
                return X @ self.coefficients

        available = types.SimpleNamespace(
            NativeCudaEngine=RecoveringEngine,
            is_available=lambda: True,
            device_count=lambda: 1,
            version=lambda: {"abi_version": 1, "python_api_version": 3},
        )
        with (
            mock.patch.dict(sys.modules, {"renewable_huber._native_cuda": available}),
            mock.patch.object(renewable_huber, "_native_cuda", available, create=True),
        ):
            model = RenewableHuberRegressor(
                backend="native_cuda", device="cuda", fit_intercept=False
            )
            X = np.arange(12, dtype=np.float64).reshape(4, 3)
            y = np.arange(4, dtype=np.float64)
            with self.assertRaisesRegex(RuntimeError, "simulated CUDA failure"):
                model.partial_fit(X, y)
            with self.assertRaises(NotFittedError):
                model.predict(X)

            model.partial_fit(X, y)
            self.assertEqual(len(engines), 2)
            self.assertEqual(model.n_samples_seen_, 4)
            with self.assertRaisesRegex(RuntimeError, "simulated predict failure"):
                model.predict(X)
            np.testing.assert_array_equal(model.predict(X), np.zeros(4))
            self.assertEqual(len(engines), 3)


if __name__ == "__main__":
    unittest.main()
