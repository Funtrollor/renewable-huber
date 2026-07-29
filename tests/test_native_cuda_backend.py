from __future__ import annotations

import json
import sys
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

import renewable_huber
from renewable_huber import RenewableHuberRegressor
from renewable_huber.backends import resolve_backend
from renewable_huber.exceptions import BackendUnavailableError, NotFittedError, ValidationError

CORPUS_PATH = Path(__file__).parent / "golden" / "native_core_v1.json"
NATIVE_TOLERANCES = {
    "float32": (3e-4, 3e-5),
    # A vendor LU solve can cross the relative convergence boundary one or
    # two iterations later than NumPy/LAPACK. The resulting optimum and
    # information matrix remain substantially closer than this bound.
    "float64": (1e-8, 2e-9),
}


def _native_cuda_ready() -> bool:
    try:
        from renewable_huber import _native_cuda

        return bool(_native_cuda.is_available() and _native_cuda.device_count())
    except (ImportError, OSError, RuntimeError):
        return False


class NativeCudaSelectionTests(unittest.TestCase):
    def test_explicit_native_request_never_falls_back(self) -> None:
        unavailable = types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            version=lambda: {"abi_version": 1, "python_api_version": 1},
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

    def test_native_cuda_rejects_cpu_device(self) -> None:
        with self.assertRaisesRegex(BackendUnavailableError, "requires a CUDA device"):
            resolve_backend("native_cuda", device="cpu")

    def test_engine_initialization_error_is_unavailable_and_not_fitted(self) -> None:
        class BrokenEngine:
            def __init__(self, dtype: str, n_parameters: int, device_id: int) -> None:
                del dtype, n_parameters, device_id
                raise RuntimeError("simulated initialization failure")

        available = types.SimpleNamespace(
            NativeCudaEngine=BrokenEngine,
            is_available=lambda: True,
            device_count=lambda: 1,
            version=lambda: {"abi_version": 1, "python_api_version": 1},
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
                tau: float,
                bandwidth_scale: float,
                max_iter: int,
                tol: float,
                ridge: float,
            ) -> dict[str, object]:
                del y, sample_weight, n_features_in, tau, bandwidth_scale, max_iter, ridge
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
            version=lambda: {"abi_version": 1, "python_api_version": 1},
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
            version=lambda: {"abi_version": 1, "python_api_version": 1},
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


@unittest.skipUnless(
    _native_cuda_ready(), "the Rust/CUDA native extension and a CUDA device are required"
)
class NativeCudaGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

    def test_unpenalized_golden_cases(self) -> None:
        for case in self.corpus["cases"]:
            if case["config"]["penalty"] != "none":
                continue
            with self.subTest(case=case["id"]):
                self._replay_case(case)

    def test_engine_is_reused_across_batches(self) -> None:
        case = next(
            case for case in self.corpus["cases"] if case["id"] == "weighted_unpenalized_stream_f64"
        )
        model = self._new_model(case)
        first, second = case["batches"]
        self._partial_fit(model, first)
        engine = model._backend._engine
        self._partial_fit(model, second)
        self.assertIs(model._backend._engine, engine)

    def test_engine_supports_sequential_cross_thread_prediction(self) -> None:
        case = next(
            case for case in self.corpus["cases"] if case["id"] == "weighted_unpenalized_stream_f64"
        )
        model = self._new_model(case)
        self._partial_fit(model, case["batches"][0])
        probe = np.asarray(case["probe_X"], dtype=np.float64)
        expected = model.predict(probe)
        with ThreadPoolExecutor(max_workers=1) as executor:
            observed = executor.submit(model.predict, probe).result()
        np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-11)

    def test_l1_is_an_explicit_error(self) -> None:
        X = np.arange(24, dtype=np.float64).reshape(8, 3)
        y = np.arange(8, dtype=np.float64)
        model = RenewableHuberRegressor(backend="native_cuda", device="cuda", penalty="l1")
        with self.assertRaisesRegex(ValidationError, "penalty='none'"):
            model.fit(X, y)

    def test_checkpoint_resume_and_numpy_migration(self) -> None:
        case = next(
            case for case in self.corpus["cases"] if case["id"] == "weighted_unpenalized_stream_f64"
        )
        uninterrupted = self._new_model(case)
        first, second = case["batches"]
        self._partial_fit(uninterrupted, first)

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "native-first-batch.npz"
            uninterrupted.save(checkpoint)
            resumed = RenewableHuberRegressor.load(checkpoint, backend="native_cuda", device="cuda")
            # Prediction immediately after loading exercises lazy device-state
            # restoration before another partial_fit call.
            probe = np.asarray(case["probe_X"], dtype=np.float64)
            np.testing.assert_allclose(
                resumed.predict(probe),
                uninterrupted.predict(probe),
                rtol=float(case["rtol"]),
                atol=float(case["atol"]),
            )

            self._partial_fit(uninterrupted, second)
            self._partial_fit(resumed, second)
            np.testing.assert_allclose(
                resumed.state_.coefficients,
                uninterrupted.state_.coefficients,
                rtol=float(case["rtol"]),
                atol=float(case["atol"]),
            )
            np.testing.assert_allclose(
                resumed.state_.information,
                uninterrupted.state_.information,
                rtol=float(case["rtol"]),
                atol=float(case["atol"]),
            )

            final_checkpoint = Path(directory) / "native-final.npz"
            resumed.save(final_checkpoint)
            migrated = RenewableHuberRegressor.load(final_checkpoint, backend="numpy", device="cpu")
            np.testing.assert_allclose(
                migrated.predict(probe),
                resumed.predict(probe),
                rtol=float(case["rtol"]),
                atol=float(case["atol"]),
            )

    def test_checkpoint_information_uses_logical_row_major_layout(self) -> None:
        case = next(
            case for case in self.corpus["cases"] if case["id"] == "weighted_unpenalized_stream_f64"
        )
        first, second = case["batches"]
        oracle_seed = RenewableHuberRegressor(
            **{**case["config"], "backend": "numpy", "device": "cpu"}
        )
        self._partial_fit(oracle_seed, first)
        asymmetric_state = oracle_seed.state_
        asymmetric_state.information[0, 1] += 0.125

        numpy_model = RenewableHuberRegressor(
            **{**case["config"], "backend": "numpy", "device": "cpu"}
        )
        native_model = self._new_model(case)
        numpy_model._restore_state(asymmetric_state.copy())
        native_model._restore_state(asymmetric_state.copy())
        self._partial_fit(numpy_model, second)
        self._partial_fit(native_model, second)

        np.testing.assert_allclose(
            native_model.state_.coefficients,
            numpy_model.state_.coefficients,
            rtol=float(case["rtol"]),
            atol=float(case["atol"]),
        )
        np.testing.assert_allclose(
            native_model.state_.information,
            numpy_model.state_.information,
            rtol=float(case["rtol"]),
            atol=float(case["atol"]),
        )

    def _replay_case(self, case: dict[str, object]) -> None:
        model = self._new_model(case)
        dtype = np.dtype(case["config"]["dtype"])
        native_rtol, native_atol = NATIVE_TOLERANCES[dtype.name]
        rtol = max(float(case["rtol"]), native_rtol)
        atol = max(float(case["atol"]), native_atol)

        for batch, expected in zip(case["batches"], case["expected"]["states"], strict=True):
            self._partial_fit(model, batch)
            state = model.state_
            diagnostics = model.diagnostics_
            np.testing.assert_allclose(
                state.coefficients,
                np.asarray(expected["coefficients"], dtype=dtype),
                rtol=rtol,
                atol=atol,
            )
            np.testing.assert_allclose(
                state.information,
                np.asarray(expected["information"], dtype=dtype),
                rtol=rtol,
                atol=atol,
            )
            self.assertEqual(state.n_samples_seen, expected["n_samples_seen"])
            self.assertEqual(state.batch_count, expected["batch_count"])
            self.assertAlmostEqual(
                state.effective_weight,
                expected["weight_sum"],
                delta=atol + rtol * abs(expected["weight_sum"]),
            )
            self.assertEqual(diagnostics.converged, expected["diagnostics"]["converged"])
            for name in ("objective", "lambda_value", "bandwidth"):
                expected_value = expected["diagnostics"][name]
                self.assertAlmostEqual(
                    getattr(diagnostics, name),
                    expected_value,
                    delta=atol + rtol * abs(expected_value),
                )

        if case["id"] == "rank_deficient_lstsq_f64":
            self.assertTrue(model.diagnostics_.used_regularized_fallback)

        prediction = model.predict(np.asarray(case["probe_X"], dtype=dtype))
        np.testing.assert_allclose(
            prediction,
            np.asarray(case["expected"]["predictions"], dtype=dtype),
            rtol=rtol,
            atol=atol,
        )

    @staticmethod
    def _new_model(case: dict[str, object]) -> RenewableHuberRegressor:
        config = dict(case["config"])
        config.update(backend="native_cuda", device="cuda")
        return RenewableHuberRegressor(**config)

    @staticmethod
    def _partial_fit(model: RenewableHuberRegressor, batch: dict[str, object]) -> None:
        dtype = np.dtype(model.dtype)
        weights = batch["sample_weight"]
        model.partial_fit(
            np.asarray(batch["X"], dtype=dtype),
            np.asarray(batch["y"], dtype=dtype),
            sample_weight=(None if weights is None else np.asarray(weights, dtype=dtype)),
        )


if __name__ == "__main__":
    unittest.main()
