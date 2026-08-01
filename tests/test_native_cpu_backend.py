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
from renewable_huber.exceptions import BackendUnavailableError, NotFittedError
from renewable_huber.state import RenewableHuberState

CORPUS_PATH = Path(__file__).parent / "golden" / "native_core_v1.json"
NATIVE_TOLERANCES = {
    "float32": (4e-4, 4e-5),
    "float64": (2e-8, 3e-9),
}


def _native_cpu_ready() -> bool:
    try:
        from renewable_huber import _native_cpu

        version = _native_cpu.version()
        return version.get("abi_version") == 1 and version.get("python_api_version") == 2
    except (ImportError, OSError, RuntimeError):
        return False


class NativeCpuSelectionTests(unittest.TestCase):
    def test_explicit_native_request_never_falls_back(self) -> None:
        with (
            mock.patch.dict(sys.modules, {"renewable_huber._native_cpu": None}),
            mock.patch.object(renewable_huber, "_native_cpu", None, create=True),
        ):
            with self.assertRaises(BackendUnavailableError):
                resolve_backend("native_cpu", device="cpu")

    def test_incompatible_protocol_fails_before_engine_creation(self) -> None:
        incompatible = types.SimpleNamespace(
            version=lambda: {"abi_version": 1, "python_api_version": 999}
        )
        with (
            mock.patch.dict(sys.modules, {"renewable_huber._native_cpu": incompatible}),
            mock.patch.object(renewable_huber, "_native_cpu", incompatible, create=True),
        ):
            with self.assertRaisesRegex(BackendUnavailableError, "incompatible"):
                resolve_backend("native_cpu", device="cpu")

    def test_native_cpu_rejects_cuda_device(self) -> None:
        with self.assertRaisesRegex(BackendUnavailableError, "requires device='cpu'"):
            resolve_backend("native_cpu", device="cuda")

    def test_resolved_thread_count_reaches_engine_and_effective_property(self) -> None:
        creations: list[tuple[str, int, int | None]] = []

        class FakeEngine:
            def __init__(
                self, dtype: str, n_parameters: int, *, n_threads: int | None = None
            ) -> None:
                creations.append((dtype, n_parameters, n_threads))
                self.n_threads = n_threads or 1

            def restore(self, *args: object) -> None:
                del args

        extension = types.SimpleNamespace(
            NativeCpuEngine=FakeEngine,
            version=lambda: {"abi_version": 1, "python_api_version": 2},
        )
        with (
            mock.patch.dict(sys.modules, {"renewable_huber._native_cpu": extension}),
            mock.patch.object(renewable_huber, "_native_cpu", extension, create=True),
        ):
            backend = resolve_backend("native_cpu", device="cpu", n_jobs=3)
            self.assertEqual(backend.effective_n_jobs, 3)
            state = RenewableHuberState.empty(2, fit_intercept=False, xp=np, dtype=np.float64)
            backend.restore_native_state(state)

        self.assertEqual(creations, [("float64", 2, 3)])
        self.assertEqual(backend.effective_n_jobs, 3)

    def test_resident_engine_restores_distinct_state_with_same_batch_count(self) -> None:
        restore_calls = 0

        class FakeEngine:
            def __init__(
                self, dtype: str, n_parameters: int, *, n_threads: int | None = None
            ) -> None:
                self.dtype = np.dtype(dtype)
                self.coefficients = np.zeros(n_parameters, dtype=self.dtype)
                self.n_threads = n_threads or 1

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
            NativeCpuEngine=FakeEngine,
            version=lambda: {"abi_version": 1, "python_api_version": 2},
        )
        with (
            mock.patch.dict(sys.modules, {"renewable_huber._native_cpu": extension}),
            mock.patch.object(renewable_huber, "_native_cpu", extension, create=True),
        ):
            backend = resolve_backend("native_cpu", device="cpu")
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

    def test_engine_initialization_failure_does_not_fit_estimator(self) -> None:
        attempts = 0

        class NamedTable:
            columns = ["a", "b", "c"]

            def __init__(self, values: np.ndarray) -> None:
                self.values = values

            def to_numpy(self) -> np.ndarray:
                return self.values

        class FlakyEngine:
            def __init__(
                self, dtype: str, n_parameters: int, *, n_threads: int | None = None
            ) -> None:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("simulated failure")
                self.dtype = np.dtype(dtype)
                self.n_parameters = n_parameters
                self.n_threads = n_threads or 1

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
                batch_weight = float(config["batch_weight"])
                return {
                    "coefficients": np.zeros(self.n_parameters, dtype=self.dtype),
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

        extension = types.SimpleNamespace(
            NativeCpuEngine=FlakyEngine,
            version=lambda: {"abi_version": 1, "python_api_version": 2},
        )
        with (
            mock.patch.dict(sys.modules, {"renewable_huber._native_cpu": extension}),
            mock.patch.object(renewable_huber, "_native_cpu", extension, create=True),
        ):
            model = RenewableHuberRegressor(backend="native_cpu", fit_intercept=False, n_jobs=3)
            X = np.arange(12, dtype=np.float64).reshape(4, 3)
            y = np.arange(4, dtype=np.float64)
            with self.assertRaisesRegex(BackendUnavailableError, "could not initialize"):
                model.partial_fit(NamedTable(X), y)
            with self.assertRaises(NotFittedError):
                model.predict(X)
            for name in (
                "backend_",
                "device_",
                "n_features_in_",
                "feature_names_in_",
                "coef_",
                "intercept_",
            ):
                self.assertFalse(hasattr(model, name), name)

            retry_X = np.arange(10, dtype=np.float64).reshape(5, 2)
            model.partial_fit(retry_X, np.arange(5, dtype=np.float64))
            self.assertEqual(model.n_features_in_, 2)
            self.assertFalse(hasattr(model, "feature_names_in_"))

    def test_failed_update_preserves_state_and_recreates_engine_for_retry(self) -> None:
        engines: list[RecoveringEngine] = []

        class RecoveringEngine:
            def __init__(
                self, dtype: str, n_parameters: int, *, n_threads: int | None = None
            ) -> None:
                self.dtype = np.dtype(dtype)
                self.n_parameters = n_parameters
                self.n_threads = n_threads or 1
                self.update_calls = 0
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
                self.update_calls += 1
                if len(engines) == 1 and self.update_calls == 2:
                    self.coefficients.fill(999.0)
                    raise RuntimeError("simulated native update failure")
                batch_weight = float(config["batch_weight"])
                return {
                    "coefficients": self.coefficients.copy(),
                    "information": self.information.copy(),
                    "n_samples_seen": self.n_samples_seen + X.shape[0],
                    "batch_count": self.batch_count + 1,
                    "previous_lambda": self.previous_lambda,
                    "weight_sum": self.weight_sum + batch_weight,
                    "iterations": 1,
                    "converged": True,
                    "used_regularized_fallback": False,
                    "objective": 0.0,
                    "lambda_value": self.previous_lambda,
                    "bandwidth": 0.5,
                }

        extension = types.SimpleNamespace(
            NativeCpuEngine=RecoveringEngine,
            version=lambda: {"abi_version": 1, "python_api_version": 2},
        )
        with (
            mock.patch.dict(sys.modules, {"renewable_huber._native_cpu": extension}),
            mock.patch.object(renewable_huber, "_native_cpu", extension, create=True),
        ):
            model = RenewableHuberRegressor(backend="native_cpu", fit_intercept=False, n_jobs=3)
            X = np.arange(8, dtype=np.float64).reshape(4, 2)
            y = np.arange(4, dtype=np.float64)
            model.partial_fit(X, y)
            self.assertEqual(model.n_jobs_, 3)
            self.assertEqual(engines[0].n_threads, 3)
            coefficients_before = model.state_.coefficients.copy()
            information_before = model.state_.information.copy()

            with self.assertRaisesRegex(RuntimeError, "simulated native update failure"):
                model.partial_fit(X, y)

            np.testing.assert_array_equal(model.state_.coefficients, coefficients_before)
            np.testing.assert_array_equal(model.state_.information, information_before)
            self.assertEqual(model.state_.n_samples_seen, 4)
            self.assertEqual(model.state_.batch_count, 1)
            self.assertIsNone(model._backend._engine)

            model.partial_fit(X, y)
            self.assertEqual(len(engines), 2)
            self.assertEqual(model.n_jobs_, 3)
            self.assertEqual(engines[1].n_threads, 3)
            self.assertEqual(model.state_.n_samples_seen, 8)
            self.assertEqual(model.state_.batch_count, 2)


@unittest.skipUnless(_native_cpu_ready(), "the Rust native CPU extension is required")
class NativeCpuGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

    def test_complete_golden_corpus(self) -> None:
        for case in self.corpus["cases"]:
            with self.subTest(case=case["id"]):
                self._replay_case(case)

    def test_checkpoint_resume_and_numpy_migration(self) -> None:
        case = next(
            case for case in self.corpus["cases"] if case["id"] == "weighted_unpenalized_stream_f64"
        )
        first, second = case["batches"]
        uninterrupted = self._new_model(case)
        self._partial_fit(uninterrupted, first)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "native-cpu.npz"
            uninterrupted.save(checkpoint)
            resumed = RenewableHuberRegressor.load(checkpoint, backend="native_cpu", device="cpu")
            probe = np.asarray(case["probe_X"], dtype=np.float64)
            np.testing.assert_allclose(
                resumed.predict(probe),
                uninterrupted.predict(probe),
                rtol=2e-8,
                atol=3e-9,
            )
            self._partial_fit(uninterrupted, second)
            self._partial_fit(resumed, second)
            np.testing.assert_allclose(
                resumed.state_.coefficients,
                uninterrupted.state_.coefficients,
                rtol=2e-8,
                atol=3e-9,
            )
            resumed.save(checkpoint)
            migrated = RenewableHuberRegressor.load(checkpoint, backend="numpy", device="cpu")
            np.testing.assert_allclose(
                migrated.predict(probe),
                resumed.predict(probe),
                rtol=2e-8,
                atol=3e-9,
            )

    def test_l1_checkpoint_resume_preserves_historical_state(self) -> None:
        case = next(case for case in self.corpus["cases"] if case["id"] == "l1_stream_f64")
        first, second = case["batches"]
        uninterrupted = self._new_model(case)
        self._partial_fit(uninterrupted, first)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "native-cpu-l1.npz"
            uninterrupted.save(checkpoint)
            resumed = RenewableHuberRegressor.load(checkpoint, backend="native_cpu", device="cpu")
            self._partial_fit(uninterrupted, second)
            self._partial_fit(resumed, second)
            np.testing.assert_allclose(
                resumed.state_.coefficients,
                uninterrupted.state_.coefficients,
                rtol=3e-7,
                atol=3e-8,
            )
            self.assertAlmostEqual(
                resumed.state_.previous_lambda,
                uninterrupted.state_.previous_lambda,
            )

    def test_engine_and_workspace_owner_are_reused_across_batches(self) -> None:
        case = next(
            case for case in self.corpus["cases"] if case["id"] == "weighted_unpenalized_stream_f64"
        )
        model = self._new_model(case)
        first, second = case["batches"]
        self._partial_fit(model, first)
        engine = model._backend._engine
        self._partial_fit(model, second)
        self.assertIs(model._backend._engine, engine)

    def test_general_asymmetric_information_state_matches_numpy(self) -> None:
        case = next(
            case for case in self.corpus["cases"] if case["id"] == "weighted_unpenalized_stream_f64"
        )
        first, second = case["batches"]
        seed = RenewableHuberRegressor(**{**case["config"], "backend": "numpy", "device": "cpu"})
        self._partial_fit(seed, first)
        asymmetric = seed.state_
        asymmetric.information[0, 1] += 0.125

        native = self._new_model(case)
        oracle = RenewableHuberRegressor(**{**case["config"], "backend": "numpy", "device": "cpu"})
        native._restore_state(asymmetric.copy())
        oracle._restore_state(asymmetric.copy())
        self._partial_fit(native, second)
        self._partial_fit(oracle, second)
        np.testing.assert_allclose(
            native.state_.coefficients,
            oracle.state_.coefficients,
            rtol=2e-8,
            atol=3e-9,
        )
        np.testing.assert_allclose(
            native.state_.information,
            oracle.state_.information,
            rtol=2e-8,
            atol=3e-9,
        )

    def test_parallel_weighted_update_matches_numpy(self) -> None:
        rng = np.random.default_rng(731)
        rows = 8_192
        features = 32
        X_source = rng.normal(size=(rows, features))
        y_source = X_source @ rng.normal(size=features) + rng.normal(scale=0.2, size=rows)
        weights_source = rng.uniform(0.25, 2.0, size=rows)
        weights_source[::17] = 0.0

        for dtype_name in ("float32", "float64"):
            for penalty in ("none", "l1"):
                with self.subTest(dtype=dtype_name, penalty=penalty):
                    dtype = np.dtype(dtype_name)
                    X = np.asarray(X_source, dtype=dtype)
                    y = np.asarray(y_source, dtype=dtype)
                    weights = np.asarray(weights_source, dtype=dtype)
                    config = {
                        "fit_intercept": True,
                        "penalty": penalty,
                        "dtype": dtype_name,
                        "max_iter": 30,
                        "tol": 1e-6,
                    }
                    oracle = RenewableHuberRegressor(
                        **config, backend="numpy", device="cpu"
                    ).partial_fit(X, y, sample_weight=weights)
                    native_single = RenewableHuberRegressor(
                        **config, backend="native_cpu", device="cpu", n_jobs=1
                    ).partial_fit(X, y, sample_weight=weights)
                    native_parallel = RenewableHuberRegressor(
                        **config, backend="native_cpu", device="cpu", n_jobs=4
                    ).partial_fit(X, y, sample_weight=weights)
                    rtol, atol = NATIVE_TOLERANCES[dtype_name]
                    np.testing.assert_allclose(
                        native_single.state_.coefficients,
                        oracle.state_.coefficients,
                        rtol=rtol,
                        atol=atol,
                    )
                    np.testing.assert_allclose(
                        native_single.state_.information,
                        oracle.state_.information,
                        rtol=rtol,
                        atol=atol,
                    )
                    np.testing.assert_allclose(
                        native_parallel.state_.coefficients,
                        native_single.state_.coefficients,
                        rtol=rtol,
                        atol=atol,
                    )
                    np.testing.assert_allclose(
                        native_parallel.state_.information,
                        native_single.state_.information,
                        rtol=rtol,
                        atol=atol,
                    )
                    self.assertAlmostEqual(
                        native_parallel.state_.effective_weight,
                        oracle.state_.effective_weight,
                        delta=atol + rtol * oracle.state_.effective_weight,
                    )
                    self.assertEqual(native_single.n_jobs_, 1)
                    self.assertEqual(native_parallel.n_jobs_, 4)

    def test_sequential_cross_thread_prediction(self) -> None:
        case = next(
            case for case in self.corpus["cases"] if case["id"] == "weighted_unpenalized_stream_f64"
        )
        model = self._new_model(case)
        self._partial_fit(model, case["batches"][0])
        probe = np.asarray(case["probe_X"], dtype=np.float64)
        expected = model.predict(probe)
        with ThreadPoolExecutor(max_workers=1) as executor:
            observed = executor.submit(model.predict, probe).result()
        np.testing.assert_allclose(observed, expected, rtol=2e-8, atol=3e-9)

    def test_direct_binding_rejects_non_contiguous_input(self) -> None:
        from renewable_huber import _native_cpu

        engine = _native_cpu.NativeCpuEngine("float64", 2)
        engine.restore(
            np.zeros(2, dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            0,
            0,
            0.0,
            0.0,
        )
        non_contiguous = np.zeros((4, 4), dtype=np.float64)[:, ::2]
        with self.assertRaisesRegex(ValueError, "C-contiguous"):
            engine.predict(non_contiguous)

    def test_direct_binding_rejects_mixed_dtype(self) -> None:
        from renewable_huber import _native_cpu

        engine = _native_cpu.NativeCpuEngine("float64", 2)
        engine.restore(
            np.zeros(2, dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            0,
            0,
            0.0,
            0.0,
        )
        with self.assertRaisesRegex(TypeError, "engine dtype"):
            engine.predict(np.zeros((4, 2), dtype=np.float32))

    def test_direct_binding_rejects_non_finite_state(self) -> None:
        from renewable_huber import _native_cpu

        engine = _native_cpu.NativeCpuEngine("float64", 2)
        with self.assertRaisesRegex(ValueError, "finite"):
            engine.restore(
                np.asarray([np.nan, 0.0], dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
                0,
                0,
                0.0,
                0.0,
            )

    def test_direct_binding_update_error_preserves_resident_state(self) -> None:
        from renewable_huber import _native_cpu

        engine = _native_cpu.NativeCpuEngine("float64", 2)
        coefficients = np.asarray([0.25, -0.5], dtype=np.float64)
        information = np.asarray([[2.0, 0.25], [0.25, 1.0]], dtype=np.float64)
        engine.restore(coefficients, information, 4, 1, 0.0, 4.0)
        X_design = np.asarray([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]], dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "target length"):
            engine.update(
                X_design,
                np.ones(3, dtype=np.float64),
                None,
                4.0,
                1,
                True,
                1.345,
                "none",
                1.0,
                1.0,
                20,
                1e-6,
                1e-8,
            )

        restored = engine.state()
        np.testing.assert_array_equal(restored["coefficients"], coefficients)
        np.testing.assert_array_equal(restored["information"], information)
        self.assertEqual(restored["n_samples_seen"], 4)
        self.assertEqual(restored["batch_count"], 1)

        result = engine.update(
            X_design,
            np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            None,
            4.0,
            1,
            True,
            1.345,
            "none",
            1.0,
            1.0,
            20,
            1e-6,
            1e-8,
        )
        self.assertEqual(result["n_samples_seen"], 8)
        self.assertEqual(result["batch_count"], 2)

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
            self.assertEqual(state.coefficients.dtype, dtype)
            self.assertEqual(state.information.dtype, dtype)
            self.assertEqual(state.n_samples_seen, expected["n_samples_seen"])
            self.assertEqual(state.batch_count, expected["batch_count"])
            self.assertAlmostEqual(
                state.previous_lambda,
                expected["previous_lambda"],
                delta=atol + rtol * abs(expected["previous_lambda"]),
            )
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
        self.assertEqual(prediction.dtype, dtype)
        np.testing.assert_allclose(
            prediction,
            np.asarray(case["expected"]["predictions"], dtype=dtype),
            rtol=rtol,
            atol=atol,
        )

    @staticmethod
    def _new_model(case: dict[str, object]) -> RenewableHuberRegressor:
        config = dict(case["config"])
        config.update(backend="native_cpu", device="cpu")
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
