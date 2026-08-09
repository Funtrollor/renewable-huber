from __future__ import annotations

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from renewable_huber import RenewableHuberRegressor
from renewable_huber.backends import resolve_backend
from renewable_huber.backends.capabilities import capabilities_of
from renewable_huber.exceptions import ValidationError

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


def _native_cpu_ready() -> bool:
    try:
        from renewable_huber import _native_cpu

        version = _native_cpu.version()
        return version.get("abi_version") == 1 and version.get("python_api_version") == 2
    except (ImportError, OSError, RuntimeError):
        return False


def _native_cuda_cupy_ready() -> bool:
    if not _native_cuda_ready():
        return False
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except (ImportError, RuntimeError):
        return False


@unittest.skipUnless(_native_cuda_cupy_ready(), "native CUDA, CuPy, and a CUDA device are required")
class NativeCudaDlpackTests(unittest.TestCase):
    def setUp(self) -> None:
        import cupy as cp

        self.cp = cp
        rng = np.random.default_rng(991)
        self.X = rng.normal(size=(256, 6)).astype(np.float32)
        beta = rng.normal(size=6).astype(np.float32)
        self.y = (self.X @ beta + rng.normal(scale=0.1, size=256)).astype(np.float32)

    def test_dlpack_device_input_matches_host_input_state(self) -> None:
        weights = np.linspace(0.25, 2.0, self.X.shape[0], dtype=np.float32)
        config = dict(
            backend="native_cuda",
            device="cuda",
            dtype="float32",
            fit_intercept=True,
            max_iter=80,
        )
        host = RenewableHuberRegressor(**config).fit(self.X, self.y, sample_weight=weights)
        device = RenewableHuberRegressor(**config).fit(
            self.cp.asarray(self.X),
            self.cp.asarray(self.y),
            sample_weight=self.cp.asarray(weights),
        )

        np.testing.assert_allclose(device.coef_, host.coef_, rtol=3e-4, atol=3e-5)
        self.assertAlmostEqual(device.intercept_, host.intercept_, places=4)
        np.testing.assert_allclose(
            device._state.information,
            host._state.information,
            rtol=5e-4,
            atol=5e-4,
        )
        self.assertEqual(device.n_samples_seen_, host.n_samples_seen_)
        self.assertEqual(device._state.effective_weight, host._state.effective_weight)

    def test_dlpack_requires_exact_dtype_and_c_contiguity(self) -> None:
        wrong_dtype = RenewableHuberRegressor(backend="native_cuda", device="cuda", dtype="float64")
        with self.assertRaisesRegex(TypeError, "dtype must exactly match"):
            wrong_dtype.fit(self.cp.asarray(self.X), self.cp.asarray(self.y))

        noncontiguous = self.cp.asarray(self.X)[:, ::2]
        matching_y = self.cp.asarray(self.y)
        model = RenewableHuberRegressor(backend="native_cuda", device="cuda", dtype="float32")
        with self.assertRaisesRegex(ValueError, "C-contiguous"):
            model.fit(noncontiguous, matching_y)

    def test_dlpack_rejects_mixed_host_and_device_batch(self) -> None:
        model = RenewableHuberRegressor(backend="native_cuda", device="cuda", dtype="float32")
        with self.assertRaisesRegex(ValidationError, "cannot mix host arrays"):
            model.fit(self.cp.asarray(self.X), self.y)

    def test_dlpack_rank_deficiency_initializes_lazy_svd_fallback(self) -> None:
        rng = np.random.default_rng(177)
        base = rng.normal(size=(96, 2))
        X = np.column_stack((base, base[:, 0])).astype(np.float64)
        y = (base @ np.asarray([1.5, -0.75]) + 0.05).astype(np.float64)
        model = RenewableHuberRegressor(
            backend="native_cuda",
            device="cuda",
            dtype="float64",
            ridge=0.0,
            max_iter=80,
        ).fit(self.cp.asarray(X), self.cp.asarray(y))

        self.assertTrue(model.diagnostics_.used_regularized_fallback)
        self.assertTrue(np.isfinite(model.coef_).all())
        self.assertTrue(np.isfinite(model._state.information).all())


@unittest.skipUnless(
    _native_cuda_ready(), "the Rust/CUDA native extension and a CUDA device are required"
)
class NativeCudaTuningTests(unittest.TestCase):
    def test_graph_and_fast_precision_preserve_the_declared_contract(self) -> None:
        rng = np.random.default_rng(2048)
        X = rng.normal(size=(8192, 48)).astype(np.float32)
        beta = rng.normal(size=48).astype(np.float32)
        y = (X @ beta + rng.normal(scale=0.2, size=X.shape[0])).astype(np.float32)
        y[::97] += 8
        common = dict(
            backend="native_cuda",
            device="cuda",
            dtype="float32",
            max_iter=40,
            tol=1e-5,
        )
        strict = RenewableHuberRegressor(**common).fit(X, y)
        graph = RenewableHuberRegressor(**common, cuda_graphs=True).fit(X, y)
        fast = RenewableHuberRegressor(**common, cuda_graphs=True, cuda_fast_math=True).fit(X, y)

        np.testing.assert_array_equal(graph.coef_, strict.coef_)
        self.assertEqual(graph.intercept_, strict.intercept_)
        np.testing.assert_allclose(fast.coef_, strict.coef_, rtol=5e-3, atol=5e-4)
        self.assertAlmostEqual(fast.intercept_, strict.intercept_, delta=5e-4)
        self.assertGreaterEqual(
            graph.cuda_features_["graph_captures"] + graph.cuda_features_["graph_fallbacks"],
            1,
        )
        self.assertTrue(fast.cuda_features_["fast_math_enabled"])


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

    @unittest.skipUnless(
        _native_cpu_ready(), "the Rust native CPU extension is required for cross-engine migration"
    )
    def test_checkpoint_migrates_between_native_cpu_and_cuda(self) -> None:
        rng = np.random.default_rng(91)
        X = rng.normal(size=(64, 5))
        y = X @ np.array([0.8, -1.2, 0.0, 2.1, -0.4]) + rng.normal(scale=0.05, size=64)
        probe = rng.normal(size=(11, 5))
        cpu = RenewableHuberRegressor(
            backend="native_cpu",
            device="cpu",
            fit_intercept=False,
            dtype="float64",
            max_iter=50,
        ).fit(X, y)

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "native-cross-engine.npz"
            cpu.save(checkpoint)
            cuda = RenewableHuberRegressor.load(checkpoint, backend="native_cuda", device="cuda")
            np.testing.assert_allclose(
                cuda.predict(probe),
                cpu.predict(probe),
                rtol=2e-8,
                atol=3e-9,
            )

            cuda.save(checkpoint)
            restored_cpu = RenewableHuberRegressor.load(
                checkpoint, backend="native_cpu", device="cpu"
            )
            np.testing.assert_allclose(
                restored_cpu.predict(probe),
                cuda.predict(probe),
                rtol=2e-8,
                atol=3e-9,
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


@unittest.skipUnless(_native_cuda_ready(), "the native CUDA extension and a device are required")
class NativeCudaCapabilityTests(unittest.TestCase):
    """What this backend advertises is what the core is able to find.

    Kept beside the other native CUDA tests so the required ``cuda`` profile
    executes them on the fixed GPU host rather than skipping them inside the
    portable capability module.
    """

    def test_native_cuda_advertises_design_ownership_and_device_reductions(self) -> None:
        capabilities = capabilities_of(resolve_backend("native_cuda", device="cuda"))
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
