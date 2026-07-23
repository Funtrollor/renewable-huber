from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from renewable_huber import RenewableHuberRegressor


def _cupy_ready() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@unittest.skipUnless(_cupy_ready(), "CuPy with an available CUDA device is required")
class CuPyBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import cupy as cp

        cls.cp = cp
        rng = np.random.default_rng(42)
        cls.X = rng.normal(size=(160, 5))
        cls.y = cls.X @ np.asarray([1.5, -2.0, 0.0, 0.8, 0.0]) + 0.4

    def test_gpu_matches_numpy_for_streamed_updates(self) -> None:
        cpu_model = RenewableHuberRegressor(max_iter=100)
        gpu_model = RenewableHuberRegressor(backend="cupy", device="cuda", max_iter=100)
        for X_batch, y_batch in ((self.X[:80], self.y[:80]), (self.X[80:], self.y[80:])):
            cpu_model.partial_fit(X_batch, y_batch)
            gpu_model.partial_fit(self.cp.asarray(X_batch), self.cp.asarray(y_batch))

        gpu_state = gpu_model.state_
        self.assertIsInstance(gpu_state.coefficients, self.cp.ndarray)
        self.assertEqual(gpu_model.backend_, "cupy")
        self.assertTrue(gpu_model.device_.startswith("cuda:"))
        np.testing.assert_allclose(
            self.cp.asnumpy(gpu_model.coef_), cpu_model.coef_, rtol=2e-5, atol=2e-5
        )
        np.testing.assert_allclose(
            self.cp.asnumpy(gpu_model.predict(self.cp.asarray(self.X))),
            cpu_model.predict(self.X),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_auto_cuda_selection_and_checkpoint_preserve_gpu_backend(self) -> None:
        model = RenewableHuberRegressor(backend="auto", device="cuda", dtype="float32")
        model.fit(self.X, self.y)
        self.assertEqual(model.backend_, "cupy")
        self.assertEqual(model.coef_.dtype, self.cp.float32)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "gpu-model.npz"
            model.save(checkpoint)
            restored = RenewableHuberRegressor.load(checkpoint)

        self.assertEqual(restored.backend_, "cupy")
        self.assertIsInstance(restored.state_.coefficients, self.cp.ndarray)
        np.testing.assert_allclose(
            self.cp.asnumpy(restored.predict(self.cp.asarray(self.X))),
            self.cp.asnumpy(model.predict(self.cp.asarray(self.X))),
            rtol=3e-5,
            atol=3e-5,
        )

    def test_l1_update_matches_numpy_reference(self) -> None:
        cpu_model = RenewableHuberRegressor(penalty="l1", lambda_scale=0.5, max_iter=150)
        gpu_model = RenewableHuberRegressor(
            backend="cupy", device="cuda", penalty="l1", lambda_scale=0.5, max_iter=150
        )
        cpu_model.fit(self.X, self.y)
        gpu_model.fit(self.cp.asarray(self.X), self.cp.asarray(self.y))

        np.testing.assert_allclose(
            self.cp.asnumpy(gpu_model.coef_), cpu_model.coef_, rtol=2e-5, atol=2e-5
        )

    def test_weighted_gpu_update_and_cpu_checkpoint_migration(self) -> None:
        weights = np.linspace(0.2, 1.8, self.X.shape[0])
        cpu_model = RenewableHuberRegressor(max_iter=100).fit(self.X, self.y, sample_weight=weights)
        gpu_model = RenewableHuberRegressor(backend="cupy", device="cuda", max_iter=100).fit(
            self.cp.asarray(self.X),
            self.cp.asarray(self.y),
            sample_weight=self.cp.asarray(weights),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cupy-to-numpy.npz"
            gpu_model.save(checkpoint)
            restored = RenewableHuberRegressor.load(checkpoint, backend="numpy", device="cpu")

        np.testing.assert_allclose(
            self.cp.asnumpy(gpu_model.coef_), cpu_model.coef_, rtol=2e-5, atol=2e-5
        )
        np.testing.assert_allclose(
            restored.predict(self.X), cpu_model.predict(self.X), rtol=2e-5, atol=2e-5
        )

    def test_singular_design_falls_back_to_a_finite_least_squares_solution(self) -> None:
        x = self.cp.linspace(-5.0, 5.0, 101)
        X = self.cp.column_stack((x, x))
        y = 3.0 * x
        model = RenewableHuberRegressor(
            backend="cupy",
            device="cuda",
            fit_intercept=False,
            ridge=0.0,
            max_iter=200,
            tol=1e-10,
        ).fit(X, y)

        self.assertTrue(bool(self.cp.all(self.cp.isfinite(model.coef_))))
        self.assertTrue(bool(self.cp.all(self.cp.isfinite(model.state_.information))))
        self.cp.testing.assert_allclose(model.predict(X), y, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
