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


if __name__ == "__main__":
    unittest.main()
