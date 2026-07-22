from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from renewable_huber import RenewableHuberRegressor
from renewable_huber.backends.cupy_backend import CuPyBackend
from renewable_huber.core.loss import huber_loss, smoothed_score_and_curvature


def _cupy_ready() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@unittest.skipUnless(_cupy_ready(), "CuPy with an available CUDA device is required")
class CudaKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import cupy as cp

        cls.cp = cp

    def test_fused_terms_match_generic_cupy_at_piecewise_boundaries(self) -> None:
        tau = 1.25
        bandwidth = 0.2
        h = min(bandwidth, tau * 0.5)
        values = np.asarray(
            [
                -3.0,
                -tau - h,
                -tau - h + 1e-4,
                -tau + h,
                -tau + h + 1e-4,
                0.0,
                tau - h - 1e-4,
                tau - h,
                tau + h,
                tau + h + 1e-4,
                3.0,
            ]
        )
        for dtype in (self.cp.float32, self.cp.float64):
            with self.subTest(dtype=dtype):
                backend = CuPyBackend(dtype=np.dtype(dtype).name)
                self.assertTrue(backend.cuda_kernels_available, repr(backend.cuda_kernel_error))
                residual = self.cp.asarray(values, dtype=dtype)
                actual = backend.cuda_smoothed_terms(residual, tau, bandwidth)
                self.assertIsNotNone(actual)
                expected = smoothed_score_and_curvature(residual, tau, bandwidth, self.cp)
                self.cp.testing.assert_allclose(actual[0], expected[0], rtol=2e-6, atol=2e-6)
                self.cp.testing.assert_allclose(actual[1], expected[1], rtol=2e-6, atol=2e-6)
                actual_loss = backend.cuda_huber_loss(residual, tau)
                self.assertIsNotNone(actual_loss)
                self.cp.testing.assert_allclose(
                    actual_loss, huber_loss(residual, tau, self.cp), rtol=2e-6, atol=2e-6
                )

    def test_unpenalized_gpu_update_reuses_one_weighted_gram_workspace(self) -> None:
        from renewable_huber.core import update

        rng = np.random.default_rng(7)
        X = self.cp.asarray(rng.normal(size=(256, 8)), dtype=self.cp.float32)
        y = self.cp.asarray(rng.normal(size=256), dtype=self.cp.float32)
        observed_workspaces = []
        original_weighted_gram = update._weighted_gram

        def record_workspace(*args, **kwargs):
            observed_workspaces.append(kwargs.get("workspace"))
            return original_weighted_gram(*args, **kwargs)

        model = RenewableHuberRegressor(backend="cupy", device="cuda", dtype="float32")
        with patch.object(update, "_weighted_gram", side_effect=record_workspace):
            model.partial_fit(X, y)

        self.assertGreaterEqual(len(observed_workspaces), 2)
        workspace = observed_workspaces[0]
        self.assertIsInstance(workspace, self.cp.ndarray)
        self.assertTrue(all(candidate is workspace for candidate in observed_workspaces))

if __name__ == "__main__":
    unittest.main()
