from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from renewable_huber import RenewableHuberRegressor
from renewable_huber.backends import resolve_backend
from renewable_huber.core.loss import huber_loss
from renewable_huber.core.update import _weighted_gram


class CPUOptimizationTests(unittest.TestCase):
    def test_numpy_weighted_gram_reuses_a_supplied_workspace(self) -> None:
        rng = np.random.default_rng(20260723)
        X = rng.normal(size=(48, 5))
        curvature = rng.choice(np.asarray([0.0, 0.5, 1.0]), size=X.shape[0])
        backend = resolve_backend("numpy")
        workspace = np.empty_like(X)

        expected = X.T @ (X * curvature[:, None])
        actual = _weighted_gram(X, curvature, backend, workspace=workspace)

        np.testing.assert_allclose(actual, expected)

    def test_l1_solver_never_builds_an_unused_hessian(self) -> None:
        rng = np.random.default_rng(12)
        X = rng.normal(size=(96, 6))
        y = X @ np.asarray([1.2, -0.8, 0.0, 0.4, 0.0, 0.3]) + 0.2

        with patch(
            "renewable_huber.core.update._gradient_and_hessian",
            side_effect=AssertionError("L1 updates must use the gradient-only path"),
        ):
            model = RenewableHuberRegressor(
                penalty="l1", lambda_scale=0.5, max_iter=100, tol=1e-7
            ).fit(X, y)

        self.assertGreater(model.score(X, y), 0.98)

    def test_l1_diagnostics_reuses_the_solver_objective(self) -> None:
        rng = np.random.default_rng(89)
        X = rng.normal(size=(128, 4))
        y = X @ np.asarray([1.0, -0.4, 0.0, 0.7]) - 0.1
        model = RenewableHuberRegressor(penalty="l1", lambda_scale=0.5, max_iter=100).fit(X, y)

        residual = y - model.predict(X)
        expected = np.mean(huber_loss(residual, model.tau, np))
        expected += model.diagnostics_.lambda_value * np.sum(np.abs(model.coef_))

        self.assertAlmostEqual(model.diagnostics_.objective, expected, places=12)

    def test_unpenalized_diagnostics_reuses_the_solver_objective(self) -> None:
        rng = np.random.default_rng(90)
        X = rng.normal(size=(128, 4))
        y = X @ np.asarray([0.8, -0.2, 0.5, 0.0]) + 0.4
        model = RenewableHuberRegressor(max_iter=100).fit(X, y)

        expected = np.mean(huber_loss(y - model.predict(X), model.tau, np))

        self.assertAlmostEqual(model.diagnostics_.objective, expected, places=12)

    def test_float32_numpy_path_tracks_a_float64_reference_without_an_intercept(self) -> None:
        rng = np.random.default_rng(91)
        X = rng.normal(size=(256, 5))
        coefficients = np.asarray([1.2, -0.6, 0.0, 0.4, -0.3])
        y = X @ coefficients + rng.normal(scale=0.03, size=X.shape[0])

        reference = RenewableHuberRegressor(fit_intercept=False, dtype="float64", max_iter=100).fit(
            X, y
        )
        optimized = RenewableHuberRegressor(fit_intercept=False, dtype="float32", max_iter=100).fit(
            X.astype(np.float32), y.astype(np.float32)
        )

        self.assertEqual(optimized.coef_.dtype, np.dtype("float32"))
        self.assertAlmostEqual(optimized.intercept_, 0.0)
        np.testing.assert_allclose(optimized.coef_, reference.coef_, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
