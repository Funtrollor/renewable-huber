from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from renewable_huber import BackendUnavailableError, NotFittedError, RenewableHuberRegressor


class _TableLike:
    def __init__(self, values: np.ndarray) -> None:
        self._values = values
        self.columns = ["feature_a", "feature_b"]

    def to_numpy(self) -> np.ndarray:
        return self._values


class RenewableHuberRegressorTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260723)
        self.X = rng.normal(size=(240, 2))
        self.y = 2.5 * self.X[:, 0] - 1.25 * self.X[:, 1] + 0.75 + rng.normal(scale=0.08, size=240)

    def test_fit_recovers_a_robust_linear_signal(self) -> None:
        model = RenewableHuberRegressor(max_iter=80).fit(self.X, self.y)

        np.testing.assert_allclose(model.coef_, [2.5, -1.25], atol=0.12)
        self.assertAlmostEqual(model.intercept_, 0.75, delta=0.12)
        self.assertGreater(model.score(self.X, self.y), 0.98)

    def test_partial_fit_keeps_only_renewable_state(self) -> None:
        model = RenewableHuberRegressor(max_iter=80)
        model.partial_fit(self.X[:120], self.y[:120])
        model.partial_fit(self.X[120:], self.y[120:])

        self.assertEqual(model.state_.n_samples_seen, 240)
        self.assertEqual(model.state_.batch_count, 2)
        self.assertEqual(model.state_.information.shape, (3, 3))
        self.assertLess(np.mean(np.abs(model.predict(self.X) - self.y)), 0.15)

    def test_l1_penalty_shrinks_irrelevant_features(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(180, 12))
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.0
        model = RenewableHuberRegressor(
            penalty="l1", lambda_scale=1.0, max_iter=150, tol=1e-7
        ).fit(X, y)

        self.assertGreater(model.diagnostics_.lambda_value, 0.0)
        self.assertGreaterEqual(np.count_nonzero(np.abs(model.coef_) < 1e-10), 8)
        self.assertAlmostEqual(model.intercept_, 1.0, delta=0.1)

    def test_table_like_input_preserves_feature_names(self) -> None:
        model = RenewableHuberRegressor().fit(_TableLike(self.X), self.y)

        np.testing.assert_array_equal(model.feature_names_in_, ["feature_a", "feature_b"])

    def test_checkpoint_round_trip(self) -> None:
        model = RenewableHuberRegressor().fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.npz"
            model.save(checkpoint)
            restored = RenewableHuberRegressor.load(checkpoint)

        np.testing.assert_allclose(restored.predict(self.X), model.predict(self.X))
        self.assertEqual(restored.state_.n_samples_seen, model.state_.n_samples_seen)

    def test_not_fitted_and_unavailable_backend_errors_are_explicit(self) -> None:
        with self.assertRaises(NotFittedError):
            RenewableHuberRegressor().predict(self.X)
        with self.assertRaises(BackendUnavailableError):
            RenewableHuberRegressor(backend="torch").fit(self.X, self.y)


if __name__ == "__main__":
    unittest.main()
