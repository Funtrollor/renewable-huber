from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from renewable_huber import (
    BackendUnavailableError,
    NotFittedError,
    RenewableHuberRegressor,
    ValidationError,
)


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
        model = RenewableHuberRegressor(penalty="l1", lambda_scale=1.0, max_iter=150, tol=1e-7).fit(
            X, y
        )

        self.assertGreater(model.diagnostics_.lambda_value, 0.0)
        self.assertGreaterEqual(np.count_nonzero(np.abs(model.coef_) < 1e-10), 8)
        self.assertAlmostEqual(model.intercept_, 1.0, delta=0.1)

    def test_table_like_input_preserves_feature_names(self) -> None:
        model = RenewableHuberRegressor().fit(_TableLike(self.X), self.y)

        np.testing.assert_array_equal(model.feature_names_in_, ["feature_a", "feature_b"])

    def test_integer_sample_weight_matches_explicit_row_replication(self) -> None:
        X = self.X[:40]
        y = self.y[:40]
        weights = np.tile(np.asarray([0, 1, 2, 3]), 10)
        repeated = np.repeat(np.arange(X.shape[0]), weights)

        weighted = RenewableHuberRegressor(max_iter=120, tol=1e-9).fit(X, y, sample_weight=weights)
        replicated = RenewableHuberRegressor(max_iter=120, tol=1e-9).fit(X[repeated], y[repeated])

        np.testing.assert_allclose(weighted.coef_, replicated.coef_, rtol=1e-9, atol=1e-9)
        self.assertAlmostEqual(weighted.intercept_, replicated.intercept_, delta=1e-9)
        np.testing.assert_allclose(
            weighted.state_.information,
            replicated.state_.information,
            rtol=1e-9,
            atol=1e-9,
        )
        self.assertEqual(weighted.state_.n_samples_seen, X.shape[0])
        self.assertEqual(weighted.state_.effective_weight, float(weights.sum()))

    def test_weighted_score_matches_manual_r2(self) -> None:
        weights = np.linspace(0.2, 2.0, self.X.shape[0])
        model = RenewableHuberRegressor().fit(self.X, self.y, sample_weight=weights)
        prediction = model.predict(self.X)
        mean = np.average(self.y, weights=weights)
        expected = 1.0 - np.sum(weights * (self.y - prediction) ** 2) / np.sum(
            weights * (self.y - mean) ** 2
        )

        self.assertAlmostEqual(
            model.score(self.X, self.y, sample_weight=weights), expected, places=12
        )

    def test_invalid_sample_weight_is_rejected(self) -> None:
        invalid_weights = (
            np.ones(self.X.shape[0] - 1),
            np.full(self.X.shape[0], -1.0),
            np.zeros(self.X.shape[0]),
            np.full(self.X.shape[0], np.nan),
        )
        for weights in invalid_weights:
            with self.subTest(weights=weights[:2]):
                with self.assertRaises(ValidationError):
                    RenewableHuberRegressor().fit(self.X, self.y, sample_weight=weights)

    def test_checkpoint_round_trip(self) -> None:
        weights = np.linspace(0.5, 1.5, self.X.shape[0])
        model = RenewableHuberRegressor().fit(_TableLike(self.X), self.y, sample_weight=weights)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.npz"
            model.save(checkpoint)
            restored = RenewableHuberRegressor.load(checkpoint)

        np.testing.assert_allclose(restored.predict(self.X), model.predict(self.X))
        self.assertEqual(restored.state_.n_samples_seen, model.state_.n_samples_seen)
        self.assertEqual(restored.state_.effective_weight, model.state_.effective_weight)
        np.testing.assert_array_equal(restored.feature_names_in_, ["feature_a", "feature_b"])

    def test_checkpoint_can_migrate_backend_device_and_dtype(self) -> None:
        model = RenewableHuberRegressor(dtype="float64").fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "portable-model.npz"
            model.save(checkpoint)
            restored = RenewableHuberRegressor.load(
                checkpoint, backend="numpy", device="cpu", dtype="float32"
            )

        self.assertEqual(restored.backend_, "numpy")
        self.assertEqual(restored.device_, "cpu")
        self.assertEqual(restored.dtype, "float32")
        self.assertEqual(restored.coef_.dtype, np.dtype("float32"))
        np.testing.assert_allclose(
            restored.predict(self.X), model.predict(self.X), rtol=3e-5, atol=3e-5
        )

    def test_weighted_checkpoint_resume_matches_uninterrupted_stream(self) -> None:
        weights = np.linspace(0.2, 1.8, self.X.shape[0])
        uninterrupted = RenewableHuberRegressor()
        uninterrupted.partial_fit(self.X[:120], self.y[:120], sample_weight=weights[:120])
        uninterrupted.partial_fit(self.X[120:], self.y[120:], sample_weight=weights[120:])

        resumable = RenewableHuberRegressor().partial_fit(
            self.X[:120], self.y[:120], sample_weight=weights[:120]
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weighted-stream.npz"
            resumable.save(checkpoint)
            resumed = RenewableHuberRegressor.load(checkpoint)
            resumed.partial_fit(self.X[120:], self.y[120:], sample_weight=weights[120:])

        np.testing.assert_array_equal(resumed.coef_, uninterrupted.coef_)
        self.assertEqual(resumed.intercept_, uninterrupted.intercept_)
        np.testing.assert_array_equal(resumed.state_.information, uninterrupted.state_.information)
        self.assertEqual(resumed.state_.effective_weight, uninterrupted.state_.effective_weight)

    def test_v1_checkpoint_loads_with_unit_weight_fallback(self) -> None:
        model = RenewableHuberRegressor().fit(self.X, self.y)
        payload = model.state_dict()
        metadata = {
            "format_version": 1,
            "config": payload["config"],
            "n_samples_seen": payload["n_samples_seen"],
            "batch_count": payload["batch_count"],
            "previous_lambda": payload["previous_lambda"],
            "n_features_in": payload["n_features_in"],
            "fit_intercept": payload["fit_intercept"],
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy-v1.npz"
            np.savez_compressed(
                checkpoint,
                coefficients=payload["coefficients"],
                information=payload["information"],
                metadata=np.asarray(json.dumps(metadata)),
            )
            restored = RenewableHuberRegressor.load(checkpoint)

        self.assertEqual(restored.state_.effective_weight, float(restored.state_.n_samples_seen))
        np.testing.assert_allclose(restored.predict(self.X), model.predict(self.X))

    def test_not_fitted_and_unavailable_backend_errors_are_explicit(self) -> None:
        with self.assertRaises(NotFittedError):
            RenewableHuberRegressor().predict(self.X)
        with self.assertRaises(BackendUnavailableError):
            RenewableHuberRegressor(backend="numpy", device="cuda").fit(self.X, self.y)


if __name__ == "__main__":
    unittest.main()
