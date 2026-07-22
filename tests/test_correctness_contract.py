from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from renewable_huber import RenewableHuberRegressor, ValidationError
from renewable_huber.backends import resolve_backend
from renewable_huber.config import EstimatorConfig
from renewable_huber.core.loss import huber_score, smoothed_curvature, smoothed_score
from renewable_huber.core.update import _gradient, _gradient_from_score
from renewable_huber.state import RenewableHuberState


class PaperReferenceContractTests(unittest.TestCase):
    def test_paper_equation_2_1_score_and_curvature_golden_points(self) -> None:
        """Match Jiang, Liang, and Yu (2024), equation (2.1), at fixed points."""

        residual = np.asarray(
            [-3.0, -2.5, -2.25, -2.0, -1.75, -1.5, -1.0, 0.0, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
        )
        expected_score = np.asarray(
            [
                -2.0,
                -2.0,
                -1.96875,
                -1.875,
                -1.71875,
                -1.5,
                -1.0,
                0.0,
                1.0,
                1.5,
                1.71875,
                1.875,
                1.96875,
                2.0,
                2.0,
            ]
        )
        expected_curvature = np.asarray(
            [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0]
        )

        actual_score = smoothed_score(residual, tau=2.0, bandwidth=0.5, xp=np)
        actual_curvature = smoothed_curvature(residual, tau=2.0, bandwidth=0.5, xp=np)

        np.testing.assert_allclose(actual_score, expected_score, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(actual_curvature, expected_curvature, rtol=0.0, atol=1e-15)

    def test_paper_bandwidth_above_half_tau_is_not_silently_narrowed(self) -> None:
        """The paper uses h=1 with tau=1.345; h only needs to stay at or below tau."""

        residual = np.asarray([-3.5, -2.0, -0.5, 0.0, 0.5, 2.0, 3.5])
        expected_score = np.asarray([-2.0, -1.625, -0.5, 0.0, 0.5, 1.625, 2.0])
        expected_curvature = np.asarray([0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0])

        actual_score = smoothed_score(residual, tau=2.0, bandwidth=1.5, xp=np)
        actual_curvature = smoothed_curvature(residual, tau=2.0, bandwidth=1.5, xp=np)

        np.testing.assert_allclose(actual_score, expected_score, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(actual_curvature, expected_curvature, rtol=0.0, atol=1e-15)

    def test_current_batch_gradient_uses_ordinary_huber_score(self) -> None:
        """Match U(D_b; beta) in equations (2.8) and (3.9), not its smoothed proxy."""

        backend = resolve_backend("numpy")
        config = EstimatorConfig(tau=2.0, bandwidth_scale=100.0, fit_intercept=False)
        state = RenewableHuberState.empty(1, fit_intercept=False, xp=np, dtype=np.float64)
        X = np.ones((3, 1))
        y = np.asarray([-2.25, -1.75, 1.75])
        beta = np.zeros(1)

        actual = _gradient(X, y, beta, state, config, backend=backend)
        expected_score = huber_score(y, config.tau, np)
        expected = np.asarray([-np.sum(expected_score) / y.size])
        smoothed_reference = np.asarray([-np.sum(smoothed_score(y, config.tau, 0.5, np)) / y.size])

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
        self.assertGreater(float(np.linalg.norm(actual - smoothed_reference)), 1e-3)


class EstimatorCorrectnessContractTests(unittest.TestCase):
    @staticmethod
    def _parameters(model: RenewableHuberRegressor) -> np.ndarray:
        return np.concatenate((np.asarray(model.coef_), np.asarray([model.intercept_])))

    def test_quadratic_huber_region_matches_closed_form_least_squares(self) -> None:
        X = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, -1.0],
                [-1.0, 2.0],
                [3.0, 0.5],
            ]
        )
        y = X @ np.asarray([1.5, -0.75]) + 0.4
        y += np.asarray([0.1, -0.2, 0.05, 0.3, -0.1, -0.15])
        design = np.column_stack((X, np.ones(X.shape[0])))
        expected = np.linalg.lstsq(design, y, rcond=None)[0]

        # tau=100 keeps every residual in the quadratic Huber region, where
        # the paper's estimating equation is exactly the OLS normal equation.
        model = RenewableHuberRegressor(tau=100.0, ridge=0.0, max_iter=20, tol=1e-12).fit(X, y)

        np.testing.assert_allclose(self._parameters(model), expected, rtol=1e-12, atol=1e-12)

    def test_large_response_outlier_is_bounded_influence(self) -> None:
        x = np.linspace(-5.0, 5.0, 101)
        X = x[:, None]
        truth = np.asarray([2.0, 1.0])
        y = truth[0] * x + truth[1]
        y[50] += 1000.0

        model = RenewableHuberRegressor(max_iter=300, tol=1e-10).fit(X, y)
        robust_error = np.linalg.norm(self._parameters(model) - truth)
        ols = np.linalg.lstsq(np.column_stack((X, np.ones(X.shape[0]))), y, rcond=None)[0]
        ols_error = np.linalg.norm(ols - truth)

        self.assertLess(robust_error, 0.03)
        self.assertLess(robust_error, ols_error * 0.01)

    def test_singular_collinear_design_uses_a_finite_least_squares_solution(self) -> None:
        x = np.linspace(-5.0, 5.0, 101)
        X = np.column_stack((x, x))
        y = 3.0 * x

        model = RenewableHuberRegressor(
            fit_intercept=False, ridge=0.0, max_iter=200, tol=1e-10
        ).fit(X, y)

        self.assertTrue(np.isfinite(model.coef_).all())
        self.assertTrue(np.isfinite(model.state_.information).all())
        np.testing.assert_allclose(model.predict(X), y, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float(np.sum(model.coef_)), 3.0, places=12)

    def test_float32_path_preserves_dtype_and_tracks_closed_form_reference(self) -> None:
        rng = np.random.default_rng(20260723)
        X = rng.normal(size=(128, 3)).astype(np.float32)
        y = (X @ np.asarray([1.25, -0.5, 0.75], dtype=np.float32) + 0.2).astype(np.float32)
        expected = np.linalg.lstsq(
            np.column_stack((X.astype(np.float64), np.ones(X.shape[0]))),
            y.astype(np.float64),
            rcond=None,
        )[0]

        model = RenewableHuberRegressor(
            tau=100.0, dtype="float32", ridge=0.0, max_iter=40, tol=1e-6
        ).fit(X, y)

        self.assertEqual(model.coef_.dtype, np.dtype("float32"))
        self.assertEqual(model.state_.information.dtype, np.dtype("float32"))
        np.testing.assert_allclose(self._parameters(model), expected, rtol=2e-5, atol=2e-5)

    def test_diagnostics_report_the_effective_bandwidth(self) -> None:
        X = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        y = np.asarray([0.0, 1.0, 2.0, 3.0])
        model = RenewableHuberRegressor(tau=1.0, bandwidth_scale=100.0).fit(X, y)

        self.assertEqual(model.diagnostics_.bandwidth, 1.0)

    def test_fit_intercept_false_keeps_the_parameter_space_through_the_origin(self) -> None:
        X = np.asarray([[-2.0, 1.0], [-1.0, -1.0], [0.0, 2.0], [1.0, 0.5], [3.0, -2.0]])
        expected = np.asarray([1.75, -0.6])
        y = X @ expected

        model = RenewableHuberRegressor(
            tau=100.0, fit_intercept=False, ridge=0.0, max_iter=20, tol=1e-12
        ).fit(X, y)

        np.testing.assert_allclose(model.coef_, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(model.intercept_, 0.0)
        self.assertEqual(model.state_.information.shape, (2, 2))

    def test_l1_matches_orthogonal_design_soft_threshold_solution(self) -> None:
        n_samples = 100
        row = np.arange(n_samples)
        X = np.column_stack(
            (
                np.where(row % 2 == 0, 1.0, -1.0),
                np.where((row // 2) % 2 == 0, 1.0, -1.0),
            )
        )
        unpenalized = np.asarray([3.0, 0.2])
        y = X @ unpenalized
        expected_lambda = 0.5 * 10.0 * np.sqrt(np.log(2.0) / n_samples)
        expected = np.sign(unpenalized) * np.maximum(np.abs(unpenalized) - expected_lambda, 0.0)

        model = RenewableHuberRegressor(
            penalty="l1",
            lambda_scale=0.5,
            tau=10.0,
            fit_intercept=False,
            ridge=0.0,
            max_iter=500,
            tol=1e-12,
        ).fit(X, y)

        self.assertAlmostEqual(model.diagnostics_.lambda_value, expected_lambda, places=15)
        np.testing.assert_allclose(model.coef_, expected, rtol=1e-10, atol=1e-10)

    def test_checkpoint_resume_is_equivalent_to_uninterrupted_stream(self) -> None:
        rng = np.random.default_rng(913)
        X = rng.normal(size=(120, 3))
        y = X @ np.asarray([1.2, -0.7, 0.4]) + 0.3
        y += rng.normal(scale=0.08, size=X.shape[0])
        first = (X[:50], y[:50])
        second = (X[50:], y[50:])

        uninterrupted = RenewableHuberRegressor(max_iter=150, tol=1e-10)
        uninterrupted.partial_fit(*first)
        uninterrupted.partial_fit(*second)

        resumable = RenewableHuberRegressor(max_iter=150, tol=1e-10)
        resumable.partial_fit(*first)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stream.npz"
            resumable.save(checkpoint)
            resumed = RenewableHuberRegressor.load(checkpoint)
            resumed.partial_fit(*second)

        np.testing.assert_array_equal(resumed.coef_, uninterrupted.coef_)
        self.assertEqual(resumed.intercept_, uninterrupted.intercept_)
        np.testing.assert_array_equal(resumed.state_.information, uninterrupted.state_.information)
        self.assertEqual(resumed.state_.n_samples_seen, 120)
        self.assertEqual(resumed.state_.batch_count, 2)

    def test_l1_checkpoint_resume_preserves_the_historical_penalty_state(self) -> None:
        rng = np.random.default_rng(914)
        X = rng.normal(size=(160, 5))
        y = X @ np.asarray([1.5, -0.8, 0.0, 0.4, 0.0]) + 0.25
        first = (X[:70], y[:70])
        second = (X[70:], y[70:])
        settings = {
            "penalty": "l1",
            "lambda_scale": 0.6,
            "max_iter": 250,
            "tol": 1e-6,
        }

        uninterrupted = RenewableHuberRegressor(**settings)
        uninterrupted.partial_fit(*first)
        self.assertTrue(uninterrupted.diagnostics_.converged)
        uninterrupted.partial_fit(*second)
        self.assertTrue(uninterrupted.diagnostics_.converged)

        resumable = RenewableHuberRegressor(**settings)
        resumable.partial_fit(*first)
        self.assertTrue(resumable.diagnostics_.converged)
        previous_lambda = resumable.state_.previous_lambda
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "l1-stream.npz"
            resumable.save(checkpoint)
            resumed = RenewableHuberRegressor.load(checkpoint)
            self.assertEqual(resumed.state_.previous_lambda, previous_lambda)
            resumed.partial_fit(*second)
            self.assertTrue(resumed.diagnostics_.converged)

        np.testing.assert_array_equal(resumed.coef_, uninterrupted.coef_)
        self.assertEqual(resumed.intercept_, uninterrupted.intercept_)
        np.testing.assert_array_equal(resumed.state_.information, uninterrupted.state_.information)
        self.assertEqual(resumed.state_.previous_lambda, uninterrupted.state_.previous_lambda)

    def test_second_batch_l1_gradient_includes_masked_historical_subgradient(self) -> None:
        backend = resolve_backend("numpy")
        state = RenewableHuberState(
            coefficients=np.asarray([2.0, -3.0, 4.0]),
            information=np.zeros((3, 3)),
            n_samples_seen=6,
            batch_count=1,
            previous_lambda=0.5,
            n_features_in=2,
            fit_intercept=True,
        )
        X = np.zeros((4, 3))
        score = np.zeros(4)
        config = EstimatorConfig(penalty="l1", fit_intercept=True)

        actual = _gradient_from_score(X, score, state.coefficients.copy(), state, config, backend)

        # Paper equation (3.9): -(N_previous / N_total) * lambda_previous
        # times sign(beta_previous), with the intercept excluded from L1.
        expected = np.asarray([-0.3, 0.3, 0.0])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    def test_two_batch_solution_satisfies_paper_equation_2_8(self) -> None:
        rng = np.random.default_rng(2028)
        X = rng.normal(size=(140, 3))
        y = X @ np.asarray([1.1, -0.6, 0.35]) + 0.2
        y += rng.normal(scale=0.12, size=X.shape[0])
        y[[8, 37, 101]] += np.asarray([8.0, -7.0, 9.0])
        first = (X[:80], y[:80])
        second = (X[80:], y[80:])

        model = RenewableHuberRegressor(tau=1.2, ridge=0.0, max_iter=300, tol=1e-10)
        model.partial_fit(*first)
        previous = model.state_
        model.partial_fit(*second)
        self.assertTrue(model.diagnostics_.converged)

        design = np.column_stack((second[0], np.ones(second[0].shape[0])))
        coefficients = self._parameters(model)
        current_score = huber_score(second[1] - design @ coefficients, model.tau, np)
        equation_residual = (
            previous.information @ (coefficients - previous.coefficients) - design.T @ current_score
        )

        np.testing.assert_allclose(equation_residual, 0.0, rtol=0.0, atol=2e-8)

    def test_two_batch_l1_solution_satisfies_paper_equation_3_6_kkt_conditions(self) -> None:
        rng = np.random.default_rng(2036)
        X = rng.normal(size=(180, 5))
        y = X @ np.asarray([1.4, -0.9, 0.0, 0.45, 0.0]) + 0.3
        y += rng.normal(scale=0.1, size=X.shape[0])
        first = (X[:80], y[:80])
        second = (X[80:], y[80:])

        model = RenewableHuberRegressor(penalty="l1", lambda_scale=0.6, max_iter=300, tol=1e-6)
        model.partial_fit(*first)
        previous = model.state_
        model.partial_fit(*second)
        self.assertTrue(model.diagnostics_.converged)

        design = np.column_stack((second[0], np.ones(second[0].shape[0])))
        coefficients = self._parameters(model)
        current_score = huber_score(second[1] - design @ coefficients, model.tau, np)
        n_total = previous.n_samples_seen + second[0].shape[0]
        penalty_mask = np.ones_like(coefficients)
        penalty_mask[-1] = 0.0
        smooth_gradient = (
            -design.T @ current_score
            + previous.information @ (coefficients - previous.coefficients)
        ) / n_total
        smooth_gradient -= (
            previous.n_samples_seen
            / n_total
            * previous.previous_lambda
            * np.sign(previous.coefficients)
            * penalty_mask
        )

        active = (np.abs(coefficients) > 1e-7) & (penalty_mask == 1.0)
        inactive = ~active & (penalty_mask == 1.0)
        stationarity = smooth_gradient + (
            model.diagnostics_.lambda_value * np.sign(coefficients) * penalty_mask
        )
        np.testing.assert_allclose(stationarity[active], 0.0, rtol=0.0, atol=2e-5)
        self.assertTrue(
            np.all(np.abs(smooth_gradient[inactive]) <= model.diagnostics_.lambda_value + 2e-5)
        )
        self.assertLess(abs(smooth_gradient[-1]), 2e-5)

    def test_structurally_valid_checkpoint_rejects_each_nonfinite_state_array(self) -> None:
        X = np.arange(12.0).reshape(6, 2)
        y = X @ np.asarray([0.5, -0.2]) + 1.0
        model = RenewableHuberRegressor().fit(X, y)

        with tempfile.TemporaryDirectory() as directory:
            for corrupted_array in ("coefficients", "information"):
                with self.subTest(corrupted_array=corrupted_array):
                    checkpoint = Path(directory) / f"corrupted-{corrupted_array}.npz"
                    model.save(checkpoint)
                    with np.load(checkpoint, allow_pickle=False) as archive:
                        coefficients = np.asarray(archive["coefficients"]).copy()
                        information = np.asarray(archive["information"]).copy()
                        metadata = np.asarray(archive["metadata"]).copy()
                    if corrupted_array == "coefficients":
                        coefficients[0] = np.nan
                    else:
                        information[0, 0] = np.inf
                    with checkpoint.open("wb") as file_handle:
                        np.savez_compressed(
                            file_handle,
                            coefficients=coefficients,
                            information=information,
                            metadata=metadata,
                        )

                    with self.assertRaises(ValidationError):
                        RenewableHuberRegressor.load(checkpoint)

    def test_checkpoint_rejects_nonfinite_configuration_during_load(self) -> None:
        X = np.arange(12.0).reshape(6, 2)
        y = X @ np.asarray([0.5, -0.2]) + 1.0
        model = RenewableHuberRegressor().fit(X, y)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "corrupted-config.npz"
            model.save(checkpoint)
            with np.load(checkpoint, allow_pickle=False) as archive:
                coefficients = np.asarray(archive["coefficients"]).copy()
                information = np.asarray(archive["information"]).copy()
                metadata = json.loads(str(archive["metadata"].item()))
            metadata["config"]["tau"] = float("nan")
            with checkpoint.open("wb") as file_handle:
                np.savez_compressed(
                    file_handle,
                    coefficients=coefficients,
                    information=information,
                    metadata=np.asarray(json.dumps(metadata)),
                )

            with self.assertRaises(ValidationError):
                RenewableHuberRegressor.load(checkpoint)

    def test_truncated_checkpoint_raises_the_public_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "truncated.npz"
            checkpoint.write_bytes(b"not a NumPy archive")

            with self.assertRaisesRegex(ValidationError, "corrupted"):
                RenewableHuberRegressor.load(checkpoint)

    def test_nan_and_infinity_are_rejected_in_features_and_target(self) -> None:
        X = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        y = np.asarray([1.0, 2.0, 3.0])
        cases = []
        for bad_value in (np.nan, np.inf, -np.inf):
            bad_X = X.copy()
            bad_X[1, 0] = bad_value
            cases.append((f"X={bad_value}", bad_X, y))
            bad_y = y.copy()
            bad_y[1] = bad_value
            cases.append((f"y={bad_value}", X, bad_y))

        for label, features, target in cases:
            with self.subTest(label=label), self.assertRaises(ValidationError):
                RenewableHuberRegressor().fit(features, target)

    def test_nonfinite_numeric_configuration_is_rejected(self) -> None:
        X = np.asarray([[0.0], [1.0]])
        y = np.asarray([0.0, 1.0])
        names = ("tau", "lambda_scale", "bandwidth_scale", "tol", "ridge")

        for name in names:
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(name=name, value=value), self.assertRaises(ValidationError):
                    RenewableHuberRegressor(**{name: value}).fit(X, y)

    def test_invalid_configuration_types_raise_the_public_validation_error(self) -> None:
        X = np.asarray([[0.0], [1.0]])
        y = np.asarray([0.0, 1.0])
        cases = {
            "penalty": [],
            "max_iter": True,
            "fit_intercept": 1,
            "backend": [],
            "device": [],
            "dtype": [],
        }

        for name, value in cases.items():
            with self.subTest(name=name), self.assertRaises(ValidationError):
                RenewableHuberRegressor(**{name: value}).fit(X, y)

    def test_empty_batches_are_rejected_without_mutating_an_existing_stream(self) -> None:
        empty_cases = [
            (np.empty((0, 2)), np.empty((0,))),
            (np.empty((1, 0)), np.asarray([1.0])),
            (np.ones((1, 2)), np.empty((0,))),
        ]
        for features, target in empty_cases:
            with self.subTest(shape=features.shape), self.assertRaises(ValidationError):
                RenewableHuberRegressor().fit(features, target)

        model = RenewableHuberRegressor().fit(
            np.asarray([[0.0, 1.0], [1.0, 2.0]]), np.asarray([1.0, 2.0])
        )
        before = model.state_
        with self.assertRaises(ValidationError):
            model.partial_fit(np.empty((0, 2)), np.empty((0,)))
        after = model.state_
        np.testing.assert_array_equal(after.coefficients, before.coefficients)
        np.testing.assert_array_equal(after.information, before.information)
        self.assertEqual(after.n_samples_seen, before.n_samples_seen)
        self.assertEqual(after.batch_count, before.batch_count)

    def test_row_order_within_one_batch_is_permutation_invariant(self) -> None:
        X, y = self._order_sensitive_data()
        permutation = np.random.default_rng(99).permutation(X.shape[0])

        original = RenewableHuberRegressor(max_iter=200, tol=1e-10).fit(X, y)
        shuffled = RenewableHuberRegressor(max_iter=200, tol=1e-10).fit(
            X[permutation], y[permutation]
        )

        np.testing.assert_allclose(
            self._parameters(shuffled), self._parameters(original), rtol=0.0, atol=1e-9
        )
        np.testing.assert_allclose(
            shuffled.state_.information,
            original.state_.information,
            rtol=2e-10,
            atol=2e-10,
        )

    def test_batch_boundaries_and_arrival_order_are_semantically_significant(self) -> None:
        X, y = self._order_sensitive_data()
        first = (X[:20], y[:20])
        second = (X[20:], y[20:])

        forward = RenewableHuberRegressor(max_iter=200, tol=1e-10)
        forward.partial_fit(*first)
        forward.partial_fit(*second)
        reverse = RenewableHuberRegressor(max_iter=200, tol=1e-10)
        reverse.partial_fit(*second)
        reverse.partial_fit(*first)
        pooled = RenewableHuberRegressor(max_iter=200, tol=1e-10).fit(X, y)

        self.assertGreater(
            np.linalg.norm(self._parameters(forward) - self._parameters(reverse)), 0.1
        )
        self.assertGreater(
            np.linalg.norm(self._parameters(forward) - self._parameters(pooled)), 0.1
        )
        self.assertEqual(forward.state_.batch_count, 2)
        self.assertEqual(reverse.state_.batch_count, 2)
        self.assertEqual(pooled.state_.batch_count, 1)

    @staticmethod
    def _order_sensitive_data() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        first = rng.normal(loc=-1.0, size=(20, 2))
        second = rng.normal(loc=1.0, size=(80, 2))
        y_first = first @ np.asarray([3.0, -2.0]) + 1.0
        y_first += rng.normal(scale=0.2, size=first.shape[0])
        y_second = second @ np.asarray([-1.0, 4.0]) - 3.0
        y_second += rng.normal(scale=0.2, size=second.shape[0])
        return np.vstack((first, second)), np.concatenate((y_first, y_second))


if __name__ == "__main__":
    unittest.main()
