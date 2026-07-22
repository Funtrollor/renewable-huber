from __future__ import annotations

import unittest

import numpy as np

from renewable_huber.core.loss import (
    smoothed_curvature,
    smoothed_score,
    smoothed_score_and_curvature,
    soft_threshold,
)


class _NumPyFacade:
    """Route calls through the generic array-namespace implementation."""

    def __getattr__(self, name: str):
        return getattr(np, name)


class LossTests(unittest.TestCase):
    def test_smoothed_score_is_continuous_at_transition_boundaries(self) -> None:
        tau = 1.345
        bandwidth = 0.1
        epsilon = 1e-10
        boundaries = [-tau - bandwidth, -tau + bandwidth, tau - bandwidth, tau + bandwidth]
        for boundary in boundaries:
            left = smoothed_score(np.asarray([boundary - epsilon]), tau, bandwidth, np)[0]
            right = smoothed_score(np.asarray([boundary + epsilon]), tau, bandwidth, np)[0]
            self.assertAlmostEqual(left, right, places=7)

    def test_curvature_and_soft_threshold(self) -> None:
        curvature = smoothed_curvature(np.asarray([0.0, 10.0]), 1.0, 0.1, np)
        np.testing.assert_allclose(curvature, [1.0, 0.0])
        values = soft_threshold(np.asarray([-3.0, 0.5, 4.0]), 1.0, np)
        np.testing.assert_allclose(values, [-2.0, 0.0, 3.0])

    def test_numpy_fast_path_matches_the_generic_terms(self) -> None:
        tau = 1.345
        bandwidth = 0.1
        generic = _NumPyFacade()
        rng = np.random.default_rng(23)
        boundaries = np.asarray(
            [-2.0, -tau - bandwidth, -tau + bandwidth, -0.1, tau - bandwidth, tau + bandwidth, 2.0]
        )
        for dtype in (np.float32, np.float64):
            residual = np.concatenate((rng.normal(size=2_048), boundaries)).astype(dtype)

            expected_score = smoothed_score(residual, tau, bandwidth, generic)
            expected_curvature = smoothed_curvature(residual, tau, bandwidth, generic)
            actual_score, actual_curvature = smoothed_score_and_curvature(
                residual, tau, bandwidth, np
            )

            np.testing.assert_array_equal(actual_score, expected_score)
            np.testing.assert_array_equal(actual_curvature, expected_curvature)


if __name__ == "__main__":
    unittest.main()
