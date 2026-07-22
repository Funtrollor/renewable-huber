from __future__ import annotations

import unittest

import numpy as np

from renewable_huber.core.loss import smoothed_curvature, smoothed_score, soft_threshold


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


if __name__ == "__main__":
    unittest.main()
