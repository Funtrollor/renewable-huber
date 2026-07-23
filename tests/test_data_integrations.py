from __future__ import annotations

import unittest
from importlib.util import find_spec

import numpy as np

from renewable_huber import RenewableHuberRegressor, ValidationError


@unittest.skipUnless(find_spec("pandas") is not None, "pandas is required")
class PandasIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import pandas as pd

        cls.pd = pd
        rng = np.random.default_rng(2028)
        cls.X_array = rng.normal(size=(120, 3))
        cls.y_array = cls.X_array @ np.asarray([1.2, -0.7, 0.4]) + 0.3
        cls.columns = ["wind_speed", "irradiance", "temperature"]

    def test_dataframe_series_and_weight_series_match_numpy(self) -> None:
        frame = self.pd.DataFrame(self.X_array, columns=self.columns)
        target = self.pd.Series(self.y_array, name="power")
        weights = self.pd.Series(np.linspace(0.5, 1.5, len(frame)), name="weight")

        pandas_model = RenewableHuberRegressor().fit(frame, target, sample_weight=weights)
        numpy_model = RenewableHuberRegressor().fit(
            self.X_array, self.y_array, sample_weight=weights.to_numpy()
        )

        np.testing.assert_array_equal(pandas_model.feature_names_in_, self.columns)
        np.testing.assert_allclose(
            pandas_model.predict(frame),
            numpy_model.predict(self.X_array),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_dataframe_prediction_requires_matching_names_and_order(self) -> None:
        frame = self.pd.DataFrame(self.X_array, columns=self.columns)
        model = RenewableHuberRegressor().fit(frame, self.y_array)

        for invalid in (
            frame[self.columns[::-1]],
            frame.rename(columns={self.columns[0]: "renamed"}),
            frame.drop(columns=self.columns[-1]),
        ):
            with self.subTest(columns=list(invalid.columns)):
                with self.assertRaisesRegex(ValidationError, "feature names"):
                    model.predict(invalid)

    def test_partial_fit_rejects_reordered_dataframe_without_mutating_state(self) -> None:
        first = self.pd.DataFrame(self.X_array[:60], columns=self.columns)
        second = self.pd.DataFrame(self.X_array[60:], columns=self.columns)
        model = RenewableHuberRegressor().partial_fit(first, self.y_array[:60])
        before = model.state_

        with self.assertRaisesRegex(ValidationError, "feature names"):
            model.partial_fit(second[self.columns[::-1]], self.y_array[60:])

        np.testing.assert_array_equal(model.state_.coefficients, before.coefficients)
        np.testing.assert_array_equal(model.state_.information, before.information)
        self.assertEqual(model.state_.batch_count, before.batch_count)

    def test_non_string_dataframe_columns_remain_positional(self) -> None:
        frame = self.pd.DataFrame(self.X_array)
        model = RenewableHuberRegressor().fit(frame, self.y_array)

        self.assertFalse(hasattr(model, "feature_names_in_"))
        self.assertEqual(model.predict(frame).shape, (len(frame),))


@unittest.skipUnless(find_spec("scipy") is not None, "SciPy is required")
class SparseInputContractTests(unittest.TestCase):
    def test_scipy_sparse_input_is_rejected_without_implicit_densification(self) -> None:
        from scipy import sparse

        X = sparse.csr_matrix(np.eye(5))
        y = np.arange(5.0)

        with self.assertRaisesRegex(TypeError, "Sparse data was passed"):
            RenewableHuberRegressor().fit(X, y)


if __name__ == "__main__":
    unittest.main()
