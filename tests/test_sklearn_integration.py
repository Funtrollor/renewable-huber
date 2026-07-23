from __future__ import annotations

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np


def _sklearn_ready() -> bool:
    return find_spec("sklearn") is not None


@unittest.skipUnless(_sklearn_ready(), "scikit-learn is required")
class SklearnIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from sklearn.base import clone, is_regressor
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from renewable_huber.integrations.sklearn import SklearnRenewableHuberRegressor

        cls.clone = staticmethod(clone)
        cls.is_regressor = staticmethod(is_regressor)
        cls.GridSearchCV = GridSearchCV
        cls.Pipeline = Pipeline
        cls.StandardScaler = StandardScaler
        cls.Regressor = SklearnRenewableHuberRegressor
        rng = np.random.default_rng(49)
        cls.X = rng.normal(size=(180, 4))
        cls.y = cls.X @ np.asarray([1.4, -0.8, 0.3, 0.0]) + 0.2

    def test_clone_is_an_unfitted_regressor(self) -> None:
        model = self.Regressor(tau=1.1, max_iter=80)
        cloned = self.clone(model)

        self.assertTrue(self.is_regressor(cloned))
        self.assertEqual(cloned.get_params(deep=False), model.get_params(deep=False))
        self.assertFalse(hasattr(cloned, "coef_"))

    def test_pipeline_and_grid_search(self) -> None:
        pipeline = self.Pipeline(
            [("scale", self.StandardScaler()), ("regressor", self.Regressor(max_iter=80))]
        )
        search = self.GridSearchCV(
            pipeline,
            param_grid={"regressor__tau": [1.0, 1.345]},
            cv=3,
            n_jobs=1,
        )
        search.fit(self.X, self.y)

        self.assertGreater(search.best_score_, 0.97)
        self.assertEqual(search.predict(self.X).shape, (self.X.shape[0],))

    def test_full_estimator_contract(self) -> None:
        from sklearn.utils.estimator_checks import check_estimator

        check_estimator(self.Regressor(max_iter=40))

    def test_checkpoint_preserves_adapter_type(self) -> None:
        model = self.Regressor(max_iter=80).fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sklearn-adapter.npz"
            model.save(checkpoint)
            restored = self.Regressor.load(checkpoint)

        self.assertIsInstance(restored, self.Regressor)
        np.testing.assert_allclose(restored.predict(self.X), model.predict(self.X))


if __name__ == "__main__":
    unittest.main()
