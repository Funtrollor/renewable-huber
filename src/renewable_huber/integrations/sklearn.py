"""scikit-learn adapter for :class:`renewable_huber.RenewableHuberRegressor`."""

from __future__ import annotations

try:
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.utils.validation import check_is_fitted, column_or_1d
except ImportError as error:  # pragma: no cover - depends on an optional package
    raise ImportError(
        "scikit-learn integration requires scikit-learn. "
        "Install it with: pip install 'renewable-huber[sklearn]'"
    ) from error

from ..estimator import RenewableHuberRegressor


class SklearnRenewableHuberRegressor(RegressorMixin, RenewableHuberRegressor, BaseEstimator):
    """Use renewable Huber regression in scikit-learn meta-estimators.

    The numerical API is inherited unchanged.  The adapter adds scikit-learn's
    regressor tags and base-estimator behaviour so that ``clone``, ``Pipeline``,
    ``GridSearchCV``, and cross-validation treat it as a regressor.
    """

    def fit(self, X, y, sample_weight=None):
        """Fit after applying scikit-learn's one-dimensional target contract."""

        return super().fit(
            X,
            column_or_1d(y, warn=True),
            sample_weight=sample_weight,
        )

    def partial_fit(self, X, y, sample_weight=None):
        """Update after applying scikit-learn's target-shape contract."""

        return super().partial_fit(
            X,
            column_or_1d(y, warn=True),
            sample_weight=sample_weight,
        )

    def predict(self, X):
        """Predict after applying scikit-learn's fitted-state contract."""

        check_is_fitted(self)
        return super().predict(X)


__all__ = ["SklearnRenewableHuberRegressor"]
