"""scikit-learn adapter for :class:`renewable_huber.RenewableHuberRegressor`."""

from __future__ import annotations

try:
    from sklearn.base import BaseEstimator, RegressorMixin
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


__all__ = ["SklearnRenewableHuberRegressor"]
