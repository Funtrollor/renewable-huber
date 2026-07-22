"""Serializable state for renewable, batch-by-batch estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .exceptions import ValidationError


@dataclass(slots=True)
class RenewableHuberState:
    """Sufficient state retained after each processed batch.

    Raw historical observations are deliberately not retained.  The state is
    enough to resume a streaming run, save a fitted model, and make predictions.
    """

    coefficients: Any
    information: Any
    n_samples_seen: int
    batch_count: int
    previous_lambda: float
    n_features_in: int
    fit_intercept: bool

    @classmethod
    def empty(
        cls, n_features_in: int, *, fit_intercept: bool, xp: Any, dtype: Any
    ) -> RenewableHuberState:
        n_parameters = n_features_in + int(fit_intercept)
        return cls(
            coefficients=xp.zeros(n_parameters, dtype=dtype),
            information=xp.zeros((n_parameters, n_parameters), dtype=dtype),
            n_samples_seen=0,
            batch_count=0,
            previous_lambda=0.0,
            n_features_in=n_features_in,
            fit_intercept=fit_intercept,
        )

    def copy(self) -> RenewableHuberState:
        """Return an independent copy suitable for inspection by callers."""

        return RenewableHuberState(
            coefficients=_copy_array(self.coefficients),
            information=_copy_array(self.information),
            n_samples_seen=self.n_samples_seen,
            batch_count=self.batch_count,
            previous_lambda=self.previous_lambda,
            n_features_in=self.n_features_in,
            fit_intercept=self.fit_intercept,
        )

    def validate(self) -> None:
        """Check the invariant required for safe continuation of a stream."""

        n_parameters = self.n_features_in + int(self.fit_intercept)
        if self.coefficients.shape != (n_parameters,):
            raise ValidationError("state coefficient shape does not match feature metadata")
        if self.information.shape != (n_parameters, n_parameters):
            raise ValidationError("state information shape does not match feature metadata")
        if self.n_samples_seen < 0 or self.batch_count < 0:
            raise ValidationError("state counters must be non-negative")


def _copy_array(value: Any) -> Any:
    """Copy NumPy/CuPy-style arrays and PyTorch tensors without importing either."""

    if clone := getattr(value, "clone", None):
        return clone()
    return value.copy()
