"""Validated configuration shared by the public estimator and numerical core."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Literal

from .exceptions import ValidationError

Penalty = Literal["none", "l1"]
BackendName = Literal["auto", "numpy", "cupy", "torch", "tensorflow"]
DeviceName = Literal["auto", "cpu", "cuda"]
DTypeName = Literal["float32", "float64"]


@dataclass(frozen=True, slots=True)
class EstimatorConfig:
    """Numerical settings for :class:`RenewableHuberRegressor`.

    Backends share the same numerical contract, allowing CPU and GPU execution
    without changing estimator construction.
    """

    tau: float = 1.345
    penalty: Penalty = "none"
    lambda_scale: float = 1.0
    bandwidth_scale: float = 1.0
    fit_intercept: bool = True
    max_iter: int = 100
    tol: float = 1e-6
    ridge: float = 1e-8
    backend: BackendName = "auto"
    device: DeviceName = "auto"
    dtype: DTypeName = "float64"

    def validate(self) -> None:
        if not _is_finite_real(self.tau) or self.tau <= 0:
            raise ValidationError("tau must be greater than zero")
        if self.penalty not in ("none", "l1"):
            raise ValidationError("penalty must be either 'none' or 'l1'")
        if not _is_finite_real(self.lambda_scale) or self.lambda_scale < 0:
            raise ValidationError("lambda_scale must be non-negative")
        if not _is_finite_real(self.bandwidth_scale) or self.bandwidth_scale <= 0:
            raise ValidationError("bandwidth_scale must be greater than zero")
        if (
            isinstance(self.max_iter, bool)
            or not isinstance(self.max_iter, Integral)
            or self.max_iter < 1
        ):
            raise ValidationError("max_iter must be at least one")
        if not isinstance(self.fit_intercept, bool):
            raise ValidationError("fit_intercept must be a boolean")
        if not _is_finite_real(self.tol) or self.tol <= 0:
            raise ValidationError("tol must be greater than zero")
        if not _is_finite_real(self.ridge) or self.ridge < 0:
            raise ValidationError("ridge must be non-negative")
        if self.backend not in ("auto", "numpy", "cupy", "torch", "tensorflow"):
            raise ValidationError("unsupported backend")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValidationError("device must be 'auto', 'cpu', or 'cuda'")
        if self.dtype not in ("float32", "float64"):
            raise ValidationError("dtype must be either 'float32' or 'float64'")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible configuration metadata."""

        return asdict(self)


def _is_finite_real(value: object) -> bool:
    """Reject booleans, non-numeric values, NaN, and infinities uniformly."""

    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(float(value))
