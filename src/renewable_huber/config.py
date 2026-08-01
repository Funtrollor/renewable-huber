"""Validated configuration shared by the public estimator and numerical core."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Literal

from .exceptions import ValidationError

Penalty = Literal["none", "l1"]
BackendName = Literal[
    "auto",
    "numpy",
    "native_cpu",
    "cupy",
    "native_cuda",
    "torch",
    "tensorflow",
]
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
    n_jobs: int | None = None
    cuda_graphs: bool = False
    cuda_fast_math: bool = False

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
        if self.backend not in (
            "auto",
            "numpy",
            "native_cpu",
            "cupy",
            "native_cuda",
            "torch",
            "tensorflow",
        ):
            raise ValidationError("unsupported backend")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValidationError("device must be 'auto', 'cpu', or 'cuda'")
        if self.dtype not in ("float32", "float64"):
            raise ValidationError("dtype must be either 'float32' or 'float64'")
        if self.n_jobs is not None and (
            isinstance(self.n_jobs, bool)
            or not isinstance(self.n_jobs, Integral)
            or self.n_jobs == 0
            or self.n_jobs < -1
        ):
            raise ValidationError("n_jobs must be None, -1, or a positive integer")
        if not isinstance(self.cuda_graphs, bool):
            raise ValidationError("cuda_graphs must be a boolean")
        if not isinstance(self.cuda_fast_math, bool):
            raise ValidationError("cuda_fast_math must be a boolean")
        if self.cuda_fast_math and self.backend == "native_cuda" and self.dtype != "float32":
            raise ValidationError("cuda_fast_math requires dtype='float32'")

    def resolved_n_jobs(self) -> int | None:
        """Resolve the native CPU worker count while preserving ``None`` defaults."""

        if self.n_jobs is None:
            return None
        if self.n_jobs == -1:
            return max(os.cpu_count() or 1, 1)
        return int(self.n_jobs)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible configuration metadata."""

        return asdict(self)


def _is_finite_real(value: object) -> bool:
    """Reject booleans, non-numeric values, NaN, and infinities uniformly."""

    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(float(value))
