"""Renewable Huber regression for robust streaming-data estimation."""

from ._version import __version__
from .estimator import RenewableHuberRegressor
from .exceptions import (
    BackendUnavailableError,
    NotFittedError,
    RenewableHuberError,
    ValidationError,
)

__all__ = [
    "BackendUnavailableError",
    "NotFittedError",
    "RenewableHuberError",
    "RenewableHuberRegressor",
    "ValidationError",
    "__version__",
]
