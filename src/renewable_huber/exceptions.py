"""Package-specific exceptions."""


class RenewableHuberError(Exception):
    """Base exception for renewable_huber."""


class NotFittedError(RenewableHuberError):
    """Raised when an operation requires a fitted estimator."""


class BackendUnavailableError(RenewableHuberError):
    """Raised when a requested numerical backend is unavailable."""


class ValidationError(RenewableHuberError, ValueError):
    """Raised when an input or configuration violates the public contract."""
