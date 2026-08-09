"""Package-specific exceptions."""


class RenewableHuberError(Exception):
    """Base exception for renewable_huber."""


class NotFittedError(RenewableHuberError):
    """Raised when an operation requires a fitted estimator."""


class BackendUnavailableError(RenewableHuberError):
    """Raised when a requested numerical backend is unavailable."""


class ValidationError(RenewableHuberError, ValueError):
    """Raised when an input or configuration violates the public contract."""


class BackendContractError(RenewableHuberError, TypeError):
    """Raised when input violates a backend's own transport contract.

    It inherits :class:`TypeError`, so every existing handler keeps working,
    while being a distinct type lets the estimator tell "this array cannot be
    coerced to a number" apart from "this backend refuses this array, and here
    is exactly why". Only the first deserves the scikit-learn-compatible
    coercion message; the second already says something more useful and must
    reach the caller intact rather than being rewritten.
    """
