"""Huber loss and its smoothed score used by renewable updates."""

from __future__ import annotations

from typing import Any


def huber_loss(residual: Any, tau: float, xp: Any) -> Any:
    """Return the elementwise classical Huber loss."""

    absolute = xp.abs(residual)
    return xp.where(
        absolute <= tau,
        0.5 * residual**2,
        tau * absolute - 0.5 * tau**2,
    )


def smoothed_score(residual: Any, tau: float, bandwidth: float, xp: Any) -> Any:
    """Smooth the Huber score over a bandwidth around ``±tau``.

    The piecewise-linear transition is continuous with the ordinary score and
    has a bounded second derivative, which is required for the renewable update.
    """

    h = min(bandwidth, tau * 0.5)
    left = 0.5 * (residual - tau + h)
    right = 0.5 * (residual + tau - h)
    return xp.where(
        residual < -tau - h,
        -tau,
        xp.where(
            residual <= -tau + h,
            left,
            xp.where(residual < tau - h, residual, xp.where(residual <= tau + h, right, tau)),
        ),
    )


def smoothed_curvature(residual: Any, tau: float, bandwidth: float, xp: Any) -> Any:
    """Return the derivative of :func:`smoothed_score`."""

    h = min(bandwidth, tau * 0.5)
    in_transition = ((residual >= -tau - h) & (residual <= -tau + h)) | (
        (residual >= tau - h) & (residual <= tau + h)
    )
    in_center = (residual > -tau + h) & (residual < tau - h)
    zeros = xp.zeros_like(residual)
    return xp.where(
        in_center,
        xp.ones_like(residual),
        xp.where(in_transition, 0.5 * xp.ones_like(residual), zeros),
    )


def soft_threshold(values: Any, threshold: Any, xp: Any) -> Any:
    """Apply elementwise soft-thresholding for L1-penalised updates."""

    return xp.sign(values) * xp.maximum(xp.abs(values) - threshold, 0.0)
