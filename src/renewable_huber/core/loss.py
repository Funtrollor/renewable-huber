"""Huber loss and its smoothed score used by renewable updates."""

from __future__ import annotations

from typing import Any

import numpy as np


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

    if xp is np:
        return _numpy_smoothed_score(residual, tau, bandwidth)

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

    if xp is np:
        return _numpy_smoothed_curvature(residual, tau, bandwidth)

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


def smoothed_score_and_curvature(
    residual: Any, tau: float, bandwidth: float, xp: Any
) -> tuple[Any, Any]:
    """Return both smoothed terms, sharing the NumPy fast path when available."""

    if xp is np:
        return _numpy_smoothed_score_and_curvature(residual, tau, bandwidth)
    return (
        smoothed_score(residual, tau, bandwidth, xp),
        smoothed_curvature(residual, tau, bandwidth, xp),
    )


def _numpy_smoothed_score(residual: np.ndarray, tau: float, bandwidth: float) -> np.ndarray:
    """Compute the piecewise score without nested temporary ``where`` arrays."""

    h = min(bandwidth, tau * 0.5)
    score = np.clip(residual, -tau, tau)
    left = (residual >= -tau - h) & (residual <= -tau + h)
    right = (residual >= tau - h) & (residual <= tau + h)
    score[left] = 0.5 * (residual[left] - tau + h)
    score[right] = 0.5 * (residual[right] + tau - h)
    return score


def _numpy_smoothed_curvature(residual: np.ndarray, tau: float, bandwidth: float) -> np.ndarray:
    """Compute curvature in place with one mask at a time."""

    h = min(bandwidth, tau * 0.5)
    curvature = np.zeros_like(residual)
    center = (residual > -tau + h) & (residual < tau - h)
    curvature[center] = 1.0
    left = (residual >= -tau - h) & (residual <= -tau + h)
    right = (residual >= tau - h) & (residual <= tau + h)
    curvature[left] = 0.5
    curvature[right] = 0.5
    return curvature


def _numpy_smoothed_score_and_curvature(
    residual: np.ndarray, tau: float, bandwidth: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute both NumPy terms while sharing the transition masks."""

    h = min(bandwidth, tau * 0.5)
    score = np.clip(residual, -tau, tau)
    curvature = np.zeros_like(residual)
    center = (residual > -tau + h) & (residual < tau - h)
    curvature[center] = 1.0
    left = (residual >= -tau - h) & (residual <= -tau + h)
    right = (residual >= tau - h) & (residual <= tau + h)
    score[left] = 0.5 * (residual[left] - tau + h)
    score[right] = 0.5 * (residual[right] + tau - h)
    curvature[left] = 0.5
    curvature[right] = 0.5
    return score, curvature


def soft_threshold(values: Any, threshold: Any, xp: Any) -> Any:
    """Apply elementwise soft-thresholding for L1-penalised updates."""

    return xp.sign(values) * xp.maximum(xp.abs(values) - threshold, xp.zeros_like(values))
