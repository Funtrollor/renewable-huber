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


def huber_score(residual: Any, tau: float, xp: Any) -> Any:
    """Return the ordinary Huber score used for the arriving batch."""

    if xp is np:
        return np.clip(residual, -tau, tau)
    absolute = xp.abs(residual)
    return xp.where(absolute <= tau, residual, tau * xp.sign(residual))


def smoothed_score(residual: Any, tau: float, bandwidth: float, xp: Any) -> Any:
    """Smooth the Huber score over a bandwidth around ``±tau``.

    The quadratic transition from Jiang, Liang, and Yu (2024), equation (2.1),
    is continuous with the ordinary score and has continuous bounded curvature.
    """

    if xp is np:
        return _numpy_smoothed_score(residual, tau, bandwidth)

    h = min(bandwidth, tau)
    negative_offset = residual + tau
    positive_offset = tau - residual
    left = h / 4.0 - tau + 0.5 * negative_offset + negative_offset**2 / (4.0 * h)
    right = -(h / 4.0 - tau + 0.5 * positive_offset + positive_offset**2 / (4.0 * h))
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

    h = min(bandwidth, tau)
    left = 0.5 + (residual + tau) / (2.0 * h)
    right = 0.5 - (residual - tau) / (2.0 * h)
    return xp.where(
        residual < -tau - h,
        xp.zeros_like(residual),
        xp.where(
            residual <= -tau + h,
            left,
            xp.where(
                residual < tau - h,
                xp.ones_like(residual),
                xp.where(residual <= tau + h, right, xp.zeros_like(residual)),
            ),
        ),
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
    """Compute the paper's quadratic transition score without nested ``where`` arrays."""

    h = min(bandwidth, tau)
    score = np.clip(residual, -tau, tau)
    left = (residual >= -tau - h) & (residual <= -tau + h)
    right = (residual >= tau - h) & (residual <= tau + h)
    negative_offset = residual[left] + tau
    positive_offset = tau - residual[right]
    score[left] = h / 4.0 - tau + 0.5 * negative_offset + negative_offset**2 / (4.0 * h)
    score[right] = -(h / 4.0 - tau + 0.5 * positive_offset + positive_offset**2 / (4.0 * h))
    return score


def _numpy_smoothed_curvature(residual: np.ndarray, tau: float, bandwidth: float) -> np.ndarray:
    """Compute the continuous transition curvature in place one mask at a time."""

    h = min(bandwidth, tau)
    curvature = np.zeros_like(residual)
    center = (residual > -tau + h) & (residual < tau - h)
    curvature[center] = 1.0
    left = (residual >= -tau - h) & (residual <= -tau + h)
    right = (residual >= tau - h) & (residual <= tau + h)
    curvature[left] = 0.5 + (residual[left] + tau) / (2.0 * h)
    curvature[right] = 0.5 - (residual[right] - tau) / (2.0 * h)
    return curvature


def _numpy_smoothed_score_and_curvature(
    residual: np.ndarray, tau: float, bandwidth: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute both NumPy terms while sharing the transition masks."""

    h = min(bandwidth, tau)
    score = np.clip(residual, -tau, tau)
    curvature = np.zeros_like(residual)
    center = (residual > -tau + h) & (residual < tau - h)
    curvature[center] = 1.0
    left = (residual >= -tau - h) & (residual <= -tau + h)
    right = (residual >= tau - h) & (residual <= tau + h)
    negative_offset = residual[left] + tau
    positive_offset = tau - residual[right]
    score[left] = h / 4.0 - tau + 0.5 * negative_offset + negative_offset**2 / (4.0 * h)
    score[right] = -(h / 4.0 - tau + 0.5 * positive_offset + positive_offset**2 / (4.0 * h))
    curvature[left] = 0.5 + negative_offset / (2.0 * h)
    curvature[right] = 0.5 + positive_offset / (2.0 * h)
    return score, curvature


def soft_threshold(values: Any, threshold: Any, xp: Any) -> Any:
    """Apply elementwise soft-thresholding for L1-penalised updates."""

    return xp.sign(values) * xp.maximum(xp.abs(values) - threshold, xp.zeros_like(values))
