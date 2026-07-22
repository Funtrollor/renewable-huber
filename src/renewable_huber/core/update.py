"""Backend-portable renewable Huber and renewable penalised Huber updates."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Any

from ..backends.protocol import ArrayBackend
from ..config import EstimatorConfig
from ..state import RenewableHuberState
from .loss import (
    huber_loss,
    smoothed_curvature,
    smoothed_score,
    smoothed_score_and_curvature,
    soft_threshold,
)


@dataclass(frozen=True, slots=True)
class UpdateDiagnostics:
    """Numerical outcome recorded for the most recent batch."""

    iterations: int
    converged: bool
    objective: float
    lambda_value: float
    bandwidth: float


def _bandwidth(n_total: int, n_predictors: int, scale: float) -> float:
    return scale / (sqrt(n_total) * log(max(n_predictors, 2)))


def _lambda(n_total: int, n_predictors: int, config: EstimatorConfig) -> float:
    if config.penalty == "none":
        return 0.0
    return config.lambda_scale * config.tau * sqrt(log(max(n_predictors, 2)) / n_total)


def _penalty_mask(reference: Any, fit_intercept: bool, xp: Any) -> Any:
    """Return an L1 mask without mutation, including immutable tensor backends."""

    if not fit_intercept:
        return xp.ones_like(reference)
    return xp.concatenate((xp.ones_like(reference[:-1]), xp.zeros_like(reference[-1:])))


def _weighted_gram(
    X: Any, curvature: Any, backend: ArrayBackend, *, workspace: Any | None = None
) -> Any:
    """Return ``X.T @ (X * curvature[:, None])`` with an optional work buffer.

    NumPy's BLAS call is already the efficient dense implementation.  In a
    Newton update it is invoked several times for the same batch, however, so
    reusing one batch-sized work buffer avoids repeatedly allocating and
    releasing ``X * curvature[:, None]``.  Other eager backends keep their
    original expression and therefore do not require mutable workspaces.
    """

    xp = backend.xp
    if workspace is not None:
        xp.multiply(X, curvature[:, None], out=workspace)
        return xp.matmul(xp.transpose(X), workspace)
    return xp.matmul(xp.transpose(X), X * curvature[:, None])


def _gradient_from_score(
    X: Any,
    score: Any,
    beta: Any,
    state: RenewableHuberState,
    backend: ArrayBackend,
) -> Any:
    """Evaluate the smooth surrogate gradient after the score is available."""

    xp = backend.xp
    n_batch = X.shape[0]
    delta = beta - state.coefficients
    transposed = xp.transpose(X)
    return -xp.matmul(transposed, score) / n_batch + xp.matmul(state.information, delta) / n_batch


def _gradient(
    X: Any,
    y: Any,
    beta: Any,
    state: RenewableHuberState,
    config: EstimatorConfig,
    bandwidth: float,
    backend: ArrayBackend,
) -> Any:
    """Evaluate only the gradient needed by the L1 proximal solver."""

    xp = backend.xp
    residual = y - xp.matmul(X, beta)
    score = smoothed_score(residual, config.tau, bandwidth, xp)
    return _gradient_from_score(X, score, beta, state, backend)


def _smooth_objective(
    X: Any,
    y: Any,
    beta: Any,
    state: RenewableHuberState,
    config: EstimatorConfig,
    bandwidth: float,
    backend: ArrayBackend,
) -> float:
    xp = backend.xp
    n_batch = X.shape[0]
    residual = y - xp.matmul(X, beta)
    current_loss = xp.mean(huber_loss(residual, config.tau, xp))
    delta = beta - state.coefficients
    historical_loss = 0.5 * xp.matmul(xp.matmul(delta, state.information), delta) / n_batch
    return backend.scalar(current_loss + historical_loss)


def _gradient_and_hessian(
    X: Any,
    y: Any,
    beta: Any,
    state: RenewableHuberState,
    config: EstimatorConfig,
    bandwidth: float,
    backend: ArrayBackend,
    *,
    workspace: Any | None = None,
) -> tuple[Any, Any]:
    xp = backend.xp
    n_batch, n_parameters = X.shape
    residual = y - xp.matmul(X, beta)
    score, curvature = smoothed_score_and_curvature(residual, config.tau, bandwidth, xp)
    gradient = _gradient_from_score(X, score, beta, state, backend)
    hessian = _weighted_gram(X, curvature, backend, workspace=workspace) + state.information
    hessian = hessian / n_batch
    hessian = hessian + config.ridge * xp.eye(n_parameters, dtype=beta.dtype)
    return gradient, hessian


def _solve_unpenalized(
    X: Any,
    y: Any,
    state: RenewableHuberState,
    config: EstimatorConfig,
    bandwidth: float,
    backend: ArrayBackend,
    *,
    workspace: Any | None = None,
) -> tuple[Any, int, bool, float]:
    beta = backend.copy(state.coefficients)
    objective = _smooth_objective(X, y, beta, state, config, bandwidth, backend)
    for iteration in range(1, config.max_iter + 1):
        gradient, hessian = _gradient_and_hessian(
            X, y, beta, state, config, bandwidth, backend, workspace=workspace
        )
        direction = backend.solve(hessian, gradient)
        step = 1.0
        while step >= 1e-8:
            candidate = beta - step * direction
            candidate_objective = _smooth_objective(
                X, y, candidate, state, config, bandwidth, backend
            )
            if candidate_objective <= objective:
                break
            step *= 0.5
        else:
            return beta, iteration, False, objective

        difference = candidate - beta
        beta = candidate
        objective = candidate_objective
        if backend.norm(difference) <= config.tol * (1.0 + backend.norm(beta)):
            return beta, iteration, True, objective
    return beta, config.max_iter, False, objective


def _solve_l1(
    X: Any,
    y: Any,
    state: RenewableHuberState,
    config: EstimatorConfig,
    bandwidth: float,
    lambda_value: float,
    backend: ArrayBackend,
) -> tuple[Any, int, bool, float]:
    """Solve the penalised surrogate with LAMM/proximal-gradient steps."""

    xp = backend.xp
    beta = backend.copy(state.coefficients)
    mask = _penalty_mask(beta, state.fit_intercept, xp)
    smooth_objective = _smooth_objective(X, y, beta, state, config, bandwidth, backend)
    phi = 1.0

    for iteration in range(1, config.max_iter + 1):
        gradient = _gradient(X, y, beta, state, config, bandwidth, backend)
        for _ in range(40):
            threshold = (lambda_value / phi) * mask
            candidate = soft_threshold(beta - gradient / phi, threshold, xp)
            difference = candidate - beta
            upper_bound = (
                smooth_objective
                + backend.scalar(xp.matmul(gradient, difference))
                + 0.5 * phi * backend.norm(difference) ** 2
            )
            candidate_smooth = _smooth_objective(X, y, candidate, state, config, bandwidth, backend)
            if candidate_smooth <= upper_bound + 1e-12:
                break
            phi *= 2.0
        else:
            return beta, iteration, False, smooth_objective

        beta = candidate
        smooth_objective = candidate_smooth
        phi = max(phi * 0.5, 1e-8)
        if backend.norm(difference) <= config.tol * (1.0 + backend.norm(beta)):
            return beta, iteration, True, smooth_objective
    return beta, config.max_iter, False, smooth_objective


def renewable_update(
    X: Any,
    y: Any,
    state: RenewableHuberState,
    config: EstimatorConfig,
    backend: ArrayBackend,
) -> tuple[RenewableHuberState, UpdateDiagnostics]:
    """Process exactly one data batch and return the next sufficient state."""

    state.validate()
    n_batch = X.shape[0]
    n_total = state.n_samples_seen + n_batch
    n_predictors = state.n_features_in
    bandwidth = _bandwidth(n_total, n_predictors, config.bandwidth_scale)
    lambda_value = _lambda(n_total, n_predictors, config)
    workspace = None
    if config.penalty == "none" and backend.name == "numpy":
        workspace = backend.xp.empty_like(X)

    if config.penalty == "l1":
        coefficients, iterations, converged, smooth_objective = _solve_l1(
            X, y, state, config, bandwidth, lambda_value, backend
        )
    else:
        coefficients, iterations, converged, smooth_objective = _solve_unpenalized(
            X, y, state, config, bandwidth, backend, workspace=workspace
        )

    xp = backend.xp
    residual = y - xp.matmul(X, coefficients)
    curvature = smoothed_curvature(residual, config.tau, bandwidth, xp)
    information = state.information + _weighted_gram(X, curvature, backend, workspace=workspace)
    new_state = RenewableHuberState(
        coefficients=backend.copy(coefficients),
        information=backend.copy(information),
        n_samples_seen=n_total,
        batch_count=state.batch_count + 1,
        previous_lambda=lambda_value,
        n_features_in=state.n_features_in,
        fit_intercept=state.fit_intercept,
    )
    diagnostics = UpdateDiagnostics(
        iterations=iterations,
        converged=converged,
        objective=_diagnostic_objective(
            smooth_objective, coefficients, state, config, lambda_value, backend
        ),
        lambda_value=lambda_value,
        bandwidth=bandwidth,
    )
    return new_state, diagnostics


def _diagnostic_objective(
    smooth_objective: float,
    coefficients: Any,
    state: RenewableHuberState,
    config: EstimatorConfig,
    lambda_value: float,
    backend: ArrayBackend,
) -> float:
    """Form diagnostics from the final smooth objective already evaluated by a solver."""

    if config.penalty != "l1":
        return smooth_objective
    xp = backend.xp
    mask = _penalty_mask(coefficients, state.fit_intercept, xp)
    return smooth_objective + lambda_value * backend.scalar(xp.sum(xp.abs(coefficients) * mask))
