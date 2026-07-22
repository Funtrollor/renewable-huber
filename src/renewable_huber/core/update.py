"""Backend-portable renewable Huber and renewable penalised Huber updates."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Any

from ..backends.protocol import ArrayBackend
from ..config import EstimatorConfig
from ..state import RenewableHuberState
from .loss import huber_loss, smoothed_curvature, smoothed_score, soft_threshold


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


def _penalty_mask(n_parameters: int, fit_intercept: bool, xp: Any, dtype: Any) -> Any:
    mask = xp.ones(n_parameters, dtype=dtype)
    if fit_intercept:
        mask[-1] = 0.0
    return mask


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
    residual = y - X @ beta
    current_loss = xp.mean(huber_loss(residual, config.tau, xp))
    delta = beta - state.coefficients
    historical_loss = 0.5 * (delta @ state.information @ delta) / n_batch
    return backend.scalar(current_loss + historical_loss)


def _full_objective(
    X: Any,
    y: Any,
    beta: Any,
    state: RenewableHuberState,
    config: EstimatorConfig,
    bandwidth: float,
    lambda_value: float,
    backend: ArrayBackend,
) -> float:
    xp = backend.xp
    result = _smooth_objective(X, y, beta, state, config, bandwidth, backend)
    if config.penalty == "l1":
        mask = _penalty_mask(beta.shape[0], state.fit_intercept, xp, beta.dtype)
        result += lambda_value * backend.scalar(xp.sum(xp.abs(beta) * mask))
    return result


def _gradient_and_hessian(
    X: Any,
    y: Any,
    beta: Any,
    state: RenewableHuberState,
    config: EstimatorConfig,
    bandwidth: float,
    backend: ArrayBackend,
) -> tuple[Any, Any]:
    xp = backend.xp
    n_batch, n_parameters = X.shape
    residual = y - X @ beta
    score = smoothed_score(residual, config.tau, bandwidth, xp)
    curvature = smoothed_curvature(residual, config.tau, bandwidth, xp)
    delta = beta - state.coefficients
    gradient = -(X.T @ score) / n_batch + (state.information @ delta) / n_batch
    hessian = (X.T @ (X * curvature[:, None]) + state.information) / n_batch
    hessian = hessian + config.ridge * xp.eye(n_parameters, dtype=beta.dtype)
    return gradient, hessian


def _solve_unpenalized(
    X: Any,
    y: Any,
    state: RenewableHuberState,
    config: EstimatorConfig,
    bandwidth: float,
    backend: ArrayBackend,
) -> tuple[Any, int, bool]:
    beta = backend.copy(state.coefficients)
    objective = _smooth_objective(X, y, beta, state, config, bandwidth, backend)
    for iteration in range(1, config.max_iter + 1):
        gradient, hessian = _gradient_and_hessian(X, y, beta, state, config, bandwidth, backend)
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
            return beta, iteration, False

        difference = candidate - beta
        beta = candidate
        objective = candidate_objective
        if backend.norm(difference) <= config.tol * (1.0 + backend.norm(beta)):
            return beta, iteration, True
    return beta, config.max_iter, False


def _solve_l1(
    X: Any,
    y: Any,
    state: RenewableHuberState,
    config: EstimatorConfig,
    bandwidth: float,
    lambda_value: float,
    backend: ArrayBackend,
) -> tuple[Any, int, bool]:
    """Solve the penalised surrogate with LAMM/proximal-gradient steps."""

    xp = backend.xp
    beta = backend.copy(state.coefficients)
    mask = _penalty_mask(beta.shape[0], state.fit_intercept, xp, beta.dtype)
    smooth_objective = _smooth_objective(X, y, beta, state, config, bandwidth, backend)
    phi = 1.0

    for iteration in range(1, config.max_iter + 1):
        gradient, _ = _gradient_and_hessian(X, y, beta, state, config, bandwidth, backend)
        for _ in range(40):
            threshold = (lambda_value / phi) * mask
            candidate = soft_threshold(beta - gradient / phi, threshold, xp)
            difference = candidate - beta
            upper_bound = (
                smooth_objective
                + backend.scalar(gradient @ difference)
                + 0.5 * phi * backend.norm(difference) ** 2
            )
            candidate_smooth = _smooth_objective(
                X, y, candidate, state, config, bandwidth, backend
            )
            if candidate_smooth <= upper_bound + 1e-12:
                break
            phi *= 2.0
        else:
            return beta, iteration, False

        beta = candidate
        smooth_objective = candidate_smooth
        phi = max(phi * 0.5, 1e-8)
        if backend.norm(difference) <= config.tol * (1.0 + backend.norm(beta)):
            return beta, iteration, True
    return beta, config.max_iter, False


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

    if config.penalty == "l1":
        coefficients, iterations, converged = _solve_l1(
            X, y, state, config, bandwidth, lambda_value, backend
        )
    else:
        coefficients, iterations, converged = _solve_unpenalized(
            X, y, state, config, bandwidth, backend
        )

    xp = backend.xp
    residual = y - X @ coefficients
    curvature = smoothed_curvature(residual, config.tau, bandwidth, xp)
    information = state.information + X.T @ (X * curvature[:, None])
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
        objective=_full_objective(
            X, y, coefficients, state, config, bandwidth, lambda_value, backend
        ),
        lambda_value=lambda_value,
        bandwidth=bandwidth,
    )
    return new_state, diagnostics
