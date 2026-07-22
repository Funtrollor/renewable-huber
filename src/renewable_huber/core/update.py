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
    huber_score,
    smoothed_curvature,
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


def _bandwidth(n_total: int, n_predictors: int, scale: float, tau: float) -> float:
    """Return the paper bandwidth, capped only where its transition regions meet."""

    return min(scale / (sqrt(n_total) * log(max(n_predictors, 2))), tau)


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
    config: EstimatorConfig,
    backend: ArrayBackend,
) -> Any:
    """Evaluate the smooth surrogate gradient after the score is available."""

    xp = backend.xp
    n_total = state.n_samples_seen + X.shape[0]
    delta = beta - state.coefficients
    transposed = xp.transpose(X)
    gradient = (-xp.matmul(transposed, score) + xp.matmul(state.information, delta)) / n_total
    if config.penalty == "l1" and state.n_samples_seen:
        mask = _penalty_mask(state.coefficients, state.fit_intercept, xp)
        historical_subgradient = xp.sign(state.coefficients) * mask
        gradient = gradient - (
            state.n_samples_seen / n_total * state.previous_lambda * historical_subgradient
        )
    return gradient


def _huber_score(residual: Any, tau: float, backend: ArrayBackend) -> Any:
    """Return the ordinary Huber score for the current batch."""

    return huber_score(residual, tau, backend.xp)


def _huber_loss(residual: Any, tau: float, backend: ArrayBackend) -> Any:
    """Use the CUDA C++ Huber-loss kernel when the active backend provides one."""

    accelerated_loss = getattr(backend, "cuda_huber_loss", None)
    if accelerated_loss is not None:
        loss = accelerated_loss(residual, tau)
        if loss is not None:
            return loss
    return huber_loss(residual, tau, backend.xp)


def _smoothed_curvature(
    residual: Any, config: EstimatorConfig, bandwidth: float, backend: ArrayBackend
) -> Any:
    """Use a fused CUDA C++ curvature kernel when the active backend provides one."""

    accelerated_curvature = getattr(backend, "cuda_smoothed_curvature", None)
    if accelerated_curvature is not None:
        curvature = accelerated_curvature(residual, config.tau, bandwidth)
        if curvature is not None:
            return curvature
    return smoothed_curvature(residual, config.tau, bandwidth, backend.xp)


def _huber_score_and_smoothed_curvature(
    residual: Any, config: EstimatorConfig, bandwidth: float, backend: ArrayBackend
) -> tuple[Any, Any]:
    """Evaluate the paper's current score and historical-information curvature."""

    accelerated_terms = getattr(backend, "cuda_huber_score_and_smoothed_curvature", None)
    if accelerated_terms is not None:
        terms = accelerated_terms(residual, config.tau, bandwidth)
        if terms is not None:
            return terms
    return (
        _huber_score(residual, config.tau, backend),
        smoothed_curvature(residual, config.tau, bandwidth, backend.xp),
    )


def _gradient(
    X: Any,
    y: Any,
    beta: Any,
    state: RenewableHuberState,
    config: EstimatorConfig,
    backend: ArrayBackend,
) -> Any:
    """Evaluate only the gradient needed by the L1 proximal solver."""

    xp = backend.xp
    residual = y - xp.matmul(X, beta)
    score = _huber_score(residual, config.tau, backend)
    return _gradient_from_score(X, score, beta, state, config, backend)


def _smooth_objective(
    X: Any,
    y: Any,
    beta: Any,
    state: RenewableHuberState,
    config: EstimatorConfig,
    backend: ArrayBackend,
) -> float:
    xp = backend.xp
    n_total = state.n_samples_seen + X.shape[0]
    residual = y - xp.matmul(X, beta)
    current_loss = xp.sum(_huber_loss(residual, config.tau, backend)) / n_total
    delta = beta - state.coefficients
    historical_loss = 0.5 * xp.matmul(xp.matmul(delta, state.information), delta) / n_total
    objective = current_loss + historical_loss
    if config.penalty == "l1" and state.n_samples_seen:
        mask = _penalty_mask(state.coefficients, state.fit_intercept, xp)
        historical_subgradient = xp.sign(state.coefficients) * mask
        objective = objective - (
            state.n_samples_seen
            / n_total
            * state.previous_lambda
            * xp.matmul(delta, historical_subgradient)
        )
    return backend.scalar(objective)


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
    n_total = state.n_samples_seen + n_batch
    residual = y - xp.matmul(X, beta)
    score, curvature = _huber_score_and_smoothed_curvature(residual, config, bandwidth, backend)
    gradient = _gradient_from_score(X, score, beta, state, config, backend)
    hessian = _weighted_gram(X, curvature, backend, workspace=workspace) + state.information
    hessian = hessian / n_total
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
    objective = _smooth_objective(X, y, beta, state, config, backend)
    for iteration in range(1, config.max_iter + 1):
        gradient, hessian = _gradient_and_hessian(
            X, y, beta, state, config, bandwidth, backend, workspace=workspace
        )
        direction = backend.solve(hessian, gradient)
        step = 1.0
        while step >= 1e-8:
            candidate = beta - step * direction
            candidate_objective = _smooth_objective(X, y, candidate, state, config, backend)
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
    smooth_objective = _smooth_objective(X, y, beta, state, config, backend)
    phi = 1.0

    for iteration in range(1, config.max_iter + 1):
        gradient = _gradient(X, y, beta, state, config, backend)
        for _ in range(40):
            threshold = (lambda_value / phi) * mask
            candidate = soft_threshold(beta - gradient / phi, threshold, xp)
            difference = candidate - beta
            upper_bound = (
                smooth_objective
                + backend.scalar(xp.matmul(gradient, difference))
                + 0.5 * phi * backend.norm(difference) ** 2
            )
            candidate_smooth = _smooth_objective(X, y, candidate, state, config, backend)
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
    bandwidth = _bandwidth(n_total, n_predictors, config.bandwidth_scale, config.tau)
    lambda_value = _lambda(n_total, n_predictors, config)
    workspace = None
    if config.penalty == "none" and backend.name in {"numpy", "cupy"}:
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
    curvature = _smoothed_curvature(residual, config, bandwidth, backend)
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
