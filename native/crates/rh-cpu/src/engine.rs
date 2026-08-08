//! The renewable update itself: validation, solver loop, state transition.
//!
//! The engine owns a `Workspace` and a `DenseSolver`; the numerical work lives
//! in `crate::kernels`. Generic parameters stay monomorphized over the concrete
//! scalar and solver, so nothing here costs a virtual call.

use std::marker::PhantomData;

use rayon::prelude::*;
use rh_core::{
    scalar_from_f64, BatchView, CoreError, Diagnostics, Penalty, State, Transition, UpdateConfig,
};

use crate::kernels::gram::{gradient_and_hessian, gradient_from_current_residual, weighted_gram};
use crate::kernels::objective::{diagnostic_objective, smooth_objective};
use crate::kernels::vector::{dot, norm, residual, smoothed_curvature, soft_threshold};
use crate::kernels::{bandwidth, lambda_value};
use crate::scalar::CpuScalar;
use crate::solver::{DenseSolver, NalgebraSolver};
use crate::workspace::Workspace;
use crate::PARALLEL_VECTOR_WORK;

/// Persistent CPU engine. State remains externally portable while temporary
/// allocations are retained here across `partial_fit` calls.
#[derive(Clone, Debug)]
pub struct CpuEngine<T: CpuScalar, S: DenseSolver<T> = NalgebraSolver> {
    solver: S,
    workspace: Workspace<T>,
    marker: PhantomData<T>,
}

impl<T: CpuScalar> Default for CpuEngine<T, NalgebraSolver> {
    fn default() -> Self {
        Self::new(NalgebraSolver)
    }
}

impl<T: CpuScalar, S: DenseSolver<T>> CpuEngine<T, S> {
    pub fn new(solver: S) -> Self {
        Self {
            solver,
            workspace: Workspace::default(),
            marker: PhantomData,
        }
    }

    /// Process exactly one batch and return the complete next state.
    pub fn update(
        &mut self,
        batch: BatchView<'_, T>,
        state: &State<T>,
        config: UpdateConfig,
    ) -> Result<Transition<T>, CoreError> {
        state.validate()?;
        config.validate()?;
        batch.validate(state)?;
        if batch
            .sample_weight
            .is_some_and(|weights| weights.iter().all(|value| *value == T::zero()))
        {
            return Err(CoreError::InvalidBatch(
                "sample_weight must contain a positive value",
            ));
        }

        let n_parameters = batch.n_parameters;
        self.workspace.reserve(batch.n_rows, n_parameters)?;
        let n_samples_seen = state
            .n_samples_seen
            .checked_add(batch.n_rows)
            .ok_or(CoreError::SizeOverflow)?;
        let batch_count = state
            .batch_count
            .checked_add(1)
            .ok_or(CoreError::SizeOverflow)?;
        let n_total = state.weight_sum + batch.batch_weight;
        if !n_total.is_finite() || n_total <= 0.0 {
            return Err(CoreError::InvalidBatch(
                "combined effective weight must be finite and positive",
            ));
        }
        let bandwidth = bandwidth(
            n_total,
            state.n_features_in,
            config.bandwidth_scale,
            config.tau,
        );
        let lambda_value = lambda_value(n_total, state.n_features_in, config);

        let solution = match config.penalty {
            Penalty::None => self.solve_unpenalized(batch, state, config, bandwidth, n_total)?,
            Penalty::L1 => self.solve_l1(batch, state, config, lambda_value, n_total)?,
        };

        // Every successful solver exit leaves `workspace.residual` evaluated
        // at the returned coefficients. Reuse it for final information rather
        // than issuing another full X @ beta pass per batch.
        smoothed_curvature(
            &self.workspace.residual,
            config.tau,
            bandwidth,
            &mut self.workspace.curvature,
        )?;
        weighted_gram(
            batch,
            &self.workspace.curvature,
            &mut self.workspace.weighted_design,
            &mut self.workspace.partial_grams,
            &mut self.workspace.gram,
        )?;
        let information = state
            .information
            .iter()
            .zip(self.workspace.gram.iter())
            .map(|(historical, current)| *historical + *current)
            .collect::<Vec<_>>();
        if information.iter().any(|value| !value.is_finite()) {
            return Err(CoreError::NonFiniteResult);
        }

        let objective = diagnostic_objective(
            solution.smooth_objective,
            &solution.coefficients,
            state,
            config.penalty,
            lambda_value,
        );
        if !objective.is_finite() {
            return Err(CoreError::NonFiniteResult);
        }
        Ok(Transition {
            state: State {
                coefficients: solution.coefficients,
                information,
                n_samples_seen,
                batch_count,
                previous_lambda: lambda_value,
                n_features_in: state.n_features_in,
                fit_intercept: state.fit_intercept,
                weight_sum: n_total,
            },
            diagnostics: Diagnostics {
                iterations: solution.iterations,
                converged: solution.converged,
                objective,
                lambda_value,
                bandwidth,
                used_regularized_fallback: solution.used_minimum_norm_fallback,
            },
        })
    }

    fn solve_unpenalized(
        &mut self,
        batch: BatchView<'_, T>,
        state: &State<T>,
        config: UpdateConfig,
        bandwidth: f64,
        n_total: f64,
    ) -> Result<SolverResult<T>, CoreError> {
        let mut beta = state.coefficients.clone();
        let mut objective =
            smooth_objective(batch, &beta, state, config, n_total, &mut self.workspace)?;
        let tolerance = scalar_from_f64::<T>(config.tolerance)?;
        let mut used_minimum_norm_fallback = false;

        for iteration in 1..=config.max_iter {
            gradient_and_hessian(
                batch,
                &beta,
                state,
                config,
                bandwidth,
                n_total,
                &mut self.workspace,
            )?;
            let solve = self.solver.solve(
                &self.workspace.hessian,
                &self.workspace.gradient,
                batch.n_parameters,
            )?;
            used_minimum_norm_fallback |= solve.used_minimum_norm_fallback;
            self.workspace.direction.copy_from_slice(&solve.solution);

            let mut step = 1.0_f64;
            let mut accepted = false;
            let mut candidate_objective = objective;
            while step >= 1.0e-8 {
                let step_t = scalar_from_f64::<T>(step)?;
                for ((candidate, beta_value), direction) in self
                    .workspace
                    .candidate
                    .iter_mut()
                    .zip(beta.iter())
                    .zip(self.workspace.direction.iter())
                {
                    *candidate = *beta_value - step_t * *direction;
                }
                let candidate = std::mem::take(&mut self.workspace.candidate);
                let objective_result = smooth_objective(
                    batch,
                    &candidate,
                    state,
                    config,
                    n_total,
                    &mut self.workspace,
                );
                self.workspace.candidate = candidate;
                candidate_objective = objective_result?;
                if candidate_objective <= objective {
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if !accepted {
                residual(
                    batch.x_design,
                    batch.n_rows,
                    batch.n_parameters,
                    &beta,
                    batch.y,
                    &mut self.workspace.residual,
                );
                return Ok(SolverResult {
                    coefficients: beta,
                    iterations: iteration,
                    converged: false,
                    smooth_objective: objective,
                    used_minimum_norm_fallback,
                });
            }

            for ((difference, candidate), previous) in self
                .workspace
                .difference
                .iter_mut()
                .zip(self.workspace.candidate.iter())
                .zip(beta.iter())
            {
                *difference = *candidate - *previous;
            }
            beta.copy_from_slice(&self.workspace.candidate);
            objective = candidate_objective;
            if norm(&self.workspace.difference) <= tolerance * (T::one() + norm(&beta)) {
                return Ok(SolverResult {
                    coefficients: beta,
                    iterations: iteration,
                    converged: true,
                    smooth_objective: objective,
                    used_minimum_norm_fallback,
                });
            }
        }
        Ok(SolverResult {
            coefficients: beta,
            iterations: config.max_iter,
            converged: false,
            smooth_objective: objective,
            used_minimum_norm_fallback,
        })
    }

    fn solve_l1(
        &mut self,
        batch: BatchView<'_, T>,
        state: &State<T>,
        config: UpdateConfig,
        lambda_value: f64,
        n_total: f64,
    ) -> Result<SolverResult<T>, CoreError> {
        let mut beta = state.coefficients.clone();
        let mut objective =
            smooth_objective(batch, &beta, state, config, n_total, &mut self.workspace)?;
        let tolerance = scalar_from_f64::<T>(config.tolerance)?;
        let mut phi = 1.0_f64;

        for iteration in 1..=config.max_iter {
            gradient_from_current_residual(
                batch,
                &beta,
                state,
                config,
                n_total,
                &mut self.workspace,
            )?;
            let mut accepted = false;
            let mut candidate_objective = objective;
            for _ in 0..40 {
                let inverse_phi = scalar_from_f64::<T>(1.0 / phi)?;
                let threshold = scalar_from_f64::<T>(lambda_value / phi)?;
                for (parameter, beta_value) in beta.iter().enumerate() {
                    let value = *beta_value - self.workspace.gradient[parameter] * inverse_phi;
                    let penalty = if state.fit_intercept && parameter + 1 == batch.n_parameters {
                        T::zero()
                    } else {
                        threshold
                    };
                    self.workspace.candidate[parameter] = soft_threshold(value, penalty);
                    self.workspace.difference[parameter] =
                        self.workspace.candidate[parameter] - *beta_value;
                }
                let gradient_dot = dot(&self.workspace.gradient, &self.workspace.difference)
                    .to_f64()
                    .ok_or(CoreError::ScalarConversion)?;
                let difference_norm = norm(&self.workspace.difference)
                    .to_f64()
                    .ok_or(CoreError::ScalarConversion)?;
                let upper_bound =
                    objective + gradient_dot + 0.5 * phi * difference_norm * difference_norm;
                let candidate = std::mem::take(&mut self.workspace.candidate);
                let objective_result = smooth_objective(
                    batch,
                    &candidate,
                    state,
                    config,
                    n_total,
                    &mut self.workspace,
                );
                self.workspace.candidate = candidate;
                candidate_objective = objective_result?;
                if candidate_objective <= upper_bound + 1.0e-12 {
                    accepted = true;
                    break;
                }
                phi *= 2.0;
            }
            if !accepted {
                residual(
                    batch.x_design,
                    batch.n_rows,
                    batch.n_parameters,
                    &beta,
                    batch.y,
                    &mut self.workspace.residual,
                );
                return Ok(SolverResult {
                    coefficients: beta,
                    iterations: iteration,
                    converged: false,
                    smooth_objective: objective,
                    used_minimum_norm_fallback: false,
                });
            }

            beta.copy_from_slice(&self.workspace.candidate);
            objective = candidate_objective;
            phi = (phi * 0.5).max(1.0e-8);
            if norm(&self.workspace.difference) <= tolerance * (T::one() + norm(&beta)) {
                return Ok(SolverResult {
                    coefficients: beta,
                    iterations: iteration,
                    converged: true,
                    smooth_objective: objective,
                    used_minimum_norm_fallback: false,
                });
            }
        }
        Ok(SolverResult {
            coefficients: beta,
            iterations: config.max_iter,
            converged: false,
            smooth_objective: objective,
            used_minimum_norm_fallback: false,
        })
    }
}

struct SolverResult<T: CpuScalar> {
    coefficients: Vec<T>,
    iterations: usize,
    converged: bool,
    smooth_objective: f64,
    used_minimum_norm_fallback: bool,
}

/// Convenience entry point for callers that do not retain a workspace.
pub fn renewable_update<T: CpuScalar>(
    batch: BatchView<'_, T>,
    state: &State<T>,
    config: UpdateConfig,
) -> Result<Transition<T>, CoreError> {
    CpuEngine::<T>::default().update(batch, state, config)
}

/// Predict from a row-major design matrix.
pub fn predict<T: CpuScalar>(
    x_design: &[T],
    n_rows: usize,
    n_parameters: usize,
    coefficients: &[T],
) -> Result<Vec<T>, CoreError> {
    if coefficients.len() != n_parameters
        || x_design.len()
            != n_rows
                .checked_mul(n_parameters)
                .ok_or(CoreError::SizeOverflow)?
    {
        return Err(CoreError::InvalidBatch("invalid prediction shape"));
    }
    if x_design.iter().any(|value| !value.is_finite()) {
        return Err(CoreError::InvalidBatch(
            "prediction input must contain only finite values",
        ));
    }
    let mut result = vec![T::zero(); n_rows];
    if x_design.len() < PARALLEL_VECTOR_WORK {
        for (row, output) in x_design.chunks_exact(n_parameters).zip(result.iter_mut()) {
            *output = dot(row, coefficients);
        }
    } else {
        result.par_iter_mut().enumerate().for_each(|(row, output)| {
            let start = row * n_parameters;
            *output = dot(&x_design[start..start + n_parameters], coefficients);
        });
    }
    Ok(result)
}
