//! Whole-batch Rust CPU engine for Renewable Huber regression.
//!
//! The hot row-wise kernels operate directly on C-contiguous input slices and
//! reuse an engine-owned workspace. Weighted Gram matrices use matrixmultiply's
//! runtime-selected SGEMM/DGEMM kernels and its portable, maximum-four-thread
//! pool (`MATMUL_NUM_THREADS` can lower the limit). Dense systems are routed
//! through the [`DenseSolver`] abstraction. P1 uses nalgebra's portable
//! partial-pivot LU with a minimum-norm SVD fallback; a tuned BLAS/LAPACK
//! provider can replace it later without changing the algorithm or Python
//! boundary.

use std::marker::PhantomData;

use nalgebra::{DMatrix, DVector, RealField};
use rh_core::{
    scalar_from_f64, BatchView, CoreError, Diagnostics, Penalty, Scalar, State, Transition,
    UpdateConfig,
};

/// Scalar supported by the CPU implementation.
pub trait CpuScalar: Scalar + RealField {
    /// Compute `X.T @ weighted_design` from two borrowed row-major matrices.
    #[doc(hidden)]
    fn weighted_gram_gemm(
        x_design: &[Self],
        weighted_design: &[Self],
        n_rows: usize,
        n_parameters: usize,
        output: &mut [Self],
    ) -> Result<(), CoreError>;
}

impl CpuScalar for f32 {
    fn weighted_gram_gemm(
        x_design: &[Self],
        weighted_design: &[Self],
        n_rows: usize,
        n_parameters: usize,
        output: &mut [Self],
    ) -> Result<(), CoreError> {
        let stride =
            validate_gemm_buffers(x_design, weighted_design, n_rows, n_parameters, output)?;
        // SAFETY: all buffers have been shape-validated by `BatchView`; A and
        // B are immutable, C is disjoint and uniquely borrowed, and the
        // strides describe X.T, weighted X, and row-major output respectively.
        unsafe {
            matrixmultiply::sgemm(
                n_parameters,
                n_rows,
                n_parameters,
                1.0,
                x_design.as_ptr(),
                1,
                stride,
                weighted_design.as_ptr(),
                stride,
                1,
                0.0,
                output.as_mut_ptr(),
                stride,
                1,
            );
        }
        Ok(())
    }
}

impl CpuScalar for f64 {
    fn weighted_gram_gemm(
        x_design: &[Self],
        weighted_design: &[Self],
        n_rows: usize,
        n_parameters: usize,
        output: &mut [Self],
    ) -> Result<(), CoreError> {
        let stride =
            validate_gemm_buffers(x_design, weighted_design, n_rows, n_parameters, output)?;
        // SAFETY: see the f32 implementation above. Runtime CPU feature
        // detection selects a portable SIMD kernel; no target-cpu flag is used.
        unsafe {
            matrixmultiply::dgemm(
                n_parameters,
                n_rows,
                n_parameters,
                1.0,
                x_design.as_ptr(),
                1,
                stride,
                weighted_design.as_ptr(),
                stride,
                1,
                0.0,
                output.as_mut_ptr(),
                stride,
                1,
            );
        }
        Ok(())
    }
}

fn validate_gemm_buffers<T>(
    x_design: &[T],
    weighted_design: &[T],
    n_rows: usize,
    n_parameters: usize,
    output: &[T],
) -> Result<isize, CoreError> {
    let design_length = n_rows
        .checked_mul(n_parameters)
        .ok_or(CoreError::SizeOverflow)?;
    let output_length = n_parameters
        .checked_mul(n_parameters)
        .ok_or(CoreError::SizeOverflow)?;
    if x_design.len() != design_length
        || weighted_design.len() != design_length
        || output.len() != output_length
    {
        return Err(CoreError::InvalidBatch(
            "GEMM buffers do not match the declared matrix shapes",
        ));
    }
    isize::try_from(n_parameters).map_err(|_| CoreError::SizeOverflow)
}

#[derive(Clone, Debug)]
pub struct SolveOutcome<T: CpuScalar> {
    pub solution: Vec<T>,
    pub used_minimum_norm_fallback: bool,
}

/// Pluggable dense solver boundary.
pub trait DenseSolver<T: CpuScalar> {
    fn solve(
        &mut self,
        row_major_matrix: &[T],
        rhs: &[T],
        dimension: usize,
    ) -> Result<SolveOutcome<T>, CoreError>;
}

/// Portable P1 provider: partial-pivot LU followed by minimum-norm SVD.
#[derive(Clone, Copy, Debug, Default)]
pub struct NalgebraSolver;

impl<T: CpuScalar> DenseSolver<T> for NalgebraSolver {
    fn solve(
        &mut self,
        row_major_matrix: &[T],
        rhs: &[T],
        dimension: usize,
    ) -> Result<SolveOutcome<T>, CoreError> {
        if row_major_matrix.len() != dimension * dimension || rhs.len() != dimension {
            return Err(CoreError::LinearSolve);
        }
        let matrix = DMatrix::from_row_slice(dimension, dimension, row_major_matrix);
        let vector = DVector::from_column_slice(rhs);
        if let Some(solution) = matrix.clone().lu().solve(&vector) {
            if solution.iter().all(|value| value.is_finite()) {
                return Ok(SolveOutcome {
                    solution: solution.as_slice().to_vec(),
                    used_minimum_norm_fallback: false,
                });
            }
        }

        let svd = matrix.svd(true, true);
        let largest = svd
            .singular_values
            .iter()
            .copied()
            .fold(
                T::zero(),
                |left, right| {
                    if left > right {
                        left
                    } else {
                        right
                    }
                },
            );
        let dimension_t = T::from_usize(dimension).ok_or(CoreError::ScalarConversion)?;
        let threshold = T::default_epsilon() * dimension_t * largest;
        let solution = svd
            .solve(&vector, threshold)
            .map_err(|_| CoreError::LinearSolve)?;
        if solution.iter().any(|value| !value.is_finite()) {
            return Err(CoreError::NonFiniteResult);
        }
        Ok(SolveOutcome {
            solution: solution.as_slice().to_vec(),
            used_minimum_norm_fallback: true,
        })
    }
}

/// Reusable buffers sized for the largest batch processed by an engine.
#[derive(Clone, Debug, Default)]
pub struct Workspace<T: CpuScalar> {
    residual: Vec<T>,
    score: Vec<T>,
    curvature: Vec<T>,
    weighted_design: Vec<T>,
    gradient: Vec<T>,
    hessian: Vec<T>,
    gram: Vec<T>,
    delta: Vec<T>,
    direction: Vec<T>,
    candidate: Vec<T>,
    difference: Vec<T>,
}

impl<T: CpuScalar> Workspace<T> {
    fn reserve(&mut self, n_rows: usize, n_parameters: usize) -> Result<(), CoreError> {
        let design_length = n_rows
            .checked_mul(n_parameters)
            .ok_or(CoreError::SizeOverflow)?;
        let matrix_length = n_parameters
            .checked_mul(n_parameters)
            .ok_or(CoreError::SizeOverflow)?;
        self.residual.resize(n_rows, T::zero());
        self.score.resize(n_rows, T::zero());
        self.curvature.resize(n_rows, T::zero());
        self.weighted_design.resize(design_length, T::zero());
        self.gradient.resize(n_parameters, T::zero());
        self.hessian.resize(matrix_length, T::zero());
        self.gram.resize(matrix_length, T::zero());
        self.delta.resize(n_parameters, T::zero());
        self.direction.resize(n_parameters, T::zero());
        self.candidate.resize(n_parameters, T::zero());
        self.difference.resize(n_parameters, T::zero());
        Ok(())
    }
}

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

        residual(
            batch.x_design,
            batch.n_rows,
            n_parameters,
            &solution.coefficients,
            batch.y,
            &mut self.workspace.residual,
        );
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
            gradient_only(batch, &beta, state, config, n_total, &mut self.workspace)?;
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
    for (row, output) in x_design.chunks_exact(n_parameters).zip(result.iter_mut()) {
        *output = dot(row, coefficients);
    }
    Ok(result)
}

fn bandwidth(n_total: f64, n_features: usize, scale: f64, tau: f64) -> f64 {
    let log_features = (n_features.max(2) as f64).ln();
    (scale / (n_total.sqrt() * log_features)).min(tau)
}

fn lambda_value(n_total: f64, n_features: usize, config: UpdateConfig) -> f64 {
    if config.penalty == Penalty::None {
        return 0.0;
    }
    config.lambda_scale * config.tau * (((n_features.max(2) as f64).ln() / n_total).sqrt())
}

fn residual<T: CpuScalar>(
    x_design: &[T],
    n_rows: usize,
    n_parameters: usize,
    beta: &[T],
    y: &[T],
    output: &mut [T],
) {
    debug_assert_eq!(output.len(), n_rows);
    for ((row, target), result) in x_design
        .chunks_exact(n_parameters)
        .zip(y.iter())
        .zip(output.iter_mut())
    {
        *result = *target - dot(row, beta);
    }
}

fn huber_loss<T: CpuScalar>(value: T, tau: T) -> T {
    let absolute = num_traits::Float::abs(value);
    if absolute <= tau {
        (T::one() / (T::one() + T::one())) * value * value
    } else {
        tau * absolute - (T::one() / (T::one() + T::one())) * tau * tau
    }
}

fn huber_score<T: CpuScalar>(value: T, tau: T) -> T {
    if value < -tau {
        -tau
    } else if value > tau {
        tau
    } else {
        value
    }
}

fn smoothed_curvature<T: CpuScalar>(
    residual: &[T],
    tau: f64,
    bandwidth: f64,
    output: &mut [T],
) -> Result<(), CoreError> {
    let tau = scalar_from_f64::<T>(tau)?;
    let h = scalar_from_f64::<T>(bandwidth.min(tau.to_f64().unwrap_or(bandwidth)))?;
    let half = T::one() / (T::one() + T::one());
    for (value, curvature) in residual.iter().zip(output.iter_mut()) {
        *curvature = if *value < -tau - h {
            T::zero()
        } else if *value <= -tau + h {
            half + (*value + tau) / (h + h)
        } else if *value < tau - h {
            T::one()
        } else if *value <= tau + h {
            half - (*value - tau) / (h + h)
        } else {
            T::zero()
        };
    }
    Ok(())
}

fn weighted_gram<T: CpuScalar>(
    batch: BatchView<'_, T>,
    curvature: &[T],
    weighted_design: &mut [T],
    output: &mut [T],
) -> Result<(), CoreError> {
    let p = batch.n_parameters;
    for (row_index, (input_row, weighted_row)) in batch
        .x_design
        .chunks_exact(p)
        .zip(weighted_design.chunks_exact_mut(p))
        .enumerate()
    {
        let weight = curvature[row_index]
            * batch
                .sample_weight
                .map_or(T::one(), |weights| weights[row_index]);
        for (weighted_value, input_value) in weighted_row.iter_mut().zip(input_row.iter()) {
            *weighted_value = *input_value * weight;
        }
    }
    T::weighted_gram_gemm(batch.x_design, weighted_design, batch.n_rows, p, output)
}

fn gradient_only<T: CpuScalar>(
    batch: BatchView<'_, T>,
    beta: &[T],
    state: &State<T>,
    config: UpdateConfig,
    n_total: f64,
    workspace: &mut Workspace<T>,
) -> Result<(), CoreError> {
    residual(
        batch.x_design,
        batch.n_rows,
        batch.n_parameters,
        beta,
        batch.y,
        &mut workspace.residual,
    );
    let tau = scalar_from_f64::<T>(config.tau)?;
    for (score, value) in workspace.score.iter_mut().zip(workspace.residual.iter()) {
        *score = huber_score(*value, tau);
    }
    gradient_from_score(batch, beta, state, config, n_total, workspace)
}

fn gradient_and_hessian<T: CpuScalar>(
    batch: BatchView<'_, T>,
    beta: &[T],
    state: &State<T>,
    config: UpdateConfig,
    bandwidth: f64,
    n_total: f64,
    workspace: &mut Workspace<T>,
) -> Result<(), CoreError> {
    residual(
        batch.x_design,
        batch.n_rows,
        batch.n_parameters,
        beta,
        batch.y,
        &mut workspace.residual,
    );
    let tau = scalar_from_f64::<T>(config.tau)?;
    for (score, value) in workspace.score.iter_mut().zip(workspace.residual.iter()) {
        *score = huber_score(*value, tau);
    }
    smoothed_curvature(
        &workspace.residual,
        config.tau,
        bandwidth,
        &mut workspace.curvature,
    )?;
    gradient_from_score(batch, beta, state, config, n_total, workspace)?;
    weighted_gram(
        batch,
        &workspace.curvature,
        &mut workspace.weighted_design,
        &mut workspace.gram,
    )?;
    let divisor = scalar_from_f64::<T>(n_total)?;
    let ridge = scalar_from_f64::<T>(config.ridge)?;
    for index in 0..workspace.hessian.len() {
        workspace.hessian[index] = (workspace.gram[index] + state.information[index]) / divisor;
    }
    for diagonal in 0..batch.n_parameters {
        workspace.hessian[diagonal * batch.n_parameters + diagonal] += ridge;
    }
    Ok(())
}

fn gradient_from_score<T: CpuScalar>(
    batch: BatchView<'_, T>,
    beta: &[T],
    state: &State<T>,
    config: UpdateConfig,
    n_total: f64,
    workspace: &mut Workspace<T>,
) -> Result<(), CoreError> {
    let p = batch.n_parameters;
    for (delta, (beta_value, state_value)) in workspace
        .delta
        .iter_mut()
        .zip(beta.iter().zip(state.coefficients.iter()))
    {
        *delta = *beta_value - *state_value;
    }
    let divisor = scalar_from_f64::<T>(n_total)?;
    let historical_scale =
        scalar_from_f64::<T>(state.weight_sum / n_total * state.previous_lambda)?;
    for parameter in 0..p {
        let mut current = T::zero();
        for row in 0..batch.n_rows {
            let sample_weight = batch.sample_weight.map_or(T::one(), |weights| weights[row]);
            current += batch.x_design[row * p + parameter] * workspace.score[row] * sample_weight;
        }
        let historical = dot(
            &state.information[parameter * p..(parameter + 1) * p],
            &workspace.delta,
        );
        let mut gradient = (-current + historical) / divisor;
        if config.penalty == Penalty::L1
            && state.n_samples_seen > 0
            && !(state.fit_intercept && parameter + 1 == p)
        {
            gradient -= historical_scale * sign(state.coefficients[parameter]);
        }
        workspace.gradient[parameter] = gradient;
    }
    Ok(())
}

fn smooth_objective<T: CpuScalar>(
    batch: BatchView<'_, T>,
    beta: &[T],
    state: &State<T>,
    config: UpdateConfig,
    n_total: f64,
    workspace: &mut Workspace<T>,
) -> Result<f64, CoreError> {
    let p = batch.n_parameters;
    residual(
        batch.x_design,
        batch.n_rows,
        p,
        beta,
        batch.y,
        &mut workspace.residual,
    );
    let tau = scalar_from_f64::<T>(config.tau)?;
    let mut current = T::zero();
    for row in 0..batch.n_rows {
        let weight = batch.sample_weight.map_or(T::one(), |weights| weights[row]);
        current += weight * huber_loss(workspace.residual[row], tau);
    }
    for (delta, (beta_value, state_value)) in workspace
        .delta
        .iter_mut()
        .zip(beta.iter().zip(state.coefficients.iter()))
    {
        *delta = *beta_value - *state_value;
    }
    let mut historical = T::zero();
    for row in 0..p {
        historical += workspace.delta[row]
            * dot(&state.information[row * p..(row + 1) * p], &workspace.delta);
    }
    let half = T::one() / (T::one() + T::one());
    let divisor = scalar_from_f64::<T>(n_total)?;
    let mut objective = (current + half * historical) / divisor;
    if config.penalty == Penalty::L1 && state.n_samples_seen > 0 {
        let historical_scale =
            scalar_from_f64::<T>(state.weight_sum / n_total * state.previous_lambda)?;
        let mut product = T::zero();
        for parameter in 0..p {
            if !(state.fit_intercept && parameter + 1 == p) {
                product += workspace.delta[parameter] * sign(state.coefficients[parameter]);
            }
        }
        objective -= historical_scale * product;
    }
    objective.to_f64().ok_or(CoreError::ScalarConversion)
}

fn diagnostic_objective<T: CpuScalar>(
    smooth_objective: f64,
    coefficients: &[T],
    state: &State<T>,
    penalty: Penalty,
    lambda_value: f64,
) -> f64 {
    if penalty == Penalty::None {
        return smooth_objective;
    }
    let last_penalized = coefficients.len() - usize::from(state.fit_intercept);
    let l1 = coefficients[..last_penalized]
        .iter()
        .filter_map(|value| num_traits::Float::abs(*value).to_f64())
        .sum::<f64>();
    smooth_objective + lambda_value * l1
}

fn dot<T: CpuScalar>(left: &[T], right: &[T]) -> T {
    left.iter()
        .zip(right.iter())
        .fold(T::zero(), |sum, (left, right)| sum + *left * *right)
}

fn norm<T: CpuScalar>(values: &[T]) -> T {
    num_traits::Float::sqrt(dot(values, values))
}

fn sign<T: CpuScalar>(value: T) -> T {
    if value > T::zero() {
        T::one()
    } else if value < T::zero() {
        -T::one()
    } else {
        T::zero()
    }
}

fn soft_threshold<T: CpuScalar>(value: T, threshold: T) -> T {
    let remainder = num_traits::Float::abs(value) - threshold;
    sign(value)
        * if remainder > T::zero() {
            remainder
        } else {
            T::zero()
        }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const GOLDEN_CORPUS: &str = include_str!("../../../../tests/golden/native_core_v1.json");

    #[test]
    fn f64_engine_replays_numpy_golden_corpus() {
        replay_cases::<f64>("float64");
    }

    #[test]
    fn f32_engine_replays_numpy_golden_corpus() {
        replay_cases::<f32>("float32");
    }

    #[test]
    fn rank_deficient_case_uses_minimum_norm_fallback() {
        let corpus: Value = serde_json::from_str(GOLDEN_CORPUS).unwrap();
        let case = corpus["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["id"] == "rank_deficient_lstsq_f64")
            .unwrap();
        let (_, diagnostics) = replay_case::<f64>(case);
        assert!(diagnostics.used_regularized_fallback);
    }

    #[test]
    fn public_gemm_dispatch_rejects_invalid_buffer_shapes() {
        let error = f64::weighted_gram_gemm(&[1.0, 2.0], &[1.0, 2.0], 1, 2, &mut [0.0])
            .expect_err("a short output buffer must be rejected before unsafe GEMM");
        assert!(matches!(error, CoreError::InvalidBatch(_)));
    }

    #[test]
    fn persistent_engine_can_grow_and_reuse_its_workspace() {
        let mut engine = CpuEngine::<f64>::default();
        let config = UpdateConfig {
            max_iter: 20,
            ..UpdateConfig::default()
        };
        let mut state = State::empty(1, false);

        let first_x = [1.0, 2.0];
        let first_y = [1.0, 2.0];
        let first = engine
            .update(
                BatchView {
                    x_design: &first_x,
                    n_rows: 2,
                    n_parameters: 1,
                    y: &first_y,
                    sample_weight: None,
                    batch_weight: 2.0,
                },
                &state,
                config,
            )
            .unwrap();
        state = first.state;

        let second_x = [1.0, 2.0, 3.0, 4.0];
        let second_y = [1.0, 2.0, 3.0, 4.0];
        let second = engine
            .update(
                BatchView {
                    x_design: &second_x,
                    n_rows: 4,
                    n_parameters: 1,
                    y: &second_y,
                    sample_weight: None,
                    batch_weight: 4.0,
                },
                &state,
                config,
            )
            .unwrap();
        assert_eq!(second.state.n_samples_seen, 6);
        assert_eq!(second.state.batch_count, 2);
        assert!((second.state.coefficients[0] - 1.0).abs() < 1.0e-7);
    }

    fn replay_cases<T: CpuScalar>(dtype: &str) {
        let corpus: Value = serde_json::from_str(GOLDEN_CORPUS).unwrap();
        let cases = corpus["cases"].as_array().unwrap();
        let selected = cases
            .iter()
            .filter(|case| case["config"]["dtype"] == dtype)
            .collect::<Vec<_>>();
        assert!(!selected.is_empty());
        for case in selected {
            replay_case::<T>(case);
        }
    }

    fn replay_case<T: CpuScalar>(case: &Value) -> (State<T>, Diagnostics) {
        let config_value = &case["config"];
        let fit_intercept = config_value["fit_intercept"].as_bool().unwrap();
        let config = UpdateConfig {
            tau: config_value["tau"].as_f64().unwrap(),
            penalty: match config_value["penalty"].as_str().unwrap() {
                "none" => Penalty::None,
                "l1" => Penalty::L1,
                other => panic!("unknown golden penalty {other}"),
            },
            lambda_scale: config_value["lambda_scale"].as_f64().unwrap(),
            bandwidth_scale: config_value["bandwidth_scale"].as_f64().unwrap(),
            max_iter: config_value["max_iter"].as_u64().unwrap() as usize,
            tolerance: config_value["tol"].as_f64().unwrap(),
            ridge: config_value["ridge"].as_f64().unwrap(),
        };
        let first_x = case["batches"][0]["X"].as_array().unwrap();
        let n_features = first_x[0].as_array().unwrap().len();
        let n_parameters = n_features + usize::from(fit_intercept);
        let mut state = State::<T>::empty(n_features, fit_intercept);
        let mut engine = CpuEngine::<T>::default();
        let expected_states = case["expected"]["states"].as_array().unwrap();
        let rtol = case["rtol"].as_f64().unwrap();
        let atol = case["atol"].as_f64().unwrap();
        let mut last_diagnostics = None;

        for (batch_value, expected) in case["batches"]
            .as_array()
            .unwrap()
            .iter()
            .zip(expected_states.iter())
        {
            let rows = batch_value["X"].as_array().unwrap();
            let mut x_design = Vec::with_capacity(rows.len() * n_parameters);
            for row in rows {
                for value in row.as_array().unwrap() {
                    x_design.push(from_json::<T>(value));
                }
                if fit_intercept {
                    x_design.push(T::one());
                }
            }
            let y = batch_value["y"]
                .as_array()
                .unwrap()
                .iter()
                .map(from_json::<T>)
                .collect::<Vec<_>>();
            let weights = batch_value["sample_weight"]
                .as_array()
                .map(|values| values.iter().map(from_json::<T>).collect::<Vec<_>>());
            let batch_weight = weights.as_ref().map_or(rows.len() as f64, |values| {
                values
                    .iter()
                    .map(|value| value.to_f64().unwrap())
                    .sum::<f64>()
            });
            let transition = engine
                .update(
                    BatchView {
                        x_design: &x_design,
                        n_rows: rows.len(),
                        n_parameters,
                        y: &y,
                        sample_weight: weights.as_deref(),
                        batch_weight,
                    },
                    &state,
                    config,
                )
                .unwrap_or_else(|error| panic!("golden case {} failed: {error}", case["id"]));
            assert_vector_close(
                &transition.state.coefficients,
                expected["coefficients"].as_array().unwrap(),
                rtol,
                atol,
                "coefficients",
            );
            let expected_information = expected["information"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap().iter().cloned())
                .collect::<Vec<_>>();
            assert_vector_close(
                &transition.state.information,
                &expected_information,
                rtol,
                atol,
                "information",
            );
            assert_eq!(
                transition.state.n_samples_seen,
                expected["n_samples_seen"].as_u64().unwrap() as usize
            );
            assert_eq!(
                transition.state.batch_count,
                expected["batch_count"].as_u64().unwrap() as usize
            );
            assert_close(
                transition.state.previous_lambda,
                expected["previous_lambda"].as_f64().unwrap(),
                rtol,
                atol,
                "previous_lambda",
            );
            assert_close(
                transition.state.weight_sum,
                expected["weight_sum"].as_f64().unwrap(),
                rtol,
                atol,
                "weight_sum",
            );
            let expected_diagnostics = &expected["diagnostics"];
            assert_eq!(
                transition.diagnostics.converged,
                expected_diagnostics["converged"].as_bool().unwrap()
            );
            for (actual, name) in [
                (transition.diagnostics.objective, "objective"),
                (transition.diagnostics.lambda_value, "lambda_value"),
                (transition.diagnostics.bandwidth, "bandwidth"),
            ] {
                assert_close(
                    actual,
                    expected_diagnostics[name].as_f64().unwrap(),
                    rtol,
                    atol,
                    name,
                );
            }
            last_diagnostics = Some(transition.diagnostics);
            state = transition.state;
        }

        let probe_rows = case["probe_X"].as_array().unwrap();
        let mut probe_design = Vec::with_capacity(probe_rows.len() * n_parameters);
        for row in probe_rows {
            for value in row.as_array().unwrap() {
                probe_design.push(from_json::<T>(value));
            }
            if fit_intercept {
                probe_design.push(T::one());
            }
        }
        let predictions = predict(
            &probe_design,
            probe_rows.len(),
            n_parameters,
            &state.coefficients,
        )
        .unwrap();
        assert_vector_close(
            &predictions,
            case["expected"]["predictions"].as_array().unwrap(),
            rtol,
            atol,
            "predictions",
        );
        (state, last_diagnostics.unwrap())
    }

    fn from_json<T: CpuScalar>(value: &Value) -> T {
        T::from_f64(value.as_f64().unwrap()).unwrap()
    }

    fn assert_vector_close<T: CpuScalar>(
        actual: &[T],
        expected: &[Value],
        rtol: f64,
        atol: f64,
        field: &str,
    ) {
        assert_eq!(actual.len(), expected.len(), "{field} length");
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_close(
                actual.to_f64().unwrap(),
                expected.as_f64().unwrap(),
                rtol,
                atol,
                &format!("{field}[{index}]"),
            );
        }
    }

    fn assert_close(actual: f64, expected: f64, rtol: f64, atol: f64, field: &str) {
        let difference = (actual - expected).abs();
        let allowed = atol + rtol * expected.abs();
        assert!(
            difference <= allowed,
            "{field}: actual={actual:?}, expected={expected:?}, difference={difference:?}, \
             allowed={allowed:?}"
        );
    }
}
