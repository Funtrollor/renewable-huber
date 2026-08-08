//! Weighted Gram and gradient accumulation.
//!
//! These are the kernels that decide whether to engage Rayon: the thresholds
//! exist because joining a worker pool costs more than the arithmetic on small
//! batches.

use rayon::prelude::*;
use rh_core::{scalar_from_f64, BatchView, CoreError, Penalty, State, UpdateConfig};

use crate::kernels::vector::{dot, huber_score, residual, sign, smoothed_curvature};
use crate::scalar::CpuScalar;
use crate::workspace::Workspace;
use crate::{PARALLEL_GRAM_WORK, PARALLEL_VECTOR_WORK};

pub(crate) fn weighted_gram<T: CpuScalar>(
    batch: BatchView<'_, T>,
    curvature: &[T],
    weighted_design: &mut [T],
    partial_grams: &mut [T],
    output: &mut [T],
) -> Result<(), CoreError> {
    let p = batch.n_parameters;
    if batch.n_rows.saturating_mul(p) < PARALLEL_VECTOR_WORK {
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
    } else {
        weighted_design
            .par_chunks_mut(p)
            .enumerate()
            .for_each(|(row_index, weighted_row)| {
                let input_row = &batch.x_design[row_index * p..(row_index + 1) * p];
                let weight = curvature[row_index]
                    * batch
                        .sample_weight
                        .map_or(T::one(), |weights| weights[row_index]);
                for (weighted_value, input_value) in weighted_row.iter_mut().zip(input_row.iter()) {
                    *weighted_value = *input_value * weight;
                }
            });
    }
    let matrix_length = p.checked_mul(p).ok_or(CoreError::SizeOverflow)?;
    let workers = partial_grams.len() / matrix_length;
    let workload = batch.n_rows.saturating_mul(matrix_length);
    if workers <= 1 || workload < PARALLEL_GRAM_WORK {
        return T::weighted_gram_gemm(batch.x_design, weighted_design, batch.n_rows, p, output);
    }

    let rows_per_worker = batch.n_rows.div_ceil(workers);
    partial_grams
        .par_chunks_mut(matrix_length)
        .enumerate()
        .try_for_each(|(worker, partial)| {
            let row_start = worker * rows_per_worker;
            let row_end = (row_start + rows_per_worker).min(batch.n_rows);
            if row_start == row_end {
                partial.fill(T::zero());
                return Ok(());
            }
            let value_start = row_start * p;
            let value_end = row_end * p;
            T::weighted_gram_gemm(
                &batch.x_design[value_start..value_end],
                &weighted_design[value_start..value_end],
                row_end - row_start,
                p,
                partial,
            )
        })?;
    output.fill(T::zero());
    for partial in partial_grams.chunks_exact(matrix_length) {
        for (total, value) in output.iter_mut().zip(partial.iter()) {
            *total += *value;
        }
    }
    Ok(())
}

pub(crate) fn gradient_from_current_residual<T: CpuScalar>(
    batch: BatchView<'_, T>,
    beta: &[T],
    state: &State<T>,
    config: UpdateConfig,
    n_total: f64,
    workspace: &mut Workspace<T>,
) -> Result<(), CoreError> {
    // L1's preceding smooth-objective evaluation used the same `beta` and
    // leaves its residual in the workspace. Each accepted candidate objective
    // establishes the invariant again for the next iteration, eliminating one
    // full X @ beta pass per proximal iteration without changing arithmetic.
    let tau = scalar_from_f64::<T>(config.tau)?;
    for (row, (score, value)) in workspace
        .score
        .iter_mut()
        .zip(workspace.residual.iter())
        .enumerate()
    {
        // Store the frequency weight in the score so the subsequent
        // parallel X.T @ score reduction applies it exactly once.
        let sample_weight = batch.sample_weight.map_or(T::one(), |weights| weights[row]);
        *score = huber_score(*value, tau) * sample_weight;
    }
    gradient_from_score(batch, beta, state, config, n_total, workspace)
}

pub(crate) fn gradient_and_hessian<T: CpuScalar>(
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
    for (row, (score, value)) in workspace
        .score
        .iter_mut()
        .zip(workspace.residual.iter())
        .enumerate()
    {
        // `gradient_from_score` consumes this already-weighted score.
        let sample_weight = batch.sample_weight.map_or(T::one(), |weights| weights[row]);
        *score = huber_score(*value, tau) * sample_weight;
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
        &mut workspace.partial_grams,
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

pub(crate) fn gradient_from_score<T: CpuScalar>(
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
    if batch.n_rows.saturating_mul(p) < PARALLEL_VECTOR_WORK {
        // Accumulate by row to walk the C-contiguous design matrix once.
        // The former column-wise reduction repeatedly traversed the same
        // matrix at a stride of `p`, which was measurably slower than NumPy's
        // BLAS path for the reference-sized micro-batches.
        workspace.gradient.fill(T::zero());
        for (row, score) in batch.x_design.chunks_exact(p).zip(workspace.score.iter()) {
            for (gradient, value) in workspace.gradient.iter_mut().zip(row.iter()) {
                *gradient += *value * *score;
            }
        }
    } else if config.penalty == Penalty::L1 || prefer_row_chunk_gradient::<T>(p) {
        // Split by contiguous row ranges and give each worker a private p-wide
        // accumulator. This reads X exactly once in cache-friendly order. The
        // previous parameter-parallel implementation made every worker walk
        // X with a p-element stride, which left much of each cache line unused.
        let workers = workspace.partial_gradients.len() / p;
        let rows_per_worker = batch.n_rows.div_ceil(workers);
        workspace
            .partial_gradients
            .par_chunks_mut(p)
            .enumerate()
            .for_each(|(worker, partial)| {
                partial.fill(T::zero());
                let row_start = worker * rows_per_worker;
                let row_end = (row_start + rows_per_worker).min(batch.n_rows);
                for row_index in row_start..row_end {
                    let row = &batch.x_design[row_index * p..(row_index + 1) * p];
                    let score = workspace.score[row_index];
                    for (sum, value) in partial.iter_mut().zip(row.iter()) {
                        *sum += *value * score;
                    }
                }
            });
        workspace.gradient.fill(T::zero());
        for partial in workspace.partial_gradients.chunks_exact(p) {
            for (sum, value) in workspace.gradient.iter_mut().zip(partial.iter()) {
                *sum += *value;
            }
        }
    } else {
        // A private p-wide accumulator per worker stops fitting comfortably in
        // cache for wider f64 designs. Parameter partitioning wins there even
        // with its strided access pattern, so retain the established path.
        workspace
            .gradient
            .par_iter_mut()
            .enumerate()
            .for_each(|(parameter, current)| {
                *current = batch
                    .x_design
                    .chunks_exact(p)
                    .zip(workspace.score.iter())
                    .fold(T::zero(), |sum, (row, score)| sum + row[parameter] * *score);
            });
    }
    for parameter in 0..p {
        let current = workspace.gradient[parameter];
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

pub(crate) fn prefer_row_chunk_gradient<T>(n_parameters: usize) -> bool {
    if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
        n_parameters <= 128
    } else {
        n_parameters <= 64
    }
}
