//! Elementwise and reduction kernels over one batch.

use rayon::prelude::*;
use rh_core::{scalar_from_f64, CoreError};

use crate::scalar::CpuScalar;
use crate::PARALLEL_VECTOR_WORK;

pub(crate) fn residual<T: CpuScalar>(
    x_design: &[T],
    n_rows: usize,
    n_parameters: usize,
    beta: &[T],
    y: &[T],
    output: &mut [T],
) {
    debug_assert_eq!(output.len(), n_rows);
    debug_assert_eq!(x_design.len(), n_rows * n_parameters);
    debug_assert_eq!(beta.len(), n_parameters);
    // Keep latency-sized calls on the current thread. For larger matrices,
    // partition by rows: this avoids the packing cost of treating GEMV as a
    // degenerate GEMM and gives each worker a long, contiguous dot product.
    if n_rows.saturating_mul(n_parameters) < PARALLEL_VECTOR_WORK {
        for ((row, target), result) in x_design
            .chunks_exact(n_parameters)
            .zip(y.iter())
            .zip(output.iter_mut())
        {
            *result = *target - dot(row, beta);
        }
    } else {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_index, result)| {
                let offset = row_index * n_parameters;
                *result = y[row_index] - dot(&x_design[offset..offset + n_parameters], beta);
            });
    }
}

pub(crate) fn huber_loss<T: CpuScalar>(value: T, tau: T) -> T {
    let absolute = num_traits::Float::abs(value);
    if absolute <= tau {
        (T::one() / (T::one() + T::one())) * value * value
    } else {
        tau * absolute - (T::one() / (T::one() + T::one())) * tau * tau
    }
}

pub(crate) fn huber_score<T: CpuScalar>(value: T, tau: T) -> T {
    if value < -tau {
        -tau
    } else if value > tau {
        tau
    } else {
        value
    }
}

pub(crate) fn smoothed_curvature<T: CpuScalar>(
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
pub(crate) fn dot<T: CpuScalar>(left: &[T], right: &[T]) -> T {
    left.iter()
        .zip(right.iter())
        .fold(T::zero(), |sum, (left, right)| sum + *left * *right)
}

pub(crate) fn norm<T: CpuScalar>(values: &[T]) -> T {
    num_traits::Float::sqrt(dot(values, values))
}

pub(crate) fn sign<T: CpuScalar>(value: T) -> T {
    if value > T::zero() {
        T::one()
    } else if value < T::zero() {
        -T::one()
    } else {
        T::zero()
    }
}

pub(crate) fn soft_threshold<T: CpuScalar>(value: T, threshold: T) -> T {
    let remainder = num_traits::Float::abs(value) - threshold;
    sign(value)
        * if remainder > T::zero() {
            remainder
        } else {
            T::zero()
        }
}
