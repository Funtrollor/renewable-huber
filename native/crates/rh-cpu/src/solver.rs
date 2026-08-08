//! Dense linear solves behind a swappable boundary.
//!
//! The default implementation is nalgebra's Cholesky fast path with partial
//! pivot LU and a minimum-norm SVD fallback. A tuned BLAS/LAPACK provider can
//! replace it without touching the algorithm or the Python boundary.

use nalgebra::{DMatrix, DVector};
use rh_core::CoreError;

use crate::scalar::CpuScalar;

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

/// Portable P1 provider: Cholesky, partial-pivot LU, then minimum-norm SVD.
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
        // The ordinary Renewable Huber Hessian is symmetric positive definite
        // once ridge is applied. Cholesky performs roughly half the work of LU
        // on the wide shapes. Restored checkpoints are allowed to contain a
        // genuinely asymmetric information matrix, so only take this path
        // after an explicit scale-aware symmetry check.
        // f32's narrower mantissa made the factorization perturb the Newton
        // path enough to add iterations on the public wide sweep. Keep LU for
        // f32; Cholesky's speedup is both stable and material for f64.
        if std::mem::size_of::<T>() == std::mem::size_of::<f64>()
            && approximately_symmetric(row_major_matrix, dimension)
        {
            if let Some(cholesky) = matrix.clone().cholesky() {
                let solution = cholesky.solve(&vector);
                if solution.iter().all(|value| value.is_finite()) {
                    return Ok(SolveOutcome {
                        solution: solution.as_slice().to_vec(),
                        used_minimum_norm_fallback: false,
                    });
                }
            }
        }
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

pub(crate) fn approximately_symmetric<T: CpuScalar>(matrix: &[T], dimension: usize) -> bool {
    let mut scale = T::one();
    for value in matrix {
        scale = num_traits::Float::max(scale, num_traits::Float::abs(*value));
    }
    let dimension_t = T::from_usize(dimension).unwrap_or(T::one());
    let tolerance =
        T::default_epsilon() * dimension_t * T::from_f64(32.0).unwrap_or_else(T::one) * scale;
    for row in 0..dimension {
        for column in 0..row {
            let difference = num_traits::Float::abs(
                matrix[row * dimension + column] - matrix[column * dimension + row],
            );
            if difference > tolerance {
                return false;
            }
        }
    }
    true
}
