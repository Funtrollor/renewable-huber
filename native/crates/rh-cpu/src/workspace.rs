//! Engine-owned scratch buffers, sized for the largest batch seen so far.
//!
//! Growth is monotonic and reuse is the point: a streaming caller runs many
//! `partial_fit` calls at one shape, and reallocating per call would dominate
//! small batches.

use rh_core::CoreError;

use crate::scalar::CpuScalar;
use crate::{MAX_PARTIAL_GRAM_BYTES, PARALLEL_GRAM_WORK, PARALLEL_VECTOR_WORK};

/// Reusable buffers sized for the largest batch processed by an engine.
#[derive(Clone, Debug, Default)]
pub struct Workspace<T: CpuScalar> {
    pub(crate) residual: Vec<T>,
    pub(crate) score: Vec<T>,
    pub(crate) curvature: Vec<T>,
    pub(crate) weighted_design: Vec<T>,
    pub(crate) partial_grams: Vec<T>,
    pub(crate) partial_gradients: Vec<T>,
    pub(crate) gradient: Vec<T>,
    pub(crate) hessian: Vec<T>,
    pub(crate) gram: Vec<T>,
    pub(crate) delta: Vec<T>,
    pub(crate) direction: Vec<T>,
    pub(crate) candidate: Vec<T>,
    pub(crate) difference: Vec<T>,
}

impl<T: CpuScalar> Workspace<T> {
    pub(crate) fn reserve(&mut self, n_rows: usize, n_parameters: usize) -> Result<(), CoreError> {
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
        let gram_work = n_rows.saturating_mul(matrix_length);
        let gram_bytes = matrix_length
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(CoreError::SizeOverflow)?;
        let memory_limited_workers = MAX_PARTIAL_GRAM_BYTES / gram_bytes.max(1);
        let gram_workers = if gram_work >= PARALLEL_GRAM_WORK && memory_limited_workers > 1 {
            rayon::current_num_threads()
                .min(n_rows)
                .min(memory_limited_workers)
        } else {
            0
        };
        self.partial_grams.resize(
            matrix_length
                .checked_mul(gram_workers)
                .ok_or(CoreError::SizeOverflow)?,
            T::zero(),
        );
        let gradient_workers = if n_rows.saturating_mul(n_parameters) >= PARALLEL_VECTOR_WORK {
            rayon::current_num_threads().min(n_rows)
        } else {
            0
        };
        self.partial_gradients.resize(
            n_parameters
                .checked_mul(gradient_workers)
                .ok_or(CoreError::SizeOverflow)?,
            T::zero(),
        );
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
