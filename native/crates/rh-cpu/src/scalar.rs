//! Scalar dispatch for the two supported element types.
//!
//! `CpuScalar` is the seam where a generic algorithm meets a concrete GEMM
//! kernel. Keeping the f32 and f64 implementations next to the shape guard they
//! both call makes the one unsafe boundary in this crate reviewable on its own.

use nalgebra::RealField;
use rh_core::{CoreError, Scalar};

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

pub(crate) fn validate_gemm_buffers<T>(
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
