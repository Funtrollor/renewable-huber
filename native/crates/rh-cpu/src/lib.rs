//! Whole-batch Rust CPU engine for Renewable Huber regression.
//!
//! The hot row-wise kernels operate directly on C-contiguous input slices and
//! reuse an engine-owned workspace. Weighted Gram matrices use matrixmultiply's
//! runtime-selected SGEMM/DGEMM kernels, partitioned across Rayon's persistent
//! worker pool for sufficiently large batches. Dense systems are routed through
//! the [`DenseSolver`] abstraction. P1 uses nalgebra's portable Cholesky fast
//! path, partial-pivot LU, and a minimum-norm SVD fallback; a tuned BLAS/LAPACK
//! provider can replace it later without changing the algorithm or Python
//! boundary.
//!
//! The modules below are an internal arrangement; everything public is
//! re-exported here, so this crate's API is exactly the list at the bottom of
//! this file.

mod engine;
mod kernels;
mod scalar;
mod solver;
mod workspace;

#[cfg(test)]
mod tests;

// Micro-batches are latency-bound: crossing the Python boundary and joining a
// worker pool costs more than the dot products themselves. These thresholds
// were tuned against the public smoke sweep on a 24-thread Ryzen host. They are
// crate-wide policy, read by the workspace, the engine, and the kernels alike.
pub(crate) const PARALLEL_VECTOR_WORK: usize = 1_000_000;
pub(crate) const PARALLEL_GRAM_WORK: usize = 16_000_000;
pub(crate) const MAX_PARTIAL_GRAM_BYTES: usize = 64 * 1024 * 1024;

pub use engine::{predict, renewable_update, CpuEngine};
pub use scalar::CpuScalar;
pub use solver::{DenseSolver, NalgebraSolver, SolveOutcome};
pub use workspace::Workspace;
