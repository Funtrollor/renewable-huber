//! Safe ownership-oriented wrapper around `native/cuda/include/rh_cuda.h`.
//!
//! The public types in this crate intentionally describe only host-contiguous
//! `f32`/`f64` buffers.  Device pointers and CUDA/C++ implementation details
//! never escape this boundary; the opaque C handle owns every device resource.
//!
//! The crate compiles without a CUDA toolchain. The `cuda` feature adds the
//! link-time half; without it every engine method returns
//! [`CudaError::NotCompiled`], and the ABI records still compile so their
//! layout stays under test.
//!
//! The modules below are an internal arrangement; everything public is
//! re-exported here.

mod engine;
mod runtime;
mod sys;
mod types;
mod validation;

#[cfg(test)]
mod tests;

pub const ABI_VERSION: u32 = 1;

pub use engine::CudaEngine;
pub use runtime::{device_count, is_available, linked_abi_version, runtime_info};
pub use types::{
    CudaDtype, CudaError, CudaScalar, DeviceBatch, Diagnostics, EngineFeatures, EngineTuning,
    HostBatch, HostMatrix, HostState, HostVector, RuntimeInfo, StateMetadata, UnpenalizedConfig,
};
