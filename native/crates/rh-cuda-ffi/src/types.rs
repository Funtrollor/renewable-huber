//! Owned and borrowed values that cross the safe boundary.
//!
//! Everything here is plain data describing host-contiguous `f32`/`f64`
//! buffers. No device pointer and no CUDA or C++ type appears in any of these
//! signatures -- that is the property that lets the rest of the workspace
//! depend on this crate without depending on a CUDA toolchain.

use std::fmt;

use thiserror::Error;

#[cfg(feature = "cuda")]
use crate::sys::ffi;

#[cfg(any(feature = "cuda", test))]
pub(crate) const ENGINE_FLAG_CUDA_GRAPHS: u64 = 1 << 0;
#[cfg(any(feature = "cuda", test))]
pub(crate) const ENGINE_FLAG_FAST_MATH: u64 = 1 << 1;

/// Strict floating-point dtype supported by the native engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaDtype {
    Float32,
    Float64,
}

impl CudaDtype {
    pub fn parse(name: &str) -> Result<Self, CudaError> {
        match name {
            "float32" => Ok(Self::Float32),
            "float64" => Ok(Self::Float64),
            _ => Err(CudaError::InvalidArgument(
                "dtype must be either 'float32' or 'float64'".to_owned(),
            )),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float64 => "float64",
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) const fn raw(self) -> i32 {
        match self {
            Self::Float32 => ffi::RH_CUDA_DTYPE_FLOAT32,
            Self::Float64 => ffi::RH_CUDA_DTYPE_FLOAT64,
        }
    }
}

impl fmt::Display for CudaDtype {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Scalar marker that prevents an `f32` engine from receiving `f64` buffers.
pub trait CudaScalar: Copy + 'static {
    const DTYPE: CudaDtype;
}

impl CudaScalar for f32 {
    const DTYPE: CudaDtype = CudaDtype::Float32;
}

impl CudaScalar for f64 {
    const DTYPE: CudaDtype = CudaDtype::Float64;
}

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("the native extension was built without CUDA support")]
    NotCompiled,
    #[error("invalid native CUDA request: {0}")]
    InvalidArgument(String),
    #[error("CUDA engine call failed with status {status}: {message}")]
    Status { status: i32, message: String },
}

/// Read-only C-row-major matrix borrowed from a Python-owned contiguous array.
#[derive(Debug, Clone, Copy)]
pub struct HostMatrix<'a, T: CudaScalar> {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) values: &'a [T],
    pub(crate) rows: usize,
    pub(crate) columns: usize,
}

impl<'a, T: CudaScalar> HostMatrix<'a, T> {
    pub fn new(values: &'a [T], rows: usize, columns: usize) -> Result<Self, CudaError> {
        let expected = rows.checked_mul(columns).ok_or_else(|| {
            CudaError::InvalidArgument("matrix dimensions overflow usize".to_owned())
        })?;
        if values.len() != expected {
            return Err(CudaError::InvalidArgument(format!(
                "matrix data has length {}, expected {expected} for shape ({rows}, {columns})",
                values.len()
            )));
        }
        Ok(Self {
            values,
            rows,
            columns,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn columns(self) -> usize {
        self.columns
    }
}

/// Read-only contiguous vector borrowed from a Python-owned array.
#[derive(Debug, Clone, Copy)]
pub struct HostVector<'a, T: CudaScalar> {
    pub(crate) values: &'a [T],
}

impl<'a, T: CudaScalar> HostVector<'a, T> {
    pub const fn new(values: &'a [T]) -> Self {
        Self { values }
    }

    pub const fn len(self) -> usize {
        self.values.len()
    }

    pub const fn is_empty(self) -> bool {
        self.values.is_empty()
    }
}

/// Caller-owned state used for safe host-to-device restoration.
#[derive(Debug, Clone, Copy)]
pub struct HostState<'a, T: CudaScalar> {
    pub coefficients: &'a [T],
    pub information: &'a [T],
    pub n_samples_seen: i64,
    pub batch_count: i64,
    pub previous_lambda: f64,
    pub weight_sum: f64,
}

/// Stateful values returned by a device-to-host `copy_state` operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateMetadata {
    pub n_samples_seen: i64,
    pub batch_count: i64,
    pub previous_lambda: f64,
    pub weight_sum: f64,
}

/// Immutable unpenalized configuration for one complete batch transition.
#[derive(Debug, Clone, Copy)]
pub struct UnpenalizedConfig {
    pub n_features_in: i64,
    pub fit_intercept: bool,
    pub tau: f64,
    pub bandwidth_scale: f64,
    pub max_iter: i64,
    pub tolerance: f64,
    pub ridge: f64,
}

/// One host-fed batch.  The `batch_weight` is supplied by Python validation,
/// not recomputed by the native layer, to preserve the frequency-weight
/// checkpoint contract exactly.
#[derive(Debug, Clone, Copy)]
pub struct HostBatch<'a, T: CudaScalar> {
    pub x_design: HostMatrix<'a, T>,
    pub y: HostVector<'a, T>,
    pub sample_weight: Option<HostVector<'a, T>>,
    pub batch_weight: f64,
}

/// One device-resident C-contiguous batch obtained from a DLPack producer.
/// Addresses are represented as integers so the descriptor can safely cross
/// PyO3's GIL-release boundary; the C ABI validates their CUDA allocation and
/// owning device before dereferencing them.
#[derive(Debug, Clone, Copy)]
pub struct DeviceBatch {
    pub x_address: usize,
    pub y_address: usize,
    pub sample_weight_address: Option<usize>,
    pub n_rows: usize,
    pub n_columns: usize,
    pub batch_weight: f64,
}

/// Diagnostics from one update.  Non-convergence is a valid outcome and does
/// not become an FFI error; the state transition is still committed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Diagnostics {
    pub iterations: i64,
    pub converged: bool,
    pub used_regularized_fallback: bool,
    pub objective: f64,
    pub lambda_value: f64,
    pub bandwidth: f64,
}

/// Runtime information copied from the CUDA C ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeInfo {
    pub runtime_version: i32,
    pub driver_version: i32,
    pub device_count: i32,
}

/// Optional CUDA execution tuning. Strict math and ordinary stream launches
/// remain the default so enabling an optimization always requires consent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EngineTuning {
    pub cuda_graphs: bool,
    pub fast_math: bool,
}

/// Engine-local capabilities and cumulative CUDA Graph activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineFeatures {
    pub requested: EngineTuning,
    pub enabled: EngineTuning,
    pub graph_captures: u64,
    pub graph_replays: u64,
    pub graph_fallbacks: u64,
}

#[cfg(any(feature = "cuda", test))]
pub(crate) const fn tuning_flags(tuning: EngineTuning) -> u64 {
    (if tuning.cuda_graphs {
        ENGINE_FLAG_CUDA_GRAPHS
    } else {
        0
    }) | (if tuning.fast_math {
        ENGINE_FLAG_FAST_MATH
    } else {
        0
    })
}

#[cfg(any(feature = "cuda", test))]
pub(crate) const fn tuning_from_flags(flags: u64) -> EngineTuning {
    EngineTuning {
        cuda_graphs: (flags & ENGINE_FLAG_CUDA_GRAPHS) != 0,
        fast_math: (flags & ENGINE_FLAG_FAST_MATH) != 0,
    }
}
