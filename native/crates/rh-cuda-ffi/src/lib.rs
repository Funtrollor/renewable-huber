//! Safe ownership-oriented wrapper around `native/cuda/include/rh_cuda.h`.
//!
//! The public types in this crate intentionally describe only host-contiguous
//! `f32`/`f64` buffers.  Device pointers and CUDA/C++ implementation details
//! never escape this boundary; the opaque C handle owns every device resource.

use std::fmt;

use thiserror::Error;

pub const ABI_VERSION: u32 = 1;

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
    const fn raw(self) -> i32 {
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
    values: &'a [T],
    rows: usize,
    columns: usize,
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
    values: &'a [T],
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

/// Opaque, single-device CUDA engine. Sequential ownership may move between
/// host threads because every C ABI call reselects the engine device.
pub struct CudaEngine {
    dtype: CudaDtype,
    n_parameters: usize,
    device_id: i32,
    #[cfg(feature = "cuda")]
    handle: std::ptr::NonNull<ffi::RhCudaEngine>,
}

// The C ABI creates one engine for one CUDA device. Every operation touching
// the opaque handle requires `&mut self`; shared references expose only
// immutable metadata. It is therefore safe to move the owner and to place it
// behind Rust/PyO3 synchronization, while Rust's mutable borrow still rejects
// concurrent CUDA calls.
unsafe impl Send for CudaEngine {}
unsafe impl Sync for CudaEngine {}

impl CudaEngine {
    pub fn create(
        dtype: CudaDtype,
        n_parameters: usize,
        device_id: i32,
    ) -> Result<Self, CudaError> {
        if n_parameters == 0 {
            return Err(CudaError::InvalidArgument(
                "n_parameters must be greater than zero".to_owned(),
            ));
        }
        if device_id < 0 {
            return Err(CudaError::InvalidArgument(
                "device_id must be non-negative".to_owned(),
            ));
        }

        #[cfg(feature = "cuda")]
        {
            let options = ffi::RhCudaEngineOptions {
                abi_version: ABI_VERSION,
                struct_size: std::mem::size_of::<ffi::RhCudaEngineOptions>() as u32,
                dtype: dtype.raw(),
                device_id,
                n_parameters: n_parameters as i64,
                reserved0: 0,
            };
            let mut raw = std::ptr::null_mut();
            let status = unsafe { ffi::rh_cuda_engine_create(&options, &mut raw) };
            if status != ffi::RH_CUDA_STATUS_SUCCESS {
                return Err(CudaError::Status {
                    status,
                    message: global_error_message("CUDA engine creation failed"),
                });
            }
            let handle = std::ptr::NonNull::new(raw).ok_or_else(|| CudaError::Status {
                status: ffi::RH_CUDA_STATUS_INTERNAL_ERROR,
                message: "CUDA ABI returned a null engine on success".to_owned(),
            })?;
            return Ok(Self {
                dtype,
                n_parameters,
                device_id,
                handle,
            });
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (dtype, n_parameters, device_id);
            Err(CudaError::NotCompiled)
        }
    }

    pub const fn dtype(&self) -> CudaDtype {
        self.dtype
    }

    pub const fn n_parameters(&self) -> usize {
        self.n_parameters
    }

    pub const fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn restore<T: CudaScalar>(&mut self, state: HostState<'_, T>) -> Result<(), CudaError> {
        self.ensure_dtype::<T>()?;
        self.validate_state(&state)?;

        #[cfg(feature = "cuda")]
        {
            let request = ffi::RhCudaHostStateView {
                abi_version: ABI_VERSION,
                struct_size: std::mem::size_of::<ffi::RhCudaHostStateView>() as u32,
                coefficients: state.coefficients.as_ptr() as *const std::ffi::c_void,
                information: state.information.as_ptr() as *const std::ffi::c_void,
                n_samples_seen: state.n_samples_seen,
                batch_count: state.batch_count,
                previous_lambda: state.previous_lambda,
                weight_sum: state.weight_sum,
            };
            let status = unsafe { ffi::rh_cuda_engine_restore(self.handle.as_ptr(), &request) };
            return self.status(status);
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = state;
            Err(CudaError::NotCompiled)
        }
    }

    pub fn copy_state<T: CudaScalar>(
        &mut self,
        coefficients: &mut [T],
        information: &mut [T],
    ) -> Result<StateMetadata, CudaError> {
        self.ensure_dtype::<T>()?;
        self.validate_state_buffers(coefficients, information)?;

        #[cfg(feature = "cuda")]
        {
            let mut request = ffi::RhCudaHostState {
                abi_version: ABI_VERSION,
                struct_size: std::mem::size_of::<ffi::RhCudaHostState>() as u32,
                coefficients: coefficients.as_mut_ptr() as *mut std::ffi::c_void,
                information: information.as_mut_ptr() as *mut std::ffi::c_void,
                n_samples_seen: 0,
                batch_count: 0,
                previous_lambda: 0.0,
                weight_sum: 0.0,
            };
            let status =
                unsafe { ffi::rh_cuda_engine_copy_state(self.handle.as_ptr(), &mut request) };
            self.status(status)?;
            return Ok(StateMetadata {
                n_samples_seen: request.n_samples_seen,
                batch_count: request.batch_count,
                previous_lambda: request.previous_lambda,
                weight_sum: request.weight_sum,
            });
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (coefficients, information);
            Err(CudaError::NotCompiled)
        }
    }

    pub fn update<T: CudaScalar>(
        &mut self,
        batch: HostBatch<'_, T>,
        config: UnpenalizedConfig,
    ) -> Result<Diagnostics, CudaError> {
        self.ensure_dtype::<T>()?;
        self.validate_batch(&batch)?;
        self.validate_config(config)?;

        #[cfg(feature = "cuda")]
        {
            let raw_batch = ffi::RhCudaHostBatch {
                abi_version: ABI_VERSION,
                struct_size: std::mem::size_of::<ffi::RhCudaHostBatch>() as u32,
                x_design: batch.x_design.values.as_ptr() as *const std::ffi::c_void,
                y: batch.y.values.as_ptr() as *const std::ffi::c_void,
                sample_weight: batch
                    .sample_weight
                    .map_or(std::ptr::null(), |weight| weight.values.as_ptr())
                    as *const std::ffi::c_void,
                n_rows: batch.x_design.rows as i64,
                n_columns: batch.x_design.columns as i64,
                batch_weight: batch.batch_weight,
            };
            let raw_config = ffi::RhCudaUnpenalizedConfig {
                abi_version: ABI_VERSION,
                struct_size: std::mem::size_of::<ffi::RhCudaUnpenalizedConfig>() as u32,
                n_features_in: config.n_features_in,
                max_iter: config.max_iter,
                tau: config.tau,
                bandwidth_scale: config.bandwidth_scale,
                tolerance: config.tolerance,
                ridge: config.ridge,
            };
            let mut raw_diagnostics = ffi::RhCudaDiagnostics {
                abi_version: ABI_VERSION,
                struct_size: std::mem::size_of::<ffi::RhCudaDiagnostics>() as u32,
                iterations: 0,
                converged: 0,
                used_regularized_fallback: 0,
                objective: 0.0,
                lambda_value: 0.0,
                bandwidth: 0.0,
            };
            let status = unsafe {
                ffi::rh_cuda_engine_update_host(
                    self.handle.as_ptr(),
                    &raw_batch,
                    &raw_config,
                    &mut raw_diagnostics,
                )
            };
            self.status(status)?;
            return Ok(Diagnostics {
                iterations: raw_diagnostics.iterations,
                converged: raw_diagnostics.converged != 0,
                used_regularized_fallback: raw_diagnostics.used_regularized_fallback != 0,
                objective: raw_diagnostics.objective,
                lambda_value: raw_diagnostics.lambda_value,
                bandwidth: raw_diagnostics.bandwidth,
            });
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (batch, config);
            Err(CudaError::NotCompiled)
        }
    }

    pub fn predict<T: CudaScalar>(
        &mut self,
        x_design: HostMatrix<'_, T>,
        prediction: &mut [T],
    ) -> Result<(), CudaError> {
        self.ensure_dtype::<T>()?;
        if x_design.columns != self.n_parameters || prediction.len() != x_design.rows {
            return Err(CudaError::InvalidArgument(
                "prediction shape does not match engine parameters".to_owned(),
            ));
        }

        #[cfg(feature = "cuda")]
        {
            let request = ffi::RhCudaHostPrediction {
                abi_version: ABI_VERSION,
                struct_size: std::mem::size_of::<ffi::RhCudaHostPrediction>() as u32,
                x_design: x_design.values.as_ptr() as *const std::ffi::c_void,
                prediction: prediction.as_mut_ptr() as *mut std::ffi::c_void,
                n_rows: x_design.rows as i64,
                n_columns: x_design.columns as i64,
            };
            let status =
                unsafe { ffi::rh_cuda_engine_predict_host(self.handle.as_ptr(), &request) };
            return self.status(status);
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (x_design, prediction);
            Err(CudaError::NotCompiled)
        }
    }

    pub fn synchronize(&mut self) -> Result<(), CudaError> {
        #[cfg(feature = "cuda")]
        {
            let status = unsafe { ffi::rh_cuda_engine_synchronize(self.handle.as_ptr()) };
            return self.status(status);
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::NotCompiled)
        }
    }

    fn ensure_dtype<T: CudaScalar>(&self) -> Result<(), CudaError> {
        if T::DTYPE == self.dtype {
            Ok(())
        } else {
            Err(CudaError::InvalidArgument(format!(
                "engine dtype is {}, but received {} buffers",
                self.dtype,
                T::DTYPE
            )))
        }
    }

    fn validate_state<T: CudaScalar>(&self, state: &HostState<'_, T>) -> Result<(), CudaError> {
        self.validate_state_buffers(state.coefficients, state.information)?;
        if state.n_samples_seen < 0 || state.batch_count < 0 {
            return Err(CudaError::InvalidArgument(
                "state counters must be non-negative".to_owned(),
            ));
        }
        if !state.previous_lambda.is_finite() || state.previous_lambda < 0.0 {
            return Err(CudaError::InvalidArgument(
                "previous_lambda must be finite and non-negative".to_owned(),
            ));
        }
        if !state.weight_sum.is_finite() || state.weight_sum < 0.0 {
            return Err(CudaError::InvalidArgument(
                "weight_sum must be finite and non-negative".to_owned(),
            ));
        }
        Ok(())
    }

    fn validate_state_buffers<T>(
        &self,
        coefficients: &[T],
        information: &[T],
    ) -> Result<(), CudaError> {
        let information_len = self
            .n_parameters
            .checked_mul(self.n_parameters)
            .ok_or_else(|| CudaError::InvalidArgument("n_parameters overflow".to_owned()))?;
        if coefficients.len() != self.n_parameters || information.len() != information_len {
            return Err(CudaError::InvalidArgument(format!(
                "state shapes must be ({},) and ({}, {})",
                self.n_parameters, self.n_parameters, self.n_parameters
            )));
        }
        Ok(())
    }

    fn validate_batch<T: CudaScalar>(&self, batch: &HostBatch<'_, T>) -> Result<(), CudaError> {
        if batch.x_design.rows == 0 || batch.x_design.columns != self.n_parameters {
            return Err(CudaError::InvalidArgument(
                "X_design must have at least one row and exactly n_parameters columns".to_owned(),
            ));
        }
        if batch.y.len() != batch.x_design.rows {
            return Err(CudaError::InvalidArgument(
                "X_design and y must have the same number of rows".to_owned(),
            ));
        }
        if let Some(weights) = batch.sample_weight {
            if weights.len() != batch.x_design.rows {
                return Err(CudaError::InvalidArgument(
                    "sample_weight and X_design must have the same number of rows".to_owned(),
                ));
            }
        }
        if !batch.batch_weight.is_finite() || batch.batch_weight <= 0.0 {
            return Err(CudaError::InvalidArgument(
                "batch_weight must be finite and greater than zero".to_owned(),
            ));
        }
        Ok(())
    }

    fn validate_config(&self, config: UnpenalizedConfig) -> Result<(), CudaError> {
        if config.n_features_in < 1 || config.n_features_in as usize > self.n_parameters {
            return Err(CudaError::InvalidArgument(
                "n_features_in must fit the engine parameter dimension".to_owned(),
            ));
        }
        if config.max_iter < 1
            || !config.tau.is_finite()
            || config.tau <= 0.0
            || !config.bandwidth_scale.is_finite()
            || config.bandwidth_scale <= 0.0
            || !config.tolerance.is_finite()
            || config.tolerance <= 0.0
            || !config.ridge.is_finite()
            || config.ridge < 0.0
        {
            return Err(CudaError::InvalidArgument(
                "received invalid unpenalized solver configuration".to_owned(),
            ));
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn status(&self, status: i32) -> Result<(), CudaError> {
        if status == ffi::RH_CUDA_STATUS_SUCCESS {
            return Ok(());
        }
        let pointer = unsafe { ffi::rh_cuda_engine_last_error(self.handle.as_ptr()) };
        let message = if pointer.is_null() {
            "CUDA ABI returned no error detail".to_owned()
        } else {
            unsafe { std::ffi::CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        };
        Err(CudaError::Status { status, message })
    }
}

impl Drop for CudaEngine {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            // Destruction is best effort: Rust cannot report an FFI failure
            // from Drop, and the C ABI guarantees it catches C++ exceptions.
            let _ = ffi::rh_cuda_engine_destroy(self.handle.as_ptr());
        }
    }
}

/// Return whether the linked CUDA engine can initialize a device right now.
pub fn is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        let mut available = 0;
        let status = unsafe { ffi::rh_cuda_is_available(&mut available) };
        return status == ffi::RH_CUDA_STATUS_SUCCESS && available != 0;
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

pub fn device_count() -> Result<i32, CudaError> {
    #[cfg(feature = "cuda")]
    {
        let mut count = 0;
        let status = unsafe { ffi::rh_cuda_device_count(&mut count) };
        if status == ffi::RH_CUDA_STATUS_SUCCESS {
            return Ok(count);
        }
        return Err(CudaError::Status {
            status,
            message: global_error_message("unable to query CUDA device count"),
        });
    }

    #[cfg(not(feature = "cuda"))]
    {
        Err(CudaError::NotCompiled)
    }
}

pub fn runtime_info() -> Result<RuntimeInfo, CudaError> {
    #[cfg(feature = "cuda")]
    {
        let mut info = ffi::RhCudaRuntimeInfo {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<ffi::RhCudaRuntimeInfo>() as u32,
            runtime_version: 0,
            driver_version: 0,
            device_count: 0,
            reserved0: 0,
        };
        let status = unsafe { ffi::rh_cuda_runtime_info(&mut info) };
        if status == ffi::RH_CUDA_STATUS_SUCCESS {
            return Ok(RuntimeInfo {
                runtime_version: info.runtime_version,
                driver_version: info.driver_version,
                device_count: info.device_count,
            });
        }
        return Err(CudaError::Status {
            status,
            message: global_error_message("unable to query CUDA runtime information"),
        });
    }

    #[cfg(not(feature = "cuda"))]
    {
        Err(CudaError::NotCompiled)
    }
}

#[cfg(feature = "cuda")]
fn global_error_message(fallback: &str) -> String {
    let pointer = unsafe { ffi::rh_cuda_last_error() };
    if pointer.is_null() {
        return fallback.to_owned();
    }
    let message = unsafe { std::ffi::CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned();
    if message.is_empty() {
        fallback.to_owned()
    } else {
        message
    }
}

#[cfg(feature = "cuda")]
mod ffi {
    use std::ffi::{c_char, c_void};

    pub const RH_CUDA_STATUS_SUCCESS: i32 = 0;
    pub const RH_CUDA_STATUS_INTERNAL_ERROR: i32 = 8;
    pub const RH_CUDA_DTYPE_FLOAT32: i32 = 1;
    pub const RH_CUDA_DTYPE_FLOAT64: i32 = 2;

    #[repr(C)]
    pub struct RhCudaEngine {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct RhCudaEngineOptions {
        pub abi_version: u32,
        pub struct_size: u32,
        pub dtype: i32,
        pub device_id: i32,
        pub n_parameters: i64,
        pub reserved0: u64,
    }

    #[repr(C)]
    pub struct RhCudaHostStateView {
        pub abi_version: u32,
        pub struct_size: u32,
        pub coefficients: *const c_void,
        pub information: *const c_void,
        pub n_samples_seen: i64,
        pub batch_count: i64,
        pub previous_lambda: f64,
        pub weight_sum: f64,
    }

    #[repr(C)]
    pub struct RhCudaHostState {
        pub abi_version: u32,
        pub struct_size: u32,
        pub coefficients: *mut c_void,
        pub information: *mut c_void,
        pub n_samples_seen: i64,
        pub batch_count: i64,
        pub previous_lambda: f64,
        pub weight_sum: f64,
    }

    #[repr(C)]
    pub struct RhCudaUnpenalizedConfig {
        pub abi_version: u32,
        pub struct_size: u32,
        pub n_features_in: i64,
        pub max_iter: i64,
        pub tau: f64,
        pub bandwidth_scale: f64,
        pub tolerance: f64,
        pub ridge: f64,
    }

    #[repr(C)]
    pub struct RhCudaHostBatch {
        pub abi_version: u32,
        pub struct_size: u32,
        pub x_design: *const c_void,
        pub y: *const c_void,
        pub sample_weight: *const c_void,
        pub n_rows: i64,
        pub n_columns: i64,
        pub batch_weight: f64,
    }

    #[repr(C)]
    pub struct RhCudaHostPrediction {
        pub abi_version: u32,
        pub struct_size: u32,
        pub x_design: *const c_void,
        pub prediction: *mut c_void,
        pub n_rows: i64,
        pub n_columns: i64,
    }

    #[repr(C)]
    pub struct RhCudaDiagnostics {
        pub abi_version: u32,
        pub struct_size: u32,
        pub iterations: i64,
        pub converged: i32,
        pub used_regularized_fallback: i32,
        pub objective: f64,
        pub lambda_value: f64,
        pub bandwidth: f64,
    }

    #[repr(C)]
    pub struct RhCudaRuntimeInfo {
        pub abi_version: u32,
        pub struct_size: u32,
        pub runtime_version: i32,
        pub driver_version: i32,
        pub device_count: i32,
        pub reserved0: i32,
    }

    extern "C" {
        pub fn rh_cuda_last_error() -> *const c_char;
        pub fn rh_cuda_is_available(available: *mut i32) -> i32;
        pub fn rh_cuda_device_count(count: *mut i32) -> i32;
        pub fn rh_cuda_runtime_info(info: *mut RhCudaRuntimeInfo) -> i32;
        pub fn rh_cuda_engine_create(
            options: *const RhCudaEngineOptions,
            out_engine: *mut *mut RhCudaEngine,
        ) -> i32;
        pub fn rh_cuda_engine_destroy(engine: *mut RhCudaEngine) -> i32;
        pub fn rh_cuda_engine_restore(
            engine: *mut RhCudaEngine,
            state: *const RhCudaHostStateView,
        ) -> i32;
        pub fn rh_cuda_engine_copy_state(
            engine: *mut RhCudaEngine,
            state: *mut RhCudaHostState,
        ) -> i32;
        pub fn rh_cuda_engine_update_host(
            engine: *mut RhCudaEngine,
            batch: *const RhCudaHostBatch,
            config: *const RhCudaUnpenalizedConfig,
            diagnostics: *mut RhCudaDiagnostics,
        ) -> i32;
        pub fn rh_cuda_engine_predict_host(
            engine: *mut RhCudaEngine,
            request: *const RhCudaHostPrediction,
        ) -> i32;
        pub fn rh_cuda_engine_synchronize(engine: *mut RhCudaEngine) -> i32;
        pub fn rh_cuda_engine_last_error(engine: *const RhCudaEngine) -> *const c_char;
    }
}
