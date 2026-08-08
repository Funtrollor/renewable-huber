//! The safe engine handle.
//!
//! `CudaEngine` owns one opaque C handle. Every method validates its arguments
//! in Rust before crossing the ABI, so `sys` is only ever reached with values
//! the C side has already agreed to accept.

use crate::types::{
    CudaDtype, CudaError, CudaScalar, DeviceBatch, Diagnostics, EngineFeatures, EngineTuning,
    HostBatch, HostMatrix, HostState, StateMetadata, UnpenalizedConfig,
};
use crate::validation::validate_unpenalized_config;

#[cfg(feature = "cuda")]
use crate::runtime::{global_error_message, linked_abi_version_matches};
#[cfg(feature = "cuda")]
use crate::sys::ffi;
#[cfg(feature = "cuda")]
use crate::types::{tuning_flags, tuning_from_flags};
#[cfg(feature = "cuda")]
use crate::validation::checked_dimension;
#[cfg(feature = "cuda")]
use crate::ABI_VERSION;

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
        Self::create_with_tuning(dtype, n_parameters, device_id, EngineTuning::default())
    }

    pub fn create_with_tuning(
        dtype: CudaDtype,
        n_parameters: usize,
        device_id: i32,
        tuning: EngineTuning,
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
        if tuning.fast_math && dtype != CudaDtype::Float32 {
            return Err(CudaError::InvalidArgument(
                "fast_math is supported only by float32 CUDA engines".to_owned(),
            ));
        }

        #[cfg(feature = "cuda")]
        {
            linked_abi_version_matches()?;
            let abi_n_parameters = checked_dimension(n_parameters, "n_parameters")?;
            let options = ffi::RhCudaEngineOptions {
                abi_version: ABI_VERSION,
                struct_size: std::mem::size_of::<ffi::RhCudaEngineOptions>() as u32,
                dtype: dtype.raw(),
                device_id,
                n_parameters: abi_n_parameters,
                reserved0: tuning_flags(tuning),
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
            let _ = (dtype, n_parameters, device_id, tuning);
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

    pub fn features(&mut self) -> Result<EngineFeatures, CudaError> {
        #[cfg(feature = "cuda")]
        {
            let mut features = ffi::RhCudaEngineFeatures {
                abi_version: ABI_VERSION,
                struct_size: std::mem::size_of::<ffi::RhCudaEngineFeatures>() as u32,
                requested_flags: 0,
                enabled_flags: 0,
                graph_captures: 0,
                graph_replays: 0,
                graph_fallbacks: 0,
            };
            let status =
                unsafe { ffi::rh_cuda_engine_features(self.handle.as_ptr(), &mut features) };
            self.status(status)?;
            return Ok(EngineFeatures {
                requested: tuning_from_flags(features.requested_flags),
                enabled: tuning_from_flags(features.enabled_flags),
                graph_captures: features.graph_captures,
                graph_replays: features.graph_replays,
                graph_fallbacks: features.graph_fallbacks,
            });
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::NotCompiled)
        }
    }

    /// Raw CUDA stream handle passed to a DLPack producer. The producer uses
    /// it only to establish the protocol-defined dependency before returning
    /// the capsule; ownership remains with this engine.
    pub fn stream_handle(&mut self) -> Result<usize, CudaError> {
        #[cfg(feature = "cuda")]
        {
            let mut stream = 0usize;
            let status = unsafe { ffi::rh_cuda_engine_stream(self.handle.as_ptr(), &mut stream) };
            self.status(status)?;
            return Ok(stream);
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::NotCompiled)
        }
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
        self.validate_config(config)?;
        self.validate_batch(&batch, config)?;

        #[cfg(feature = "cuda")]
        {
            let n_rows = checked_dimension(batch.x_design.rows, "batch row count")?;
            let n_columns = checked_dimension(batch.x_design.columns, "batch column count")?;
            let raw_batch = ffi::RhCudaHostBatch {
                abi_version: ABI_VERSION,
                struct_size: std::mem::size_of::<ffi::RhCudaHostBatch>() as u32,
                x_design: batch.x_design.values.as_ptr() as *const std::ffi::c_void,
                y: batch.y.values.as_ptr() as *const std::ffi::c_void,
                sample_weight: batch
                    .sample_weight
                    .map_or(std::ptr::null(), |weight| weight.values.as_ptr())
                    as *const std::ffi::c_void,
                n_rows,
                n_columns,
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

    /// Execute one update and export its committed state with the same CUDA
    /// stream completion. This is the hot path used by the Python backend,
    /// which returns a portable state after every renewable batch.
    pub fn update_with_state<T: CudaScalar>(
        &mut self,
        batch: HostBatch<'_, T>,
        config: UnpenalizedConfig,
        coefficients: &mut [T],
        information: &mut [T],
    ) -> Result<(Diagnostics, StateMetadata), CudaError> {
        self.ensure_dtype::<T>()?;
        self.validate_config(config)?;
        self.validate_batch(&batch, config)?;
        self.validate_state_buffers(coefficients, information)?;

        #[cfg(feature = "cuda")]
        {
            let n_rows = checked_dimension(batch.x_design.rows, "batch row count")?;
            let n_columns = checked_dimension(batch.x_design.columns, "batch column count")?;
            let raw_batch = ffi::RhCudaHostBatch {
                abi_version: ABI_VERSION,
                struct_size: std::mem::size_of::<ffi::RhCudaHostBatch>() as u32,
                x_design: batch.x_design.values.as_ptr() as *const std::ffi::c_void,
                y: batch.y.values.as_ptr() as *const std::ffi::c_void,
                sample_weight: batch
                    .sample_weight
                    .map_or(std::ptr::null(), |weight| weight.values.as_ptr())
                    as *const std::ffi::c_void,
                n_rows,
                n_columns,
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
            let mut raw_state = ffi::RhCudaHostState {
                abi_version: ABI_VERSION,
                struct_size: std::mem::size_of::<ffi::RhCudaHostState>() as u32,
                coefficients: coefficients.as_mut_ptr() as *mut std::ffi::c_void,
                information: information.as_mut_ptr() as *mut std::ffi::c_void,
                n_samples_seen: 0,
                batch_count: 0,
                previous_lambda: 0.0,
                weight_sum: 0.0,
            };
            let status = unsafe {
                ffi::rh_cuda_engine_update_host_with_state(
                    self.handle.as_ptr(),
                    &raw_batch,
                    &raw_config,
                    &mut raw_diagnostics,
                    &mut raw_state,
                )
            };
            self.status(status)?;
            return Ok((
                Diagnostics {
                    iterations: raw_diagnostics.iterations,
                    converged: raw_diagnostics.converged != 0,
                    used_regularized_fallback: raw_diagnostics.used_regularized_fallback != 0,
                    objective: raw_diagnostics.objective,
                    lambda_value: raw_diagnostics.lambda_value,
                    bandwidth: raw_diagnostics.bandwidth,
                },
                StateMetadata {
                    n_samples_seen: raw_state.n_samples_seen,
                    batch_count: raw_state.batch_count,
                    previous_lambda: raw_state.previous_lambda,
                    weight_sum: raw_state.weight_sum,
                },
            ));
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (batch, config, coefficients, information);
            Err(CudaError::NotCompiled)
        }
    }

    /// Execute an update from DLPack-validated device pointers and export the
    /// committed state. The CUDA ABI performs a second allocation/device check
    /// before enqueueing any device-to-device copies.
    pub fn update_device_with_state<T: CudaScalar>(
        &mut self,
        batch: DeviceBatch,
        config: UnpenalizedConfig,
        coefficients: &mut [T],
        information: &mut [T],
    ) -> Result<(Diagnostics, StateMetadata), CudaError> {
        self.ensure_dtype::<T>()?;
        self.validate_config(config)?;
        self.validate_device_batch(batch, config)?;
        self.validate_state_buffers(coefficients, information)?;

        #[cfg(feature = "cuda")]
        {
            let raw_batch = ffi::RhCudaDeviceBatch {
                abi_version: ABI_VERSION,
                struct_size: std::mem::size_of::<ffi::RhCudaDeviceBatch>() as u32,
                x_design: batch.x_address as *const std::ffi::c_void,
                y: batch.y_address as *const std::ffi::c_void,
                sample_weight: batch
                    .sample_weight_address
                    .map_or(std::ptr::null(), |address| {
                        address as *const std::ffi::c_void
                    }),
                n_rows: checked_dimension(batch.n_rows, "batch row count")?,
                n_columns: checked_dimension(batch.n_columns, "batch column count")?,
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
            let mut raw_state = ffi::RhCudaHostState {
                abi_version: ABI_VERSION,
                struct_size: std::mem::size_of::<ffi::RhCudaHostState>() as u32,
                coefficients: coefficients.as_mut_ptr() as *mut std::ffi::c_void,
                information: information.as_mut_ptr() as *mut std::ffi::c_void,
                n_samples_seen: 0,
                batch_count: 0,
                previous_lambda: 0.0,
                weight_sum: 0.0,
            };
            let status = unsafe {
                ffi::rh_cuda_engine_update_device_with_state(
                    self.handle.as_ptr(),
                    &raw_batch,
                    &raw_config,
                    &mut raw_diagnostics,
                    &mut raw_state,
                )
            };
            self.status(status)?;
            return Ok((
                Diagnostics {
                    iterations: raw_diagnostics.iterations,
                    converged: raw_diagnostics.converged != 0,
                    used_regularized_fallback: raw_diagnostics.used_regularized_fallback != 0,
                    objective: raw_diagnostics.objective,
                    lambda_value: raw_diagnostics.lambda_value,
                    bandwidth: raw_diagnostics.bandwidth,
                },
                StateMetadata {
                    n_samples_seen: raw_state.n_samples_seen,
                    batch_count: raw_state.batch_count,
                    previous_lambda: raw_state.previous_lambda,
                    weight_sum: raw_state.weight_sum,
                },
            ));
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (batch, config, coefficients, information);
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
            let n_rows = checked_dimension(x_design.rows, "prediction row count")?;
            let n_columns = checked_dimension(x_design.columns, "prediction column count")?;
            let request = ffi::RhCudaHostPrediction {
                abi_version: ABI_VERSION,
                struct_size: std::mem::size_of::<ffi::RhCudaHostPrediction>() as u32,
                x_design: x_design.values.as_ptr() as *const std::ffi::c_void,
                prediction: prediction.as_mut_ptr() as *mut std::ffi::c_void,
                n_rows,
                n_columns,
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

    fn validate_batch<T: CudaScalar>(
        &self,
        batch: &HostBatch<'_, T>,
        config: UnpenalizedConfig,
    ) -> Result<(), CudaError> {
        let feature_columns = usize::try_from(config.n_features_in).map_err(|_| {
            CudaError::InvalidArgument("n_features_in is outside the host shape range".to_owned())
        })?;
        if batch.x_design.rows == 0
            || (batch.x_design.columns != self.n_parameters
                && batch.x_design.columns != feature_columns)
        {
            return Err(CudaError::InvalidArgument(
                "X must have at least one row and either n_features_in or n_parameters columns"
                    .to_owned(),
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

    fn validate_device_batch(
        &self,
        batch: DeviceBatch,
        config: UnpenalizedConfig,
    ) -> Result<(), CudaError> {
        let feature_columns = usize::try_from(config.n_features_in).map_err(|_| {
            CudaError::InvalidArgument("n_features_in is outside the host shape range".to_owned())
        })?;
        if batch.x_address == 0
            || batch.y_address == 0
            || batch.n_rows == 0
            || (batch.n_columns != self.n_parameters && batch.n_columns != feature_columns)
        {
            return Err(CudaError::InvalidArgument(
                "device X must be non-empty and have n_features_in or n_parameters columns"
                    .to_owned(),
            ));
        }
        if !batch.batch_weight.is_finite() || batch.batch_weight <= 0.0 {
            return Err(CudaError::InvalidArgument(
                "batch_weight must be finite and greater than zero".to_owned(),
            ));
        }
        Ok(())
    }

    fn validate_config(&self, config: UnpenalizedConfig) -> Result<(), CudaError> {
        validate_unpenalized_config(self.n_parameters, config)
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
