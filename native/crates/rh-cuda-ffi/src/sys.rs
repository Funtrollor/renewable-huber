//! Raw C ABI: the `#[repr(C)]` records and the `extern "C"` declarations.
//!
//! This is the only module that names the C contract directly, and the only
//! one whose correctness cannot be checked by the Rust type system. It is kept
//! apart from the safe wrapper so that reviewing "does this match
//! native/cuda/include/rh_cuda.h" is a self-contained task.
//!
//! Hand-written rather than generated: bindgen would need libclang and the CUDA
//! headers wherever it runs, would regenerate only the part that is already
//! correct, and would cover none of the status codes, dtype codes, flag bits or
//! Python dict keys that the contract manifest also governs.

/// Plain-data half of the C ABI: constants and `#[repr(C)]` records.
///
/// Deliberately *not* gated on the `cuda` feature. Struct layout is a property
/// of the contract, not of whether a CUDA toolchain happened to be present at
/// build time, so compiling this unconditionally lets `cargo test -p
/// rh-cuda-ffi` verify it on a runner with no GPU and no CUDA Toolkit. The
/// numbers mirror native/contracts/rh_cuda_contract.json; `mod abi_layout`
/// below is what actually holds them to it.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) mod abi {
    use std::ffi::c_void;

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
    pub struct RhCudaDeviceBatch {
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

    #[repr(C)]
    pub struct RhCudaEngineFeatures {
        pub abi_version: u32,
        pub struct_size: u32,
        pub requested_flags: u64,
        pub enabled_flags: u64,
        pub graph_captures: u64,
        pub graph_replays: u64,
        pub graph_fallbacks: u64,
    }
}

/// Linkage half of the C ABI. Only this part needs the built CUDA library.
#[cfg(feature = "cuda")]
pub(crate) mod ffi {
    pub use super::abi::*;

    use std::ffi::c_char;

    extern "C" {
        /// Reports the ABI version of the library that is actually linked.
        /// Source-level mirrors cannot catch a stale binary; this can.
        pub fn rh_cuda_abi_version() -> u32;
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
        pub fn rh_cuda_engine_update_host_with_state(
            engine: *mut RhCudaEngine,
            batch: *const RhCudaHostBatch,
            config: *const RhCudaUnpenalizedConfig,
            diagnostics: *mut RhCudaDiagnostics,
            state: *mut RhCudaHostState,
        ) -> i32;
        pub fn rh_cuda_engine_stream(engine: *mut RhCudaEngine, stream: *mut usize) -> i32;
        pub fn rh_cuda_engine_update_device_with_state(
            engine: *mut RhCudaEngine,
            batch: *const RhCudaDeviceBatch,
            config: *const RhCudaUnpenalizedConfig,
            diagnostics: *mut RhCudaDiagnostics,
            state: *mut RhCudaHostState,
        ) -> i32;
        pub fn rh_cuda_engine_predict_host(
            engine: *mut RhCudaEngine,
            request: *const RhCudaHostPrediction,
        ) -> i32;
        pub fn rh_cuda_engine_synchronize(engine: *mut RhCudaEngine) -> i32;
        pub fn rh_cuda_engine_features(
            engine: *mut RhCudaEngine,
            features: *mut RhCudaEngineFeatures,
        ) -> i32;
        pub fn rh_cuda_engine_last_error(engine: *const RhCudaEngine) -> *const c_char;
    }
}
/// Compile-and-run mirror of native/contracts/rh_cuda_contract.json.
///
/// Not gated on the `cuda` feature, so the workspace's ordinary
/// `cargo test -p rh-cuda-ffi` covers it with no CUDA toolchain. The C++ side
/// asserts the same numbers against rh_cuda.h in
/// native/cuda/src/abi_contract.cpp; between them, a layout change has to break
/// one of the two builds before it can reach a caller.
#[cfg(test)]
mod abi_layout {
    use super::abi::*;
    use core::mem::{offset_of, size_of};

    #[test]
    fn struct_layout_matches_the_contract_manifest() {
        assert_eq!(size_of::<RhCudaEngineOptions>(), 32);
        assert_eq!(offset_of!(RhCudaEngineOptions, abi_version), 0);
        assert_eq!(offset_of!(RhCudaEngineOptions, struct_size), 4);
        assert_eq!(offset_of!(RhCudaEngineOptions, dtype), 8);
        assert_eq!(offset_of!(RhCudaEngineOptions, device_id), 12);
        assert_eq!(offset_of!(RhCudaEngineOptions, n_parameters), 16);
        assert_eq!(offset_of!(RhCudaEngineOptions, reserved0), 24);

        assert_eq!(size_of::<RhCudaHostStateView>(), 56);
        assert_eq!(offset_of!(RhCudaHostStateView, abi_version), 0);
        assert_eq!(offset_of!(RhCudaHostStateView, struct_size), 4);
        assert_eq!(offset_of!(RhCudaHostStateView, coefficients), 8);
        assert_eq!(offset_of!(RhCudaHostStateView, information), 16);
        assert_eq!(offset_of!(RhCudaHostStateView, n_samples_seen), 24);
        assert_eq!(offset_of!(RhCudaHostStateView, batch_count), 32);
        assert_eq!(offset_of!(RhCudaHostStateView, previous_lambda), 40);
        assert_eq!(offset_of!(RhCudaHostStateView, weight_sum), 48);

        assert_eq!(size_of::<RhCudaHostState>(), 56);
        assert_eq!(offset_of!(RhCudaHostState, abi_version), 0);
        assert_eq!(offset_of!(RhCudaHostState, struct_size), 4);
        assert_eq!(offset_of!(RhCudaHostState, coefficients), 8);
        assert_eq!(offset_of!(RhCudaHostState, information), 16);
        assert_eq!(offset_of!(RhCudaHostState, n_samples_seen), 24);
        assert_eq!(offset_of!(RhCudaHostState, batch_count), 32);
        assert_eq!(offset_of!(RhCudaHostState, previous_lambda), 40);
        assert_eq!(offset_of!(RhCudaHostState, weight_sum), 48);

        assert_eq!(size_of::<RhCudaUnpenalizedConfig>(), 56);
        assert_eq!(offset_of!(RhCudaUnpenalizedConfig, abi_version), 0);
        assert_eq!(offset_of!(RhCudaUnpenalizedConfig, struct_size), 4);
        assert_eq!(offset_of!(RhCudaUnpenalizedConfig, n_features_in), 8);
        assert_eq!(offset_of!(RhCudaUnpenalizedConfig, max_iter), 16);
        assert_eq!(offset_of!(RhCudaUnpenalizedConfig, tau), 24);
        assert_eq!(offset_of!(RhCudaUnpenalizedConfig, bandwidth_scale), 32);
        assert_eq!(offset_of!(RhCudaUnpenalizedConfig, tolerance), 40);
        assert_eq!(offset_of!(RhCudaUnpenalizedConfig, ridge), 48);

        assert_eq!(size_of::<RhCudaHostBatch>(), 56);
        assert_eq!(offset_of!(RhCudaHostBatch, abi_version), 0);
        assert_eq!(offset_of!(RhCudaHostBatch, struct_size), 4);
        assert_eq!(offset_of!(RhCudaHostBatch, x_design), 8);
        assert_eq!(offset_of!(RhCudaHostBatch, y), 16);
        assert_eq!(offset_of!(RhCudaHostBatch, sample_weight), 24);
        assert_eq!(offset_of!(RhCudaHostBatch, n_rows), 32);
        assert_eq!(offset_of!(RhCudaHostBatch, n_columns), 40);
        assert_eq!(offset_of!(RhCudaHostBatch, batch_weight), 48);

        assert_eq!(size_of::<RhCudaDeviceBatch>(), 56);
        assert_eq!(offset_of!(RhCudaDeviceBatch, abi_version), 0);
        assert_eq!(offset_of!(RhCudaDeviceBatch, struct_size), 4);
        assert_eq!(offset_of!(RhCudaDeviceBatch, x_design), 8);
        assert_eq!(offset_of!(RhCudaDeviceBatch, y), 16);
        assert_eq!(offset_of!(RhCudaDeviceBatch, sample_weight), 24);
        assert_eq!(offset_of!(RhCudaDeviceBatch, n_rows), 32);
        assert_eq!(offset_of!(RhCudaDeviceBatch, n_columns), 40);
        assert_eq!(offset_of!(RhCudaDeviceBatch, batch_weight), 48);

        assert_eq!(size_of::<RhCudaHostPrediction>(), 40);
        assert_eq!(offset_of!(RhCudaHostPrediction, abi_version), 0);
        assert_eq!(offset_of!(RhCudaHostPrediction, struct_size), 4);
        assert_eq!(offset_of!(RhCudaHostPrediction, x_design), 8);
        assert_eq!(offset_of!(RhCudaHostPrediction, prediction), 16);
        assert_eq!(offset_of!(RhCudaHostPrediction, n_rows), 24);
        assert_eq!(offset_of!(RhCudaHostPrediction, n_columns), 32);

        assert_eq!(size_of::<RhCudaDiagnostics>(), 48);
        assert_eq!(offset_of!(RhCudaDiagnostics, abi_version), 0);
        assert_eq!(offset_of!(RhCudaDiagnostics, struct_size), 4);
        assert_eq!(offset_of!(RhCudaDiagnostics, iterations), 8);
        assert_eq!(offset_of!(RhCudaDiagnostics, converged), 16);
        assert_eq!(offset_of!(RhCudaDiagnostics, used_regularized_fallback), 20);
        assert_eq!(offset_of!(RhCudaDiagnostics, objective), 24);
        assert_eq!(offset_of!(RhCudaDiagnostics, lambda_value), 32);
        assert_eq!(offset_of!(RhCudaDiagnostics, bandwidth), 40);

        assert_eq!(size_of::<RhCudaRuntimeInfo>(), 24);
        assert_eq!(offset_of!(RhCudaRuntimeInfo, abi_version), 0);
        assert_eq!(offset_of!(RhCudaRuntimeInfo, struct_size), 4);
        assert_eq!(offset_of!(RhCudaRuntimeInfo, runtime_version), 8);
        assert_eq!(offset_of!(RhCudaRuntimeInfo, driver_version), 12);
        assert_eq!(offset_of!(RhCudaRuntimeInfo, device_count), 16);
        assert_eq!(offset_of!(RhCudaRuntimeInfo, reserved0), 20);

        assert_eq!(size_of::<RhCudaEngineFeatures>(), 48);
        assert_eq!(offset_of!(RhCudaEngineFeatures, abi_version), 0);
        assert_eq!(offset_of!(RhCudaEngineFeatures, struct_size), 4);
        assert_eq!(offset_of!(RhCudaEngineFeatures, requested_flags), 8);
        assert_eq!(offset_of!(RhCudaEngineFeatures, enabled_flags), 16);
        assert_eq!(offset_of!(RhCudaEngineFeatures, graph_captures), 24);
        assert_eq!(offset_of!(RhCudaEngineFeatures, graph_replays), 32);
        assert_eq!(offset_of!(RhCudaEngineFeatures, graph_fallbacks), 40);
    }

    #[test]
    fn constants_match_the_contract_manifest() {
        assert_eq!(crate::ABI_VERSION, 1);
        assert_eq!(size_of::<*const core::ffi::c_void>(), 8);
        assert_eq!(RH_CUDA_STATUS_SUCCESS, 0);
        assert_eq!(RH_CUDA_STATUS_INTERNAL_ERROR, 8);
        assert_eq!(RH_CUDA_DTYPE_FLOAT32, 1);
        assert_eq!(RH_CUDA_DTYPE_FLOAT64, 2);
        assert_eq!(crate::types::ENGINE_FLAG_CUDA_GRAPHS, 1);
        assert_eq!(crate::types::ENGINE_FLAG_FAST_MATH, 2);
    }
}
