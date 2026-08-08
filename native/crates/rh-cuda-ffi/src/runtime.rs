//! Process-wide queries about the linked CUDA library and device.
//!
//! None of these need an engine, and all of them have to work on a build with
//! no CUDA support at all -- reporting absence is a valid answer.

use crate::types::{CudaError, RuntimeInfo};

#[cfg(feature = "cuda")]
use crate::sys::ffi;
#[cfg(feature = "cuda")]
use crate::ABI_VERSION;

/// ABI version reported by the CUDA library that is actually linked.
///
/// Returns `None` for a build without CUDA support, where no library exists to
/// disagree with [`crate::ABI_VERSION`]. Callers that surface a version to
/// users should prefer this over the compile-time constant: the constant can only
/// ever agree with itself, while this can catch a stale `renewable_huber_cuda`
/// binary left behind by an earlier build.
pub fn linked_abi_version() -> Option<u32> {
    #[cfg(feature = "cuda")]
    {
        use std::sync::OnceLock;

        static LINKED: OnceLock<u32> = OnceLock::new();
        // rh_cuda_abi_version returns a constant and touches no CUDA state, so
        // this is safe to call before any device or engine exists.
        Some(*LINKED.get_or_init(|| unsafe { ffi::rh_cuda_abi_version() }))
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

/// Confirm the linked library speaks the ABI version this crate was built
/// against, before the first engine exists.
///
/// The public structs carry `abi_version` too, but by the time the library
/// reads one the caller has already handed it a record laid out to whatever
/// contract this build believes in.
#[cfg(feature = "cuda")]
pub(crate) fn linked_abi_version_matches() -> Result<(), CudaError> {
    match linked_abi_version() {
        Some(linked) if linked != ABI_VERSION => Err(CudaError::InvalidArgument(format!(
            "the linked CUDA library reports ABI version {linked}, but this build expects \
             {ABI_VERSION}; rebuild the native CUDA extension"
        ))),
        _ => Ok(()),
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
pub(crate) fn global_error_message(fallback: &str) -> String {
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
