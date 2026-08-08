//! Legacy DLPack capsule consumption.
//!
//! Isolated deliberately: this is the only place in the extension that
//! dereferences a pointer handed over by another framework, and the only place
//! whose correctness depends on a protocol rather than on Rust's type system.
//! Keeping it in one file makes "is the unsafe here sound" a self-contained
//! review rather than a search through the PyO3 bindings.
//!
//! The contract enforced here: the producer's tensor must be CUDA-resident on
//! the engine's device, exactly the engine dtype, and C-contiguous. The
//! capsule is consumed exactly once -- renamed to `used_dltensor` so the
//! producer will not free it -- and the producer's own deleter runs when this
//! wrapper drops, after the native call has synchronized.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use rh_cuda_ffi::CudaDtype;

const DL_DEVICE_CUDA: i32 = 2;
const DL_DTYPE_FLOAT: u8 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct DlDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DlDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DlTensor {
    data: *mut std::ffi::c_void,
    device: DlDevice,
    ndim: i32,
    dtype: DlDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DlManagedTensor {
    dl_tensor: DlTensor,
    manager_ctx: *mut std::ffi::c_void,
    deleter: Option<unsafe extern "C" fn(*mut DlManagedTensor)>,
}

/// Owns one consumed legacy DLPack capsule until the native CUDA call returns.
pub(crate) struct DlpackTensor {
    _capsule: Py<PyAny>,
    managed: *mut DlManagedTensor,
    pub(crate) address: usize,
    pub(crate) shape: Vec<usize>,
}

impl DlpackTensor {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn consume(
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
        stream: usize,
        expected_dtype: CudaDtype,
        expected_device: i32,
        expected_ndim: usize,
        name: &str,
    ) -> PyResult<Self> {
        let device: (i32, i32) = value
            .call_method0("__dlpack_device__")
            .and_then(|result| result.extract())
            .map_err(|_| PyTypeError::new_err(format!("{name} must implement CUDA DLPack")))?;
        if device != (DL_DEVICE_CUDA, expected_device) {
            return Err(PyValueError::new_err(format!(
                "{name} must be on CUDA device {expected_device}"
            )));
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("stream", stream)?;
        let capsule = value.call_method("__dlpack__", (), Some(&kwargs))?;
        let managed = unsafe {
            pyo3::ffi::PyCapsule_GetPointer(capsule.as_ptr(), c"dltensor".as_ptr())
                as *mut DlManagedTensor
        };
        if managed.is_null() {
            return Err(PyTypeError::new_err(format!(
                "{name} returned an invalid or already-consumed DLPack capsule"
            )));
        }
        let tensor = unsafe { &(*managed).dl_tensor };
        if tensor.device.device_type != DL_DEVICE_CUDA || tensor.device.device_id != expected_device
        {
            return Err(PyValueError::new_err(format!(
                "{name} DLPack tensor is on the wrong CUDA device"
            )));
        }
        let expected_bits = match expected_dtype {
            CudaDtype::Float32 => 32,
            CudaDtype::Float64 => 64,
        };
        if tensor.dtype.code != DL_DTYPE_FLOAT
            || tensor.dtype.bits != expected_bits
            || tensor.dtype.lanes != 1
        {
            return Err(PyTypeError::new_err(format!(
                "{name} DLPack dtype must exactly match {}",
                expected_dtype.name()
            )));
        }
        if tensor.ndim != expected_ndim as i32 || tensor.shape.is_null() {
            return Err(PyValueError::new_err(format!(
                "{name} must be a {expected_ndim}-dimensional tensor"
            )));
        }
        let raw_shape = unsafe { std::slice::from_raw_parts(tensor.shape, expected_ndim) };
        let shape = raw_shape
            .iter()
            .map(|&dimension| {
                usize::try_from(dimension).map_err(|_| {
                    PyValueError::new_err(format!("{name} has an invalid negative dimension"))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        if shape.contains(&0) {
            return Err(PyValueError::new_err(format!("{name} must not be empty")));
        }
        if !tensor.strides.is_null() {
            let strides = unsafe { std::slice::from_raw_parts(tensor.strides, expected_ndim) };
            let mut expected_stride = 1i64;
            for axis in (0..expected_ndim).rev() {
                if strides[axis] != expected_stride {
                    return Err(PyValueError::new_err(format!(
                        "{name} must be C-contiguous for native CUDA"
                    )));
                }
                expected_stride = expected_stride
                    .checked_mul(raw_shape[axis])
                    .ok_or_else(|| PyValueError::new_err(format!("{name} shape is too large")))?;
            }
        }
        let base = tensor.data as usize;
        let offset = usize::try_from(tensor.byte_offset)
            .map_err(|_| PyValueError::new_err(format!("{name} byte offset is too large")))?;
        let address = base
            .checked_add(offset)
            .filter(|address| *address != 0)
            .ok_or_else(|| PyValueError::new_err(format!("{name} has an invalid data pointer")))?;
        let rename_status =
            unsafe { pyo3::ffi::PyCapsule_SetName(capsule.as_ptr(), c"used_dltensor".as_ptr()) };
        if rename_status != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "unable to claim {name} DLPack capsule"
            )));
        }
        Ok(Self {
            _capsule: capsule.unbind(),
            managed,
            address,
            shape,
        })
    }
}

impl Drop for DlpackTensor {
    fn drop(&mut self) {
        let deleter = unsafe { (*self.managed).deleter };
        if let Some(deleter) = deleter {
            unsafe { deleter(self.managed) };
        }
    }
}
