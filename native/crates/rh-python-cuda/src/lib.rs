//! PyO3 boundary for the optional host/DLPack CUDA engine.
//!
//! Python remains responsible for public estimator validation and for
//! calculating `batch_weight`. This extension accepts C-contiguous NumPy host
//! arrays or CUDA DLPack tensors in strict `float32`/`float64`, then forwards
//! one fully validated batch to the safe Rust CUDA wrapper.

use numpy::{ndarray::Array2, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};
use rh_cuda_ffi::{
    device_count as cuda_device_count, is_available as cuda_is_available, runtime_info, CudaDtype,
    CudaEngine, CudaError, CudaScalar, DeviceBatch, Diagnostics, EngineTuning, HostBatch,
    HostMatrix, HostState, HostVector, StateMetadata, UnpenalizedConfig,
};

mod dlpack;

use dlpack::DlpackTensor;

const PYTHON_API_VERSION: u32 = 3;

/// Host-fed adapter for one persistent, single-device native CUDA engine.
///
/// Sequential calls may move between Python threads. The CUDA C ABI selects
/// the owning device on every call, and PyO3's mutable borrow prevents
/// concurrent access to one engine.
#[pyclass(module = "renewable_huber._native_cuda")]
struct NativeCudaEngine {
    engine: CudaEngine,
}

#[pymethods]
impl NativeCudaEngine {
    #[new]
    #[pyo3(signature = (dtype, n_parameters, device_id=0, cuda_graphs=false, fast_math=false))]
    fn new(
        dtype: &str,
        n_parameters: usize,
        device_id: i32,
        cuda_graphs: bool,
        fast_math: bool,
    ) -> PyResult<Self> {
        let dtype = CudaDtype::parse(dtype).map_err(to_py_error)?;
        let engine = CudaEngine::create_with_tuning(
            dtype,
            n_parameters,
            device_id,
            EngineTuning {
                cuda_graphs,
                fast_math,
            },
        )
        .map_err(to_py_error)?;
        Ok(Self { engine })
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.engine.dtype().name()
    }

    #[getter]
    fn n_parameters(&self) -> usize {
        self.engine.n_parameters()
    }

    #[getter]
    fn device_id(&self) -> i32 {
        self.engine.device_id()
    }

    fn features<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let features = self.engine.features().map_err(to_py_error)?;
        let result = PyDict::new(py);
        result.set_item("cuda_graphs_requested", features.requested.cuda_graphs)?;
        result.set_item("cuda_graphs_enabled", features.enabled.cuda_graphs)?;
        result.set_item("fast_math_requested", features.requested.fast_math)?;
        result.set_item("fast_math_enabled", features.enabled.fast_math)?;
        result.set_item("graph_captures", features.graph_captures)?;
        result.set_item("graph_replays", features.graph_replays)?;
        result.set_item("graph_fallbacks", features.graph_fallbacks)?;
        Ok(result)
    }

    /// Restore host checkpoint state into persistent device allocations.
    #[pyo3(signature = (
        coefficients,
        information,
        n_samples_seen,
        batch_count,
        previous_lambda,
        weight_sum,
    ))]
    #[allow(clippy::too_many_arguments)] // Stable, explicit Python checkpoint ABI.
    fn restore(
        &mut self,
        py: Python<'_>,
        coefficients: &Bound<'_, PyAny>,
        information: &Bound<'_, PyAny>,
        n_samples_seen: i64,
        batch_count: i64,
        previous_lambda: f64,
        weight_sum: f64,
    ) -> PyResult<()> {
        match self.engine.dtype() {
            CudaDtype::Float32 => restore_typed::<f32>(
                py,
                &mut self.engine,
                coefficients,
                information,
                n_samples_seen,
                batch_count,
                previous_lambda,
                weight_sum,
            ),
            CudaDtype::Float64 => restore_typed::<f64>(
                py,
                &mut self.engine,
                coefficients,
                information,
                n_samples_seen,
                batch_count,
                previous_lambda,
                weight_sum,
            ),
        }
    }

    /// Execute one validated unpenalized batch and return flat state/diagnostic
    /// fields consumed by the Python native backend.
    ///
    /// `batch_weight` is positional immediately after `sample_weight` to match
    /// the backend dispatcher.  It is calculated by the Python estimator from
    /// the original validated sample-weight vector so the native engine cannot
    /// accidentally change streaming semantics.
    #[pyo3(signature = (
        x_design,
        y,
        sample_weight,
        batch_weight,
        n_features_in,
        fit_intercept,
        tau,
        bandwidth_scale,
        max_iter,
        tol,
        ridge,
    ))]
    #[allow(clippy::too_many_arguments)] // Stable, explicit Python update ABI.
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        x_design: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        sample_weight: Option<&Bound<'py, PyAny>>,
        batch_weight: f64,
        n_features_in: i64,
        fit_intercept: bool,
        tau: f64,
        bandwidth_scale: f64,
        max_iter: i64,
        tol: f64,
        ridge: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let config = UnpenalizedConfig {
            n_features_in,
            fit_intercept,
            tau,
            bandwidth_scale,
            max_iter,
            tolerance: tol,
            ridge,
        };
        match self.engine.dtype() {
            CudaDtype::Float32 => update_typed::<f32>(
                py,
                &mut self.engine,
                x_design,
                y,
                sample_weight,
                batch_weight,
                config,
            ),
            CudaDtype::Float64 => update_typed::<f64>(
                py,
                &mut self.engine,
                x_design,
                y,
                sample_weight,
                batch_weight,
                config,
            ),
        }
    }

    /// Consume CUDA tensors through the Python DLPack protocol. The producer
    /// is asked to make this engine's private stream a valid consumer stream;
    /// no host staging or implicit dtype/device conversion is performed.
    #[pyo3(signature = (
        x_design,
        y,
        sample_weight,
        batch_weight,
        n_features_in,
        fit_intercept,
        tau,
        bandwidth_scale,
        max_iter,
        tol,
        ridge,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn update_device<'py>(
        &mut self,
        py: Python<'py>,
        x_design: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        sample_weight: Option<&Bound<'py, PyAny>>,
        batch_weight: f64,
        n_features_in: i64,
        fit_intercept: bool,
        tau: f64,
        bandwidth_scale: f64,
        max_iter: i64,
        tol: f64,
        ridge: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let config = UnpenalizedConfig {
            n_features_in,
            fit_intercept,
            tau,
            bandwidth_scale,
            max_iter,
            tolerance: tol,
            ridge,
        };
        let stream = self.engine.stream_handle().map_err(to_py_error)?;
        match self.engine.dtype() {
            CudaDtype::Float32 => update_device_typed::<f32>(
                py,
                &mut self.engine,
                stream,
                x_design,
                y,
                sample_weight,
                batch_weight,
                config,
            ),
            CudaDtype::Float64 => update_device_typed::<f64>(
                py,
                &mut self.engine,
                stream,
                x_design,
                y,
                sample_weight,
                batch_weight,
                config,
            ),
        }
    }

    /// Predict from a C-contiguous host design matrix and return a NumPy array
    /// in this engine's strict dtype.
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        x_design: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        match self.engine.dtype() {
            CudaDtype::Float32 => Ok(predict_typed::<f32>(py, &mut self.engine, x_design)?
                .into_any()
                .unbind()),
            CudaDtype::Float64 => Ok(predict_typed::<f64>(py, &mut self.engine, x_design)?
                .into_any()
                .unbind()),
        }
    }

    /// Export a portable host mirror of the persistent device state.
    fn state<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        match self.engine.dtype() {
            CudaDtype::Float32 => state_dict_typed::<f32>(py, &mut self.engine),
            CudaDtype::Float64 => state_dict_typed::<f64>(py, &mut self.engine),
        }
    }

    fn synchronize(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.engine.synchronize())
            .map_err(to_py_error)
    }
}

#[pyfunction]
fn is_available() -> bool {
    cuda_is_available()
}

#[pyfunction]
fn device_count() -> i32 {
    cuda_device_count().unwrap_or(0)
}

#[pyfunction]
fn version<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    // Report what the linked library says, not what this crate was compiled to
    // believe. The Python backend gates on this value, and a constant echoing
    // itself would let a stale native build past that gate.
    result.set_item(
        "abi_version",
        rh_cuda_ffi::linked_abi_version().unwrap_or(rh_cuda_ffi::ABI_VERSION),
    )?;
    result.set_item("python_api_version", PYTHON_API_VERSION)?;
    result.set_item("engine_version", env!("CARGO_PKG_VERSION"))?;
    result.set_item("initial_state", "canonical_empty")?;
    result.set_item("device_input", "dlpack")?;
    result.set_item("cuda_graphs", "opt_in_best_effort")?;
    result.set_item("fast_math", "opt_in_float32_tf32")?;
    result.set_item("supports_cuda_graphs", true)?;
    result.set_item("supports_fast_math", true)?;
    result.set_item("cuda_available", cuda_is_available())?;
    match runtime_info() {
        Ok(info) => {
            result.set_item("runtime_version", info.runtime_version)?;
            result.set_item("driver_version", info.driver_version)?;
            result.set_item("device_count", info.device_count)?;
        }
        Err(_) => {
            result.set_item("runtime_version", py.None())?;
            result.set_item("driver_version", py.None())?;
            result.set_item("device_count", 0)?;
        }
    }
    Ok(result)
}

#[pymodule]
fn _renewable_huber_native_cuda(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCudaEngine>()?;
    module.add_function(wrap_pyfunction!(is_available, module)?)?;
    module.add_function(wrap_pyfunction!(device_count, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)] // Mirrors the public checkpoint fields.
fn restore_typed<T: CudaScalar + numpy::Element + Send + Sync>(
    py: Python<'_>,
    engine: &mut CudaEngine,
    coefficients: &Bound<'_, PyAny>,
    information: &Bound<'_, PyAny>,
    n_samples_seen: i64,
    batch_count: i64,
    previous_lambda: f64,
    weight_sum: f64,
) -> PyResult<()> {
    let coefficients = readonly_vector::<T>(coefficients, "coefficients")?;
    let information = readonly_matrix::<T>(information, "information")?;
    if information.1 != engine.n_parameters() || information.2 != engine.n_parameters() {
        return Err(PyValueError::new_err(
            "information has the wrong square shape",
        ));
    }
    let coefficients_slice = contiguous_vector(&coefficients.0, "coefficients")?;
    let information_slice = contiguous_matrix(&information.0, "information")?;
    let state = HostState {
        coefficients: coefficients_slice,
        information: information_slice,
        n_samples_seen,
        batch_count,
        previous_lambda,
        weight_sum,
    };
    py.allow_threads(|| engine.restore(state))
        .map_err(to_py_error)
}

fn update_typed<'py, T: CudaScalar + numpy::Element + Default + Send + Sync>(
    py: Python<'py>,
    engine: &mut CudaEngine,
    x_design: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    sample_weight: Option<&Bound<'py, PyAny>>,
    batch_weight: f64,
    config: UnpenalizedConfig,
) -> PyResult<Bound<'py, PyDict>> {
    let x_design = readonly_matrix::<T>(x_design, "X_design")?;
    let y = readonly_vector::<T>(y, "y")?;
    let x_values = contiguous_matrix(&x_design.0, "X_design")?;
    let y_values = contiguous_vector(&y.0, "y")?;
    let x_matrix = HostMatrix::new(x_values, x_design.1, x_design.2).map_err(to_py_error)?;
    let y_vector = HostVector::new(y_values);

    let weight = match sample_weight {
        Some(value) => {
            let array = readonly_vector::<T>(value, "sample_weight")?;
            Some((array,))
        }
        None => None,
    };
    let weight_vector = match weight.as_ref() {
        Some((array,)) => Some(HostVector::new(contiguous_vector(
            &array.0,
            "sample_weight",
        )?)),
        None => None,
    };
    let n_parameters = engine.n_parameters();
    let information_len = n_parameters
        .checked_mul(n_parameters)
        .ok_or_else(|| PyValueError::new_err("n_parameters squared overflows usize"))?;
    let mut coefficients = vec![T::default(); n_parameters];
    let mut information = vec![T::default(); information_len];
    let (diagnostics, metadata) = py
        .allow_threads(|| {
            engine.update_with_state(
                HostBatch {
                    x_design: x_matrix,
                    y: y_vector,
                    sample_weight: weight_vector,
                    batch_weight,
                },
                config,
                &mut coefficients,
                &mut information,
            )
        })
        .map_err(to_py_error)?;
    let result = state_dict_from_parts(py, n_parameters, coefficients, information, metadata)?;
    // These arrays were allocated specifically for this return value and do
    // not alias the engine's resident device state or mutable host scratch.
    // The Python adapter can therefore adopt them without a second p + p^2
    // host copy while retaining the defensive-copy fallback for test doubles
    // and older extension builds that do not advertise detached ownership.
    result.set_item("state_is_detached", true)?;
    add_diagnostics(&result, diagnostics)?;
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn update_device_typed<'py, T: CudaScalar + numpy::Element + Default + Send + Sync>(
    py: Python<'py>,
    engine: &mut CudaEngine,
    stream: usize,
    x_design: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    sample_weight: Option<&Bound<'py, PyAny>>,
    batch_weight: f64,
    config: UnpenalizedConfig,
) -> PyResult<Bound<'py, PyDict>> {
    let expected = engine.dtype();
    let device_id = engine.device_id();
    let x = DlpackTensor::consume(py, x_design, stream, expected, device_id, 2, "X_design")?;
    let target = DlpackTensor::consume(py, y, stream, expected, device_id, 1, "y")?;
    if target.shape[0] != x.shape[0] {
        return Err(PyValueError::new_err(
            "X_design and y must have the same number of rows",
        ));
    }
    let weight = match sample_weight {
        Some(value) => {
            let tensor =
                DlpackTensor::consume(py, value, stream, expected, device_id, 1, "sample_weight")?;
            if tensor.shape[0] != x.shape[0] {
                return Err(PyValueError::new_err(
                    "sample_weight and X_design must have the same number of rows",
                ));
            }
            Some(tensor)
        }
        None => None,
    };
    let n_parameters = engine.n_parameters();
    let information_len = n_parameters
        .checked_mul(n_parameters)
        .ok_or_else(|| PyValueError::new_err("n_parameters squared overflows usize"))?;
    let mut coefficients = vec![T::default(); n_parameters];
    let mut information = vec![T::default(); information_len];
    let batch = DeviceBatch {
        x_address: x.address,
        y_address: target.address,
        sample_weight_address: weight.as_ref().map(|tensor| tensor.address),
        n_rows: x.shape[0],
        n_columns: x.shape[1],
        batch_weight,
    };
    // The three capsule owners remain alive across this GIL release. Their
    // producer deleters run only after the CUDA ABI has completed its D2D
    // copies and committed the state.
    let (diagnostics, metadata) = py
        .allow_threads(|| {
            engine.update_device_with_state::<T>(batch, config, &mut coefficients, &mut information)
        })
        .map_err(to_py_error)?;
    drop(weight);
    drop(target);
    drop(x);
    let result = state_dict_from_parts(py, n_parameters, coefficients, information, metadata)?;
    result.set_item("state_is_detached", true)?;
    result.set_item("input_transport", "dlpack")?;
    add_diagnostics(&result, diagnostics)?;
    Ok(result)
}

fn predict_typed<'py, T: CudaScalar + numpy::Element + Default + Send + Sync>(
    py: Python<'py>,
    engine: &mut CudaEngine,
    x_design: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<T>>> {
    let x_design = readonly_matrix::<T>(x_design, "X_design")?;
    let x_values = contiguous_matrix(&x_design.0, "X_design")?;
    let matrix = HostMatrix::new(x_values, x_design.1, x_design.2).map_err(to_py_error)?;
    let mut prediction = vec![T::default(); matrix.rows()];
    py.allow_threads(|| engine.predict(matrix, &mut prediction))
        .map_err(to_py_error)?;
    Ok(prediction.into_pyarray(py))
}

fn state_dict_typed<'py, T: CudaScalar + numpy::Element + Default + Send + Sync>(
    py: Python<'py>,
    engine: &mut CudaEngine,
) -> PyResult<Bound<'py, PyDict>> {
    let n_parameters = engine.n_parameters();
    let information_len = n_parameters
        .checked_mul(n_parameters)
        .ok_or_else(|| PyValueError::new_err("n_parameters squared overflows usize"))?;
    let mut coefficients = vec![T::default(); n_parameters];
    let mut information = vec![T::default(); information_len];
    let metadata = py
        .allow_threads(|| engine.copy_state(&mut coefficients, &mut information))
        .map_err(to_py_error)?;
    state_dict_from_parts(py, n_parameters, coefficients, information, metadata)
}

fn state_dict_from_parts<'py, T: numpy::Element>(
    py: Python<'py>,
    n_parameters: usize,
    coefficients: Vec<T>,
    information: Vec<T>,
    metadata: StateMetadata,
) -> PyResult<Bound<'py, PyDict>> {
    let information = Array2::from_shape_vec((n_parameters, n_parameters), information)
        .map_err(|_| PyRuntimeError::new_err("native information matrix has an invalid shape"))?;
    let result = PyDict::new(py);
    result.set_item("coefficients", coefficients.into_pyarray(py))?;
    result.set_item("information", information.into_pyarray(py))?;
    result.set_item("n_samples_seen", metadata.n_samples_seen)?;
    result.set_item("batch_count", metadata.batch_count)?;
    result.set_item("previous_lambda", metadata.previous_lambda)?;
    result.set_item("weight_sum", metadata.weight_sum)?;
    Ok(result)
}

fn add_diagnostics(result: &Bound<'_, PyDict>, diagnostics: Diagnostics) -> PyResult<()> {
    result.set_item("iterations", diagnostics.iterations)?;
    result.set_item("converged", diagnostics.converged)?;
    result.set_item(
        "used_regularized_fallback",
        diagnostics.used_regularized_fallback,
    )?;
    result.set_item("objective", diagnostics.objective)?;
    result.set_item("lambda_value", diagnostics.lambda_value)?;
    result.set_item("bandwidth", diagnostics.bandwidth)?;
    Ok(())
}

fn readonly_matrix<'py, T: numpy::Element>(
    value: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<(PyReadonlyArray2<'py, T>, usize, usize)> {
    let array = value.extract::<PyReadonlyArray2<'py, T>>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{name} must be a C-contiguous NumPy array with the engine dtype"
        ))
    })?;
    let (rows, columns) = {
        let view = array.as_array();
        (view.nrows(), view.ncols())
    };
    Ok((array, rows, columns))
}

fn readonly_vector<'py, T: numpy::Element>(
    value: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<(PyReadonlyArray1<'py, T>, usize)> {
    let array = value.extract::<PyReadonlyArray1<'py, T>>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{name} must be a C-contiguous NumPy array with the engine dtype"
        ))
    })?;
    let length = { array.as_array().len() };
    Ok((array, length))
}

fn contiguous_matrix<'array, 'py, T: numpy::Element>(
    array: &'array PyReadonlyArray2<'py, T>,
    name: &str,
) -> PyResult<&'array [T]> {
    array.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be C-contiguous; copy explicitly before native CUDA use"
        ))
    })
}

fn contiguous_vector<'array, 'py, T: numpy::Element>(
    array: &'array PyReadonlyArray1<'py, T>,
    name: &str,
) -> PyResult<&'array [T]> {
    array.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be C-contiguous; copy explicitly before native CUDA use"
        ))
    })
}

fn to_py_error(error: CudaError) -> PyErr {
    match error {
        CudaError::InvalidArgument(message) => PyValueError::new_err(message),
        CudaError::NotCompiled | CudaError::Status { .. } => {
            PyRuntimeError::new_err(error.to_string())
        }
    }
}
