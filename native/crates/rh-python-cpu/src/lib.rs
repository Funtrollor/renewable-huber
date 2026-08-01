//! PyO3 boundary for the opt-in Rust CPU P1 engine.
//!
//! Python owns public validation and checkpoint serialization. This module
//! accepts only exact-dtype C-contiguous NumPy buffers, releases the GIL for
//! native work, and returns portable row-major NumPy state.

use numpy::{ndarray::Array2, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};
use rayon::{ThreadPool, ThreadPoolBuilder};
use rh_core::{BatchView, CoreError, Diagnostics, Penalty, State, UpdateConfig};
use rh_cpu::{predict, CpuEngine, CpuScalar};

const ABI_VERSION: u32 = 1;
const PYTHON_API_VERSION: u32 = 2;

struct EngineData<T: CpuScalar> {
    engine: CpuEngine<T>,
    coefficients: Vec<T>,
    information: Vec<T>,
    n_samples_seen: usize,
    batch_count: usize,
    previous_lambda: f64,
    weight_sum: f64,
}

impl<T: CpuScalar> EngineData<T> {
    fn new(n_parameters: usize) -> PyResult<Self> {
        if n_parameters == 0 {
            return Err(PyValueError::new_err("n_parameters must be positive"));
        }
        let information_length = n_parameters
            .checked_mul(n_parameters)
            .ok_or_else(|| PyValueError::new_err("n_parameters is too large"))?;
        Ok(Self {
            engine: CpuEngine::default(),
            coefficients: vec![T::zero(); n_parameters],
            information: vec![T::zero(); information_length],
            n_samples_seen: 0,
            batch_count: 0,
            previous_lambda: 0.0,
            weight_sum: 0.0,
        })
    }

    fn n_parameters(&self) -> usize {
        self.coefficients.len()
    }

    fn take_state(&mut self, n_features_in: usize, fit_intercept: bool) -> State<T> {
        State {
            coefficients: std::mem::take(&mut self.coefficients),
            information: std::mem::take(&mut self.information),
            n_samples_seen: self.n_samples_seen,
            batch_count: self.batch_count,
            previous_lambda: self.previous_lambda,
            n_features_in,
            fit_intercept,
            weight_sum: self.weight_sum,
        }
    }

    fn replace_state(&mut self, state: State<T>) {
        self.coefficients = state.coefficients;
        self.information = state.information;
        self.n_samples_seen = state.n_samples_seen;
        self.batch_count = state.batch_count;
        self.previous_lambda = state.previous_lambda;
        self.weight_sum = state.weight_sum;
    }
}

enum TypedEngine {
    Float32(EngineData<f32>),
    Float64(EngineData<f64>),
}

/// Selects the Rayon scheduler used by one Python-visible engine.
///
/// The global variant preserves the original behavior and honours an
/// application-wide Rayon configuration. A dedicated pool lets callers bound
/// CPU use independently for each estimator without mutating process-global
/// state.
enum ExecutionPool {
    Global,
    Dedicated(ThreadPool),
}

impl ExecutionPool {
    fn new(n_threads: Option<i64>) -> PyResult<Self> {
        let Some(n_threads) = n_threads else {
            return Ok(Self::Global);
        };
        if n_threads < 1 {
            return Err(PyValueError::new_err(
                "n_threads must be a positive integer",
            ));
        }
        let n_threads = usize::try_from(n_threads)
            .map_err(|_| PyValueError::new_err("n_threads is too large"))?;
        let pool = ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .thread_name(|index| format!("renewable-huber-cpu-{index}"))
            .build()
            .map_err(|error| {
                PyRuntimeError::new_err(format!(
                    "failed to create the native CPU thread pool: {error}"
                ))
            })?;
        Ok(Self::Dedicated(pool))
    }

    fn n_threads(&self) -> usize {
        match self {
            Self::Global => rayon::current_num_threads(),
            Self::Dedicated(pool) => pool.current_num_threads(),
        }
    }

    fn is_dedicated(&self) -> bool {
        matches!(self, Self::Dedicated(_))
    }

    fn install<R: Send>(&self, operation: impl FnOnce() -> R + Send) -> R {
        match self {
            Self::Global => operation(),
            Self::Dedicated(pool) => pool.install(operation),
        }
    }
}

/// Persistent Rust CPU engine with reusable numerical workspaces.
#[pyclass(module = "renewable_huber._native_cpu")]
struct NativeCpuEngine {
    inner: TypedEngine,
    execution_pool: ExecutionPool,
}

#[pymethods]
impl NativeCpuEngine {
    #[new]
    #[pyo3(signature = (dtype, n_parameters, n_threads=None))]
    fn new(dtype: &str, n_parameters: usize, n_threads: Option<i64>) -> PyResult<Self> {
        let inner = match dtype {
            "float32" => TypedEngine::Float32(EngineData::new(n_parameters)?),
            "float64" => TypedEngine::Float64(EngineData::new(n_parameters)?),
            _ => {
                return Err(PyValueError::new_err(
                    "dtype must be exactly 'float32' or 'float64'",
                ))
            }
        };
        Ok(Self {
            inner,
            execution_pool: ExecutionPool::new(n_threads)?,
        })
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.inner {
            TypedEngine::Float32(_) => "float32",
            TypedEngine::Float64(_) => "float64",
        }
    }

    #[getter]
    fn n_parameters(&self) -> usize {
        match &self.inner {
            TypedEngine::Float32(engine) => engine.n_parameters(),
            TypedEngine::Float64(engine) => engine.n_parameters(),
        }
    }

    /// Effective number of Rayon workers available to this engine.
    #[getter]
    fn n_threads(&self) -> usize {
        self.execution_pool.n_threads()
    }

    /// Whether this engine owns a private Rayon pool instead of using global state.
    #[getter]
    fn has_dedicated_thread_pool(&self) -> bool {
        self.execution_pool.is_dedicated()
    }

    #[pyo3(signature = (
        coefficients,
        information,
        n_samples_seen,
        batch_count,
        previous_lambda,
        weight_sum,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn restore(
        &mut self,
        coefficients: &Bound<'_, PyAny>,
        information: &Bound<'_, PyAny>,
        n_samples_seen: i64,
        batch_count: i64,
        previous_lambda: f64,
        weight_sum: f64,
    ) -> PyResult<()> {
        if n_samples_seen < 0 || batch_count < 0 {
            return Err(PyValueError::new_err("state counters must be non-negative"));
        }
        match &mut self.inner {
            TypedEngine::Float32(engine) => restore_typed(
                engine,
                coefficients,
                information,
                n_samples_seen as usize,
                batch_count as usize,
                previous_lambda,
                weight_sum,
            ),
            TypedEngine::Float64(engine) => restore_typed(
                engine,
                coefficients,
                information,
                n_samples_seen as usize,
                batch_count as usize,
                previous_lambda,
                weight_sum,
            ),
        }
    }

    #[pyo3(signature = (
        x_design,
        y,
        sample_weight,
        batch_weight,
        n_features_in,
        fit_intercept,
        tau,
        penalty,
        lambda_scale,
        bandwidth_scale,
        max_iter,
        tol,
        ridge,
    ))]
    #[allow(clippy::too_many_arguments)]
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
        penalty: &str,
        lambda_scale: f64,
        bandwidth_scale: f64,
        max_iter: i64,
        tol: f64,
        ridge: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        if n_features_in < 1 {
            return Err(PyValueError::new_err("n_features_in must be positive"));
        }
        if max_iter < 1 {
            return Err(PyValueError::new_err("max_iter must be positive"));
        }
        let penalty = match penalty {
            "none" => Penalty::None,
            "l1" => Penalty::L1,
            _ => return Err(PyValueError::new_err("unsupported native penalty")),
        };
        let config = UpdateConfig {
            tau,
            penalty,
            lambda_scale,
            bandwidth_scale,
            max_iter: max_iter as usize,
            tolerance: tol,
            ridge,
        };
        let Self {
            inner,
            execution_pool,
        } = self;
        match inner {
            TypedEngine::Float32(engine) => update_typed(
                py,
                execution_pool,
                engine,
                x_design,
                y,
                sample_weight,
                batch_weight,
                n_features_in as usize,
                fit_intercept,
                config,
            ),
            TypedEngine::Float64(engine) => update_typed(
                py,
                execution_pool,
                engine,
                x_design,
                y,
                sample_weight,
                batch_weight,
                n_features_in as usize,
                fit_intercept,
                config,
            ),
        }
    }

    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        x_design: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let Self {
            inner,
            execution_pool,
        } = self;
        match inner {
            TypedEngine::Float32(engine) => {
                Ok(predict_typed(py, execution_pool, engine, x_design)?
                    .into_any()
                    .unbind())
            }
            TypedEngine::Float64(engine) => {
                Ok(predict_typed(py, execution_pool, engine, x_design)?
                    .into_any()
                    .unbind())
            }
        }
    }

    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        match &self.inner {
            TypedEngine::Float32(engine) => state_dict(py, engine),
            TypedEngine::Float64(engine) => state_dict(py, engine),
        }
    }
}

#[pyfunction]
fn version<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("abi_version", ABI_VERSION)?;
    result.set_item("python_api_version", PYTHON_API_VERSION)?;
    result.set_item("engine_version", env!("CARGO_PKG_VERSION"))?;
    result.set_item("linear_algebra_provider", "nalgebra+matrixmultiply")?;
    result.set_item("parallel_provider", "rayon")?;
    result.set_item("parallel_threads", rayon::current_num_threads())?;
    result.set_item("supports_engine_thread_pool", true)?;
    result.set_item("engine_thread_pool_default", "global")?;
    result.set_item("strict_dtypes", ("float32", "float64"))?;
    Ok(result)
}

#[pymodule]
fn _renewable_huber_native_cpu(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCpuEngine>()?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    Ok(())
}

fn restore_typed<T: CpuScalar + numpy::Element>(
    engine: &mut EngineData<T>,
    coefficients: &Bound<'_, PyAny>,
    information: &Bound<'_, PyAny>,
    n_samples_seen: usize,
    batch_count: usize,
    previous_lambda: f64,
    weight_sum: f64,
) -> PyResult<()> {
    let coefficients = readonly_vector::<T>(coefficients, "coefficients")?;
    let information = readonly_matrix::<T>(information, "information")?;
    let n_parameters = engine.n_parameters();
    if coefficients.1 != n_parameters {
        return Err(PyValueError::new_err("coefficients has the wrong length"));
    }
    if information.1 != n_parameters || information.2 != n_parameters {
        return Err(PyValueError::new_err(
            "information has the wrong square shape",
        ));
    }
    if !previous_lambda.is_finite() || previous_lambda < 0.0 {
        return Err(PyValueError::new_err(
            "previous_lambda must be finite and non-negative",
        ));
    }
    if !weight_sum.is_finite() || weight_sum < 0.0 || (n_samples_seen > 0 && weight_sum == 0.0) {
        return Err(PyValueError::new_err(
            "weight_sum is inconsistent with the state counters",
        ));
    }
    let coefficient_values = contiguous_vector(&coefficients.0, "coefficients")?;
    let information_values = contiguous_matrix(&information.0, "information")?;
    if coefficient_values.iter().any(|value| !value.is_finite())
        || information_values.iter().any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err(
            "state arrays must contain only finite values",
        ));
    }
    engine.coefficients = coefficient_values.to_vec();
    engine.information = information_values.to_vec();
    engine.n_samples_seen = n_samples_seen;
    engine.batch_count = batch_count;
    engine.previous_lambda = previous_lambda;
    engine.weight_sum = weight_sum;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn update_typed<'py, T: CpuScalar + numpy::Element>(
    py: Python<'py>,
    execution_pool: &ExecutionPool,
    engine: &mut EngineData<T>,
    x_design: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    sample_weight: Option<&Bound<'py, PyAny>>,
    batch_weight: f64,
    n_features_in: usize,
    fit_intercept: bool,
    config: UpdateConfig,
) -> PyResult<Bound<'py, PyDict>> {
    let x_design = readonly_matrix::<T>(x_design, "X_design")?;
    let y = readonly_vector::<T>(y, "y")?;
    let x_values = contiguous_matrix(&x_design.0, "X_design")?;
    let y_values = contiguous_vector(&y.0, "y")?;
    let weight = match sample_weight {
        Some(value) => Some(readonly_vector::<T>(value, "sample_weight")?),
        None => None,
    };
    let weight_values = match weight.as_ref() {
        Some(array) => Some(contiguous_vector(&array.0, "sample_weight")?),
        None => None,
    };
    // Move the resident state into the transition instead of cloning its p^2
    // information matrix on every partial_fit. The core only borrows it while
    // solving. If anything fails, put the exact buffers back before exposing
    // the exception so update remains transactional.
    let state = engine.take_state(n_features_in, fit_intercept);
    let batch = BatchView {
        x_design: x_values,
        n_rows: x_design.1,
        n_parameters: x_design.2,
        y: y_values,
        sample_weight: weight_values,
        batch_weight,
    };
    let transition = match py
        .allow_threads(|| execution_pool.install(|| engine.engine.update(batch, &state, config)))
    {
        Ok(transition) => transition,
        Err(error) => {
            engine.replace_state(state);
            return Err(to_py_error(error));
        }
    };
    let diagnostics = transition.diagnostics;
    engine.replace_state(transition.state);
    let result = state_dict(py, engine)?;
    add_diagnostics(&result, diagnostics)?;
    Ok(result)
}

fn predict_typed<'py, T: CpuScalar + numpy::Element>(
    py: Python<'py>,
    execution_pool: &ExecutionPool,
    engine: &mut EngineData<T>,
    x_design: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<T>>> {
    let x_design = readonly_matrix::<T>(x_design, "X_design")?;
    let x_values = contiguous_matrix(&x_design.0, "X_design")?;
    let values = py
        .allow_threads(|| {
            execution_pool
                .install(|| predict(x_values, x_design.1, x_design.2, &engine.coefficients))
        })
        .map_err(to_py_error)?;
    Ok(values.into_pyarray(py))
}

fn state_dict<'py, T: CpuScalar + numpy::Element>(
    py: Python<'py>,
    engine: &EngineData<T>,
) -> PyResult<Bound<'py, PyDict>> {
    let n_parameters = engine.n_parameters();
    let information =
        Array2::from_shape_vec((n_parameters, n_parameters), engine.information.clone())
            .map_err(|_| PyRuntimeError::new_err("native information shape is invalid"))?;
    let result = PyDict::new(py);
    result.set_item("coefficients", engine.coefficients.clone().into_pyarray(py))?;
    result.set_item("information", information.into_pyarray(py))?;
    // Both arrays above own cloned Rust buffers and are never reused by the
    // resident engine. The Python adapter can therefore adopt them directly
    // instead of copying the p^2 information matrix a second time.
    result.set_item("state_is_detached", true)?;
    result.set_item("n_samples_seen", engine.n_samples_seen)?;
    result.set_item("batch_count", engine.batch_count)?;
    result.set_item("previous_lambda", engine.previous_lambda)?;
    result.set_item("weight_sum", engine.weight_sum)?;
    Ok(result)
}

fn add_diagnostics(result: &Bound<'_, PyDict>, diagnostics: Diagnostics) -> PyResult<()> {
    result.set_item("iterations", diagnostics.iterations)?;
    result.set_item("converged", diagnostics.converged)?;
    result.set_item("objective", diagnostics.objective)?;
    result.set_item("lambda_value", diagnostics.lambda_value)?;
    result.set_item("bandwidth", diagnostics.bandwidth)?;
    result.set_item(
        "used_regularized_fallback",
        diagnostics.used_regularized_fallback,
    )?;
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
    let shape = {
        let view = array.as_array();
        (view.nrows(), view.ncols())
    };
    Ok((array, shape.0, shape.1))
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
    let length = array.as_array().len();
    Ok((array, length))
}

fn contiguous_matrix<'array, 'py, T: numpy::Element>(
    array: &'array PyReadonlyArray2<'py, T>,
    name: &str,
) -> PyResult<&'array [T]> {
    array.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be C-contiguous; copy explicitly before native CPU use"
        ))
    })
}

fn contiguous_vector<'array, 'py, T: numpy::Element>(
    array: &'array PyReadonlyArray1<'py, T>,
    name: &str,
) -> PyResult<&'array [T]> {
    array.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be C-contiguous; copy explicitly before native CPU use"
        ))
    })
}

fn to_py_error(error: CoreError) -> PyErr {
    match error {
        CoreError::InvalidConfig(_)
        | CoreError::InvalidState(_)
        | CoreError::InvalidBatch(_)
        | CoreError::SizeOverflow
        | CoreError::ScalarConversion => PyValueError::new_err(error.to_string()),
        CoreError::LinearSolve | CoreError::NonFiniteResult => {
            PyRuntimeError::new_err(error.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ExecutionPool;

    #[test]
    fn global_execution_pool_preserves_rayon_default() {
        let pool = ExecutionPool::new(None).unwrap();
        assert!(!pool.is_dedicated());
        assert_eq!(pool.install(rayon::current_num_threads), pool.n_threads());
    }

    #[test]
    fn dedicated_execution_pool_uses_requested_worker_count() {
        let pool = ExecutionPool::new(Some(3)).unwrap();
        assert!(pool.is_dedicated());
        assert_eq!(pool.n_threads(), 3);
        assert_eq!(pool.install(rayon::current_num_threads), 3);

        let worker_indices = pool.install(|| {
            use rayon::prelude::*;
            (0..10_000)
                .into_par_iter()
                .filter_map(|_| rayon::current_thread_index())
                .collect::<std::collections::BTreeSet<_>>()
        });
        assert!(!worker_indices.is_empty());
        assert!(worker_indices.iter().all(|index| *index < 3));
    }
}
