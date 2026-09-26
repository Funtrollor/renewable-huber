//! Argument checks performed before anything crosses the ABI.
//!
//! Rejecting here rather than relying on the C side keeps the failure a typed
//! `CudaError` with a Rust message, and means a malformed request never reaches
//! code holding device resources.

use crate::types::{CudaError, UpdateConfig};

/// Accept `n_features_in` only when it and the intercept flag reproduce the
/// engine's parameter count.
pub(crate) fn validate_intercept_layout(
    n_parameters: usize,
    n_features_in: i64,
    fit_intercept: bool,
) -> Result<usize, CudaError> {
    let n_features_in = usize::try_from(n_features_in).map_err(|_| {
        CudaError::InvalidArgument("n_features_in must be greater than zero".to_owned())
    })?;
    let expected_parameters = n_features_in
        .checked_add(usize::from(fit_intercept))
        .ok_or_else(|| {
            CudaError::InvalidArgument("feature and intercept dimensions are too large".to_owned())
        })?;
    if n_features_in == 0 || expected_parameters != n_parameters {
        return Err(CudaError::InvalidArgument(
            "n_parameters must equal n_features_in plus the intercept column".to_owned(),
        ));
    }
    Ok(n_features_in)
}

pub(crate) fn validate_update_config(
    n_parameters: usize,
    config: UpdateConfig,
) -> Result<(), CudaError> {
    validate_intercept_layout(n_parameters, config.n_features_in, config.fit_intercept)?;
    let max_iter = usize::try_from(config.max_iter)
        .ok()
        .filter(|value| *value >= 1)
        .ok_or_else(|| CudaError::InvalidArgument("max_iter must be positive".to_owned()))?;
    // The numerical policy (finite positive tau/bandwidth/tolerance,
    // non-negative ridge and lambda_scale) is the engine-independent one the
    // CPU engine enforces; share it rather than restate it.
    rh_core::UpdateConfig {
        tau: config.tau,
        penalty: config.penalty,
        lambda_scale: config.lambda_scale,
        bandwidth_scale: config.bandwidth_scale,
        max_iter,
        tolerance: config.tolerance,
        ridge: config.ridge,
    }
    .validate()
    .map_err(|error| CudaError::InvalidArgument(error.to_string()))
}

#[cfg(feature = "cuda")]
pub(crate) fn checked_dimension(value: usize, name: &str) -> Result<i64, CudaError> {
    i64::try_from(value)
        .map_err(|_| CudaError::InvalidArgument(format!("{name} exceeds the CUDA ABI limit")))
}
