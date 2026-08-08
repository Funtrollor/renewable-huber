//! Argument checks performed before anything crosses the ABI.
//!
//! Rejecting here rather than relying on the C side keeps the failure a typed
//! `CudaError` with a Rust message, and means a malformed request never reaches
//! code holding device resources.

use crate::types::{CudaError, UnpenalizedConfig};

pub(crate) fn validate_unpenalized_config(
    n_parameters: usize,
    config: UnpenalizedConfig,
) -> Result<(), CudaError> {
    let n_features_in = usize::try_from(config.n_features_in).map_err(|_| {
        CudaError::InvalidArgument("n_features_in must be greater than zero".to_owned())
    })?;
    let expected_parameters = n_features_in
        .checked_add(usize::from(config.fit_intercept))
        .ok_or_else(|| {
            CudaError::InvalidArgument("feature and intercept dimensions are too large".to_owned())
        })?;
    if n_features_in == 0 || expected_parameters != n_parameters {
        return Err(CudaError::InvalidArgument(
            "n_parameters must equal n_features_in plus the intercept column".to_owned(),
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
pub(crate) fn checked_dimension(value: usize, name: &str) -> Result<i64, CudaError> {
    i64::try_from(value)
        .map_err(|_| CudaError::InvalidArgument(format!("{name} exceeds the CUDA ABI limit")))
}
