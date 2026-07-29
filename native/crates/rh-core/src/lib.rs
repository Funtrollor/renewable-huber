//! Engine-independent contracts for Renewable Huber state transitions.
//!
//! This crate deliberately has no Python, CUDA, or linear-algebra provider
//! dependency. Buffers use portable row-major storage so checkpoints can move
//! between the Python, CPU, and future CUDA engines without transposition.

use std::fmt::Debug;

use num_traits::{Float, FromPrimitive};
use thiserror::Error;

/// Floating-point values supported by the strict native engines.
pub trait Scalar: Float + FromPrimitive + Debug + Send + Sync + Copy + Default + 'static {}

impl Scalar for f32 {}
impl Scalar for f64 {}

/// Estimator penalty, frozen by the native-core RFC.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Penalty {
    None,
    L1,
}

/// Immutable numerical configuration for one state transition.
#[derive(Clone, Copy, Debug)]
pub struct UpdateConfig {
    pub tau: f64,
    pub penalty: Penalty,
    pub lambda_scale: f64,
    pub bandwidth_scale: f64,
    pub max_iter: usize,
    pub tolerance: f64,
    pub ridge: f64,
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            tau: 1.345,
            penalty: Penalty::None,
            lambda_scale: 1.0,
            bandwidth_scale: 1.0,
            max_iter: 100,
            tolerance: 1.0e-6,
            ridge: 1.0e-8,
        }
    }
}

impl UpdateConfig {
    pub fn validate(self) -> Result<(), CoreError> {
        if !self.tau.is_finite() || self.tau <= 0.0 {
            return Err(CoreError::InvalidConfig("tau must be finite and positive"));
        }
        if !self.lambda_scale.is_finite() || self.lambda_scale < 0.0 {
            return Err(CoreError::InvalidConfig(
                "lambda_scale must be finite and non-negative",
            ));
        }
        if !self.bandwidth_scale.is_finite() || self.bandwidth_scale <= 0.0 {
            return Err(CoreError::InvalidConfig(
                "bandwidth_scale must be finite and positive",
            ));
        }
        if self.max_iter == 0 {
            return Err(CoreError::InvalidConfig("max_iter must be positive"));
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(CoreError::InvalidConfig(
                "tolerance must be finite and positive",
            ));
        }
        if !self.ridge.is_finite() || self.ridge < 0.0 {
            return Err(CoreError::InvalidConfig(
                "ridge must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

/// Portable sufficient state. `information` is row-major and may be asymmetric
/// when restored from an external checkpoint.
#[derive(Clone, Debug)]
pub struct State<T: Scalar> {
    pub coefficients: Vec<T>,
    pub information: Vec<T>,
    pub n_samples_seen: usize,
    pub batch_count: usize,
    pub previous_lambda: f64,
    pub n_features_in: usize,
    pub fit_intercept: bool,
    pub weight_sum: f64,
}

impl<T: Scalar> State<T> {
    pub fn empty(n_features_in: usize, fit_intercept: bool) -> Self {
        let n_parameters = n_features_in + usize::from(fit_intercept);
        Self {
            coefficients: vec![T::zero(); n_parameters],
            information: vec![T::zero(); n_parameters * n_parameters],
            n_samples_seen: 0,
            batch_count: 0,
            previous_lambda: 0.0,
            n_features_in,
            fit_intercept,
            weight_sum: 0.0,
        }
    }

    pub fn n_parameters(&self) -> usize {
        self.n_features_in + usize::from(self.fit_intercept)
    }

    pub fn validate(&self) -> Result<(), CoreError> {
        let n_parameters = self.n_parameters();
        if self.coefficients.len() != n_parameters {
            return Err(CoreError::InvalidState(
                "coefficient shape does not match feature metadata",
            ));
        }
        if self.information.len() != n_parameters * n_parameters {
            return Err(CoreError::InvalidState(
                "information shape does not match feature metadata",
            ));
        }
        if !self.previous_lambda.is_finite() || self.previous_lambda < 0.0 {
            return Err(CoreError::InvalidState(
                "previous_lambda must be finite and non-negative",
            ));
        }
        if !self.weight_sum.is_finite() || self.weight_sum < 0.0 {
            return Err(CoreError::InvalidState(
                "weight_sum must be finite and non-negative",
            ));
        }
        if self.n_samples_seen > 0 && self.weight_sum == 0.0 {
            return Err(CoreError::InvalidState(
                "weight_sum must be positive after observing samples",
            ));
        }
        if self.coefficients.iter().any(|value| !value.is_finite())
            || self.information.iter().any(|value| !value.is_finite())
        {
            return Err(CoreError::InvalidState(
                "state arrays must contain only finite values",
            ));
        }
        Ok(())
    }
}

/// Borrowed C-contiguous row-major input for one complete batch.
#[derive(Clone, Copy, Debug)]
pub struct BatchView<'a, T: Scalar> {
    pub x_design: &'a [T],
    pub n_rows: usize,
    pub n_parameters: usize,
    pub y: &'a [T],
    pub sample_weight: Option<&'a [T]>,
    /// Frequency-weight sum calculated by the validated Python layer.
    pub batch_weight: f64,
}

impl<'a, T: Scalar> BatchView<'a, T> {
    pub fn validate(&self, state: &State<T>) -> Result<(), CoreError> {
        if self.n_rows == 0 {
            return Err(CoreError::InvalidBatch(
                "batch must contain at least one row",
            ));
        }
        if self.n_parameters != state.n_parameters() {
            return Err(CoreError::InvalidBatch(
                "design width does not match state parameters",
            ));
        }
        let expected_x = self
            .n_rows
            .checked_mul(self.n_parameters)
            .ok_or(CoreError::SizeOverflow)?;
        if self.x_design.len() != expected_x {
            return Err(CoreError::InvalidBatch(
                "design buffer is not a complete contiguous matrix",
            ));
        }
        if self.y.len() != self.n_rows {
            return Err(CoreError::InvalidBatch(
                "target length does not match design rows",
            ));
        }
        if !self.batch_weight.is_finite() || self.batch_weight <= 0.0 {
            return Err(CoreError::InvalidBatch(
                "batch_weight must be finite and positive",
            ));
        }
        if self.x_design.iter().any(|value| !value.is_finite())
            || self.y.iter().any(|value| !value.is_finite())
        {
            return Err(CoreError::InvalidBatch(
                "batch arrays must contain only finite values",
            ));
        }
        if let Some(weights) = self.sample_weight {
            if weights.len() != self.n_rows {
                return Err(CoreError::InvalidBatch(
                    "sample_weight length does not match design rows",
                ));
            }
            if weights
                .iter()
                .any(|value| !value.is_finite() || *value < T::zero())
            {
                return Err(CoreError::InvalidBatch(
                    "sample_weight must be finite and non-negative",
                ));
            }
        }
        Ok(())
    }
}

/// Numerical outcome of the most recent transition.
#[derive(Clone, Copy, Debug)]
pub struct Diagnostics {
    pub iterations: usize,
    pub converged: bool,
    pub objective: f64,
    pub lambda_value: f64,
    pub bandwidth: f64,
    pub used_regularized_fallback: bool,
}

/// One atomic state transition.
#[derive(Clone, Debug)]
pub struct Transition<T: Scalar> {
    pub state: State<T>,
    pub diagnostics: Diagnostics,
}

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("invalid native configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("invalid native state: {0}")]
    InvalidState(&'static str),
    #[error("invalid native batch: {0}")]
    InvalidBatch(&'static str),
    #[error("native buffer size overflow")]
    SizeOverflow,
    #[error("native scalar conversion failed")]
    ScalarConversion,
    #[error("native linear solve failed")]
    LinearSolve,
    #[error("native calculation produced a non-finite value")]
    NonFiniteResult,
}

pub fn scalar_from_f64<T: Scalar>(value: f64) -> Result<T, CoreError> {
    T::from_f64(value).ok_or(CoreError::ScalarConversion)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state_shape_follows_intercept_contract() {
        let with_intercept = State::<f64>::empty(3, true);
        assert_eq!(with_intercept.coefficients.len(), 4);
        assert_eq!(with_intercept.information.len(), 16);
        with_intercept.validate().unwrap();

        let without_intercept = State::<f32>::empty(3, false);
        assert_eq!(without_intercept.coefficients.len(), 3);
        assert_eq!(without_intercept.information.len(), 9);
        without_intercept.validate().unwrap();
    }

    #[test]
    fn validation_rejects_non_finite_portable_state() {
        let mut state = State::<f64>::empty(1, false);
        state.coefficients[0] = f64::NAN;
        assert!(matches!(state.validate(), Err(CoreError::InvalidState(_))));
    }
}
