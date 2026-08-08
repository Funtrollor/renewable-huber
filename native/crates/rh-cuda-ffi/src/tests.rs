//! Behavioural tests for the safe wrapper.
//!
//! The ABI layout tests live in `sys`, next to the declarations they
//! constrain.

use crate::engine::CudaEngine;
use crate::types::{
    tuning_flags, tuning_from_flags, CudaDtype, EngineTuning, UnpenalizedConfig,
    ENGINE_FLAG_CUDA_GRAPHS, ENGINE_FLAG_FAST_MATH,
};
use crate::validation::validate_unpenalized_config;

fn config(fit_intercept: bool) -> UnpenalizedConfig {
    UnpenalizedConfig {
        n_features_in: 3,
        fit_intercept,
        tau: 1.345,
        bandwidth_scale: 1.0,
        max_iter: 100,
        tolerance: 1e-6,
        ridge: 1e-8,
    }
}

#[test]
fn parameter_shape_matches_intercept_contract() {
    assert!(validate_unpenalized_config(3, config(false)).is_ok());
    assert!(validate_unpenalized_config(4, config(true)).is_ok());
    assert!(validate_unpenalized_config(4, config(false)).is_err());
    assert!(validate_unpenalized_config(3, config(true)).is_err());
}

#[test]
fn tuning_flags_round_trip() {
    let tuning = EngineTuning {
        cuda_graphs: true,
        fast_math: true,
    };
    assert_eq!(
        tuning_flags(tuning),
        ENGINE_FLAG_CUDA_GRAPHS | ENGINE_FLAG_FAST_MATH
    );
    assert_eq!(tuning_from_flags(tuning_flags(tuning)), tuning);
}

#[test]
fn fast_math_rejects_float64_before_cuda_initialization() {
    let error = CudaEngine::create_with_tuning(
        CudaDtype::Float64,
        2,
        0,
        EngineTuning {
            cuda_graphs: false,
            fast_math: true,
        },
    )
    .err()
    .expect("float64 fast math must fail");
    assert!(error.to_string().contains("float32"));
}
