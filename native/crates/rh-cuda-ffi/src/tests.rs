//! Behavioural tests for the safe wrapper.
//!
//! The ABI layout tests live in `sys`, next to the declarations they
//! constrain.

use crate::engine::CudaEngine;
use crate::types::{
    parse_penalty, tuning_flags, tuning_from_flags, CudaDtype, EngineTuning, UpdateConfig,
    ENGINE_FLAG_CUDA_GRAPHS, ENGINE_FLAG_FAST_MATH, SUPPORTED_PENALTIES,
};
use crate::validation::validate_update_config;
use rh_core::Penalty;

fn config(fit_intercept: bool) -> UpdateConfig {
    UpdateConfig {
        n_features_in: 3,
        fit_intercept,
        tau: 1.345,
        bandwidth_scale: 1.0,
        max_iter: 100,
        tolerance: 1e-6,
        ridge: 1e-8,
        penalty: Penalty::None,
        lambda_scale: 1.0,
    }
}

#[test]
fn parameter_shape_matches_intercept_contract() {
    assert!(validate_update_config(3, config(false)).is_ok());
    assert!(validate_update_config(4, config(true)).is_ok());
    assert!(validate_update_config(4, config(false)).is_err());
    assert!(validate_update_config(3, config(true)).is_err());
}

#[test]
fn l1_configuration_shares_the_core_numerical_policy() {
    let l1 = UpdateConfig {
        penalty: Penalty::L1,
        lambda_scale: 0.0,
        ..config(true)
    };
    assert!(validate_update_config(4, l1).is_ok());
    for lambda_scale in [-0.5, f64::NAN, f64::INFINITY] {
        let rejected = UpdateConfig { lambda_scale, ..l1 };
        assert!(validate_update_config(4, rejected).is_err());
    }
    let zero_iterations = UpdateConfig { max_iter: 0, ..l1 };
    assert!(validate_update_config(4, zero_iterations).is_err());
    let bad_tau = UpdateConfig { tau: 0.0, ..l1 };
    assert!(validate_update_config(4, bad_tau).is_err());
}

#[test]
fn penalty_names_parse_and_unknown_names_fail_closed() {
    assert_eq!(parse_penalty("none").unwrap(), Penalty::None);
    assert_eq!(parse_penalty("l1").unwrap(), Penalty::L1);
    assert!(parse_penalty("l2").is_err());
    assert!(parse_penalty("L1").is_err());
    assert_eq!(SUPPORTED_PENALTIES, ["none", "l1"]);
    for name in SUPPORTED_PENALTIES {
        assert!(parse_penalty(name).is_ok());
    }
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
