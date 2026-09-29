//! Property tests for the engine-independent validation contracts.
//!
//! Every native engine trusts `UpdateConfig::validate`, `State::validate` and
//! `BatchView::validate` to have rejected bad input before any arithmetic, so
//! these check each rule against an independently written statement of it
//! rather than against hand-picked examples.
//!
//! Failure persistence is off crate-wide (see `config`): a failing case prints
//! its minimal shrunk input, which belongs in a named regression test, not in
//! an untracked `proptest-regressions/` file.

use proptest::prelude::*;
use rh_core::{scalar_from_f64, BatchView, CoreError, Penalty, State, UpdateConfig};

fn config() -> ProptestConfig {
    ProptestConfig {
        // Modest by default so `cargo test` stays fast; PROPTEST_CASES raises
        // it for a local soak run.
        cases: std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|cases| cases.parse().ok())
            .unwrap_or(64),
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

/// Any `f64`, deliberately biased towards the values that sit on a boundary
/// of some validation rule.
fn edge_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0),
        Just(-0.0),
        Just(f64::MIN_POSITIVE),
        Just(-f64::MIN_POSITIVE),
        Just(f64::NAN),
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
        Just(f64::MAX),
        -10.0..10.0_f64,
        any::<f64>(),
    ]
}

fn penalty() -> impl Strategy<Value = Penalty> {
    prop_oneof![Just(Penalty::None), Just(Penalty::L1)]
}

prop_compose! {
    fn valid_update_config()(
        tau in 1.0e-3..10.0_f64,
        penalty in penalty(),
        lambda_scale in 0.0..10.0_f64,
        bandwidth_scale in 1.0e-3..10.0_f64,
        max_iter in 1_usize..1_000,
        tolerance in 1.0e-12..1.0_f64,
        ridge in 0.0..1.0_f64,
    ) -> UpdateConfig {
        UpdateConfig { tau, penalty, lambda_scale, bandwidth_scale, max_iter, tolerance, ridge }
    }
}

/// A valid configuration with up to three fields overwritten by boundary
/// values, so every rule's edge is reached often rather than by luck.
fn any_update_config() -> impl Strategy<Value = UpdateConfig> {
    (
        valid_update_config(),
        prop::collection::vec((0_usize..6, edge_f64()), 0..=3),
    )
        .prop_map(|(mut config, edits)| {
            for (field, value) in edits {
                match field {
                    0 => config.tau = value,
                    1 => config.lambda_scale = value,
                    2 => config.bandwidth_scale = value,
                    3 => config.max_iter = 0,
                    4 => config.tolerance = value,
                    _ => config.ridge = value,
                }
            }
            config
        })
}

/// The field `validate` must name first, written from the documented domain of
/// each field rather than from the implementation. `None` means valid.
fn first_invalid_field(config: &UpdateConfig) -> Option<&'static str> {
    let positive = |value: f64| value.is_finite() && value > 0.0;
    let non_negative = |value: f64| value.is_finite() && value >= 0.0;
    if !positive(config.tau) {
        Some("tau")
    } else if !non_negative(config.lambda_scale) {
        Some("lambda_scale")
    } else if !positive(config.bandwidth_scale) {
        Some("bandwidth_scale")
    } else if config.max_iter == 0 {
        Some("max_iter")
    } else if !positive(config.tolerance) {
        Some("tolerance")
    } else if !non_negative(config.ridge) {
        Some("ridge")
    } else {
        None
    }
}

fn finite_values(length: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-1.0e6..1.0e6_f64, length)
}

fn non_finite() -> impl Strategy<Value = f64> {
    prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)]
}

prop_compose! {
    /// A well-formed batch for `State::empty(n_features_in, fit_intercept)`.
    fn valid_batch()(
        n_features_in in 0_usize..6,
        fit_intercept in any::<bool>(),
        n_rows in 1_usize..24,
    )(
        x in finite_values(n_rows * (n_features_in + usize::from(fit_intercept))),
        y in finite_values(n_rows),
        weights in prop::option::of(prop::collection::vec(0.0..10.0_f64, n_rows)),
        batch_weight in 1.0e-6..1.0e6_f64,
        n_features_in in Just(n_features_in),
        fit_intercept in Just(fit_intercept),
        n_rows in Just(n_rows),
    ) -> (State<f64>, Vec<f64>, Vec<f64>, Option<Vec<f64>>, usize, f64) {
        (State::empty(n_features_in, fit_intercept), x, y, weights, n_rows, batch_weight)
    }
}

fn view<'a>(
    state: &State<f64>,
    x: &'a [f64],
    y: &'a [f64],
    weights: Option<&'a [f64]>,
    n_rows: usize,
    batch_weight: f64,
) -> BatchView<'a, f64> {
    BatchView {
        x_design: x,
        n_rows,
        n_parameters: state.n_parameters(),
        y,
        sample_weight: weights,
        batch_weight,
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn update_config_validate_names_the_first_invalid_field(config in any_update_config()) {
        match (config.validate(), first_invalid_field(&config)) {
            (Ok(()), None) => {}
            (Err(CoreError::InvalidConfig(message)), Some(field)) => {
                prop_assert!(
                    message.starts_with(field),
                    "{config:?}: expected a {field} error, got {message:?}"
                );
            }
            (result, expected) => {
                prop_assert!(false, "{config:?}: got {result:?}, expected invalid field {expected:?}");
            }
        }
    }

    #[test]
    fn a_non_finite_field_is_always_rejected(
        config in any_update_config(),
        field in 0_usize..5,
        value in non_finite(),
    ) {
        let mut config = config;
        match field {
            0 => config.tau = value,
            1 => config.lambda_scale = value,
            2 => config.bandwidth_scale = value,
            3 => config.tolerance = value,
            _ => config.ridge = value,
        }
        prop_assert!(matches!(config.validate(), Err(CoreError::InvalidConfig(_))));
    }

    #[test]
    fn empty_state_has_the_intercept_shape_and_validates(
        n_features_in in 0_usize..64,
        fit_intercept in any::<bool>(),
    ) {
        let state = State::<f64>::empty(n_features_in, fit_intercept);
        let p = n_features_in + usize::from(fit_intercept);
        prop_assert_eq!(state.n_parameters(), p);
        prop_assert_eq!(state.coefficients.len(), p);
        prop_assert_eq!(state.information.len(), p * p);
        prop_assert!(state.coefficients.iter().chain(&state.information).all(|v| *v == 0.0));
        prop_assert!(state.validate().is_ok());
    }

    #[test]
    fn state_rejects_any_single_non_finite_entry(
        n_features_in in 1_usize..8,
        fit_intercept in any::<bool>(),
        in_information in any::<bool>(),
        index in any::<prop::sample::Index>(),
        value in non_finite(),
    ) {
        let mut state = State::<f64>::empty(n_features_in, fit_intercept);
        let target = if in_information {
            &mut state.information
        } else {
            &mut state.coefficients
        };
        let position = index.index(target.len());
        target[position] = value;
        prop_assert!(matches!(state.validate(), Err(CoreError::InvalidState(_))));
    }

    #[test]
    fn state_scalars_must_be_finite_non_negative_and_consistent(
        previous_lambda in edge_f64(),
        weight_sum in edge_f64(),
        n_samples_seen in 0_usize..4,
    ) {
        let mut state = State::<f64>::empty(2, true);
        state.previous_lambda = previous_lambda;
        state.weight_sum = weight_sum;
        state.n_samples_seen = n_samples_seen;
        let valid = previous_lambda.is_finite()
            && previous_lambda >= 0.0
            && weight_sum.is_finite()
            && weight_sum >= 0.0
            && !(n_samples_seen > 0 && weight_sum == 0.0);
        prop_assert_eq!(state.validate().is_ok(), valid);
    }

    #[test]
    fn well_formed_batches_validate((state, x, y, weights, n_rows, batch_weight) in valid_batch()) {
        let batch = view(&state, &x, &y, weights.as_deref(), n_rows, batch_weight);
        prop_assert!(batch.validate(&state).is_ok());
    }

    #[test]
    fn batch_rejects_a_single_non_finite_observation(
        (state, mut x, mut y, weights, n_rows, batch_weight) in valid_batch(),
        in_design in any::<bool>(),
        index in any::<prop::sample::Index>(),
        value in non_finite(),
    ) {
        // A zero-width design (no features, no intercept) has no entry to corrupt.
        let target = if in_design && !x.is_empty() { &mut x } else { &mut y };
        let position = index.index(target.len());
        target[position] = value;
        let batch = view(&state, &x, &y, weights.as_deref(), n_rows, batch_weight);
        prop_assert!(matches!(batch.validate(&state), Err(CoreError::InvalidBatch(_))));
    }

    #[test]
    fn batch_rejects_negative_or_non_finite_sample_weights(
        (state, x, y, _weights, n_rows, batch_weight) in valid_batch(),
        index in any::<prop::sample::Index>(),
        value in prop_oneof![-1.0e6..-1.0e-300_f64, non_finite()],
    ) {
        let mut weights = vec![1.0; n_rows];
        weights[index.index(n_rows)] = value;
        let batch = view(&state, &x, &y, Some(&weights), n_rows, batch_weight);
        prop_assert!(matches!(batch.validate(&state), Err(CoreError::InvalidBatch(_))));
    }

    #[test]
    fn batch_rejects_shape_mismatches(
        (state, x, y, weights, n_rows, batch_weight) in valid_batch(),
        mismatch in 0_usize..4,
    ) {
        let short_y = &y[..y.len() - 1];
        let short_weights = vec![1.0; n_rows - 1];
        let mut batch = view(&state, &x, &y, weights.as_deref(), n_rows, batch_weight);
        match mismatch {
            0 => batch.y = short_y,
            1 => batch.sample_weight = Some(&short_weights),
            2 => batch.n_parameters += 1,
            _ => batch.n_rows += 1,
        }
        prop_assert!(matches!(batch.validate(&state), Err(CoreError::InvalidBatch(_))));
    }

    #[test]
    fn batch_weight_must_be_finite_and_positive(
        (state, x, y, weights, n_rows, _batch_weight) in valid_batch(),
        batch_weight in edge_f64(),
    ) {
        let batch = view(&state, &x, &y, weights.as_deref(), n_rows, batch_weight);
        let valid = batch_weight.is_finite() && batch_weight > 0.0;
        prop_assert_eq!(batch.validate(&state).is_ok(), valid);
    }

    #[test]
    fn scalar_conversion_is_total_and_matches_a_cast(value in any::<f64>()) {
        // Both engines' scalars accept every f64, so `ScalarConversion` can
        // only ever come from a future scalar type.
        let wide = scalar_from_f64::<f64>(value).unwrap();
        let narrow = scalar_from_f64::<f32>(value).unwrap();
        prop_assert_eq!(wide.to_bits(), value.to_bits());
        if value.is_nan() {
            prop_assert!(narrow.is_nan());
        } else {
            prop_assert_eq!(narrow.to_bits(), (value as f32).to_bits());
        }
    }
}
