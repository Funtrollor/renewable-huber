//! Property tests for the numeric kernels and the invariants the engine builds
//! on them.
//!
//! These live inside the crate because the kernels are `pub(crate)`.
//!
//! Tolerances. Exact (`==`) comparisons are used wherever both sides perform
//! the same IEEE operations on the same operands, e.g. odd symmetry of
//! `huber_score`, or a parallel path that computes the same per-row dot
//! product as the serial one. Where two sides sum the same `n` terms in a
//! different order (GEMM blocking, partial Gram matrices, parallel gradient
//! reductions), the tolerance is the standard worst-case bound for recursive
//! summation, `|computed - exact| <= (n + c) * eps * sum(|term|)` with a small
//! constant `c` for the per-term products, and a factor two because the naive
//! reference carries its own rounding error. `eps` is the scalar's own machine
//! epsilon, so the same check is meaningful for `f32` and `f64`.
//!
//! Large-batch cases fill their matrices from a seeded generator rather than
//! from per-element strategies: a million-element `Vec` strategy would build a
//! million shrinkable value trees per case. Shrinking still minimises the shape
//! and the seed. Those cases run few iterations and in a fixed four-thread
//! Rayon pool, so the parallel branches are taken on any host.
//!
//! Failure persistence is off (see `config`), consistently with the rh-core
//! property tests: a failure prints its minimal input, which belongs in a named
//! regression test rather than an untracked `proptest-regressions/` file.

use num_traits::Float;
use proptest::prelude::*;
use rh_core::{scalar_from_f64, BatchView, Penalty, State, UpdateConfig};

use crate::kernels::gram::{gradient_from_score, prefer_row_chunk_gradient, weighted_gram};
use crate::kernels::objective::{diagnostic_objective, smooth_objective};
use crate::kernels::vector::{
    dot, huber_loss, huber_score, norm, residual, sign, smoothed_curvature, soft_threshold,
};
use crate::kernels::{bandwidth, lambda_value};
use crate::scalar::CpuScalar;
use crate::workspace::Workspace;
use crate::{predict, CpuEngine, PARALLEL_GRAM_WORK, PARALLEL_VECTOR_WORK};

fn cases(default: u32) -> u32 {
    // PROPTEST_CASES raises every count for a local soak run; the defaults
    // keep `cargo test` fast in the unoptimised test profile.
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn config(default_cases: u32) -> ProptestConfig {
    ProptestConfig {
        cases: cases(default_cases),
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

fn eps<T: CpuScalar>() -> f64 {
    <T as Float>::epsilon().to_f64().unwrap()
}

fn to<T: CpuScalar>(value: f64) -> T {
    scalar_from_f64(value).unwrap()
}

fn from<T: CpuScalar>(value: T) -> f64 {
    value.to_f64().unwrap()
}

/// Multiples of a positive scale, biased towards the knots at +-1.
fn around_one() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0),
        Just(1.0),
        Just(-1.0),
        0.999..1.001_f64,
        -1.001..-0.999_f64,
        -5.0..5.0_f64,
    ]
}

/// SplitMix64: a dependency-free deterministic filler for large matrices.
fn filled(seed: u64, length: usize, low: f64, high: f64) -> Vec<f64> {
    let mut state = seed;
    (0..length)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            low + (high - low) * ((z >> 11) as f64 / (1_u64 << 53) as f64)
        })
        .collect()
}

fn four_threads<R: Send>(work: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(work)
}

// -- scalar kernels ----------------------------------------------------------

fn check_huber<T: CpuScalar>(tau: f64, a: f64, b: f64) -> Result<(), TestCaseError> {
    let tau_t = to::<T>(tau);
    let tau = from(tau_t);
    let (a_t, b_t) = (to::<T>(a), to::<T>(b));
    let (a, b) = (from(a_t), from(b_t));
    let score = from(huber_score(a_t, tau_t));

    // Bounded by tau, the identity inside [-tau, tau], clipped outside.
    prop_assert!(score.abs() <= tau);
    if a.abs() <= tau {
        prop_assert_eq!(score, a);
    } else {
        prop_assert_eq!(score, tau.copysign(a));
    }
    // Odd symmetry is exact: negation commutes with each comparison.
    prop_assert_eq!(huber_score(-a_t, tau_t), -huber_score(a_t, tau_t));
    // Monotone non-decreasing.
    let (low, high) = if a <= b { (a_t, b_t) } else { (b_t, a_t) };
    prop_assert!(huber_score(low, tau_t) <= huber_score(high, tau_t));

    let loss_a = from(huber_loss(a_t, tau_t));
    let loss_b = from(huber_loss(b_t, tau_t));
    prop_assert!(loss_a >= 0.0);
    prop_assert_eq!(huber_loss(-a_t, tau_t), huber_loss(a_t, tau_t));
    // Below both the quadratic and the linear envelope, up to a few roundings.
    let slack = 8.0 * eps::<T>() * (a * a + tau * a.abs());
    prop_assert!(loss_a <= 0.5 * a * a + slack, "{loss_a} > a^2/2 at a={a}");
    prop_assert!(
        loss_a <= tau * a.abs() + slack,
        "{loss_a} > tau|a| at a={a}"
    );
    // huber_score is the derivative of huber_loss: the loss lies above every
    // tangent line, which is convexity plus consistency of the two kernels.
    let score_b = from(huber_score(b_t, tau_t));
    let tangent = loss_b + score_b * (a - b);
    let slack = 16.0 * eps::<T>() * (loss_a.abs() + loss_b.abs() + (score_b * (a - b)).abs());
    prop_assert!(
        loss_a >= tangent - slack,
        "loss {loss_a} below tangent {tangent}"
    );
    Ok(())
}

fn check_soft_threshold<T: CpuScalar>(threshold: f64, a: f64, b: f64) -> Result<(), TestCaseError> {
    let t = to::<T>(threshold);
    let (a_t, b_t) = (to::<T>(a), to::<T>(b));
    let out = soft_threshold(a_t, t);
    let (a, out_f) = (from(a_t), from(out));

    // Shrinkage: never grows, never flips sign, zero inside the threshold.
    prop_assert!(out_f.abs() <= a.abs());
    prop_assert!(out_f == 0.0 || out_f.signum() == a.signum());
    if a.abs() <= from(t) {
        prop_assert_eq!(out_f, 0.0);
    }
    prop_assert_eq!(soft_threshold(-a_t, t), -out);
    // A zero threshold is the identity, which is what leaves the intercept
    // coordinate untouched in the proximal step.
    prop_assert_eq!(soft_threshold(a_t, T::zero()), a_t);
    // Proximal operators are non-expansive.
    let distance = (from(soft_threshold(b_t, t)) - out_f).abs();
    let b = from(b_t);
    let slack = 4.0 * eps::<T>() * (a.abs() + b.abs() + from(t));
    prop_assert!(distance <= (a - b).abs() + slack);
    // `sign` reconstructs its argument.
    prop_assert_eq!(sign(a_t) * Float::abs(a_t), a_t);
    Ok(())
}

fn check_curvature<T: CpuScalar>(
    tau: f64,
    bandwidth_ratio: f64,
    multiples: &[f64],
) -> Result<(), TestCaseError> {
    let tau_t = to::<T>(tau);
    let tau = from(tau_t);
    let bandwidth = tau * bandwidth_ratio;
    let h = from(to::<T>(bandwidth.min(tau)));
    let residuals = multiples
        .iter()
        .map(|multiple| to::<T>(multiple * tau))
        .collect::<Vec<_>>();
    let negated = residuals.iter().map(|value| -*value).collect::<Vec<_>>();
    let mut curvature = vec![T::zero(); residuals.len()];
    let mut mirrored = vec![T::zero(); residuals.len()];
    smoothed_curvature(&residuals, tau, bandwidth, &mut curvature).unwrap();
    smoothed_curvature(&negated, tau, bandwidth, &mut mirrored).unwrap();

    // Inside the band, `v` may sit up to one rounding of `tau - h` beyond the
    // exact knot, which the division by `2h` amplifies by `tau / h`.
    let slack = 4.0 * eps::<T>() * (1.0 + tau / h);
    for ((value, weight), mirror) in residuals.iter().zip(&curvature).zip(&mirrored) {
        let (value, weight) = (from(*value), from(*weight));
        prop_assert!(
            (-slack..=1.0 + slack).contains(&weight),
            "{weight} at {value}"
        );
        // Even in the residual, exactly.
        prop_assert_eq!(*mirror, to::<T>(weight));
        // Exactly 1 well inside the quadratic zone and 0 well outside it.
        if value.abs() < tau - h - slack * h {
            prop_assert_eq!(weight, 1.0);
        }
        if value.abs() > tau + h + slack * h {
            prop_assert_eq!(weight, 0.0);
        }
    }
    Ok(())
}

fn check_dot<T: CpuScalar>(left: &[f64], right: &[f64]) -> Result<(), TestCaseError> {
    let left = left.iter().map(|value| to::<T>(*value)).collect::<Vec<_>>();
    let right = right
        .iter()
        .map(|value| to::<T>(*value))
        .collect::<Vec<_>>();
    prop_assert_eq!(dot(&left, &right), dot(&right, &left));
    let (left_norm, right_norm) = (from(norm(&left)), from(norm(&right)));
    prop_assert!(left_norm >= 0.0);
    // Cauchy-Schwarz, with the summation bound for n terms on each side.
    let n = left.len() as f64;
    let bound = left_norm * right_norm * (1.0 + 2.0 * (n + 2.0) * eps::<T>());
    prop_assert!(from(dot(&left, &right)).abs() <= bound);
    Ok(())
}

proptest! {
    #![proptest_config(config(64))]

    #[test]
    fn huber_score_and_loss_are_bounded_odd_even_and_consistent(
        tau in 1.0e-2..10.0_f64,
        a in around_one(),
        b in around_one(),
    ) {
        check_huber::<f64>(tau, a * tau, b * tau)?;
        check_huber::<f32>(tau, a * tau, b * tau)?;
    }

    #[test]
    fn soft_threshold_is_a_sign_preserving_non_expansive_shrinkage(
        threshold in prop_oneof![Just(0.0), 1.0e-3..10.0_f64],
        a in around_one(),
        b in around_one(),
    ) {
        let scale = threshold.max(1.0);
        check_soft_threshold::<f64>(threshold, a * scale, b * scale)?;
        check_soft_threshold::<f32>(threshold, a * scale, b * scale)?;
    }

    #[test]
    fn smoothed_curvature_is_an_even_unit_interval_weight(
        tau in 1.0e-2..10.0_f64,
        bandwidth_ratio in prop_oneof![1.0e-4..1.0_f64, 1.0..3.0_f64],
        multiples in prop::collection::vec(
            prop_oneof![around_one(), (0.0..2.0_f64).prop_map(|m| 1.0 - m), -3.0..3.0_f64],
            1..32,
        ),
    ) {
        check_curvature::<f64>(tau, bandwidth_ratio, &multiples)?;
        check_curvature::<f32>(tau, bandwidth_ratio, &multiples)?;
    }

    #[test]
    fn dot_is_symmetric_and_obeys_cauchy_schwarz(
        pairs in prop::collection::vec((-1.0e3..1.0e3_f64, -1.0e3..1.0e3_f64), 0..64),
    ) {
        let (left, right): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
        check_dot::<f64>(&left, &right)?;
        check_dot::<f32>(&left, &right)?;
    }

    #[test]
    fn bandwidth_is_positive_capped_by_tau_and_shrinks_with_data(
        n_total in 1.0..1.0e9_f64,
        growth in 1.0..1.0e3_f64,
        n_features in 0_usize..10_000,
        scale in 1.0e-3..10.0_f64,
        tau in 1.0e-3..10.0_f64,
    ) {
        let h = bandwidth(n_total, n_features, scale, tau);
        prop_assert!(h > 0.0 && h <= tau);
        prop_assert!(bandwidth(n_total * growth, n_features, scale, tau) <= h);
    }

    #[test]
    fn lambda_is_zero_unpenalized_and_linear_in_its_scale(
        n_total in 1.0..1.0e9_f64,
        growth in 1.0..1.0e3_f64,
        n_features in 0_usize..10_000,
        lambda_scale in 0.0..100.0_f64,
        tau in 1.0e-3..10.0_f64,
    ) {
        let penalized = UpdateConfig { tau, lambda_scale, penalty: Penalty::L1, ..UpdateConfig::default() };
        let unpenalized = UpdateConfig { penalty: Penalty::None, ..penalized };
        prop_assert_eq!(lambda_value(n_total, n_features, unpenalized), 0.0);
        let lambda = lambda_value(n_total, n_features, penalized);
        prop_assert!(lambda >= 0.0 && lambda.is_finite());
        // Doubling is exact in binary floating point, so this is exact too.
        let doubled = UpdateConfig { lambda_scale: 2.0 * lambda_scale, ..penalized };
        prop_assert_eq!(lambda_value(n_total, n_features, doubled), 2.0 * lambda);
        prop_assert!(lambda_value(n_total * growth, n_features, penalized) <= lambda);
    }
}

// -- batch kernels -------------------------------------------------------------

prop_compose! {
    /// A small dense problem: `(n_rows, n_parameters, x, y, weights)`.
    fn small_problem()(n_rows in 1_usize..48, n_parameters in 1_usize..8)(
        x in prop::collection::vec(-4.0..4.0_f64, n_rows * n_parameters),
        y in prop::collection::vec(-8.0..8.0_f64, n_rows),
        weights in prop::option::of(prop::collection::vec(0.0..3.0_f64, n_rows)),
        n_rows in Just(n_rows),
        n_parameters in Just(n_parameters),
    ) -> (usize, usize, Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
        (n_rows, n_parameters, x, y, weights)
    }
}

fn batch<'a, T: CpuScalar>(
    x: &'a [T],
    y: &'a [T],
    weights: Option<&'a [T]>,
    n_rows: usize,
    n_parameters: usize,
) -> BatchView<'a, T> {
    BatchView {
        x_design: x,
        n_rows,
        n_parameters,
        y,
        sample_weight: weights,
        batch_weight: n_rows as f64,
    }
}

fn cast<T: CpuScalar>(values: &[f64]) -> Vec<T> {
    values.iter().map(|value| to::<T>(*value)).collect()
}

/// `weighted_gram` against a naive triple loop, entry by entry.
fn check_weighted_gram<T: CpuScalar>(
    n_rows: usize,
    p: usize,
    x: &[f64],
    curvature: &[f64],
    weights: Option<&[f64]>,
) -> Result<(), TestCaseError> {
    let (x_t, y_t, curvature_t) = (cast::<T>(x), vec![T::zero(); n_rows], cast::<T>(curvature));
    let weights_t = weights.map(cast::<T>);
    let view = batch(&x_t, &y_t, weights_t.as_deref(), n_rows, p);
    let mut workspace = Workspace::<T>::default();
    workspace.reserve(n_rows, p).unwrap();
    weighted_gram(
        view,
        &curvature_t,
        &mut workspace.weighted_rows,
        &mut workspace.partial_grams,
        &mut workspace.gram,
    )
    .unwrap();
    if n_rows * p * p >= PARALLEL_GRAM_WORK {
        // Confirms the partial-Gram reduction ran, not the single GEMM.
        prop_assert!(workspace.partial_grams.len() >= 2 * p * p);
    }

    let x = x_t.iter().copied().map(from).collect::<Vec<_>>();
    let row_weight = (0..n_rows)
        .map(|row| from(curvature_t[row]) * weights_t.as_deref().map_or(1.0, |w| from(w[row])))
        .collect::<Vec<_>>();
    let unit = eps::<T>() * 2.0 * (n_rows as f64 + 4.0);
    for i in 0..p {
        for j in 0..p {
            let (mut expected, mut magnitude) = (0.0, 0.0);
            for (row, weight) in row_weight.iter().enumerate() {
                let term = x[row * p + i] * x[row * p + j] * weight;
                expected += term;
                magnitude += term.abs();
            }
            let actual = from(workspace.gram[i * p + j]);
            let mirror = from(workspace.gram[j * p + i]);
            prop_assert!(
                (actual - expected).abs() <= unit * magnitude,
                "G[{i},{j}] = {actual}, reference {expected}, bound {}",
                unit * magnitude
            );
            prop_assert!((actual - mirror).abs() <= 2.0 * unit * magnitude);
        }
    }
    Ok(())
}

/// `gradient_from_score` with no history reduces to `-X^T score / n_total`.
fn check_gradient_reduction(
    n_rows: usize,
    p: usize,
    x: &[f64],
    score: &[f64],
    penalty: Penalty,
) -> Result<(), TestCaseError> {
    let y = vec![0.0; n_rows];
    let view = batch(x, &y, None, n_rows, p);
    let mut state = State::<f64>::empty(p, false);
    // n_samples_seen = 0 keeps the L1 historical sign term out, so both
    // penalties must reduce to the same product.
    state.coefficients = filled(p as u64, p, -1.0, 1.0);
    let beta = state.coefficients.clone();
    let config = UpdateConfig {
        penalty,
        ..UpdateConfig::default()
    };
    let n_total = n_rows as f64;
    let mut workspace = Workspace::<f64>::default();
    workspace.reserve(n_rows, p).unwrap();
    workspace.score.copy_from_slice(score);
    gradient_from_score(view, &beta, &state, config, n_total, &mut workspace).unwrap();

    let unit = eps::<f64>() * 2.0 * (n_rows as f64 + 4.0);
    for j in 0..p {
        let (mut expected, mut magnitude) = (0.0, 0.0);
        for row in 0..n_rows {
            let term = x[row * p + j] * score[row];
            expected += term;
            magnitude += term.abs();
        }
        let actual = workspace.gradient[j];
        prop_assert!(
            (actual + expected / n_total).abs() <= unit * magnitude / n_total,
            "gradient[{j}] = {actual}, reference {}",
            -expected / n_total
        );
    }
    Ok(())
}

proptest! {
    #![proptest_config(config(64))]

    #[test]
    fn weighted_gram_matches_a_naive_reference(
        (n_rows, p, x, _y, weights) in small_problem(),
        curvature_seed in any::<u64>(),
    ) {
        let curvature = filled(curvature_seed, n_rows, 0.0, 1.0);
        check_weighted_gram::<f64>(n_rows, p, &x, &curvature, weights.as_deref())?;
        check_weighted_gram::<f32>(n_rows, p, &x, &curvature, weights.as_deref())?;
    }

    #[test]
    fn serial_gradient_matches_a_naive_reduction(
        (n_rows, p, x, y, _weights) in small_problem(),
        penalty in prop_oneof![Just(Penalty::None), Just(Penalty::L1)],
    ) {
        check_gradient_reduction(n_rows, p, &x, &y, penalty)?;
    }

    #[test]
    fn residual_is_target_minus_prediction_row_by_row(
        (n_rows, p, x, y, _weights) in small_problem(),
        beta_seed in any::<u64>(),
    ) {
        let beta = filled(beta_seed, p, -2.0, 2.0);
        let mut output = vec![0.0; n_rows];
        residual(&x, n_rows, p, &beta, &y, &mut output);
        let prediction = predict(&x, n_rows, p, &beta).unwrap();
        for row in 0..n_rows {
            prop_assert_eq!(output[row], y[row] - prediction[row]);
        }
    }

    /// The L1 gradient differs from the unpenalized one only by the
    /// historical sign term, and never on the intercept coordinate.
    #[test]
    fn l1_gradient_term_never_touches_the_intercept(
        (n_rows, p, x, y, _weights) in small_problem(),
        fit_intercept in any::<bool>(),
        seed in any::<u64>(),
        previous_lambda in 0.0..5.0_f64,
    ) {
        let n_features_in = p - usize::from(fit_intercept);
        let mut state = State::<f64>::empty(n_features_in, fit_intercept);
        state.coefficients = filled(seed, p, -2.0, 2.0);
        state.information = filled(seed ^ 1, p * p, -1.0, 1.0);
        state.n_samples_seen = 10;
        state.weight_sum = 10.0;
        state.previous_lambda = previous_lambda;
        let beta = filled(seed ^ 2, p, -2.0, 2.0);
        let score = filled(seed ^ 3, n_rows, -1.0, 1.0);
        let view = batch(&x, &y, None, n_rows, p);
        let n_total = state.weight_sum + n_rows as f64;

        let mut gradients = Vec::new();
        for penalty in [Penalty::None, Penalty::L1] {
            let config = UpdateConfig { penalty, ..UpdateConfig::default() };
            let mut workspace = Workspace::<f64>::default();
            workspace.reserve(n_rows, p).unwrap();
            workspace.score.copy_from_slice(&score);
            gradient_from_score(view, &beta, &state, config, n_total, &mut workspace).unwrap();
            gradients.push(workspace.gradient.clone());
        }
        let historical_scale = state.weight_sum / n_total * state.previous_lambda;
        let (unpenalized, penalized) = (&gradients[0], &gradients[1]);
        for (j, coefficient) in state.coefficients.iter().enumerate() {
            if fit_intercept && j + 1 == p {
                prop_assert_eq!(penalized[j], unpenalized[j]);
            } else {
                prop_assert_eq!(
                    penalized[j],
                    unpenalized[j] - historical_scale * sign(*coefficient)
                );
            }
        }
    }

    /// Likewise the reported L1 objective does not depend on the intercept.
    #[test]
    fn l1_diagnostic_objective_ignores_the_intercept(
        coefficients in prop::collection::vec(-5.0..5.0_f64, 1..8),
        other_intercept in -5.0..5.0_f64,
        fit_intercept in any::<bool>(),
        smooth in -10.0..10.0_f64,
        lambda in 0.0..3.0_f64,
    ) {
        let p = coefficients.len();
        // An intercept needs at least one other coefficient to be meaningful.
        let fit_intercept = fit_intercept && p > 1;
        let state = State::<f64>::empty(p - usize::from(fit_intercept), fit_intercept);
        let penalized = diagnostic_objective(smooth, &coefficients, &state, Penalty::L1, lambda);
        let expected_l1 = coefficients[..p - usize::from(fit_intercept)]
            .iter()
            .map(|value| value.abs())
            .sum::<f64>();
        prop_assert_eq!(penalized, smooth + lambda * expected_l1);
        prop_assert_eq!(diagnostic_objective(smooth, &coefficients, &state, Penalty::None, lambda), smooth);
        if fit_intercept {
            let mut moved = coefficients.clone();
            moved[p - 1] = other_intercept;
            prop_assert_eq!(
                diagnostic_objective(smooth, &moved, &state, Penalty::L1, lambda),
                penalized
            );
        }
    }
}

// -- parallel branches ---------------------------------------------------------

proptest! {
    // Each case moves a million or more elements through the unoptimised test
    // profile, so a handful of cases is what keeps `cargo test` quick.
    #![proptest_config(config(4))]

    #[test]
    fn parallel_residual_and_predict_match_the_serial_row_kernel(
        p in 1_usize..16,
        extra_rows in 0_usize..64,
        seed in any::<u64>(),
    ) {
        let n_rows = PARALLEL_VECTOR_WORK.div_ceil(p) + extra_rows;
        let x = filled(seed, n_rows * p, -3.0, 3.0);
        let y = filled(seed ^ 1, n_rows, -3.0, 3.0);
        let beta = filled(seed ^ 2, p, -1.0, 1.0);
        let (whole_residual, whole_prediction) = four_threads(|| {
            let mut output = vec![0.0; n_rows];
            residual(&x, n_rows, p, &beta, &y, &mut output);
            (output, predict(&x, n_rows, p, &beta).unwrap())
        });
        // A one-row call is always below the parallel threshold.
        for row in 0..n_rows {
            let slice = &x[row * p..(row + 1) * p];
            let mut single = [0.0];
            residual(slice, 1, p, &beta, &y[row..=row], &mut single);
            prop_assert_eq!(whole_residual[row], single[0]);
            prop_assert_eq!(whole_prediction[row], predict(slice, 1, p, &beta).unwrap()[0]);
        }
    }

    #[test]
    fn partial_gram_reduction_matches_a_naive_reference(
        p in 6_usize..14,
        extra_rows in 0_usize..512,
        seed in any::<u64>(),
        weighted in any::<bool>(),
    ) {
        let n_rows = PARALLEL_GRAM_WORK.div_ceil(p * p) + extra_rows;
        let x = filled(seed, n_rows * p, -2.0, 2.0);
        let curvature = filled(seed ^ 1, n_rows, 0.0, 1.0);
        let weights = weighted.then(|| filled(seed ^ 2, n_rows, 0.0, 3.0));
        four_threads(|| check_weighted_gram::<f64>(n_rows, p, &x, &curvature, weights.as_deref()))?;
    }

    #[test]
    fn parallel_gradient_paths_match_a_naive_reduction(
        row_chunked in any::<bool>(),
        width in 0_usize..8,
        extra_rows in 0_usize..64,
        seed in any::<u64>(),
    ) {
        // Row-chunked accumulation for narrow f64 designs, parameter
        // partitioning for wide ones; see `prefer_row_chunk_gradient`.
        let p = if row_chunked { 2 + width } else { 65 + width };
        prop_assert_eq!(prefer_row_chunk_gradient::<f64>(p), row_chunked);
        let n_rows = PARALLEL_VECTOR_WORK.div_ceil(p) + extra_rows;
        let x = filled(seed, n_rows * p, -2.0, 2.0);
        let score = filled(seed ^ 1, n_rows, -1.345, 1.345);
        four_threads(|| check_gradient_reduction(n_rows, p, &x, &score, Penalty::None))?;
    }
}

// -- engine invariants ---------------------------------------------------------

prop_compose! {
    /// A well-posed first batch: a design with an optional trailing intercept
    /// column and targets from a linear model plus bounded noise.
    fn regression_problem()(
        n_features_in in 1_usize..4,
        fit_intercept in any::<bool>(),
        extra_rows in 4_usize..40,
        seed in any::<u64>(),
    ) -> (State<f64>, Vec<f64>, Vec<f64>, usize) {
        let p = n_features_in + usize::from(fit_intercept);
        let n_rows = p + extra_rows;
        let mut x = filled(seed, n_rows * p, -3.0, 3.0);
        if fit_intercept {
            for row in 0..n_rows {
                x[row * p + p - 1] = 1.0;
            }
        }
        let beta = filled(seed ^ 1, p, -2.0, 2.0);
        let noise = filled(seed ^ 2, n_rows, -0.5, 0.5);
        let y = (0..n_rows)
            .map(|row| dot(&x[row * p..(row + 1) * p], &beta) + noise[row])
            .collect();
        (State::empty(n_features_in, fit_intercept), x, y, n_rows)
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

proptest! {
    #![proptest_config(config(32))]

    /// Unit frequency weights are the same arithmetic as no weights at all.
    #[test]
    fn unit_sample_weights_are_bitwise_identical_to_none(
        (state, x, y, n_rows) in regression_problem(),
        penalty in prop_oneof![Just(Penalty::None), Just(Penalty::L1)],
    ) {
        let p = state.n_parameters();
        let config = UpdateConfig { penalty, max_iter: 25, ..UpdateConfig::default() };
        let ones = vec![1.0; n_rows];
        let plain = CpuEngine::<f64>::default()
            .update(batch(&x, &y, None, n_rows, p), &state, config)
            .unwrap();
        let weighted = CpuEngine::<f64>::default()
            .update(batch(&x, &y, Some(&ones), n_rows, p), &state, config)
            .unwrap();
        prop_assert_eq!(bits(&plain.state.coefficients), bits(&weighted.state.coefficients));
        prop_assert_eq!(bits(&plain.state.information), bits(&weighted.state.information));
        prop_assert_eq!(plain.diagnostics.objective.to_bits(), weighted.diagnostics.objective.to_bits());
        prop_assert_eq!(plain.diagnostics.iterations, weighted.diagnostics.iterations);
    }

    /// A reused engine carries nothing between batches but its buffers: after
    /// a larger batch has grown the workspace, the next result is bitwise the
    /// result a fresh engine produces.
    #[test]
    fn workspace_reuse_never_changes_a_result(
        (state, x, y, n_rows) in regression_problem(),
        penalty in prop_oneof![Just(Penalty::None), Just(Penalty::L1)],
    ) {
        let p = state.n_parameters();
        let config = UpdateConfig { penalty, max_iter: 25, ..UpdateConfig::default() };
        let view = batch(&x, &y, None, n_rows, p);
        let fresh = CpuEngine::<f64>::default().update(view, &state, config).unwrap();

        let mut reused = CpuEngine::<f64>::default();
        let big_x = [x.as_slice(), x.as_slice()].concat();
        let big_y = [y.as_slice(), y.as_slice()].concat();
        reused.update(batch(&big_x, &big_y, None, 2 * n_rows, p), &state, config).unwrap();
        let second = reused.update(view, &state, config).unwrap();

        prop_assert_eq!(bits(&second.state.coefficients), bits(&fresh.state.coefficients));
        prop_assert_eq!(bits(&second.state.information), bits(&fresh.state.information));
        prop_assert_eq!(second.diagnostics.objective.to_bits(), fresh.diagnostics.objective.to_bits());
    }

    /// Bookkeeping of one transition, and the unpenalized solver's line search
    /// only ever accepting a non-increasing smooth objective.
    #[test]
    fn unpenalized_update_advances_state_and_never_raises_the_objective(
        (state, x, y, n_rows) in regression_problem(),
    ) {
        let p = state.n_parameters();
        let config = UpdateConfig { max_iter: 25, ..UpdateConfig::default() };
        let view = batch(&x, &y, None, n_rows, p);
        let transition = CpuEngine::<f64>::default().update(view, &state, config).unwrap();

        let next = &transition.state;
        prop_assert_eq!(next.n_samples_seen, state.n_samples_seen + n_rows);
        prop_assert_eq!(next.batch_count, state.batch_count + 1);
        prop_assert_eq!(next.weight_sum, state.weight_sum + n_rows as f64);
        prop_assert_eq!(next.previous_lambda, 0.0);
        prop_assert_eq!(transition.diagnostics.lambda_value, 0.0);
        prop_assert!(next.validate().is_ok());

        let mut workspace = Workspace::<f64>::default();
        workspace.reserve(n_rows, p).unwrap();
        let starting = smooth_objective(
            view,
            &state.coefficients,
            &state,
            config,
            state.weight_sum + n_rows as f64,
            &mut workspace,
        )
        .unwrap();
        prop_assert!(transition.diagnostics.objective <= starting);
    }

    /// With a penalty large enough to zero every penalized coefficient, the
    /// intercept is still fitted: the proximal step never shrinks it.
    #[test]
    fn l1_zeroes_penalized_coefficients_but_never_the_intercept(
        n_features_in in 1_usize..4,
        extra_rows in 4_usize..40,
        location in prop_oneof![1.0..5.0_f64, -5.0..-1.0_f64],
        seed in any::<u64>(),
    ) {
        let p = n_features_in + 1;
        let n_rows = p + extra_rows;
        let mut x = filled(seed, n_rows * p, -3.0, 3.0);
        for row in 0..n_rows {
            x[row * p + p - 1] = 1.0;
        }
        let y = filled(seed ^ 1, n_rows, -0.5, 0.5)
            .into_iter()
            .map(|noise| location + noise)
            .collect::<Vec<_>>();
        let state = State::<f64>::empty(n_features_in, true);
        let config = UpdateConfig {
            penalty: Penalty::L1,
            lambda_scale: 1.0e6,
            max_iter: 50,
            ..UpdateConfig::default()
        };
        let transition = CpuEngine::<f64>::default()
            .update(batch(&x, &y, None, n_rows, p), &state, config)
            .unwrap();
        let coefficients = &transition.state.coefficients;
        prop_assert!(coefficients[..p - 1].iter().all(|value| *value == 0.0), "{coefficients:?}");
        // A product, not `signum`: a thresholded intercept would be a signed
        // zero, and `(-0.0).signum()` is -1.
        prop_assert!(coefficients[p - 1] * location > 0.0, "{coefficients:?}");
    }
}
