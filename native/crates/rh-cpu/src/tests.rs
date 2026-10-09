//! Cross-module tests for the CPU engine.
//!
//! These replay the same golden corpus the Python suite uses and check the
//! parallel paths against their single-threaded equivalents, so they reach
//! into crate-private kernels on purpose.

use serde_json::Value;

use crate::kernels::gram::{weighted_gram, GRAM_ROW_CHUNK};
use crate::kernels::vector::{dot, DOT_LANES};
use crate::scalar::CpuScalar;
use crate::workspace::Workspace;
use crate::{predict, CpuEngine, PARALLEL_GRAM_WORK, PARALLEL_VECTOR_WORK};
use rh_core::{BatchView, CoreError, Diagnostics, Penalty, State, UpdateConfig};

const GOLDEN_CORPUS: &str = include_str!("../../../../tests/golden/native_core_v1.json");

#[test]
fn f64_engine_replays_numpy_golden_corpus() {
    replay_cases::<f64>("float64");
}

#[test]
fn f32_engine_replays_numpy_golden_corpus() {
    replay_cases::<f32>("float32");
}

#[test]
fn rank_deficient_case_uses_minimum_norm_fallback() {
    let corpus: Value = serde_json::from_str(GOLDEN_CORPUS).unwrap();
    let case = corpus["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["id"] == "rank_deficient_lstsq_f64")
        .unwrap();
    let (_, diagnostics) = replay_case::<f64>(case);
    assert!(diagnostics.used_regularized_fallback);
}

#[test]
fn public_gemm_dispatch_rejects_invalid_buffer_shapes() {
    let error = f64::weighted_gram_gemm(&[1.0, 2.0], &[1.0, 2.0], 1, 2, &mut [0.0], false)
        .expect_err("a short output buffer must be rejected before unsafe GEMM");
    assert!(matches!(error, CoreError::InvalidBatch(_)));
}

#[test]
fn parallel_weighted_gram_matches_direct_weighted_sum() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| {
            let n_rows = 4_096;
            let p = 64;
            let x = (0..n_rows * p)
                .map(|index| ((index % 97) as f64 - 48.0) / 37.0)
                .collect::<Vec<_>>();
            let y = vec![0.0; n_rows];
            let curvature = (0..n_rows)
                .map(|row| 0.25 + (row % 11) as f64 / 20.0)
                .collect::<Vec<_>>();
            let sample_weight = (0..n_rows)
                .map(|row| 0.5 + (row % 7) as f64 / 5.0)
                .collect::<Vec<_>>();
            let batch = BatchView {
                x_design: &x,
                n_rows,
                n_parameters: p,
                y: &y,
                sample_weight: Some(&sample_weight),
                batch_weight: sample_weight.iter().sum(),
            };
            let mut workspace = Workspace::<f64>::default();
            workspace.reserve(n_rows, p).unwrap();
            weighted_gram(
                batch,
                &curvature,
                &mut workspace.weighted_rows,
                &mut workspace.partial_grams,
                &mut workspace.gram,
            )
            .unwrap();
            assert!(workspace.partial_grams.len() > p * p);

            for row in 0..p {
                for column in 0..p {
                    let expected = (0..n_rows).fold(0.0, |sum, sample| {
                        let weight = curvature[sample] * sample_weight[sample];
                        sum + x[sample * p + row] * x[sample * p + column] * weight
                    });
                    let actual = workspace.gram[row * p + column];
                    assert!(
                        (actual - expected).abs() <= 1.0e-10 * (1.0 + expected.abs()),
                        "Gram[{row}, {column}] expected {expected}, got {actual}"
                    );
                }
            }
        });
}

#[test]
fn parallel_predict_matches_single_thread_pool() {
    let n_rows = 20_000;
    let n_parameters = 64;
    assert!(n_rows * n_parameters >= PARALLEL_VECTOR_WORK);
    let x = (0..n_rows * n_parameters)
        .map(|index| ((index % 113) as f64 - 56.0) / 31.0)
        .collect::<Vec<_>>();
    let coefficients = (0..n_parameters)
        .map(|index| ((index % 17) as f64 - 8.0) / 19.0)
        .collect::<Vec<_>>();
    let single_thread = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| predict(&x, n_rows, n_parameters, &coefficients).unwrap());
    let multi_thread = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| predict(&x, n_rows, n_parameters, &coefficients).unwrap());
    assert_eq!(multi_thread, single_thread);
}

#[test]
fn persistent_engine_can_grow_and_reuse_its_workspace() {
    let mut engine = CpuEngine::<f64>::default();
    let config = UpdateConfig {
        max_iter: 20,
        ..UpdateConfig::default()
    };
    let mut state = State::empty(1, false);

    let first_x = [1.0, 2.0];
    let first_y = [1.0, 2.0];
    let first = engine
        .update(
            BatchView {
                x_design: &first_x,
                n_rows: 2,
                n_parameters: 1,
                y: &first_y,
                sample_weight: None,
                batch_weight: 2.0,
            },
            &state,
            config,
        )
        .unwrap();
    state = first.state;

    let second_x = [1.0, 2.0, 3.0, 4.0];
    let second_y = [1.0, 2.0, 3.0, 4.0];
    let second = engine
        .update(
            BatchView {
                x_design: &second_x,
                n_rows: 4,
                n_parameters: 1,
                y: &second_y,
                sample_weight: None,
                batch_weight: 4.0,
            },
            &state,
            config,
        )
        .unwrap();
    assert_eq!(second.state.n_samples_seen, 6);
    assert_eq!(second.state.batch_count, 2);
    assert!((second.state.coefficients[0] - 1.0).abs() < 1.0e-7);
}

fn replay_cases<T: CpuScalar>(dtype: &str) {
    let corpus: Value = serde_json::from_str(GOLDEN_CORPUS).unwrap();
    let cases = corpus["cases"].as_array().unwrap();
    let selected = cases
        .iter()
        .filter(|case| case["config"]["dtype"] == dtype)
        .collect::<Vec<_>>();
    assert!(!selected.is_empty());
    for case in selected {
        replay_case::<T>(case);
    }
}

fn replay_case<T: CpuScalar>(case: &Value) -> (State<T>, Diagnostics) {
    let config_value = &case["config"];
    let fit_intercept = config_value["fit_intercept"].as_bool().unwrap();
    let config = UpdateConfig {
        tau: config_value["tau"].as_f64().unwrap(),
        penalty: match config_value["penalty"].as_str().unwrap() {
            "none" => Penalty::None,
            "l1" => Penalty::L1,
            other => panic!("unknown golden penalty {other}"),
        },
        lambda_scale: config_value["lambda_scale"].as_f64().unwrap(),
        bandwidth_scale: config_value["bandwidth_scale"].as_f64().unwrap(),
        max_iter: config_value["max_iter"].as_u64().unwrap() as usize,
        tolerance: config_value["tol"].as_f64().unwrap(),
        ridge: config_value["ridge"].as_f64().unwrap(),
    };
    let first_x = case["batches"][0]["X"].as_array().unwrap();
    let n_features = first_x[0].as_array().unwrap().len();
    let n_parameters = n_features + usize::from(fit_intercept);
    let mut state = State::<T>::empty(n_features, fit_intercept);
    let mut engine = CpuEngine::<T>::default();
    let expected_states = case["expected"]["states"].as_array().unwrap();
    let rtol = case["rtol"].as_f64().unwrap();
    let atol = case["atol"].as_f64().unwrap();
    let mut last_diagnostics = None;

    for (batch_value, expected) in case["batches"]
        .as_array()
        .unwrap()
        .iter()
        .zip(expected_states.iter())
    {
        let rows = batch_value["X"].as_array().unwrap();
        let mut x_design = Vec::with_capacity(rows.len() * n_parameters);
        for row in rows {
            for value in row.as_array().unwrap() {
                x_design.push(from_json::<T>(value));
            }
            if fit_intercept {
                x_design.push(T::one());
            }
        }
        let y = batch_value["y"]
            .as_array()
            .unwrap()
            .iter()
            .map(from_json::<T>)
            .collect::<Vec<_>>();
        let weights = batch_value["sample_weight"]
            .as_array()
            .map(|values| values.iter().map(from_json::<T>).collect::<Vec<_>>());
        let batch_weight = weights.as_ref().map_or(rows.len() as f64, |values| {
            values
                .iter()
                .map(|value| value.to_f64().unwrap())
                .sum::<f64>()
        });
        let transition = engine
            .update(
                BatchView {
                    x_design: &x_design,
                    n_rows: rows.len(),
                    n_parameters,
                    y: &y,
                    sample_weight: weights.as_deref(),
                    batch_weight,
                },
                &state,
                config,
            )
            .unwrap_or_else(|error| panic!("golden case {} failed: {error}", case["id"]));
        assert_vector_close(
            &transition.state.coefficients,
            expected["coefficients"].as_array().unwrap(),
            rtol,
            atol,
            "coefficients",
        );
        let expected_information = expected["information"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().cloned())
            .collect::<Vec<_>>();
        assert_vector_close(
            &transition.state.information,
            &expected_information,
            rtol,
            atol,
            "information",
        );
        assert_eq!(
            transition.state.n_samples_seen,
            expected["n_samples_seen"].as_u64().unwrap() as usize
        );
        assert_eq!(
            transition.state.batch_count,
            expected["batch_count"].as_u64().unwrap() as usize
        );
        assert_close(
            transition.state.previous_lambda,
            expected["previous_lambda"].as_f64().unwrap(),
            rtol,
            atol,
            "previous_lambda",
        );
        assert_close(
            transition.state.weight_sum,
            expected["weight_sum"].as_f64().unwrap(),
            rtol,
            atol,
            "weight_sum",
        );
        let expected_diagnostics = &expected["diagnostics"];
        assert_eq!(
            transition.diagnostics.converged,
            expected_diagnostics["converged"].as_bool().unwrap()
        );
        for (actual, name) in [
            (transition.diagnostics.objective, "objective"),
            (transition.diagnostics.lambda_value, "lambda_value"),
            (transition.diagnostics.bandwidth, "bandwidth"),
        ] {
            assert_close(
                actual,
                expected_diagnostics[name].as_f64().unwrap(),
                rtol,
                atol,
                name,
            );
        }
        last_diagnostics = Some(transition.diagnostics);
        state = transition.state;
    }

    let probe_rows = case["probe_X"].as_array().unwrap();
    let mut probe_design = Vec::with_capacity(probe_rows.len() * n_parameters);
    for row in probe_rows {
        for value in row.as_array().unwrap() {
            probe_design.push(from_json::<T>(value));
        }
        if fit_intercept {
            probe_design.push(T::one());
        }
    }
    let predictions = predict(
        &probe_design,
        probe_rows.len(),
        n_parameters,
        &state.coefficients,
    )
    .unwrap();
    assert_vector_close(
        &predictions,
        case["expected"]["predictions"].as_array().unwrap(),
        rtol,
        atol,
        "predictions",
    );
    (state, last_diagnostics.unwrap())
}

fn from_json<T: CpuScalar>(value: &Value) -> T {
    T::from_f64(value.as_f64().unwrap()).unwrap()
}

fn assert_vector_close<T: CpuScalar>(
    actual: &[T],
    expected: &[Value],
    rtol: f64,
    atol: f64,
    field: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{field} length");
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(
            actual.to_f64().unwrap(),
            expected.as_f64().unwrap(),
            rtol,
            atol,
            &format!("{field}[{index}]"),
        );
    }
}

fn assert_close(actual: f64, expected: f64, rtol: f64, atol: f64, field: &str) {
    let difference = (actual - expected).abs();
    let allowed = atol + rtol * expected.abs();
    assert!(
        difference <= allowed,
        "{field}: actual={actual:?}, expected={expected:?}, difference={difference:?}, \
         allowed={allowed:?}"
    );
}

/// The whole-range product the chunked Gram must reproduce: weight every row,
/// then one GEMM per worker row range, reduced in worker order.
fn reference_gram<T: CpuScalar>(
    x: &[T],
    weights: &[T],
    n_rows: usize,
    p: usize,
    workers: usize,
) -> Vec<T> {
    let weighted = x
        .chunks_exact(p)
        .zip(weights.iter())
        .flat_map(|(row, weight)| row.iter().map(move |value| *value * *weight))
        .collect::<Vec<_>>();
    let mut output = vec![T::zero(); p * p];
    let rows_per_worker = n_rows.div_ceil(workers.max(1));
    let mut partial = vec![T::zero(); p * p];
    for worker in 0..workers.max(1) {
        let start = (worker * rows_per_worker).min(n_rows);
        let end = (start + rows_per_worker).min(n_rows);
        if start == end {
            partial.fill(T::zero());
        } else {
            T::weighted_gram_gemm(
                &x[start * p..end * p],
                &weighted[start * p..end * p],
                end - start,
                p,
                &mut partial,
                false,
            )
            .unwrap();
        }
        if workers <= 1 {
            return partial;
        }
        for (total, value) in output.iter_mut().zip(partial.iter()) {
            *total += *value;
        }
    }
    output
}

fn check_chunked_gram<T: CpuScalar>(n_rows: usize, p: usize, threads: usize) {
    let x = (0..n_rows * p)
        .map(|index| T::from_f64((((index * 7919) % 1009) as f64 - 504.0) / 97.3).unwrap())
        .collect::<Vec<_>>();
    let curvature = (0..n_rows)
        .map(|row| T::from_f64(0.13 + ((row * 31) % 17) as f64 / 7.1).unwrap())
        .collect::<Vec<_>>();
    let y = vec![T::zero(); n_rows];
    let batch = BatchView {
        x_design: &x,
        n_rows,
        n_parameters: p,
        y: &y,
        sample_weight: None,
        batch_weight: n_rows as f64,
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(|| {
            let mut workspace = Workspace::<T>::default();
            workspace.reserve(n_rows, p).unwrap();
            let workers = workspace.partial_grams.len() / (p * p);
            weighted_gram(
                batch,
                &curvature,
                &mut workspace.weighted_rows,
                &mut workspace.partial_grams,
                &mut workspace.gram,
            )
            .unwrap();
            let parallel = workers > 1 && n_rows * p * p >= PARALLEL_GRAM_WORK;
            let expected = reference_gram(
                &x,
                &curvature,
                n_rows,
                p,
                if parallel { workers } else { 1 },
            );
            let bits = |values: &[T]| {
                values
                    .iter()
                    .map(|value| value.to_f64().unwrap().to_bits())
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                bits(&workspace.gram),
                bits(&expected),
                "n_rows={n_rows} p={p} threads={threads}"
            );
        });
}

#[test]
fn chunked_gram_is_bitwise_identical_to_one_gemm() {
    // Serial path: several full chunks plus a ragged tail, and a batch
    // shorter than one chunk.
    for (n_rows, p) in [
        (3 * GRAM_ROW_CHUNK + 77, 13),
        (GRAM_ROW_CHUNK - 5, 9),
        (1, 4),
    ] {
        check_chunked_gram::<f64>(n_rows, p, 1);
        check_chunked_gram::<f32>(n_rows, p, 1);
    }
    // Partitioned path: each worker range is itself chunked.
    let p = 64;
    let n_rows = PARALLEL_GRAM_WORK.div_ceil(p * p) + 333;
    check_chunked_gram::<f64>(n_rows, p, 4);
    check_chunked_gram::<f32>(n_rows, p, 3);
}

#[test]
fn dot_lanes_cover_every_length_and_stop_at_the_shorter_slice() {
    let values = (0..4 * DOT_LANES + 3)
        .map(|index| ((index * 37) % 23) as f64 / 7.0 - 1.5)
        .collect::<Vec<_>>();
    for length in 0..values.len() {
        let left = &values[..length];
        let right = values
            .iter()
            .rev()
            .take(length)
            .copied()
            .collect::<Vec<_>>();
        let expected = left
            .iter()
            .zip(right.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let magnitude = left
            .iter()
            .zip(right.iter())
            .map(|(a, b)| (a * b).abs())
            .sum::<f64>();
        let actual = dot(left, &right);
        assert!(
            (actual - expected).abs() <= 4.0 * f64::EPSILON * (length as f64 + 1.0) * magnitude,
            "length {length}: {actual} vs {expected}"
        );
        assert_eq!(actual.to_bits(), dot(&right, left).to_bits());
        // A longer right-hand side contributes nothing past the left's end.
        assert_eq!(
            actual.to_bits(),
            dot(left, &[right.as_slice(), &[9.0; 5]].concat()).to_bits()
        );
    }
    assert_eq!(dot::<f64>(&[], &[1.0]), 0.0);
}
